"""LingLong (20-DoF) velocity environment configurations.

- 20 controlled joints: 12 leg + 8 arm (no waist).
- Foot bodies are plain ``left/right_ankle_roll_link`` (no ``link_`` prefix).
- IMU site is named ``imu`` in the XML.
"""

import torch

from mjlab.asset_zoo.robots import (
  LINGLONG_ACTION_SCALE,
  get_linglong_robot_cfg,
)
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensor,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains import config as terrain_config
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.noise.noise_cfg import GaussianNoiseCfg, UniformNoiseCfg

# --------------------------------------------------------------------------- #
# Reward helper functions
# --------------------------------------------------------------------------- #


def body_height_reward(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float = 0.1,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward for maintaining a target body height."""
  asset: Entity = env.scene[asset_cfg.name]
  root_z = asset.data.body_link_pos_w[:, 0, 2]
  error = root_z - target_height
  return torch.exp(-error.pow(2) / std**2)


def foot_flat_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  std: float = 0.3,
  std_stance: float = 0.15,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward keeping foot soles flat in ALL phases (world frame)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]

  body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]

  B, N, _ = body_quat_w.shape
  quat_flat = body_quat_w.reshape(B * N, 4)

  up_local = torch.tensor(
    [0.0, 0.0, 1.0], device=env.device, dtype=torch.float32
  ).expand(B * N, 3)
  foot_normal_w = quat_apply(quat_flat, up_local).reshape(B, N, 3)

  # tilt² = sin²(θ) = 1 - cos²(θ) = 1 - normal_z²
  tilt_sq = 1.0 - torch.square(foot_normal_w[:, :, 2])

  assert contact_sensor.data.found is not None
  in_air = (contact_sensor.data.found == 0).float()
  on_ground = 1.0 - in_air

  swing_reward = torch.exp(-tilt_sq / std**2) * in_air
  stance_reward = torch.exp(-tilt_sq / std_stance**2) * on_ground

  reward = torch.sum(swing_reward + stance_reward, dim=1)
  num_feet = torch.clamp(torch.sum(in_air + on_ground, dim=1), min=1.0)
  return reward / num_feet


def contact_force_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_force: float = 400.0,
) -> torch.Tensor:
  """Penalize contact force exceeding threshold during stance."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.force is not None
  assert contact_sensor.data.found is not None

  force_mag = torch.norm(contact_sensor.data.force, dim=-1)  # [B, N]
  on_ground = (contact_sensor.data.found > 0).float()
  excess = torch.relu(force_mag - max_force)
  return torch.sum(excess * on_ground, dim=1)


def yaw_roll_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize hip yaw/roll joints deviating from default using L2 squared."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  diff = joint_pos - default_pos
  return torch.sum(torch.square(diff), dim=1)


def joint_power_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize joint mechanical power (torque × velocity)."""
  asset: Entity = env.scene[asset_cfg.name]
  torque = asset.data.actuator_force[:, asset_cfg.actuator_ids]
  vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return torch.sum(torch.abs(torque * vel), dim=1)


def limb_distance_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  min_dist: float = 0.13,
  max_dist: float = 0.50,
) -> torch.Tensor:
  """Reward maintaining limb pair XY distance within [min_dist, max_dist]."""
  asset: Entity = env.scene[asset_cfg.name]
  pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
  diff = pos[:, 0, :2] - pos[:, 1, :2]
  dist = torch.norm(diff, dim=1)
  too_close = torch.clamp(min_dist - dist, min=0.0)
  too_far = torch.clamp(dist - max_dist, min=0.0)
  penalty = too_close + too_far
  return torch.exp(-penalty * 50.0)


def stand_still_penalty(
  env: ManagerBasedRlEnv,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize joint deviation from default when velocity command is zero."""
  asset: Entity = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command("twist")
  assert cmd is not None
  cmd_mag = torch.norm(cmd, dim=1)
  standing = (cmd_mag < command_threshold).float()

  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  deviation = torch.sum(torch.square(joint_pos - default_pos), dim=1)
  return deviation * standing


def feet_contact_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward both feet on ground when standing."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  both_in_contact = sensor.data.found.float()
  num_contact = torch.sum(both_in_contact, dim=1)

  cmd = env.command_manager.get_command("twist")
  assert cmd is not None
  cmd_mag = torch.norm(cmd, dim=1)
  standing = (cmd_mag < command_threshold).float()
  return (num_contact / 2.0) * standing


def gait_phase_signal(
  env: ManagerBasedRlEnv,
  freq: float = 1.0,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Fixed-frequency gait phase clock: sin(2*pi*phi)."""
  phase = getattr(env, "_gait_phase", None)
  if phase is None:
    phase = torch.zeros(env.num_envs, device=env.device)

  cmd = env.command_manager.get_command("twist")
  assert cmd is not None
  cmd_mag = torch.norm(cmd, dim=1)
  moving = (cmd_mag > command_threshold).float()
  phase += moving * freq * env.step_dt
  phase = torch.remainder(phase, 1.0)
  setattr(env, "_gait_phase", phase)

  return torch.sin(2 * torch.pi * phase).unsqueeze(-1)


def gait_phase_tracking(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  freq: float = 1.0,
  duty_ratio: float = 0.5,
  phase_offset: float = 0.5,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward for tracking desired alternating gait contact pattern."""
  phase = getattr(env, "_gait_phase", None)
  if phase is None:
    return torch.zeros(env.num_envs, device=env.device)

  contact_sensor = env.scene.sensors[sensor_name]
  foot_contact = contact_sensor.data.found

  left_desired = (phase < duty_ratio).float()
  right_phase = torch.remainder(phase + phase_offset, 1.0)
  right_desired = (right_phase < duty_ratio).float()
  desired_contact = torch.stack([left_desired, right_desired], dim=1)

  match = (foot_contact == desired_contact).float()
  reward = match.mean(dim=1)

  cmd = env.command_manager.get_command("twist")
  assert cmd is not None
  cmd_mag = torch.norm(cmd, dim=1)
  moving = (cmd_mag > command_threshold).float()
  return reward * moving


def step_symmetry_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward symmetric left-right step timing."""
  sensor: ContactSensor = env.scene[sensor_name]
  last_air = sensor.data.last_air_time
  assert last_air is not None
  left_at = last_air[:, 0]
  right_at = last_air[:, 1]

  has_walked = (left_at > 0.01).float() * (right_at > 0.01).float()

  diff = torch.abs(left_at - right_at)
  reward = torch.exp(-diff / 0.1)

  cmd = env.command_manager.get_command("twist")
  assert cmd is not None
  cmd_mag = torch.norm(cmd, dim=1)
  alpha = torch.sigmoid((cmd_mag - command_threshold) * 20.0)

  return reward * has_walked * alpha


def step_length_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str = "twist",
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward forward foot swing velocity to encourage larger step length."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]

  foot_vel_w = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :]
  base_vel_w = asset.data.root_link_lin_vel_w.unsqueeze(1)
  rel_vel_w = foot_vel_w - base_vel_w

  base_quat_w = asset.data.root_link_quat_w  # [B, 4]
  B, N, _ = rel_vel_w.shape
  rel_vel_b = quat_apply_inverse(
    base_quat_w.unsqueeze(1).expand(B, N, 4).reshape(B * N, 4),
    rel_vel_w.reshape(B * N, 3),
  ).reshape(B, N, 3)

  foot_vx_rel = rel_vel_b[:, :, 0]  # [B, N]

  assert contact_sensor.data.found is not None
  in_air = (contact_sensor.data.found == 0).float()  # [B, N]

  forward_vel = torch.clamp(foot_vx_rel, min=0.0)
  num_in_air = torch.clamp(in_air.sum(dim=1), min=1.0)
  reward = (forward_vel * in_air).sum(dim=1) / num_in_air

  cmd = env.command_manager.get_command(command_name)
  assert cmd is not None
  cmd_mag = torch.norm(cmd, dim=1)
  alpha = torch.sigmoid((cmd_mag - command_threshold) * 20.0)

  return reward * alpha


# --------------------------------------------------------------------------- #
# Leg / body name lists
# --------------------------------------------------------------------------- #

# Joints controlled by the locomotion policy.
LEG_JOINTS = (
  "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
  "left_knee", "left_ankle_pitch", "left_ankle_roll",
  "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
  "right_knee", "right_ankle_pitch", "right_ankle_roll",
)
ARM_JOINTS = (
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint", "left_elbow_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint", "right_elbow_joint",
)
HIP_YAW_ROLL_JOINTS = (
  "left_hip_yaw", "left_hip_roll",
  "right_hip_yaw", "right_hip_roll",
)

# Foot sites: ankle_roll bodies are the lowest limb segment.
FOOT_SITE_NAMES = ("left_foot", "right_foot")
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
KNEE_BODY_NAMES = ("left_knee_link", "right_knee_link")


# --------------------------------------------------------------------------- #
# Environment config builder
# --------------------------------------------------------------------------- #


def linglong_flat_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
  """Create LingLong 20-DoF velocity training configuration.

  Args:
    play: If True, configure for playback/visualization.
    rough: If True, keep rough terrain + height scan; otherwise flat plane.
  """
  cfg = make_velocity_env_cfg()

  # Simulation.
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.mujoco.impratio = 3.0
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.njmax = 500

  # Robot.
  cfg.scene.entities = {"robot": get_linglong_robot_cfg()}

  # mjlab 1.6: foot clearance uses TerrainHeightSensor, not site z.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITE_NAMES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  # Foot contact sensor — ankle_roll bodies are the foot contact bodies.
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg,)

  # Actions — all 20 joints.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = LINGLONG_ACTION_SCALE

  # Viewer.
  cfg.viewer.body_name = "base_link"

  # Commands.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.0
  twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
  twist_cmd.ranges.lin_vel_y = (-0.3, 0.3)
  twist_cmd.ranges.ang_vel_z = (-0.8, 0.8)
  twist_cmd.rel_standing_envs = 0.3

  # Curriculum.
  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=mdp.commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {"step": 0,          "lin_vel_x": (-0.3, 0.5), "ang_vel_z": (-0.3, 0.3)},
        {"step": 2500 * 24,  "lin_vel_x": (-0.5, 1.0), "ang_vel_z": (-0.8, 0.8)},
      ],
    },
  )

  # Observations — no named foot sites in XML beyond the ones added by spec.
  if "foot_height" in cfg.observations["critic"].terms:
    del cfg.observations["critic"].terms["foot_height"]

  # Remove base_lin_vel from actor.
  del cfg.observations["actor"].terms["base_lin_vel"]

  # Add gait phase signal.
  cfg.observations["actor"].terms["gait_phase"] = ObservationTermCfg(
    func=gait_phase_signal,
    params={"freq": 1.0, "command_threshold": 0.1},
  )

  # Observation scales.
  cfg.observations["actor"].terms["base_ang_vel"].scale = 0.25
  cfg.observations["actor"].terms["joint_pos"].scale = 1.0
  cfg.observations["actor"].terms["joint_vel"].scale = 0.05
  cfg.observations["critic"].terms["base_lin_vel"].scale = 2.0
  cfg.observations["critic"].terms["base_ang_vel"].scale = 0.25
  cfg.observations["critic"].terms["joint_pos"].scale = 1.0
  cfg.observations["critic"].terms["joint_vel"].scale = 0.05

  # Actor observation order.
  actor_terms = cfg.observations["actor"].terms
  actor_order = [
    "base_ang_vel", "projected_gravity", "command",
    "joint_pos", "joint_vel", "actions", "gait_phase",
  ]
  cfg.observations["actor"].terms = {k: actor_terms[k] for k in actor_order}

  # History.
  cfg.observations["actor"].history_length = 10

  # Observation delay (simulate IMU + encoder latency).
  cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
  cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 3
  cfg.observations["actor"].terms["projected_gravity"].delay_min_lag = 0
  cfg.observations["actor"].terms["projected_gravity"].delay_max_lag = 3
  cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 0
  cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1

  # IMU noise.
  cfg.observations["actor"].terms["projected_gravity"].noise = UniformNoiseCfg(
    n_min=-0.05, n_max=0.05,
  )
  cfg.observations["actor"].terms["base_ang_vel"].noise = GaussianNoiseCfg(
    mean=0.0, std=0.02,
  )

  # Critic observation order.
  critic_terms = cfg.observations["critic"].terms
  privileged = ["base_lin_vel"]
  critic_order = (
    privileged
    + [k for k in actor_order if k in critic_terms]
    + [k for k in critic_terms if k not in privileged and k not in actor_order]
  )
  cfg.observations["critic"].terms = {k: critic_terms[k] for k in critic_order}

  # Events — foot friction (URDF sole cylinders).
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 6)
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  cfg.events["base_com"].params["ranges"] = {
      0: (-0.02, 0.02),
      1: (-0.02, 0.02),
      2: (-0.02, 0.02),
    }

  # Encoder bias — joint encoder reading offset (±0.05 rad ≈ ±2.9°).
  cfg.events["encoder_bias"].params["bias_range"] = (-0.05, 0.05)

  # Physics-consistent inertia randomization.
  cfg.events["pseudo_inertia"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
      "alpha_range": (-0.20, 0.20),
      "d_range": (-0.02, 0.02),
      "t_range": (-0.005, 0.005),
    },
  )

  # Actuator delay is on BuiltinPositionActuatorCfg (delay_min/max_lag) in 1.6.

  # Payload mass randomization on base_link (general mass variation).
  cfg.events["payload_mass"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      "operation": "add",
      "ranges": (-3.0, 5.0),  # -3 to +5 kg payload variation
    },
  )

  # Terminations.
  cfg.terminations["root_height"] = TerminationTermCfg(
    func=envs_mdp.root_height_below_minimum,
    params={"minimum_height": -2.0},
  )
  cfg.terminations["nan_detection"] = TerminationTermCfg(
    func=envs_mdp.nan_detection,
  )

  # ------------------------------------------------------------------ #
  # Rewards
  # ------------------------------------------------------------------ #

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02

  # Pose constraint: speed-dependent joint tolerance.
  cfg.rewards["pose"].params["std_standing"] = {
    # Hip yaw/roll: tight
    "left_hip_yaw":      0.03,
    "left_hip_roll":     0.03,
    "right_hip_yaw":     0.03,
    "right_hip_roll":    0.03,
    # Hip pitch, knee, ankle: tight when standing
    "left_hip_pitch":    0.08,
    "left_knee":         0.08,
    "left_ankle_pitch":  0.08,
    "left_ankle_roll":   0.03,
    "right_hip_pitch":   0.08,
    "right_knee":        0.08,
    "right_ankle_pitch": 0.08,
    "right_ankle_roll":  0.03,
    # Arms: loose when standing
    "left_shoulder_pitch_joint":  0.15,
    "left_shoulder_roll_joint":   0.15,
    "left_shoulder_yaw_joint":    0.15,
    "left_elbow_joint":           0.15,
    "right_shoulder_pitch_joint": 0.15,
    "right_shoulder_roll_joint":  0.15,
    "right_shoulder_yaw_joint":   0.15,
    "right_elbow_joint":          0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    "left_hip_yaw":      0.3,
    "left_hip_roll":     0.3,
    "right_hip_yaw":     0.3,
    "right_hip_roll":    0.3,
    "left_hip_pitch":    0.48,
    "left_knee":         0.95,
    "left_ankle_pitch":  0.48,
    "left_ankle_roll":   0.50,
    "right_hip_pitch":   0.48,
    "right_knee":        0.95,
    "right_ankle_pitch": 0.48,
    "right_ankle_roll":  0.50,
    # Arms: free to swing while walking
    "left_shoulder_pitch_joint":  0.6,
    "left_shoulder_roll_joint":   0.6,
    "left_shoulder_yaw_joint":    0.6,
    "left_elbow_joint":           0.6,
    "right_shoulder_pitch_joint": 0.6,
    "right_shoulder_roll_joint":  0.6,
    "right_shoulder_yaw_joint":   0.6,
    "right_elbow_joint":          0.6,
  }
  cfg.rewards["pose"].params["std_running"] = cfg.rewards["pose"].params["std_walking"]

  # Energy penalties — scaled for ~30kg robot.
  cfg.rewards["torques"] = RewardTermCfg(
    func=envs_mdp.joint_torques_l2, weight=-5.0e-5,
  )
  cfg.rewards["dof_vel"] = RewardTermCfg(
    func=envs_mdp.joint_vel_l2, weight=-1.0e-7,
  )
  cfg.rewards["dof_acc"] = RewardTermCfg(
    func=envs_mdp.joint_acc_l2, weight=-1.0e-7,
  )
  cfg.rewards["joint_power"] = RewardTermCfg(
    func=joint_power_l2, weight=-2.0e-6,
  )
  cfg.rewards["action_smoothness"] = RewardTermCfg(
    func=envs_mdp.action_acc_l2, weight=-0.03,
  )
  cfg.rewards["action_rate_l2"] = RewardTermCfg(
    func=envs_mdp.action_rate_l2, weight=-0.1,
  )

  # Body height: ~0.95m at default pose (XML spawns base at 0.99).
  cfg.rewards["body_height"] = RewardTermCfg(
    func=body_height_reward,
    weight=2.0,
    params={"target_height": 0.95, "std": 0.04},
  )

  # Foot flat reward.
  cfg.rewards["foot_flat"] = RewardTermCfg(
    func=foot_flat_reward,
    weight=0.5,
    params={
      "sensor_name": "feet_ground_contact",
      "std": 0.3,
      "std_stance": 0.15,
      "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
    },
  )

  # Foot clearance / slip need site names.
  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = FOOT_SITE_NAMES
  cfg.rewards["foot_clearance"].params["target_height"] = 0.05
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.05

  # Contact force constraint.
  cfg.rewards["soft_landing"] = RewardTermCfg(
    func=mdp.soft_landing,
    weight=-1.0e-3,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.1,
    },
  )
  cfg.rewards["contact_force"] = RewardTermCfg(
    func=contact_force_penalty,
    weight=-1.0e-3,
    params={
      "sensor_name": "feet_ground_contact",
      "max_force": 400.0,
    },
  )

  # Hip yaw/roll penalty (L2 squared).
  cfg.rewards["yaw_roll_penalty"] = RewardTermCfg(
    func=yaw_roll_penalty,
    weight=-1.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=HIP_YAW_ROLL_JOINTS),
    },
  )

  # Lateral velocity tracking.
  cfg.rewards["track_lateral_vel"] = RewardTermCfg(
    func=mdp.track_lateral_velocity,
    weight=1.0,
    params={"command_name": "twist", "std": 0.15},
  )

  # Stand still penalty.
  cfg.rewards["stand_still"] = RewardTermCfg(
    func=stand_still_penalty,
    weight=-0.3,
    params={"command_threshold": 0.1},
  )

  # Feet contact when standing.
  cfg.rewards["feet_contact"] = RewardTermCfg(
    func=feet_contact_penalty,
    weight=0.2,
    params={
      "sensor_name": "feet_ground_contact",
      "command_threshold": 0.1,
    },
  )

  # Feet XY distance.
  cfg.rewards["feet_distance"] = RewardTermCfg(
    func=limb_distance_reward,
    weight=0.2,
    params={
      "min_dist": 0.08,
      "max_dist": 0.70,
      "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
    },
  )

  # Knee distance.
  cfg.rewards["knee_distance"] = RewardTermCfg(
    func=limb_distance_reward,
    weight=0.2,
    params={
      "min_dist": 0.1,
      "max_dist": 0.50,
      "asset_cfg": SceneEntityCfg("robot", body_names=KNEE_BODY_NAMES),
    },
  )

  # Gait phase tracking.
  cfg.rewards["gait_phase_tracking"] = RewardTermCfg(
    func=gait_phase_tracking,
    weight=2.0,
    params={
      "sensor_name": "feet_ground_contact",
      "freq": 1.0,
      "duty_ratio": 0.5,
      "phase_offset": 0.5,
      "command_threshold": 0.1,
    },
  )

  # Step symmetry.
  cfg.rewards["step_symmetry"] = RewardTermCfg(
    func=step_symmetry_reward,
    weight=0.5,
    params={
      "sensor_name": "feet_ground_contact",
      "command_threshold": 0.1,
    },
  )

  # Step length.
  cfg.rewards["step_length"] = RewardTermCfg(
    func=step_length_reward,
    weight=0.5,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
    },
  )

  # Air time.
  cfg.rewards["air_time"].weight = 1.0
  cfg.rewards["air_time"].params["threshold_min"] = 0.05
  cfg.rewards["air_time"].params["threshold_max"] = 0.15
  cfg.rewards["air_time"].params["command_threshold"] = 0.1

  # Upright / body references.
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

  # ------------------------------------------------------------------ #
  # Terrain
  # ------------------------------------------------------------------ #

  assert cfg.scene.terrain is not None
  if rough:
    import mjlab.terrains as tg
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
      size=(8.0, 8.0),
      border_width=20.0,
      num_rows=10,
      num_cols=20,
      sub_terrains={
        "flat": tg.BoxFlatTerrainCfg(proportion=0.2),
        "pyramid_stairs": tg.BoxPyramidStairsTerrainCfg(
          proportion=0.2,
          step_height_range=(0.0, 0.06),
          step_width=0.4,
          platform_width=3.0,
          border_width=1.0,
        ),
        "pyramid_stairs_inv": tg.BoxInvertedPyramidStairsTerrainCfg(
          proportion=0.2,
          step_height_range=(0.0, 0.06),
          step_width=0.4,
          platform_width=3.0,
          border_width=1.0,
        ),
        "pyramid_slope": tg.HfPyramidSlopedTerrainCfg(
          proportion=0.15,
          slope_range=(0.0, 0.04),
          platform_width=2.0,
          border_width=0.25,
        ),
        "pyramid_slope_inv": tg.HfPyramidSlopedTerrainCfg(
          proportion=0.15,
          slope_range=(0.0, 0.04),
          platform_width=2.0,
          border_width=0.25,
          inverted=True,
        ),
        "random_grid": tg.BoxRandomGridTerrainCfg(
          proportion=0.1,
          grid_width=0.2,
          grid_height_range=(0.01, 0.05),
          platform_width=2.0,
          border_width=0.25,
        ),
      },
      add_lights=True,
    )
    for sensor in cfg.scene.sensors or ():
      if sensor.name == "terrain_scan":
        assert isinstance(sensor, RayCastSensorCfg)
        sensor.frame.name = "base_link"
    if cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = True
    cfg.scene.terrain.max_init_terrain_level = 0
  else:
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.sensors = tuple(
      s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )
    if "height_scan" in cfg.observations["actor"].terms:
      del cfg.observations["actor"].terms["height_scan"]
    if "height_scan" in cfg.observations["critic"].terms:
      del cfg.observations["critic"].terms["height_scan"]
    cfg.curriculum.pop("terrain_levels", None)

  # ------------------------------------------------------------------ #
  # Play mode
  # ------------------------------------------------------------------ #

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    if rough:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0
    else:
      cfg.events["randomize_terrain"] = EventTermCfg(
        func=envs_mdp.randomize_terrain,
        mode="reset",
        params={},
      )
    twist_cmd.ranges.lin_vel_x = (-0.5, 0.8)
    twist_cmd.ranges.ang_vel_z = (-0.8, 0.8)
    twist_cmd.rel_standing_envs = 0.5

  return cfg


def linglong_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create LingLong 20-DoF rough terrain velocity configuration."""
  return linglong_flat_env_cfg(play=play, rough=True)
