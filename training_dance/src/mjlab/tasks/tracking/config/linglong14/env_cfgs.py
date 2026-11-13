"""Linglong 1.4 simulation dance training with fixed nominal physics."""

from pathlib import Path

from mjlab.asset_zoo.robots.linglong14.linglong14_constants import (
  LINGLONG14_ACTION_SCALE, LINGLONG14_TRACKED_BODIES, get_linglong14_robot_cfg,
)
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from . import mdp as local_mdp

LINGLONG14_REFERENCE_FRAME_DIM = 35
LINGLONG14_PROPRIO_FRAME_DIM = 84
LINGLONG14_ACTOR_OBS_DIM = 21 * 35 + 35 + 10 * 84
DEFAULT_MOTION = Path(__file__).resolve().parents[4] / "asset_zoo/motions/linglong14/xiao_speed075_modified.npz"


def linglong14_xiao_env_cfg(play=False):
  cfg = make_tracking_env_cfg()
  cfg.scene.num_envs = 1024
  cfg.scene.entities = {"robot": get_linglong14_robot_cfg(play=play)}
  cfg.scene.sensors = tuple(
    ContactSensorCfg(name=name, primary=ContactMatch(mode="body", pattern=".*", entity="robot", exclude=exclude),
                     fields=("found", "force"), reduce="netforce", history_length=3)
    for name, exclude in (
      ("illegal_contact", ("left_ankle_roll_link", "right_ankle_roll_link")),
      ("undesired_contact", ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")),
    )
  )
  cfg.actions["joint_pos"] = mdp.MotionReferenceJointPositionActionCfg(
    entity_name="robot", actuator_names=(".*",), scale=LINGLONG14_ACTION_SCALE,
    use_default_offset=False, command_name="motion",
  )
  cfg.commands["motion"] = local_mdp.Linglong14MotionCommandCfg(
    entity_name="robot", motion_file=str(DEFAULT_MOTION), anchor_body_name="base_link",
    body_names=LINGLONG14_TRACKED_BODIES, pose_range={},
    velocity_range={}, joint_position_range=(0., 0.),
    resampling_time_range=(1e9, 1e9), debug_vis=False, sampling_mode="uniform",
  )
  # The original actor has no measured global position or base linear velocity.
  cfg.observations["actor"] = ObservationGroupCfg(
    terms={
      "local_ref_window": ObservationTermCfg(func=mdp.centered_motion_reference,
        params={"command_name": "motion", "num_past_frames": 10, "num_future_frames": 10, "step_size": 1}),
      "local_ref_current": ObservationTermCfg(func=mdp.current_motion_reference, params={"command_name": "motion"}),
      "proprio_history": ObservationTermCfg(func=mdp.proprioception, history_length=10, flatten_history_dim=True),
    }, concatenate_terms=True, enable_corruption=False,
  )
  critic = cfg.observations["critic"].terms
  critic["command"].func = local_mdp.critic_motion_command
  critic["motion_anchor_ori_b"].func = local_mdp.anchor_ori_without_yaw
  # Dict order reproduces the original privileged observation ordering.
  cfg.observations["critic"].terms = {
    **{k: critic[k] for k in ("command", "motion_anchor_pos_b", "motion_anchor_ori_b", "body_pos", "body_ori", "base_lin_vel")},
    "base_height": ObservationTermCfg(func=local_mdp.robot_base_height),
    "base_ang_vel": critic["base_ang_vel"],
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    **{k: critic[k] for k in ("joint_pos", "joint_vel", "actions")},
  }
  cfg.rewards["motion_global_root_pos"].params["std"] = .5
  cfg.rewards["motion_global_root_ori"].params["std"] = .5
  cfg.rewards["motion_body_pos"].func = local_mdp.current_relative_body_position_reward
  cfg.rewards["motion_body_ori"].func = local_mdp.current_relative_body_orientation_reward
  cfg.rewards["motion_body_pos"].params["std"] = 1.
  cfg.rewards["motion_body_ori"].params["std"] = 1.
  cfg.rewards["joint_limit"].weight = -1.
  cfg.rewards["action_rate_l2"] = RewardTermCfg(func=local_mdp.reference_target_rate, weight=-.08)
  cfg.rewards.pop("self_collisions")
  cfg.rewards["undesired_contacts"] = RewardTermCfg(func=local_mdp.contact_penalty, weight=-1., params={"sensor_name": "undesired_contact", "threshold": 1.})
  feet = ("left_ankle_roll_link", "right_ankle_roll_link")
  knees = ("left_knee_link", "right_knee_link")
  legs = ("left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link")
  for name, weight, std, bodies in (("legs", 1., .45, legs), ("feet", 1.5, .3, feet), ("knees", 1.2, .35, knees)):
    cfg.rewards[f"motion_body_pos_{name}"] = RewardTermCfg(func=local_mdp.current_relative_body_position_reward, weight=weight,
      params={"command_name": "motion", "std": std, "body_names": bodies})
  cfg.rewards["motion_body_height_feet"] = RewardTermCfg(func=mdp.motion_relative_body_height_error_exp, weight=1.2,
    params={"command_name": "motion", "std": .06, "body_names": feet})
  cfg.rewards["motion_knee_width"] = RewardTermCfg(func=local_mdp.knee_separation, weight=.8)
  cfg.rewards["motion_body_lin_vel_feet"] = RewardTermCfg(func=mdp.motion_global_body_linear_velocity_error_exp, weight=.4,
    params={"command_name": "motion", "std": .75, "body_names": feet})
  cfg.rewards["zero_base_xy_vel"] = RewardTermCfg(func=local_mdp.zero_base_xy_velocity, weight=-.3)

  cfg.events = {}
  cfg.terminations.pop("ee_body_pos")
  cfg.terminations["anchor_pos"] = TerminationTermCfg(func=local_mdp.anchor_height_termination)
  cfg.terminations["anchor_ori"] = TerminationTermCfg(func=local_mdp.anchor_tilt_termination)
  cfg.terminations["hand_body_pos"] = TerminationTermCfg(func=local_mdp.body_height_termination,
    params={"body_names": ("left_wrist_yaw_link", "right_wrist_yaw_link")})
  cfg.terminations["feet_body_pos"] = TerminationTermCfg(func=local_mdp.body_height_termination,
    params={"body_names": feet})
  cfg.terminations["illegal_body_contact"] = TerminationTermCfg(func=local_mdp.illegal_contact)
  cfg.viewer.body_name, cfg.viewer.distance = "base_link", 3.5
  cfg.sim.nconmax, cfg.sim.njmax = 128, 1500
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.mujoco.timestep, cfg.decimation = .005, 4
  if play:
    cfg.events.clear()
    cfg.episode_length_s = 1e9
    command = cfg.commands["motion"]
    command.sampling_mode = "start"
    command.pose_range = command.velocity_range = {}
    command.joint_position_range = (0., 0.)
  return cfg
