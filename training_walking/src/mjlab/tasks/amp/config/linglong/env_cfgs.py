"""LingLong 20-DoF AMP environment configuration.

Builds on the LingLong velocity task and adds the AMP machinery: style
reward mixing, contact/termination terms, a stand-stage curriculum and a
light domain-randomization set.
"""

from dataclasses import replace

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.amp import mdp as amp_mdp
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.config.linglong.env_cfgs import (
  FOOT_SITE_NAMES,
  linglong_flat_env_cfg,
)
from mjlab.terrains.config import (
  hf_pyramid_slope,
  hf_pyramid_slope_inv,
  random_rough,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise.noise_cfg import UniformNoiseCfg

FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
KNEE_BODY_NAMES = ("left_knee_link", "right_knee_link")
FOOT_CONTACT_PATTERN = r"^(left_ankle_roll_link|right_ankle_roll_link)$"
# LingLong hip_yaw has no collision primitive; base_link's analog is the torso box.
ILLEGAL_GROUND_GEOMS = ("torso_collision",)
# Penalize ground contact on hip_pitch + arms. Knees/feet stay legal.
PENALIZE_GROUND_GEOMS = (
  "left_hip_pitch_collision",
  "right_hip_pitch_collision",
  "left_shoulder_pitch_collision",
  "right_shoulder_pitch_collision",
  "left_shoulder_roll_collision",
  "right_shoulder_roll_collision",
  "left_shoulder_yaw_collision",
  "right_shoulder_yaw_collision",
  "left_elbow_collision",
  "right_elbow_collision",
)

# Standing task stage: gate lerp=1 + stand_still -1. Pose skip is on from iter 0.
# At the same iter, step_amplitude_symmetry turns on (pure fwd/back only).
AMP_STAND_SWITCH_ITER = 7500
AMP_STEPS_PER_ITER = 24
AMP_STAND_SWITCH_STEP = AMP_STAND_SWITCH_ITER * AMP_STEPS_PER_ITER
# Yaw command: warmup ±0.3 (inherited), then ±0.6 (stays; rotate is covered by style).
AMP_YAW_STAGE1 = (-0.6, 0.6)
# Expert walk motion is ~1 m/s; keep vx cap at 1.0.
AMP_VX_PLAY = (-0.5, 1.0)

# AMP Rough only. Does not change the default ROUGH_TERRAINS_CFG.
AMP_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  sub_terrains={
    # Not BoxFlat: one 8 m box vs the 26 cm sole cylinders only makes ~1
    # contact per pad, so feet sink and stumble. A 2 mm heightfield looks
    # flat but matches the other patches' dense contacts.
    "flat": random_rough(
      proportion=0.3,
      noise_range=(0.0, 0.002),
      noise_step=0.001,
      horizontal_scale=0.1,
      vertical_scale=0.001,
    ),
    "hf_pyramid_slope": hf_pyramid_slope(proportion=0.1, slope_range=(0.0, 0.05)),
    "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
      proportion=0.1, slope_range=(0.0, 0.05)
    ),
    "random_rough": random_rough(
      proportion=0.5,
      noise_range=(0.01, 0.025),
      noise_step=0.005,
      horizontal_scale=0.1,
    ),
  },
  add_lights=True,
)


def _set_amp_yaw_curriculum(cfg: ManagerBasedRlEnvCfg) -> None:
  """AMP yaw caps at ±0.6; vx stays at the base walking curriculum (max 1.0)."""
  twist_cmd = cfg.commands["twist"]
  command_vel = cfg.curriculum.get("command_vel") if cfg.curriculum else None
  if command_vel is None:
    twist_cmd.ranges.ang_vel_z = AMP_YAW_STAGE1
    twist_cmd.ranges.lin_vel_x = AMP_VX_PLAY
    return
  twist_cmd.ranges.ang_vel_z = AMP_YAW_STAGE1
  stages = command_vel.params["velocity_stages"]
  for stage in stages:
    if "ang_vel_z" in stage:
      lo, hi = stage["ang_vel_z"]
      if max(abs(lo), abs(hi)) > 0.3:
        stage["ang_vel_z"] = AMP_YAW_STAGE1


def amp_stand_stage_curriculum(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor
) -> dict[str, torch.Tensor]:
  """Keep pose skip on; stand stage raises stand_still and step amplitude."""
  del env_ids
  on = env.common_step_counter >= AMP_STAND_SWITCH_STEP
  amp_mdp.apply_amp_stand_stage_weights(env, on=on)
  flag = torch.tensor(float(on), device=env.device)
  return {"stand_stage": flag, "step_amplitude": flag}


def _install_amp_stand_stage(cfg: ManagerBasedRlEnvCfg) -> None:
  if not cfg.curriculum:
    return
  cfg.curriculum["amp_stand_stage"] = CurriculumTermCfg(func=amp_stand_stage_curriculum)


def _apply_amp_dr(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
  """Light domain-randomization set + actor observation noise."""
  del play
  robot_geoms = SceneEntityCfg("robot", geom_names=(".*",))
  robot_base = SceneEntityCfg("robot", body_names=("base_link",))

  cfg.events["foot_friction"].params["asset_cfg"] = robot_geoms
  cfg.events["foot_friction"].params["ranges"] = (0.25, 1.75)
  cfg.events["foot_friction"].params["shared_random"] = True
  cfg.events["foot_friction"].params["operation"] = "abs"

  cfg.events["base_com"].params["asset_cfg"] = robot_base
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.03, 0.03),
    1: (-0.03, 0.03),
    2: (-0.03, 0.03),
  }
  cfg.events["payload_mass"].params["ranges"] = (-1.0, 1.0)
  cfg.events["pseudo_inertia"].params["alpha_range"] = amp_mdp.inertia_alpha_range()

  if "encoder_bias" in cfg.events:
    cfg.events["encoder_bias"].params["bias_range"] = (-0.035, 0.035)

  if "push_robot" in cfg.events:
    cfg.events["push_robot"].func = amp_mdp.push_robots_gym
    cfg.events["push_robot"].interval_range_s = (5.0, 5.0)
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-0.3, 0.3),
      "y": (-0.3, 0.3),
    }

  actor = cfg.observations["actor"].terms
  actor["base_ang_vel"].noise = UniformNoiseCfg(n_min=-0.2, n_max=0.2)
  actor["joint_pos"].noise = UniformNoiseCfg(n_min=-0.01, n_max=0.01)
  actor["joint_vel"].noise = UniformNoiseCfg(n_min=-1.5, n_max=1.5)
  actor["projected_gravity"].noise = UniformNoiseCfg(n_min=-0.05, n_max=0.05)


def _install_amp_collision_terms(cfg: ManagerBasedRlEnvCfg) -> None:
  """Torso vs ground kills; hip/arm vs ground penalize; keep self-collision."""
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  illegal_ground_cfg = ContactSensorCfg(
    name="illegal_ground_contact",
    primary=ContactMatch(mode="geom", entity="robot", pattern=ILLEGAL_GROUND_GEOMS),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  penalize_ground_cfg = ContactSensorCfg(
    name="penalize_ground_contact",
    primary=ContactMatch(mode="geom", entity="robot", pattern=PENALIZE_GROUND_GEOMS),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (
    self_collision_cfg,
    illegal_ground_cfg,
    penalize_ground_cfg,
  )
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=vel_mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": "self_collision", "force_threshold": 10.0},
  )
  cfg.rewards["collision"] = RewardTermCfg(
    func=vel_mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": "penalize_ground_contact", "force_threshold": 0.1},
  )
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=vel_mdp.illegal_contact,
    params={"sensor_name": "illegal_ground_contact", "force_threshold": 1.0},
  )


def _install_amp_step_amplitude(cfg: ManagerBasedRlEnvCfg) -> None:
  """Register at weight 0; the stand-stage curriculum raises it."""
  cfg.rewards["step_amplitude_symmetry"] = RewardTermCfg(
    func=amp_mdp.step_amplitude_symmetry_longitudinal,
    weight=0.0,
    params={
      "sensor_name": "feet_ground_contact",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
    },
  )


def _apply_amp_rough_terrains(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
  """AMP Rough terrain mix."""
  gen = replace(AMP_ROUGH_TERRAINS_CFG)
  if play:
    gen.curriculum = False
    gen.num_cols = 5
    gen.num_rows = 5
    gen.border_width = 10.0
  else:
    gen.curriculum = True
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = gen
  cfg.scene.terrain.max_init_terrain_level = 0
  # Default velocity env uses 10 Newton iterations; too few to keep the
  # 5-cylinder soles from sinking on large box / heightfield contacts.
  cfg.sim.mujoco.iterations = max(cfg.sim.mujoco.iterations, 30)
  cfg.sim.mujoco.ls_iterations = max(cfg.sim.mujoco.ls_iterations, 50)
  _install_amp_nan_safe_obs(cfg)


def _install_amp_nan_safe_obs(cfg: ManagerBasedRlEnvCfg) -> None:
  """Raycast/critic NaN insurance."""
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    if "height_scan" in terms:
      terms["height_scan"].func = amp_mdp.safe_height_scan
  for name, term in cfg.observations["critic"].terms.items():
    if name == "height_scan":
      continue
    term.func = amp_mdp.nan_safe_obs(term.func)


def _finish_amp_env(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
  _set_amp_yaw_curriculum(cfg)
  _install_amp_step_amplitude(cfg)
  _install_amp_stand_stage(cfg)
  _install_amp_collision_terms(cfg)
  cfg.terminations["fell_over"].func = amp_mdp.gym_fallen
  cfg.terminations["fell_over"].params = {"roll_limit": 0.7, "pitch_limit": 0.7}
  _apply_amp_dr(cfg, play=play)


def linglong_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat-terrain AMP locomotion for LingLong 20-DoF."""
  cfg = linglong_flat_env_cfg(play=play, rough=False)

  cfg.sim.njmax = 700
  cfg.sim.nconmax = 256
  cfg.sim.contact_sensor_maxmatch = 500

  _finish_amp_env(cfg, play=play)
  return cfg


def linglong_amp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Rough-terrain AMP locomotion for LingLong 20-DoF."""
  cfg = linglong_flat_env_cfg(play=play, rough=True)

  cfg.sim.njmax = 700
  cfg.sim.nconmax = 256
  cfg.sim.contact_sensor_maxmatch = 500

  _finish_amp_env(cfg, play=play)
  _apply_amp_rough_terrains(cfg, play=play)
  return cfg
