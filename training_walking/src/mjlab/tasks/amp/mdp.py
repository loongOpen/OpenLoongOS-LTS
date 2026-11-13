"""Generic AMP helpers: events, terminations, observations and stand-stage logic."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.events import _sample_se3_range, resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply, sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET = SceneEntityCfg("robot")

INERTIA_SCALE = (0.8, 1.2)

# Standing task stage: step_amplitude_symmetry weight once the stage turns on.
AMP_STEP_AMPLITUDE_WEIGHT = 1.0


def inertia_alpha_range(scale: tuple[float, float] = INERTIA_SCALE) -> tuple[float, float]:
  """``pseudo_inertia`` scales mass and inertia by ``e^{2α}``; invert a ×range."""
  lo, hi = scale
  return (0.5 * math.log(lo), 0.5 * math.log(hi))


def push_robots_gym(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET,
) -> None:
  """Overwrite only the root velocity axes present in ``velocity_range``."""
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  vel_w = asset.data.root_link_vel_w[env_ids].clone()
  samples = _sample_se3_range(velocity_range, vel_w.shape, env.device)
  key_to_idx = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}
  for key, idx in key_to_idx.items():
    if key in velocity_range:
      vel_w[:, idx] = samples[:, idx]
  asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)


def gym_fallen(
  env: ManagerBasedRlEnv,
  roll_limit: float = 0.7,
  pitch_limit: float = 0.7,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET,
) -> torch.Tensor:
  """Fall termination: ``|roll| > roll_limit`` or ``|pitch| > pitch_limit``."""
  asset: Entity = env.scene[asset_cfg.name]
  roll, pitch, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)
  return (roll.abs() > roll_limit) | (pitch.abs() > pitch_limit)


def safe_height_scan(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Height scan with NaN protection."""
  result = envs_mdp.height_scan(env, sensor_name=sensor_name)
  return torch.nan_to_num(result, nan=5.0)


def nan_safe_obs(func):
  """Wrap an observation function with NaN/Inf protection."""

  def wrapper(env, **kwargs):
    result = func(env, **kwargs)
    return torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)

  wrapper.__name__ = getattr(func, "__name__", "nan_safe_obs")
  wrapper.__wrapped__ = func  # type: ignore[attr-defined]
  return wrapper


def longitudinal_vx_gate(
  cmd: torch.Tensor, command_threshold: float = 0.1
) -> torch.Tensor:
  """1 only for pure fwd/back: |vx|>=thresh and |vy|,|wz| both below thresh."""
  return (
    (cmd[:, 0].abs() >= command_threshold)
    & (cmd[:, 1].abs() < command_threshold)
    & (cmd[:, 2].abs() < command_threshold)
  ).float()


def step_amplitude_symmetry(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward symmetric step amplitude between left and right feet.

  Tracks each foot's world-frame position at liftoff and touchdown,
  computes the swing displacement, then compares the forward (body-x)
  component between left and right.  Asymmetric steps are penalised
  with  exp(-Δ² / σ²).

  State is stored on the env object (``_sas_*``) and lazily initialised.
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None

  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]  # [B, 2, 3]
  on_ground = sensor.data.found > 0  # [B, 2]

  # ---- lazy-init persistent state (same pattern as _gait_phase) ----
  _lo = getattr(env, "_sas_lo", None)
  if _lo is None:
    _td = foot_pos_w.clone()          # touchdown position (world)
    _lo = foot_pos_w.clone()          # liftoff position (world)
    _prev = on_ground.clone()          # previous contact state
    _fwd = torch.zeros(env.num_envs, 2, device=foot_pos_w.device)
  else:
    _td = getattr(env, "_sas_td")
    _prev = getattr(env, "_sas_prev")
    _fwd = getattr(env, "_sas_fwd")

  # ---- detect transitions ----
  just_td = on_ground & ~_prev          # foot just landed
  just_lo = ~on_ground & _prev          # foot just lifted off

  # ---- record liftoff position ----
  lo_mask = just_lo.unsqueeze(-1).expand_as(foot_pos_w)
  _lo.copy_(torch.where(lo_mask, foot_pos_w, _lo))

  # ---- compute swing amplitude at touchdown ----
  td_mask = just_td.unsqueeze(-1).expand_as(foot_pos_w)
  swing_w = foot_pos_w - _lo  # [B, 2, 3]

  # project onto body forward (x) axis
  base_quat = asset.data.root_link_quat_w  # [B, 4]
  B = base_quat.shape[0]
  body_x = quat_apply(base_quat, torch.tensor(
    [1.0, 0.0, 0.0], device=foot_pos_w.device
  ).unsqueeze(0).expand(B, 3))  # [B, 3]

  step_fwd = (swing_w * body_x.unsqueeze(1)).sum(-1)  # [B, 2]
  _fwd.copy_(torch.where(just_td, step_fwd, _fwd))

  # ---- record touchdown position (for potential future use) ----
  _td.copy_(torch.where(td_mask, foot_pos_w, _td))

  # ---- persist state ----
  setattr(env, "_sas_td", _td)
  setattr(env, "_sas_lo", _lo)
  setattr(env, "_sas_prev", on_ground.clone())
  setattr(env, "_sas_fwd", _fwd)

  # ---- symmetry reward ----
  left_fwd = _fwd[:, 0]
  right_fwd = _fwd[:, 1]
  has_stepped = just_td.any(dim=1).float()

  diff = (left_fwd - right_fwd).abs()
  reward = torch.exp(-diff / 0.10)  # 10 cm tolerance

  cmd = env.command_manager.get_command("twist")
  assert cmd is not None
  cmd_mag = torch.norm(cmd, dim=1)
  alpha = torch.sigmoid((cmd_mag - command_threshold) * 20.0)

  return reward * has_stepped * alpha


def step_amplitude_symmetry_longitudinal(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Left-right swing amplitude, only for pure forward/back walking."""
  reward = step_amplitude_symmetry(
    env,
    sensor_name=sensor_name,
    asset_cfg=asset_cfg,
    command_threshold=command_threshold,
  )
  cmd = env.command_manager.get_command("twist")
  assert cmd is not None
  return reward * longitudinal_vx_gate(cmd, command_threshold)


def apply_amp_stand_stage_weights(env: ManagerBasedRlEnv, *, on: bool) -> None:
  """Pose skip always on; standing stage raises stand_still and step symmetry."""
  env._amp_skip_zero_pose_std = True
  env.reward_manager.get_term_cfg("stand_still").weight = -1.0 if on else -0.3
  env.reward_manager.get_term_cfg("step_amplitude_symmetry").weight = (
    AMP_STEP_AMPLITUDE_WEIGHT if on else 0.0
  )
