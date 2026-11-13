"""AMP runner / discriminator configuration (additive; does not change PPO)."""

from dataclasses import dataclass, field
from typing import Literal, Tuple

from mjlab.rl import RslRlOnPolicyRunnerCfg


@dataclass
class AmpAlgoCfg:
  """Adversarial Motion Priors extras used by AmpOnPolicyRunner."""

  motion_files: Tuple[str, ...] = ()
  """Motion JSON files (77-D frames, 20-DoF LingLong)."""
  num_preload_transitions: int = 200_000
  replay_buffer_size: int = 1_000_000
  hidden_dims: Tuple[int, ...] = (1024, 512, 256)
  reward_coef: float = 0.3
  task_reward_lerp: float = 0.5
  """total = (1 - lerp) * style + lerp * task"""
  mix_switch_iter: int = 0
  """If ``> 0``, switch to ``stage2_*`` mix after this many PPO iterations."""
  stage2_reward_coef: float = 0.5
  stage2_task_reward_lerp: float = 0.5
  stand_switch_iter: int = 0
  """If ``> 0`` with ``low_vx_task_gate``, delay the standing task mix until this iter."""
  low_vx_task_gate: bool = False
  """If True, |vx|, |vy|, |wz| below abs thresholds use `low_vx_task_reward_lerp`.

  Lateral and yaw must be gated too: otherwise vx≈0 + vz≈0 with nonzero vy
  would drop AMP style during side-step. 1.0 = task only, no AMP style.
  """
  low_vx_abs: float = 0.3
  low_vy_abs: float = 0.3
  low_wz_abs: float = 0.15
  low_vx_task_reward_lerp: float = 0.7
  style_reward_function: Literal[
    "quad_mapping", "log_mapping", "wasserstein_mapping"
  ] = "quad_mapping"
  loss_function: Literal["MSELoss", "PairwiseLoss", "BCEWithLogitsLoss"] = "MSELoss"
  learning_rate: float = 3e-4
  weight_decay: float = 1e-3
  gradient_penalty_coef: float = 10.0
  num_mini_batches: int = 80


@dataclass
class AmpOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  amp: AmpAlgoCfg = field(default_factory=AmpAlgoCfg)
