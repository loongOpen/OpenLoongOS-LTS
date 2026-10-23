"""RL configuration for LingLong (20-DoF) AMP task."""

from mjlab import MJLAB_SRC_PATH
from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.amp.config.linglong.env_cfgs import AMP_STAND_SWITCH_ITER
from mjlab.tasks.amp.rl import AmpAlgoCfg, AmpOnPolicyRunnerCfg

_MOTION_DIR = MJLAB_SRC_PATH / "tasks" / "amp" / "motions" / "linglong"


def _motion_files() -> tuple[str, ...]:
  files = sorted(str(p) for p in _MOTION_DIR.glob("*.json"))
  if not files:
    raise FileNotFoundError(
      f"No AMP motion JSON found in {_MOTION_DIR}."
    )
  return tuple(files)


def linglong_amp_ppo_runner_cfg() -> AmpOnPolicyRunnerCfg:
  return AmpOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="linglong_amp",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10_000,
    amp=AmpAlgoCfg(
      motion_files=_motion_files(),
      num_preload_transitions=200_000,
      replay_buffer_size=1_000_000,
      hidden_dims=(1024, 512, 256),

      # Mixing schedule:
      # lerp08 (iter 0–5000); iter 5000+ more AMP style (lerp 0.65, coef 0.4).
      # iter 7500: standing task stage (lerp 1.0 = no AMP style, stand_still -1)
      # plus step_amplitude_symmetry 1.0 on pure fwd/back.
      reward_coef=0.3,
      task_reward_lerp=0.8,
      mix_switch_iter=5000,
      stage2_reward_coef=0.4,
      stage2_task_reward_lerp=0.65,
      low_vx_task_gate=True,
      low_vx_abs=0.1,
      low_vy_abs=0.1,
      low_wz_abs=0.1,
      low_vx_task_reward_lerp=1.0,
      stand_switch_iter=AMP_STAND_SWITCH_ITER,

      style_reward_function="quad_mapping",
      loss_function="MSELoss",
      learning_rate=3e-4,
      weight_decay=1e-3,
      gradient_penalty_coef=10.0,
      num_mini_batches=80,
    ),
  )
