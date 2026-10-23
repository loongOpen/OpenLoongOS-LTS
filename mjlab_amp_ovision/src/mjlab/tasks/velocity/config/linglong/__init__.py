from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import linglong_flat_env_cfg, linglong_rough_env_cfg
from .rl_cfg import linglong_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-LingLong",
  env_cfg=linglong_flat_env_cfg(),
  play_env_cfg=linglong_flat_env_cfg(play=True),
  rl_cfg=linglong_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-LingLong",
  env_cfg=linglong_rough_env_cfg(),
  play_env_cfg=linglong_rough_env_cfg(play=True),
  rl_cfg=linglong_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
