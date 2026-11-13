"""Register the standalone Linglong 1.4 dance task."""
from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl import MjlabOnPolicyRunner
from .env_cfgs import linglong14_xiao_env_cfg
from .rl_cfg import linglong14_xiao_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-LingLong14-Xiao",
  env_cfg=linglong14_xiao_env_cfg(),
  play_env_cfg=linglong14_xiao_env_cfg(play=True),
  rl_cfg=linglong14_xiao_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)
