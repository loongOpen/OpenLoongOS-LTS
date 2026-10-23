from mjlab import MJLAB_SRC_PATH
from mjlab.tasks.registry import register_mjlab_task

_MOTION_DIR = MJLAB_SRC_PATH / "tasks" / "amp" / "motions" / "linglong"

if not any(_MOTION_DIR.glob("*.json")):
  print(
    f"[WARN] Skipping Mjlab-Amp-*-LingLong: no motion JSON in {_MOTION_DIR}"
  )
else:
  from mjlab.tasks.amp.rl import AmpOnPolicyRunner

  from .env_cfgs import linglong_amp_flat_env_cfg, linglong_amp_rough_env_cfg
  from .rl_cfg import linglong_amp_ppo_runner_cfg

  register_mjlab_task(
    task_id="Mjlab-Amp-Flat-LingLong",
    env_cfg=linglong_amp_flat_env_cfg(),
    play_env_cfg=linglong_amp_flat_env_cfg(play=True),
    rl_cfg=linglong_amp_ppo_runner_cfg(),
    runner_cls=AmpOnPolicyRunner,
  )

  register_mjlab_task(
    task_id="Mjlab-Amp-Rough-LingLong",
    env_cfg=linglong_amp_rough_env_cfg(),
    play_env_cfg=linglong_amp_rough_env_cfg(play=True),
    rl_cfg=linglong_amp_ppo_runner_cfg(),
    runner_cls=AmpOnPolicyRunner,
  )
