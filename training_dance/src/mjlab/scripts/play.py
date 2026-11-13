"""Play or headlessly evaluate a locally trained Linglong 1.4 dance policy."""

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass
class PlayConfig:
  checkpoint_file: Path
  motion_file: Path | None = None
  device: str = "auto"
  viewer: Literal["auto", "native", "viser", "none"] = "auto"
  steps: int = 0
  """Headless evaluation length; zero means one full motion including the hold."""
  report: Path | None = None


def run_play(task_id: str, cfg: PlayConfig):
  import os

  configure_torch_backends(allow_tf32=False)
  device = ("cuda:0" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
  if not cfg.checkpoint_file.is_file():
    raise FileNotFoundError(cfg.checkpoint_file)
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = 42
  if cfg.motion_file is not None:
    env_cfg.commands["motion"].motion_file = str(cfg.motion_file.resolve())
  agent_cfg = load_rl_cfg(task_id)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  try:
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
    runner.load(str(cfg.checkpoint_file), load_cfg={"actor": True}, map_location=device)
    policy = runner.get_inference_policy(device=device)
    viewer = cfg.viewer
    if viewer == "auto":
      viewer = "native" if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") else "viser"
    if viewer == "native":
      NativeMujocoViewer(wrapped, policy, enable_perturbations=False).run()
    elif viewer == "viser":
      ViserPlayViewer(wrapped, policy).run()
    else:
      command = env.command_manager.get_term("motion")
      steps = cfg.steps or command.motion.time_step_total - 1
      obs = wrapped.get_observations()
      failures = 0
      reward_sum = 0.
      max_joint_error = 0.
      min_height = float("inf")
      start = time.monotonic()
      with torch.inference_mode():
        for step in range(steps):
          actions = policy(obs)
          if not torch.isfinite(actions).all():
            raise RuntimeError(f"Non-finite policy output at step {step}")
          obs, reward, done, _ = wrapped.step(actions)
          failures += int(done.sum())
          reward_sum += float(reward.sum())
          max_joint_error = max(max_joint_error, float((command.robot_joint_pos - command.joint_pos).abs().max()))
          min_height = min(min_height, float(command.robot_anchor_pos_w[0, 2]))
          if (step + 1) % 250 == 0:
            print(f"[EVAL] {step + 1}/{steps}, resets={failures}", flush=True)
      report = dict(
        task=task_id, device=device, steps=steps,
        simulated_seconds=steps * env.step_dt,
        source_motion_frames=command.real_motion_frames,
        evaluated_motion_frames=command.motion.time_step_total,
        resets=failures, full_motion_completed=steps >= command.motion.time_step_total - 1 and failures == 0,
        reward_sum=reward_sum, min_base_height_m=min_height,
        max_joint_tracking_error_rad=max_joint_error,
        elapsed_seconds=time.monotonic() - start,
      )
      if cfg.report is not None:
        cfg.report.parent.mkdir(parents=True, exist_ok=True)
        cfg.report.write_text(json.dumps(report, indent=2) + "\n")
      print(json.dumps(report, indent=2), flush=True)
      if failures:
        raise RuntimeError(f"Policy triggered {failures} termination resets during evaluation")
      return report
  finally:
    wrapped.close()


def main():
  maybe_print_top_level_help("play")
  import mjlab.tasks  # noqa: F401

  task, remaining = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()), add_help=False,
    return_unknown_args=True, config=mjlab.TYRO_FLAGS,
  )
  cfg = tyro.cli(PlayConfig, args=remaining, prog=sys.argv[0] + " " + task, config=mjlab.TYRO_FLAGS)
  run_play(task, cfg)


if __name__ == "__main__":
  main()
