"""Train the bundled dance task with standard RSL-RL PPO."""

import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends


@dataclass
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlOnPolicyRunnerCfg
  device: str = "cuda:0"
  log_root: str = "logs/rsl_rl"
  checkpoint: Path | None = None

  @staticmethod
  def from_task(task_id):
    return TrainConfig(env=load_env_cfg(task_id), agent=load_rl_cfg(task_id))


def run_train(task_id: str, cfg: TrainConfig) -> Path:
  del task_id
  if cfg.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. For a small CPU check, use --device cpu --env.scene.num-envs 2.")
  motion = Path(cfg.env.commands["motion"].motion_file)
  if not motion.is_file():
    raise FileNotFoundError(f"Motion file not found: {motion}")
  configure_torch_backends()
  cfg.env.seed = cfg.agent.seed
  cfg.agent.logger = "tensorboard"
  cfg.agent.upload_model = False
  run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
  if cfg.agent.run_name:
    run_name += "_" + cfg.agent.run_name
  log_dir = Path(cfg.log_root).resolve() / cfg.agent.experiment_name / run_name
  dump_yaml(log_dir / "params/env.yaml", asdict(cfg.env))
  dump_yaml(log_dir / "params/agent.yaml", asdict(cfg.agent))
  env = ManagerBasedRlEnv(cfg=cfg.env, device=cfg.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
  try:
    runner = MjlabOnPolicyRunner(wrapped, asdict(cfg.agent), str(log_dir), cfg.device)
    if cfg.checkpoint is not None:
      runner.load(str(cfg.checkpoint), map_location=cfg.device)
    runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)
  finally:
    wrapped.close()
  print(f"[INFO] Checkpoints and TensorBoard logs: {log_dir}", flush=True)
  return log_dir


def main():
  maybe_print_top_level_help("train")
  import mjlab.tasks  # noqa: F401

  task, remaining = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()), add_help=False,
    return_unknown_args=True, config=mjlab.TYRO_FLAGS,
  )
  cfg = tyro.cli(TrainConfig, args=remaining, default=TrainConfig.from_task(task),
                 prog=sys.argv[0] + " " + task, config=mjlab.TYRO_FLAGS)
  run_train(task, cfg)


if __name__ == "__main__":
  main()
