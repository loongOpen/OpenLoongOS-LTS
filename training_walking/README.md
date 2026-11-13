# training_walking

中文教程：[技术使用文档](../docs/technical-guide.md) ·[从训练到全链仿真](../docs/training-to-simulation.md) · [技术架构与策略接口](../docs/architecture.md) · [项目首页](../README.md)

A minimal [mjlab](https://github.com/mujocolab/mjlab)-based demo for training
the **LingLong 20-DoF humanoid** with **AMP (Adversarial Motion Priors)** on
MuJoCo-Warp. This is a stripped-down distribution: one robot, one reference
motion (forward walking), and the core training framework — nothing else.

## What's inside

- `Mjlab-Amp-Flat-LingLong` / `Mjlab-Amp-Rough-LingLong` — AMP locomotion tasks
  (PPO + discriminator style reward), with a velocity-tracking task reward.
- `Mjlab-Velocity-Flat-LingLong` / `Mjlab-Velocity-Rough-LingLong` — plain
  velocity-tracking baselines for the same robot (the AMP tasks build on these).
- `src/mjlab/asset_zoo/robots/LingLong/` — robot model (XML/URDF + meshes) and
  actuator/PD configuration.
- `src/mjlab/tasks/amp/motions/linglong/` — one expert motion clip
  (forward walking, 77-D frames → 53-D AMP observations).

## Install

Requires an NVIDIA GPU (CUDA) on Linux x86_64, and [uv](https://docs.astral.sh/uv/).
Use a separate virtual environment for this project: the dance and parkour
packages use different versions of their training dependencies.

```bash
git clone https://atomgit.com/openloong/OpenLoongOS-LTS.git
cd OpenLoongOS-LTS
git lfs pull
cd training_walking
uv sync --locked --extra cu128 --group dev
```

## Train

From the repository root, `./train.sh walking [OPTIONS]` selects the flat AMP
task by default. The older `./train.sh Mjlab-Amp-Flat-LingLong [OPTIONS]` also
works. Logs use local TensorBoard by default; W&B is optional via
`--agent.logger wandb` and requires your own account configuration.

```bash
# AMP on flat terrain (adjust --env.scene.num-envs to fit your GPU;
# run the small-scale check in the Chinese tutorial before a full run)
uv run --locked --extra cu128 train Mjlab-Amp-Flat-LingLong --env.scene.num-envs 1024 --agent.logger tensorboard

# Rough-terrain variant
uv run --locked --extra cu128 train Mjlab-Amp-Rough-LingLong --env.scene.num-envs 1024 --agent.logger tensorboard

# Plain velocity-tracking baseline (no AMP)
uv run --locked --extra cu128 train Mjlab-Velocity-Flat-LingLong --env.scene.num-envs 1024 --agent.logger tensorboard
```

Before a long run, check sampling, PPO/discriminator updates, checkpoint saving
and ONNX export with a small run (this does not train a useful policy):

```bash
uv run --locked --extra cu128 train Mjlab-Amp-Flat-LingLong \
  --env.scene.num-envs 16 --agent.max-iterations 2 --agent.save-interval 1 \
  --agent.run-name smoke --agent.amp.num-preload-transitions 256 \
  --agent.amp.replay-buffer-size 4096 --agent.amp.num-mini-batches 2
```

Keep `uv.lock` and use `--locked` so a fresh installation uses the same
dependency versions. Warp is obtained from the ordinary package index; CUDA
PyTorch is obtained from its explicit official PyTorch index. Do not copy an
unrelated mjlab environment into this directory.

## Play / visualize a checkpoint

Replace `<run>` and the checkpoint filename with an existing training output.

```bash
uv run --locked --extra cu128 play Mjlab-Amp-Flat-LingLong --checkpoint-file 'logs/rsl_rl/linglong_amp/<run>/model_1000.pt'
```

List all registered tasks:

```bash
uv run --locked --extra cu128 list-envs
```

## Test

The current short-run evidence and its limits are recorded in [VALIDATION.md](VALIDATION.md).

```bash
uv run --locked --extra cu128 pytest tests/
```

## Notes

- The AMP pipeline loads expert motion JSON (`Frames` × 77-D) and distills it
  into a 53-D observation: `joint_q20 | joint_dq20 | base_lin3 | base_ang3 |
  foot6 | root_z1`. To use your own motion clip, drop a JSON with the same
  layout into `src/mjlab/tasks/amp/motions/linglong/` — it is picked up
  automatically (optional `MotionWeight` field per file).
- Domain randomization is intentionally light: foot friction, base CoM,
  payload mass, inertia scaling, encoder bias, and random pushes.

## License & credits

Apache-2.0. Built on [mjlab](https://github.com/mujocolab/mjlab) by the MjLab
developers; RL training uses [rsl_rl](https://github.com/leggedrobotics/rsl_rl).
