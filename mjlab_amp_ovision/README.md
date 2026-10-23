# mjlab_amp_ovision

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

```bash
git clone https://atomgit.com/openloong/LoongMC-2026-LTS.git
cd LoongMC-2026-LTS
git lfs pull
cd mjlab_amp_ovision
uv sync --locked --extra cu128 --group dev
```

## Train

```bash
# AMP on flat terrain (adjust --env.scene.num-envs to fit your GPU;
# ~1024 envs needs about 6 GB of VRAM for this robot)
uv run --locked --extra cu128 train Mjlab-Amp-Flat-LingLong --env.scene.num-envs 1024 --agent.logger tensorboard

# Rough-terrain variant
uv run --locked --extra cu128 train Mjlab-Amp-Rough-LingLong --env.scene.num-envs 1024 --agent.logger tensorboard

# Plain velocity-tracking baseline (no AMP)
uv run --locked --extra cu128 train Mjlab-Velocity-Flat-LingLong --env.scene.num-envs 1024 --agent.logger tensorboard
```

## Play / visualize a checkpoint

```bash
uv run --locked --extra cu128 play Mjlab-Amp-Flat-LingLong --checkpoint-file logs/rsl_rl/linglong_amp/<run>/model_1000.pt
```

List all registered tasks:

```bash
uv run --locked --extra cu128 list-envs
```

## Test

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
