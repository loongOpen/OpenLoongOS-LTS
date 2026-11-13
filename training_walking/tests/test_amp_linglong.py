"""Tests for the LingLong 20-DoF AMP task."""

import json
from pathlib import Path

import torch

from mjlab import MJLAB_SRC_PATH
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

MOTION_DIR = MJLAB_SRC_PATH / "tasks" / "amp" / "motions" / "linglong"
ASSET_DIR = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "LingLong"


def _motion_files() -> list[Path]:
  return sorted(MOTION_DIR.glob("*.json"))


def test_linglong_assets_exist() -> None:
  assert (ASSET_DIR / "linglong_20dof.xml").exists()
  assert (ASSET_DIR / "linglong_20dof.urdf").exists()
  meshes = list((ASSET_DIR / "meshes").glob("*.STL"))
  assert len(meshes) == 27


def test_linglong_constants_spec() -> None:
  import mujoco

  from mjlab.asset_zoo.robots.LingLong.linglong_constants import (
    LINGLONG_ACTION_SCALE,
    get_linglong_robot_cfg,
    get_spec,
  )

  spec = get_spec()
  hinges = [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]
  assert len(hinges) == 20
  sensor_names = {s.name for s in spec.sensors}
  assert {"imu_ang_vel", "imu_lin_vel", "root_angmom"} <= sensor_names
  cfg = get_linglong_robot_cfg()
  assert len(cfg.articulation.actuators) == 20
  assert len(LINGLONG_ACTION_SCALE) == 20


def test_amp_motion_files_are_complete() -> None:
  files = _motion_files()
  assert len(files) == 1
  for path in files:
    motion = json.load(open(path))
    frames = motion["Frames"]
    dims = {len(f) for f in frames}
    assert dims == {77}, f"{path.name}: {dims}"
    assert motion["MotionWeight"] > 0


def test_amp_loader_extracts_53d() -> None:
  from mjlab.tasks.amp.rl.motion_loader import AmpMotionLoader

  loader = AmpMotionLoader(
    device="cpu",
    time_between_frames=0.02,
    motion_files=[str(p) for p in _motion_files()],
    preload_transitions=True,
    num_preload_transitions=64,
  )
  assert loader.observation_dim == 53
  assert loader.preloaded_s.shape[-1] == 53
  s, s_next = next(loader.feed_forward_generator(1, 8))
  assert s.shape == (8, 53)
  assert s_next.shape == (8, 53)
  assert torch.isfinite(s).all()


def test_amp_tasks_registered() -> None:
  ids = list_tasks()
  assert "Mjlab-Amp-Flat-LingLong" in ids
  assert "Mjlab-Amp-Rough-LingLong" in ids
  assert "Mjlab-Velocity-Flat-LingLong" in ids
  assert "Mjlab-Velocity-Rough-LingLong" in ids

  cfg = load_env_cfg("Mjlab-Amp-Flat-LingLong")
  assert cfg.rewards["stand_still"].weight == -0.3
  assert "amp_stand_stage" in cfg.curriculum
  twist = cfg.commands["twist"]
  assert twist.ranges.ang_vel_z == (-0.6, 0.6)

  rl = load_rl_cfg("Mjlab-Amp-Flat-LingLong")
  assert rl.experiment_name == "linglong_amp"
  assert len(rl.amp.motion_files) == 1

  runner = load_runner_cls("Mjlab-Amp-Flat-LingLong")
  assert runner is not None
  assert runner.amp_loader_cls.RAW_FRAME_SIZE == 77


def test_linglong_play_mode() -> None:
  for tid in ("Mjlab-Amp-Flat-LingLong", "Mjlab-Velocity-Flat-LingLong"):
    play = load_env_cfg(tid, play=True)
    assert play.episode_length_s >= 1e9
    assert not play.observations["actor"].enable_corruption
    assert "push_robot" not in play.events
