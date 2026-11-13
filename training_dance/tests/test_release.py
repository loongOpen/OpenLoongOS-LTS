"""Release contract: portable assets and no task-specific sim-to-real recipes."""

from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.robots.linglong14.linglong14_constants import (
  LINGLONG14_BODY_NAMES,
  LINGLONG14_JOINT_NAMES,
  get_linglong14_robot_cfg,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg


TASK = "Mjlab-Tracking-Flat-LingLong14-Xiao"


def test_task_has_nominal_physics_and_clean_observations():
  assert list_tasks() == [TASK]
  for play in (False, True):
    cfg = load_env_cfg(TASK, play=play)
    assert cfg.events == {}
    for group in cfg.observations.values():
      assert not group.enable_corruption
      assert all(term.noise is None for term in group.terms.values())
    for actuator in cfg.scene.entities["robot"].articulation.actuators:
      assert actuator.delay_min_lag == actuator.delay_max_lag == 0
    motion = cfg.commands["motion"]
    assert motion.pose_range == motion.velocity_range == {}
    assert motion.joint_position_range == (0., 0.)
    assert motion.sampling_mode == ("start" if play else "uniform")
    assert not hasattr(motion, "adaptive_alpha")
    assert not hasattr(motion, "terminal_hold_sample_prob")


def test_complete_model_and_named_motion_are_bundled():
  robot = get_linglong14_robot_cfg().build()
  model = robot.compile()
  assert (model.nq, model.nv, model.nu) == (33, 32, 26)
  assert tuple(robot.joint_names) == LINGLONG14_JOINT_NAMES
  assert tuple(robot.body_names) == LINGLONG14_BODY_NAMES
  cfg = load_env_cfg(TASK)
  path = Path(cfg.commands["motion"].motion_file)
  assert path.is_file()
  with np.load(path, allow_pickle=False) as data:
    assert data["joint_pos"].shape == (1781, 26)
    assert float(data["fps"].reshape(-1)[0]) == 50.
    assert tuple(data["joint_names"]) == LINGLONG14_JOINT_NAMES
    assert tuple(data["body_names"]) == LINGLONG14_BODY_NAMES
    for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
      assert np.isfinite(data[key]).all()
    # Independent FK verifies that the bundled trajectory is in this model's order.
    mjd = mujoco.MjData(model)
    body_ids = [model.body(name).id for name in LINGLONG14_BODY_NAMES]
    for frame in (0, 356, 890, 1424, 1780):
      mjd.qpos[:3] = data["body_pos_w"][frame, 0]
      mjd.qpos[3:7] = data["body_quat_w"][frame, 0]
      mjd.qpos[7:] = data["joint_pos"][frame]
      mujoco.mj_forward(model, mjd)
      np.testing.assert_allclose(mjd.xpos[body_ids], data["body_pos_w"][frame], atol=2e-6)


def test_example_policy_has_no_experiment_or_optimizer_payload():
  root = Path(__file__).resolve().parents[1]
  checkpoint = torch.load(root / "checkpoints/dance_demo.pt", map_location="cpu", weights_only=True)
  assert set(checkpoint) == {"actor_state_dict", "iter", "infos"}
  assert checkpoint["infos"] is None
  actor = checkpoint["actor_state_dict"]
  assert actor["mlp.0.weight"].shape == (512, 1610)
  assert actor["mlp.6.weight"].shape == (26, 128)
  assert all(torch.isfinite(value).all() for value in actor.values())


def test_private_recipes_are_absent_from_the_distribution():
  root = Path(__file__).resolve().parents[1]
  assert not (root / "src/mjlab/envs/mdp/dr").exists()
  assert not (root / "src/mjlab/tasks/tracking/mdp/events.py").exists()
  assert not (root / "src/mjlab/tasks/tracking/mdp/noise.py").exists()
  assert not (root / "src/mjlab/tasks/tracking/rl").exists()
  assert {p.name for p in (root / "src/mjlab/asset_zoo/robots").iterdir() if p.is_dir() and p.name != "__pycache__"} == {"linglong14"}
