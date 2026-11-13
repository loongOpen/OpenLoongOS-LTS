"""Standalone Lite model, history/depth/PD parity and model selection."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("lite_sim2sim", REPO / "scripts/sim2sim_mujoco_lite.py")
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)
_spec = importlib.util.spec_from_file_location("lite_mjcf", REPO / "scripts/tools/build_lite_mujoco.py")
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)


@pytest.fixture(scope="module")
def metadata():
    return sim.deployment.load_lite_config(None)


@pytest.fixture(scope="module")
def model(metadata):
    return sim.build_model(sim.DEFAULT_XML, sim.DEFAULT_URDF, metadata, 0.005, "flat", 0.2)


def make_sim(model, metadata, policy=None, **kwargs):
    args = SimpleNamespace(no_self_occlusion=True,
                           cmd_vx=0.6, cmd_vy=0.0, cmd_dyaw=0.0, heading=None)
    for key, value in kwargs.items():
        setattr(args, key, value)
    policy = policy or (lambda obs, depth: np.zeros(26))
    return sim.LiteSimulation(args, model, metadata, policy)


def test_generated_model_matches_urdf_and_training_parameters(model, metadata):
    check = builder.check_model(sim.DEFAULT_XML, sim.DEFAULT_URDF, samples=16)
    assert check["max_position_error_m"] < 1e-10
    assert check["max_inertia_error_kg_m2"] < 1e-10
    assert (model.nq, model.nv, model.nu) == (33, 32, 26)
    for i, name in enumerate(metadata["joint_names"]):
        joint = model.joint(name).id
        motor = model.actuator(name + "_motor").id
        np.testing.assert_allclose(model.dof_armature[model.jnt_dofadr[joint]], metadata["armature"][i])
        np.testing.assert_allclose(model.actuator_ctrlrange[motor], [-metadata["effort_limit"][i], metadata["effort_limit"][i]])
        assert model.dof_damping[model.jnt_dofadr[joint]] == 0
        if "wrist" in name:
            assert metadata["effort_limit"][i] == 7
    contact = np.asarray([model.geom(n).id for n in ["contact_base_link_0", "contact_left_wrist_yaw_link_0"]])
    np.testing.assert_array_equal(model.geom_group[contact], 3)
    assert not model.geom_contype[model.geom_group == 2].any()


def test_absolute_proprioception_immediate_control_and_reset(model, metadata):
    actions = np.arange(26, dtype=float) / 100
    robot = make_sim(model, metadata, lambda obs, depth: actions)
    obs, depth = robot.observation()
    assert obs.shape == (1, 450) and depth.shape == (1, 8, 18, 32)
    assert obs.dtype == depth.dtype == np.float32
    np.testing.assert_allclose(obs[0, 15:45].reshape(5, 6), np.tile([1, 0, 0, 0, 0, 1], (5, 1)))
    np.testing.assert_allclose(obs[0, 45:60].reshape(5, 3), np.tile([0.6, 0, 0], (5, 1)))
    np.testing.assert_allclose(obs[0, 60:190].reshape(5, 26), np.tile(metadata["default_joint_pos"], (5, 1)), atol=1e-7)
    assert not obs[0, 190:].any()
    robot.control_step()
    np.testing.assert_allclose(robot.target, robot.q0 + robot.action_scale * actions)
    robot.physics_step()
    # The first physics step already receives the new policy target.
    np.testing.assert_allclose(robot.data.ctrl[robot.motor_ids],
                               robot.kp * robot.action_scale * actions, atol=1e-10)
    robot.physics_step()
    robot.physics_step()
    assert np.isfinite(robot.data.qpos).all()
    robot.reset()
    np.testing.assert_allclose(robot.data.qpos[robot.qpos_ids], metadata["default_joint_pos"])
    assert not robot.last_action.any()
    assert not any(history.initialized for history in robot.histories + [robot.camera.history])
    np.testing.assert_allclose(robot.target, robot.q0)


def test_camera_plane_rays_processing_and_current_depth(model, metadata):
    robot = make_sim(model, metadata)
    camera = robot.camera
    pos, rotation = camera.pose(robot.data)
    np.testing.assert_allclose(pos, np.asarray(metadata["initial_root_pos"]) + [0.1237, 0, 0.40966])
    np.testing.assert_allclose(rotation[:, 0], [2 ** -0.5, 0, -2 ** -0.5], atol=1e-7)
    world_rays = camera.local_rays @ rotation.T
    distances = -pos[2] / world_rays[:, 2]
    optical = np.minimum(distances * camera.local_rays[:, 0], 2.5).reshape(36, 64)[18:, 16:-16]
    # Independent torch preprocessing checks both ray metric and reflect padding.
    import torch
    import torch.nn.functional as F
    kernel = torch.exp(-torch.arange(-1, 2, dtype=torch.float32) ** 2 / 2)
    kernel /= kernel.sum()
    weight = (kernel[:, None] * kernel[None, :])[None, None]
    expected = F.conv2d(F.pad(torch.tensor(optical, dtype=torch.float32)[None, None], (1, 1, 1, 1), mode="reflect"), weight)[0, 0].numpy() / 2.5
    actual = camera.capture(model, robot.data)
    np.testing.assert_allclose(actual[0], np.tile(expected, (8, 1, 1)), atol=2e-7)
    first = actual.copy()
    robot.data.qpos[robot.root_qpos + 2] += 0.2
    mujoco.mj_forward(model, robot.data)
    current = camera.capture(model, robot.data)
    assert not np.allclose(current[:, -1], first[:, -1])
    np.testing.assert_allclose(current[:, 0], first[:, 0])



@pytest.mark.parametrize("terrain", ["stairs", "box"])
def test_terrain_variants_step_with_finite_observations(metadata, terrain):
    model = sim.build_model(sim.DEFAULT_XML, sim.DEFAULT_URDF, metadata, 0.005, terrain, 0.1)
    robot = make_sim(model, metadata)
    for _ in range(10):
        robot.control_step()
        for _ in range(4):
            robot.physics_step()
    obs, depth = robot.observation()
    assert np.isfinite(obs).all() and np.isfinite(depth).all()
    assert 0 <= depth.min() <= depth.max() <= 1
    assert not robot.data.warning.number.any()
    if terrain == "box":
        assert model.nhfield == 1 and model.hfield_data.max() == pytest.approx(1)


def test_export_directory_selects_highest_iteration_and_metadata_checksums(tmp_path, metadata):
    for number in (2, 9, 100):
        (tmp_path / f"policy_{number}.onnx").write_bytes(b"example")
    policy = sim.resolve_policy(tmp_path)
    assert policy.name == "policy_100.onnx"
    data = {**metadata, "onnx_sha256": "bad checksum"}
    policy.with_suffix(".json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        sim.load_metadata(policy, sim.DEFAULT_URDF)
    data["onnx_sha256"] = sim.sha256(policy)
    data["urdf_sha256"] = "wrong model"
    policy.with_suffix(".json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="URDF differs"):
        sim.load_metadata(policy, sim.DEFAULT_URDF)
