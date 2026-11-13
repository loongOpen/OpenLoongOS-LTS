"""Standalone Lite deployment/export checks; no IsaacLab app is required."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/sim2sim/export_lite_pt_to_onnx.py"
CHECKPOINT = ROOT / "checkpoints/lite/model_10200.pt"
SAVED_CONFIG = CHECKPOINT.parent / "params/env.yaml"


@pytest.fixture(scope="module")
def exporter():
    spec = importlib.util.spec_from_file_location("lite_onnx_export_test_module", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_source_interface_and_motor_limits(exporter):
    config = exporter.load_lite_config()
    assert config["training_config"] is None
    assert config["onnx_inputs"] == {"obs": [1, 450], "depth": [1, 8, 18, 32]}
    assert config["onnx_outputs"] == {"actions": [1, 26]}
    assert len(set(config["joint_names"])) == 26
    assert not any("waist" in name for name in config["joint_names"])
    assert config["initial_root_pos"] == [0.0, 0.0, 0.99]
    assert config["depth"]["mount_link"] == "base_link"
    assert config["depth"]["mount_pos_m"] == [0.1237, 0.0, 0.40966]
    assert config["obs_slices"]["last_action"] == {"start": 320, "end": 450, "dim_per_frame": 26}
    np.testing.assert_allclose(config["action_scale"], 0.25 * np.array(config["effort_limit"]) / config["kp"])
    for i, name in enumerate(config["joint_names"]):
        if "wrist" in name:
            assert config["effort_limit"][i] == 7.0
            assert config["action_scale"][i] == 0.035
    left = config["joint_names"].index("left_wrist_yaw_joint")
    right = config["joint_names"].index("right_wrist_yaw_joint")
    assert config["velocity_limit"][left] == 20.943
    assert config["velocity_limit"][right] == 20.94


def test_configuration_import_and_loading_need_no_torch_or_isaaclab():
    code = f"""
import importlib.util, sys
spec = importlib.util.spec_from_file_location('lite_config_only', {str(SCRIPT)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.load_lite_config(None)
assert 'torch' not in sys.modules
assert not any(name.startswith(('isaaclab', 'omni', 'pxr')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.skipif(not SAVED_CONFIG.is_file(), reason="local Lite training configuration is unavailable")
def test_saved_training_config_agrees_with_current_asset(exporter):
    pytest.importorskip("yaml")
    current = exporter.load_lite_config(None)
    saved = exporter.load_lite_config(CHECKPOINT)
    assert saved["training_config_sha256"] == exporter.sha256(SAVED_CONFIG)
    for key in ("joint_names", "kp", "kd", "armature", "effort_limit", "velocity_limit", "position_lower",
                "position_upper", "default_joint_pos", "action_scale", "initial_root_pos", "policy_dt"):
        assert current[key] == saved[key], key
    assert current["actuator_delay_physics_steps_training"] == [[0, 0]] * 26
    assert saved["actuator_delay_physics_steps_training"] == [[0, 2]] * 26
    current_depth, saved_depth = dict(current["depth"]), dict(saved["depth"])
    assert current_depth.pop("delay_frames_training") == [0, 0]
    assert saved_depth.pop("delay_frames_training") == [0, 1]
    assert current_depth == saved_depth


def test_yaml_python_tags_are_inert_or_rejected(exporter, tmp_path):
    yaml = pytest.importorskip("yaml")
    config = tmp_path / "safe.yaml"
    config.write_text("position: !!python/tuple [1, 2, 3]\nindices: !!python/object/apply:builtins.slice [null, null, null]\n")
    assert exporter.read_training_yaml(config) == {"position": [1, 2, 3], "indices": [None, None, None]}
    marker = tmp_path / "must_not_exist"
    config.write_text(f"!!python/object/apply:os.system ['touch {marker}']\n")
    with pytest.raises(yaml.constructor.ConstructorError):
        exporter.read_training_yaml(config)
    assert not marker.exists()


@pytest.mark.parametrize(("path", "value"), [
    (("sim", "dt"), 0.004),
    (("scene", "depth_camera", "update_period"), 0.04),
    (("scene", "depth_camera", "pattern_cfg", "width"), 80),
    (("scene", "depth_camera", "pattern_cfg", "height"), 48),
    (("scene", "depth_camera", "pattern_cfg", "horizontal_aperture"), 1.0),
    (("scene", "depth_camera", "pattern_cfg", "vertical_aperture"), 1.0),
    (("observations", "depth", "depth_image", "params", "crop_region"), [17, 1, 16, 16]),
    (("observations", "depth", "depth_image", "params", "depth_range"), [0.1, 2.5]),
    (("observations", "depth", "depth_image", "params", "data_type"), "distance_to_camera"),
    (("observations", "depth", "depth_image", "params", "blur_kernel_size"), 5),
    (("observations", "depth", "depth_image", "params", "blur_sigma"), 1.5),
    (("observations", "depth", "depth_image", "params", "history_length"), 40),
    (("observations", "depth", "depth_image", "params", "history_skip_frames"), 4),
    (("observations", "depth", "depth_image", "params", "num_output_frames"), 9),
])
def test_unsupported_saved_runtime_settings_fail_before_export(exporter, tmp_path, path, value):
    yaml = pytest.importorskip("yaml")
    # Keep each mutation independent and exercise the public YAML/config path.
    config = json.loads(json.dumps(exporter._config_from_source(), default=float))
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    output = tmp_path / "env.yaml"
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="Unsupported"):
        exporter.load_lite_config(None, env_yaml=output)


def test_saved_action_order_and_motor_limit_mismatch_are_rejected(exporter, tmp_path):
    yaml = pytest.importorskip("yaml")
    original = json.loads(json.dumps(exporter._config_from_source(), default=float))
    for field in ("joint_names", "effort_limit_sim"):
        config = copy.deepcopy(original)
        if field == "joint_names":
            config["actions"]["joint_pos"][field].reverse()
        else:
            config["scene"]["robot"]["actuators"]["left_wrist_yaw_joint"][field] = 27.0
        output = tmp_path / f"bad_{field}.yaml"
        output.write_text(yaml.safe_dump(config, sort_keys=False))
        with pytest.raises(ValueError):
            exporter.load_lite_config(None, env_yaml=output)


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="local Lite PT smoke-test checkpoint is unavailable")
def test_real_pt_export_matches_onnxruntime_and_metadata(exporter, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("yaml")
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    output = exporter.export_lite_checkpoint(CHECKPOINT, tmp_path)
    assert output.name == "policy_10200.onnx"
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["onnx_sha256"] == exporter.sha256(output)
    assert metadata["checkpoint_sha256"] == exporter.sha256(CHECKPOINT)
    assert metadata["joint_names"] == exporter.load_lite_config(CHECKPOINT)["joint_names"]
    onnx.checker.check_model(onnx.load(str(output)))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    assert {node.name: node.shape for node in session.get_inputs()} == metadata["onnx_inputs"]
    assert {node.name: node.shape for node in session.get_outputs()} == metadata["onnx_outputs"]
    policy, obs_dim, frames, actions = exporter.load_lite_policy(CHECKPOINT)
    assert (obs_dim, frames, actions) == (450, 8, 26)
    rng = np.random.default_rng(12)
    for _ in range(3):
        obs = rng.normal(size=(1, 450)).astype(np.float32)
        depth = rng.uniform(size=(1, 8, 18, 32)).astype(np.float32)
        with torch.no_grad():
            expected = policy(torch.from_numpy(obs), torch.from_numpy(depth)).numpy()
        actual = session.run(["actions"], {"obs": obs, "depth": depth})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_saved_config_without_domain_randomization_can_be_exported(exporter, tmp_path):
    yaml = pytest.importorskip("yaml")
    config = json.loads(json.dumps(exporter._config_from_source(), default=float))
    output = tmp_path / "env.yaml"
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    metadata = exporter.load_lite_config(None, env_yaml=output)
    assert metadata["actuator_delay_physics_steps_training"] == [[0, 0]] * 26
    assert metadata["depth"]["delay_frames_training"] == [0, 0]
