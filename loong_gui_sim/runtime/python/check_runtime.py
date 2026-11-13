#!/usr/bin/env python3
"""Check the standalone GUI policy environment without starting a control chain."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import struct
import sys


def check_policies():
    """Load all shipped policies and compare both new native plans to CPU ORT."""
    import numpy as np
    import onnxruntime as ort
    from dance_runtime import DancePolicy
    from native_policy import NativeParkourPolicy

    gui = Path(__file__).resolve().parents[2]
    definitions = {
        "walking": (gui / "runtime/model/onnx/policy.onnx", {"obs": [1, 700]}, [1, 20]),
        "dance": (gui / "runtime/model/dance/linglong14_mjlab_dance_29999.onnx", {"obs": [1, 1610]}, [1, 26]),
        "parkour": (gui.parent / "training_parkour/checkpoints/lite/exported/policy_10200.onnx",
                    {"obs": [1, 450], "depth": [1, 8, 18, 32]}, [1, 26]),
    }
    sessions, outputs, records = {}, {}, {}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    for name, (path, shapes, output_shape) in definitions.items():
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        actual = {item.name: item.shape for item in session.get_inputs()}
        if actual != shapes or any(item.type != "tensor(float)" for item in session.get_inputs()):
            raise RuntimeError(f"{name}: unexpected input contract: {actual}")
        output = session.get_outputs()
        if len(output) != 1 or output[0].name != "actions" or output[0].shape != output_shape or output[0].type != "tensor(float)":
            raise RuntimeError(f"{name}: unexpected action contract")
        values = {key: np.zeros(shape, np.float32) for key, shape in shapes.items()}
        result = session.run(["actions"], values)[0]
        if result.shape != tuple(output_shape) or not np.isfinite(result).all():
            raise RuntimeError(f"{name}: inference returned invalid actions")
        sessions[name], outputs[name] = session, result
        records[name] = {"inputs": shapes, "actions": output_shape, "finite": True}

    parkour = NativeParkourPolicy(definitions["parkour"][0])
    try:
        actual = parkour(np.zeros((1, 450), np.float32), np.zeros((1, 8, 18, 32), np.float32))
        expected = outputs["parkour"][0]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        records["parkour"]["native_max_abs_error"] = float(np.max(np.abs(actual - expected)))
    finally:
        parkour.close()

    reference_path = gui / "runtime/model/dance/linglong14_mjlab_xiao.refbin"
    raw = reference_path.read_bytes()
    magic, frames, actions, dimension, dt = struct.unpack("<8sIIIf", raw[:24])
    if (magic, actions, dimension) != (b"BMJLAB01", 26, 35) or abs(dt - .02) > 1e-7:
        raise RuntimeError("dance: unexpected reference contract")
    reference = np.frombuffer(raw, dtype="<f4", offset=24).reshape(frames, dimension)
    q, dq, quat, gyro = np.zeros(26, np.float32), np.zeros(26, np.float32), np.array([1, 0, 0, 0], np.float32), np.zeros(3, np.float32)
    proprio = np.concatenate((np.array([0, 0, -1], np.float32), gyro, q, dq, np.zeros(26, np.float32)))
    window = reference[np.clip(np.arange(-10, 11), 0, frames - 1)].ravel()
    observation = np.concatenate((window, reference[0], np.tile(proprio, 10))).astype(np.float32)[None]
    expected = sessions["dance"].run(["actions"], {"obs": observation})[0][0]
    dance = DancePolicy(definitions["dance"][0], reference_path)
    try:
        target = dance.step(q, dq, quat, gyro)
        if not np.isfinite(target).all():
            raise RuntimeError("dance: native target contains invalid values")
        np.testing.assert_allclose(dance.action, expected, atol=2e-5, rtol=2e-5)
        records["dance"]["native_max_abs_error"] = float(np.max(np.abs(dance.action - expected)))
    finally:
        dance.close()
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="also test a tiny MuJoCo offscreen depth render")
    parser.add_argument("--policies", action="store_true", help="also test all three ONNX policies and both new native plans")
    args = parser.parse_args()
    render_result = None
    if args.render:
        from render_probe import probe_rendering
        render_result = probe_rendering(check_policies=False)
        if not render_result["ok"]:
            print(json.dumps(render_result, ensure_ascii=False), file=sys.stderr)
            return 1
        os.environ["MUJOCO_GL"] = render_result["backend"]
    import numpy as np
    import mujoco
    import onnxruntime as ort
    import scipy
    import PIL.Image

    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <worldbody><light pos="0 0 3"/>
        <geom type="plane" size="2 2 .1"/>
        <body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
        <camera name="probe" pos="0 -2 1" xyaxes="1 0 0 0 .5 1"/>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    result = {"status": "ok", "python": sys.version.split()[0], "executable": sys.executable,
              "architecture": platform.machine(), "mujoco_step": True,
              "providers": ort.get_available_providers(),
              "packages": {name: version(name) for name in
                           ("numpy", "mujoco", "onnxruntime", "scipy", "Pillow")}}
    if "CPUExecutionProvider" not in result["providers"]:
        raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")
    if args.render:
        result["depth_render"] = True
        result["render_backend"] = render_result["backend"]
        result["render_probe"] = render_result
    if args.policies:
        result["policies"] = check_policies()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
