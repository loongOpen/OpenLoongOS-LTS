"""Export a Lite AMP/depth PT checkpoint and an audited deployment interface.

This entry point has no IsaacLab dependency. It reuses the existing actor
reconstruction/export functions, but rejects policies outside Lite's
obs[1,450] + depth[1,8,18,32] -> actions[1,26] interface. Saved params/env.yaml
is the source of deployment settings when present; otherwise current Lite
source literals are read with AST without importing the training environment.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "source/lite_parkour/lite_parkour"
DEFAULT_URDF = PACKAGE / "data/Robots/Lite/lite_26dof.urdf"
ASSET_SOURCE = PACKAGE / "assets/lite.py"
ENV_SOURCE = PACKAGE / "tasks/amp/lite_env_cfg.py"
OBS_HISTORY, DEPTH_HISTORY, DEPTH_SKIP, DEPTH_FRAMES = 5, 37, 5, 8
DEPTH_SHAPE, CAMERA_SIZE = (18, 32), (64, 36)
DEPTH_CROP, DEPTH_RANGE = (18, 0, 16, 16), (0.0, 2.5)
POLICY_DT, CAMERA_PITCH_DEG, CAMERA_FOVY_DEG = 0.02, 45.0, 58.29
OBS_TERMS = (("base_ang_vel", 3), ("root_local_rot_tan_norm", 6), ("velocity_commands", 3),
             ("joint_pos", 26), ("joint_vel", 26), ("last_action", 26))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assignment(tree: ast.Module, name: str):
    return next(node.value for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))


def _source_defaults():
    """Read data literals only: no imports, execution, or IsaacLab initialization."""
    asset = ast.parse(ASSET_SOURCE.read_text(encoding="utf-8"))
    env = ast.parse(ENV_SOURCE.read_text(encoding="utf-8"))
    gains = ast.literal_eval(_assignment(asset, "_ACTUATORS"))
    initial = next(node for node in ast.walk(asset) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == "InitialStateCfg")
    initial = {kw.arg: ast.literal_eval(kw.value) for kw in initial.keywords}
    return gains, initial, ast.literal_eval(_assignment(env, "CAMERA_POS"))


_GAINS, _INITIAL, CAMERA_POS = _source_defaults()
CAMERA_ROT = (0.0, math.sin(math.pi / 8), 0.0, math.cos(math.pi / 8))  # xyzw
DEFAULT_JOINT_POS = _INITIAL["joint_pos"]
_URDF_JOINTS = [j for j in ET.parse(DEFAULT_URDF).getroot().findall("joint") if j.get("type") != "fixed"]
JOINT_ORDER = tuple(j.get("name") for j in _URDF_JOINTS)
ACTUATORS = {}
JOINT_ACTUATORS = {}
for _joint in _URDF_JOINTS:
    _kind = _joint.get("name").removeprefix("left_").removeprefix("right_").removesuffix("_joint")
    _limit = _joint.find("limit")
    ACTUATORS[_kind] = (*_GAINS[_kind], float(_limit.get("effort")), float(_limit.get("velocity")))
    JOINT_ACTUATORS[_joint.get("name")] = ACTUATORS[_kind]


def _resolve(value, name: str, *, default=None) -> float:
    if isinstance(value, dict):
        matches = [v for pattern, v in value.items() if re.fullmatch(pattern, name)]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous settings for {name}: {value}")
        value = matches[0] if matches else default
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"Missing or invalid scalar for {name}: {value}")
    return float(value)


def joint_params():
    """Return current source kp, kd, armature, effort, default and scale arrays."""
    rows = []
    for name in JOINT_ORDER:
        kind = name.removeprefix("left_").removeprefix("right_").removesuffix("_joint")
        kp, kd, armature, effort, _ = JOINT_ACTUATORS[name]
        rows.append((kp, kd, armature, effort, _resolve(DEFAULT_JOINT_POS, name, default=0.0), 0.25 * effort / kp))
    return tuple(np.asarray(column, dtype=np.float64) for column in zip(*rows))


def read_training_yaml(path: Path) -> dict:
    """Allow only data tags used by IsaacLab's YAML serializer.

    Python tuples become plain lists and slice tags become inert lists. No
    Python object constructors/imports, unknown tags, or unsafe loader run.
    """
    import yaml

    class ConfigLoader(yaml.SafeLoader):
        pass

    for tag in ("tag:yaml.org,2002:python/tuple", "tag:yaml.org,2002:python/object/apply:builtins.slice"):
        ConfigLoader.add_constructor(tag, lambda loader, node: loader.construct_sequence(node, deep=True))
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=ConfigLoader)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a training configuration mapping: {path}")
    return data


def _config_from_source() -> dict:
    kp, kd, armature, effort, default, scale = joint_params()
    actuators = {name: {"joint_names_expr": [name], "stiffness": kp[i], "damping": kd[i],
                       "armature": armature[i], "effort_limit_sim": effort[i],
                       "velocity_limit_sim": JOINT_ACTUATORS[name][4]} for i, name in enumerate(JOINT_ORDER)}
    policy = {name: {"history_length": OBS_HISTORY} for name in
              ("base_ang_vel", "root_local_rot_tan_norm", "velocity_commands", "joint_pos", "joint_vel", "actions")}
    for name in ("joint_pos", "joint_vel"):
        policy[name]["params"] = {"asset_cfg": {"joint_names": list(JOINT_ORDER), "preserve_order": True}}
    return {
        "sim": {"dt": 0.005}, "decimation": 4,
        "scene": {"robot": {"init_state": _INITIAL, "actuators": actuators},
                  "depth_camera": {"prim_path": "{ENV_REGEX_NS}/Robot/Geometry/base_link",
                                   "offset": {"pos": CAMERA_POS, "rot": CAMERA_ROT, "convention": "world"},
                                   "pattern_cfg": {"width": 64, "height": 36, "focal_length": 1.0,
                                                   "horizontal_aperture": 2 * math.tan(math.radians(89.51) / 2),
                                                   "vertical_aperture": 2 * math.tan(math.radians(58.29) / 2)},
                                   "update_period": POLICY_DT}},
        "actions": {"joint_pos": {"joint_names": list(JOINT_ORDER), "preserve_order": True,
                                  "use_default_offset": True, "scale": dict(zip(JOINT_ORDER, scale))}},
        "observations": {"policy": policy, "depth": {"depth_image": {"params": {
            "data_type": "distance_to_image_plane", "crop_region": DEPTH_CROP,
            "blur_kernel_size": 3, "blur_sigma": 1.0, "depth_range": DEPTH_RANGE,
            "history_length": DEPTH_HISTORY, "history_skip_frames": DEPTH_SKIP,
            "num_output_frames": DEPTH_FRAMES}}}},
    }


def load_lite_config(checkpoint: Path | None = None, env_yaml: Path | None = None, urdf: Path | None = None) -> dict:
    """Resolve Lite deployment metadata, preferring the saved training config."""
    checkpoint = Path(checkpoint).expanduser().resolve() if checkpoint is not None else None
    config_path = (Path(env_yaml).expanduser().resolve() if env_yaml else
                   checkpoint.parent / "params/env.yaml" if checkpoint is not None else None)
    if env_yaml and not config_path.is_file():
        raise FileNotFoundError(config_path)
    has_saved_config = config_path is not None and config_path.is_file()
    cfg = read_training_yaml(config_path) if has_saved_config else _config_from_source()
    robot = cfg["scene"]["robot"]
    source_urdf = robot.get("spawn", {}).get("asset_path")
    urdf_path = Path(urdf or (source_urdf if source_urdf and Path(source_urdf).is_file() else DEFAULT_URDF))
    urdf_path = urdf_path.expanduser().resolve()
    xml = ET.parse(urdf_path).getroot()
    joints = [j for j in xml.findall("joint") if j.get("type") != "fixed"]
    names = [j.get("name") for j in joints]
    action = cfg["actions"]["joint_pos"]
    if names != list(JOINT_ORDER) or action["joint_names"] != names or len(names) != 26:
        raise ValueError("Lite URDF/action joint order must match all 26 policy joints")
    if not action.get("preserve_order") or not action.get("use_default_offset") or action.get("clip") is not None:
        raise ValueError("Lite deployment requires ordered joint positions, default offset, and unclipped actions")
    policy = cfg["observations"]["policy"]
    term_order = [k for k, v in policy.items() if isinstance(v, dict) and "history_length" in v]
    expected_terms = ["base_ang_vel", "root_local_rot_tan_norm", "velocity_commands", "joint_pos", "joint_vel", "actions"]
    if term_order != expected_terms:
        raise ValueError(f"Unsupported policy term order: {term_order}")
    for name in expected_terms:
        term = policy[name]
        if term["history_length"] != OBS_HISTORY or term.get("scale") is not None or term.get("clip") is not None:
            raise ValueError(f"Unsupported observation history/scale/clip: {name}")
    for name in ("joint_pos", "joint_vel"):
        asset = policy[name]["params"]["asset_cfg"]
        if asset["joint_names"] != names or not asset["preserve_order"]:
            raise ValueError(f"Policy {name} joint order does not match actions")

    rows, delays = [], []
    for joint in joints:
        name, limit = joint.get("name"), joint.find("limit")
        matches = [a for a in robot["actuators"].values()
                   if any(re.fullmatch(pattern, name) for pattern in a["joint_names_expr"])]
        if len(matches) != 1:
            raise ValueError(f"Expected one actuator for {name}, found {len(matches)}")
        actuator = matches[0]
        p, d, a, effort, velocity = [_resolve(actuator[k], name) for k in
                                    ("stiffness", "damping", "armature", "effort_limit_sim", "velocity_limit_sim")]
        if p <= 0 or effort <= 0 or velocity <= 0 or d < 0 or a < 0:
            raise ValueError(f"Invalid PD, inertia or motor limits: {name}")
        if not np.allclose([effort, velocity], [float(limit.get("effort")), float(limit.get("velocity"))], rtol=1e-6):
            raise ValueError(f"Saved training motor limits differ from current URDF: {name}")
        rows.append((p, d, a, effort, velocity, float(limit.get("lower")), float(limit.get("upper")),
                     _resolve(robot["init_state"]["joint_pos"], name, default=0.0), _resolve(action["scale"], name)))
        delays.append([actuator.get("min_delay", 0), actuator.get("max_delay", 0)])
    values = dict(zip(("kp", "kd", "armature", "effort_limit", "velocity_limit", "position_lower", "position_upper",
                       "default_joint_pos", "action_scale"), map(list, zip(*rows))))
    if not np.isfinite(np.asarray(rows)).all():
        raise ValueError("Lite deployment parameters contain non-finite values")
    depth = cfg["observations"]["depth"]["depth_image"]["params"]
    camera = cfg["scene"]["depth_camera"]
    pattern, offset = camera["pattern_cfg"], camera["offset"]
    policy_dt = cfg["sim"]["dt"] * cfg["decimation"]
    hfov = math.degrees(2 * math.atan(pattern["horizontal_aperture"] / (2 * pattern["focal_length"])))
    vfov = math.degrees(2 * math.atan(pattern["vertical_aperture"] / (2 * pattern["focal_length"])))
    fixed_scalars = {
        "policy_dt": (policy_dt, POLICY_DT),
        "depth_camera.update_period": (camera["update_period"], POLICY_DT),
        "depth_camera.width": (pattern["width"], CAMERA_SIZE[0]),
        "depth_camera.height": (pattern["height"], CAMERA_SIZE[1]),
        "depth_camera.hfov_deg": (hfov, 89.51),
        "depth_camera.vfov_deg": (vfov, CAMERA_FOVY_DEG),
        "depth_camera.horizontal_aperture_offset": (pattern.get("horizontal_aperture_offset", 0.0), 0.0),
        "depth_camera.vertical_aperture_offset": (pattern.get("vertical_aperture_offset", 0.0), 0.0),
        "depth_camera.max_distance": (camera.get("max_distance", DEPTH_RANGE[1]), DEPTH_RANGE[1]),
        "depth.blur_kernel_size": (depth["blur_kernel_size"], 3),
        "depth.blur_sigma": (depth["blur_sigma"], 1.0),
        "depth.history_length": (depth["history_length"], DEPTH_HISTORY),
        "depth.history_skip_frames": (depth["history_skip_frames"], DEPTH_SKIP),
        "depth.num_output_frames": (depth["num_output_frames"], DEPTH_FRAMES),
    }
    for setting, (actual, expected) in fixed_scalars.items():
        if not math.isclose(float(actual), float(expected), rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f"Unsupported {setting}: {actual}; Lite runtime requires {expected}")
    fixed_values = {
        "depth.crop_region": (list(depth["crop_region"]), list(DEPTH_CROP)),
        "depth.depth_range": (list(depth["depth_range"]), list(DEPTH_RANGE)),
        "depth.data_type": (depth["data_type"], "distance_to_image_plane"),
        "depth_camera.ray_alignment": (camera.get("ray_alignment", "base"), "base"),
    }
    for setting, (actual, expected) in fixed_values.items():
        if actual != expected:
            raise ValueError(f"Unsupported {setting}: {actual}; Lite runtime requires {expected}")
    # Legacy checkpoint metadata is preserved for provenance only; the runtime
    # always applies the current motor target and the current depth frame.
    depth_delays = list(depth.get("delayed_frame_ranges", [0, 0]))
    if depth_delays not in ([0, 0], [0, 1]):
        raise ValueError(f"Unsupported depth.delayed_frame_ranges: {depth_delays}")
    crop = list(depth["crop_region"])
    cropped = [pattern["height"] - crop[0] - crop[1], pattern["width"] - crop[2] - crop[3]]
    if cropped != list(DEPTH_SHAPE) or depth["num_output_frames"] != DEPTH_FRAMES:
        raise ValueError(f"Unsupported Lite depth interface: {depth['num_output_frames']} x {cropped}")
    if camera["prim_path"].split("/")[-1] != "base_link" or offset["convention"] != "world":
        raise ValueError("Lite camera must use the base_link mount and world axis convention")
    slices, start = {}, 0
    for name, dim in OBS_TERMS:
        slices[name] = {"start": start, "end": start + OBS_HISTORY * dim, "dim_per_frame": dim}
        start += OBS_HISTORY * dim
    result = {
        "schema_version": 1, "task": "Lite-AMP-Depth-v0", "robot": "lite_26dof", "root_link": "base_link",
        "training_config": str(config_path) if has_saved_config else None,
        "training_config_sha256": sha256(config_path) if has_saved_config else None,
        "config_source": "saved params/env.yaml" if has_saved_config else "current Lite source literals",
        "urdf_path": str(urdf_path), "urdf_sha256": sha256(urdf_path),
        "onnx_inputs": {"obs": [1, start], "depth": [1, DEPTH_FRAMES, *DEPTH_SHAPE]},
        "onnx_outputs": {"actions": [1, len(names)]},
        "policy_dt": policy_dt, "training_physics_dt": cfg["sim"]["dt"], "joint_names": names,
        **values, "actuator_delay_physics_steps_training": delays,
        "initial_root_pos": list(robot["init_state"]["pos"]),
        "default_root_pos": list(robot["init_state"]["pos"]),
        "default_root_rot_xyzw": list(robot["init_state"].get("rot", [0.0, 0.0, 0.0, 1.0])),
        "action_to_target": "target[i] = default_joint_pos[i] + action_scale[i] * actions[i]; no filter, no clip",
        "last_action": "raw output from the previous policy step; zeros on entry",
        "obs_history": OBS_HISTORY, "obs_slices": slices,
        "obs_layout": "per-term histories oldest to newest, flattened; terms concatenated in obs_slices order",
        "obs_reset": "fill every history slot with the first frame",
        "obs_definitions": {
            "base_ang_vel": "base_link angular velocity in its local frame, rad/s, unscaled",
            "root_local_rot_tan_norm": "remove root yaw; concatenate columns 0 and 2 of the local rotation matrix",
            "velocity_commands": "heading-frame [vx m/s, vy m/s, wz rad/s], unscaled",
            "joint_pos": "absolute joint positions in joint_names order, rad",
            "joint_vel": "absolute joint velocities in joint_names order, rad/s",
            "last_action": "previous raw policy output",
        },
        "depth": {
            "mount_link": "base_link", "mount_pos_m": list(offset["pos"]), "mount_rot_xyzw": list(offset["rot"]),
            "mount_convention": "world: x forward, y left, z up; camera offset relative to base_link",
            "image_wh": [pattern["width"], pattern["height"]],
            "hfov_deg": hfov, "vfov_deg": vfov,
            "value": depth["data_type"], "sees_robot_itself": True,
            "crop_up_down_left_right": crop,
            "blur": {"kernel": depth["blur_kernel_size"], "sigma": depth["blur_sigma"], "padding": "reflect"},
            "depth_range_m": list(depth["depth_range"]),
            "normalize": "crop, blur; replace NaN/+inf with max and -inf with min; clamp and map depth_range to [0,1]",
            "update_period_s": camera["update_period"], "history_length": depth["history_length"],
            "history_skip": depth["history_skip_frames"], "output_frames": depth["num_output_frames"],
            "output_order": "oldest to newest",
            "delay_frames_training": depth_delays,
            "reset": "fill every history slot with the first frame",
        },
    }
    return result


def load_lite_policy(checkpoint: Path):
    if __package__:
        from .export_pt_to_onnx import load_policy
    else:
        from export_pt_to_onnx import load_policy
    net, obs_dim, frames, num_actions = load_policy(Path(checkpoint))
    if (obs_dim, frames, num_actions) != (450, 8, 26):
        raise ValueError(f"Expected Lite interface obs450/depth8/actions26, got {obs_dim}/{frames}/{num_actions}")
    return net, obs_dim, frames, num_actions


def deploy_meta(checkpoint: Path, env_yaml: Path | None = None, urdf: Path | None = None) -> dict:
    checkpoint = Path(checkpoint).expanduser().resolve()
    return {"checkpoint": checkpoint.name, "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            **load_lite_config(checkpoint, env_yaml, urdf)}


def export_lite_checkpoint(checkpoint: Path, output_dir: Path | None = None, *, env_yaml: Path | None = None,
                           urdf: Path | None = None, filename: str | None = None) -> Path:
    """Convert PT to ONNX and write same-stem JSON metadata; return the ONNX path."""
    if __package__:
        from .export_pt_to_onnx import export_onnx
    else:
        from export_pt_to_onnx import export_onnx
    checkpoint = Path(checkpoint).expanduser().resolve()
    net, obs_dim, frames, _ = load_lite_policy(checkpoint)
    metadata = deploy_meta(checkpoint, env_yaml, urdf)
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else checkpoint.parent / "exported"
    filename = filename or f"policy_{checkpoint.stem.removeprefix('model_')}.onnx"
    onnx_path = export_onnx(net, obs_dim, frames, output_dir, depth_shape=DEPTH_SHAPE, filename=filename)
    import onnx
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    # Keep bundled artifacts relocatable when the whole folder is shared.
    for field in ("checkpoint_path", "training_config", "urdf_path"):
        if metadata.get(field):
            path = Path(metadata[field])
            if path.is_relative_to(ROOT):
                metadata[field] = str(path.relative_to(ROOT))
    metadata["onnx_sha256"] = sha256(onnx_path)
    onnx_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="Lite training checkpoint (.pt)")
    parser.add_argument("--output-dir", type=Path, help="default: exported/ beside the checkpoint")
    parser.add_argument("--env-yaml", type=Path, help="default: params/env.yaml beside the checkpoint")
    parser.add_argument("--urdf", type=Path, help="optional relocated Lite URDF")
    args = parser.parse_args()
    if not args.checkpoint.expanduser().is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    path = export_lite_checkpoint(args.checkpoint, args.output_dir, env_yaml=args.env_yaml, urdf=args.urdf)
    print(f"Lite interface: obs[1,450] + depth[1,8,18,32] -> actions[1,26]")
    print(f"Deployment metadata: {path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
