#!/usr/bin/env python3
"""Lite 26-DoF AMP-depth sim2sim: PT -> ONNX -> ONNX Runtime -> MuJoCo.

Examples (from the repository root, or use an absolute script path):
    python scripts/sim2sim_mujoco_lite.py --checkpoint /path/to/run/model_200.pt --follow-view
    python scripts/sim2sim_mujoco_lite.py --checkpoint /path/to/run --terrain stairs --follow-view
    python scripts/sim2sim_mujoco_lite.py --onnx /path/to/exported/policy_200.onnx
    python scripts/sim2sim_mujoco_lite.py --checkpoint /path/to/model_200.pt --headless --duration 5

PT input is always exported before inference; a run directory selects its highest
model_<iteration>.pt. Outputs are saved under the run's exported/ directory.
Inputs: obs(1,450), depth(1,8,18,32); output: actions(1,26).
Observation/depth, PD, keyboard and viewer helpers are local to this project.

W/S forward/back, A/D lateral, Q/E turn, hold R accelerate, T reset.
Terrain choices exclude stepping stones and gaps: flat, stairs, box.
Dependencies: numpy, mujoco, onnxruntime; PT export: torch, onnx, PyYAML;
GUI: pygame and mujoco-python-viewer. Isaac Lab is not imported.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "source/lite_parkour/lite_parkour"
DEFAULT_URDF = PACKAGE_ROOT / "data/Robots/Lite/lite_26dof.urdf"
DEFAULT_XML = DEFAULT_URDF.with_suffix(".xml")
EXPORT_ROOT = REPO_ROOT / "logs/rsl_rl/lite_amp_depth"


def _load_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = _load_helper("_lite_sim2sim_helpers", REPO_ROOT / "scripts/sim2sim/runtime.py")
deployment = _load_helper("_lite_deployment", REPO_ROOT / "scripts/sim2sim/export_lite_pt_to_onnx.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_checkpoint_onnx(path):
    checkpoint = shared.resolve_checkpoint(path)
    exporter = REPO_ROOT / "scripts/sim2sim/export_lite_pt_to_onnx.py"
    print(f"[CHECKPOINT] {checkpoint}", flush=True)
    subprocess.run([sys.executable, str(exporter), "--checkpoint", str(checkpoint)], check=True)
    output = checkpoint.parent / "exported" / f"policy_{checkpoint.stem.removeprefix('model_')}.onnx"
    if not output.is_file():
        raise FileNotFoundError(f"Lite exporter did not produce {output}")
    return output


def resolve_policy(path=None):
    if path is None:
        checkpoints = list(EXPORT_ROOT.glob("*/model_*.pt"))
        if not checkpoints:
            checkpoints = list((REPO_ROOT / "checkpoints/lite").glob("model_*.pt"))
        if checkpoints:
            newest = max(checkpoints, key=lambda p: p.stat().st_mtime_ns)
            return export_checkpoint_onnx(newest.parent)
        policies = list(EXPORT_ROOT.glob("*/exported/policy*.onnx"))
        if not policies:
            raise FileNotFoundError(f"No Lite checkpoint or ONNX under {EXPORT_ROOT}; pass --checkpoint or --onnx.")
        path = max(policies, key=lambda p: p.stat().st_mtime_ns)
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".pt" or (path.is_dir() and any(path.glob("model_*.pt"))):
        return export_checkpoint_onnx(path)
    if path.is_dir():
        candidates = list(path.glob("policy*.onnx"))
        if not candidates:
            raise FileNotFoundError(f"No policy*.onnx in {path}")
        numbered = [p for p in candidates if p.stem.removeprefix("policy_").isdecimal()]
        path = max(numbered, key=lambda p: int(p.stem.removeprefix("policy_"))) if numbered else max(
            candidates, key=lambda p: p.stat().st_mtime_ns)
    if not path.is_file() or path.suffix.lower() != ".onnx":
        raise FileNotFoundError(f"Expected an ONNX policy: {path}")
    return path


def load_metadata(policy, urdf_path, joint_order=None):
    iteration = policy.stem.removeprefix("policy_")
    candidates = (policy.with_suffix(".json"),
                  policy.with_name(f"deploy_{iteration}.json"), policy.with_name("deploy.json"))
    metadata = None
    for path in candidates:
        if not path.is_file():
            continue
        candidate = json.loads(path.read_text())
        digest = candidate.get("onnx_sha256")
        if digest is not None and digest != sha256(policy):
            if path == candidates[-1]:
                continue  # A legacy deploy.json can belong to another numbered policy.
            raise ValueError(f"ONNX checksum does not match {path}; re-export the checkpoint.")
        metadata = candidate
        break
    if metadata is None:
        print("[INFO] No matching deployment metadata; using current Lite training parameters.")
        metadata = deployment.load_lite_config(None, urdf=urdf_path)
    if metadata.get("urdf_sha256") is not None and metadata["urdf_sha256"] != sha256(urdf_path):
        raise ValueError("Lite URDF differs from the exported deployment metadata; regenerate the model and re-export.")
    expected = [j.get("name") for j in ET.parse(urdf_path).getroot().findall("joint") if j.get("type") != "fixed"]
    names = metadata["joint_names"]
    if len(names) != 26 or len(set(names)) != 26 or set(names) != set(expected):
        raise ValueError("Policy metadata must contain the 26 unlocked Lite joints exactly once.")
    if joint_order is not None:
        manifest = json.loads(joint_order.expanduser().resolve().read_text())
        selected = manifest.get("policy_action_joint_names", manifest.get("joint_names"))
        if selected != names:
            raise ValueError("--joint-order disagrees with the Lite deployment metadata; re-export the correct policy.")
    for key in ("kp", "kd", "armature", "effort_limit", "velocity_limit", "default_joint_pos", "action_scale"):
        value = np.asarray(metadata[key], dtype=float)
        if value.shape != (26,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid deployment parameter {key}: expected 26 finite values")
    if metadata["depth"]["mount_link"] != "base_link":
        raise ValueError("Lite's camera must be attached to base_link")
    return metadata


def add_stairs(world, step_height):
    """Two-metre-wide stairs with light-gray treads over darker risers."""
    shared.add_stairs(world, step_height)
    for step in list(world.findall("geom")):
        if not step.get("name", "").startswith("stairs_"):
            continue
        step.set("rgba", "0.44 0.44 0.44 1")
        center = np.fromstring(step.get("pos"), sep=" ")
        half_size = np.fromstring(step.get("size"), sep=" ")
        half_size[1] = 1.0  # MuJoCo box sizes are half-extents: Y spans [-1, 1] m.
        step.set("size", shared.numbers(half_size))
        # Group 1 is visible in RGB but excluded from policy depth rays. The
        # 0.2 mm visual offset prevents z-fighting; contact surfaces stay intact.
        ET.SubElement(world, "geom", name=step.get("name") + "_tread", type="plane",
                      pos=shared.numbers([*center[:2], center[2] + half_size[2] + 0.0002]),
                      size=shared.numbers([*half_size[:2], 0.001]),
                      group="1", contype="0", conaffinity="0", rgba="0.60 0.60 0.60 1")


def build_model(xml_path, urdf_path, metadata, dt, terrain, step_height):
    """Prepare a private MJCF copy; the generated source file is never modified."""
    root = ET.parse(xml_path).getroot()
    shared.configure_scene_lighting(root)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    for name in ("meshdir", "texturedir"):
        compiler.set(name, str((xml_path.parent / compiler.get(name, ".")).resolve()))
    compiler.set("inertiafromgeom", "false")
    for include in root.iter("include"):
        include.set("file", str((xml_path.parent / include.get("file")).resolve()))
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.attrib.update(timestep=str(dt), integrator="implicitfast", solver="Newton", iterations="50", gravity="0 0 -9.81")
    world = root.find("worldbody")
    if world is None:
        raise ValueError("Lite MJCF has no worldbody")
    bodies = {b.get("name"): b for b in world.iter("body")}
    if "base_link" not in bodies:
        raise ValueError("Lite MJCF has no base_link body")
    # Rebuild collisions from the current URDF. Mesh contact proxies stay in group
    # 3, while visual meshes in group 2 are visible to the depth rays.
    mesh_names = {Path(m.get("file")).name: m.get("name") for m in root.findall("asset/mesh")}
    for link in ET.parse(urdf_path).getroot().findall("link"):
        body = bodies.get(link.get("name"))
        if body is None:
            if link.get("name") == "pe" and len(link) == 0:
                continue  # Empty fixed pelvis may be fused by a custom MJCF.
            raise ValueError(f"Lite MJCF is missing URDF body {link.get('name')}")
        for geom in list(body.findall("geom")):
            if geom.get("group") == "2":
                geom.attrib.update(contype="0", conaffinity="0")
            else:
                body.remove(geom)
        for index, collision in enumerate(link.findall("collision")):
            origin = collision.find("origin")
            pos = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
            shape = next(iter(collision.find("geometry")))
            attr = dict(name=f"contact_{link.get('name')}_{index}", pos=pos,
                        quat=shared.numbers(shared.rpy_quaternion(np.fromstring(rpy, sep=" "))),
                        group="3", contype="1", conaffinity="1", condim="3",
                        friction="1 0.005 0.0001", solref="0.01 1", solimp="0.9 0.95 0.001",
                        rgba="0.9 0.25 0.2 0.5")
            if shape.tag == "cylinder":
                radius, length = float(shape.get("radius")), float(shape.get("length"))
                if length <= 2 * radius:
                    attr.update(type="sphere", size=shared.numbers([radius]))
                else:
                    attr.update(type="capsule", size=shared.numbers([radius, length / 2 - radius]))
            elif shape.tag == "box":
                attr.update(type="box", size=shared.numbers(np.fromstring(shape.get("size"), sep=" ") / 2))
            elif shape.tag == "sphere":
                attr.update(type="sphere", size=shape.get("radius"))
            elif shape.tag == "mesh":
                filename = Path(shape.get("filename")).name
                if filename not in mesh_names:
                    raise ValueError(f"MJCF is missing collision mesh {filename}")
                attr.update(type="mesh", mesh=mesh_names[filename])
            else:
                raise ValueError(f"Unsupported Lite collision shape: {shape.tag}")
            ET.SubElement(body, "geom", attr)
    joints = {j.get("name"): j for b in bodies.values() for j in b.findall("joint")}
    movable = {j.get("name"): j for j in ET.parse(urdf_path).getroot().findall("joint") if j.get("type") != "fixed"}
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    actuator.clear()
    for i, name in enumerate(metadata["joint_names"]):
        joint = joints.get(name)
        if joint is None:
            joint = joints.get(name.removesuffix("_joint"))
        if joint is None:
            raise ValueError(f"MJCF is missing joint {name}")
        limit = movable[name].find("limit")
        joint.attrib.update(range=f"{limit.get('lower')} {limit.get('upper')}", damping="0", frictionloss="0",
                            armature=str(metadata["armature"][i]))
        effort = metadata["effort_limit"][i]
        ET.SubElement(actuator, "motor", name=name + "_motor", joint=joint.get("name"), gear="1",
                      ctrllimited="true", ctrlrange=shared.numbers([-effort, effort]),
                      forcelimited="true", forcerange=shared.numbers([-effort, effort]))
    heights = None
    if terrain == "stairs":
        add_stairs(world, step_height)
    elif terrain == "box":
        heights = shared.add_box_terrain(root)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if heights is not None:
        hfield = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "box_rough_hfield")
        address = model.hfield_adr[hfield]
        model.hfield_data[address:address + heights.size] = heights.ravel()
    if model.nu != 26 or model.nq != 33 or model.nv != 32:
        raise ValueError("Expected a floating-base Lite with 26 actuated hinges (nq=33, nv=32, nu=26).")
    return model


class OnnxPolicy:
    def __init__(self, path, device="cpu"):
        try:
            import onnxruntime as ort
        except ImportError:
            raise SystemExit(f"Install ONNX Runtime into this Python: {sys.executable} -m pip install onnxruntime") from None
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise ValueError("--device cuda requires onnxruntime-gpu; use --device cpu otherwise.")
            providers.insert(0, "CUDAExecutionProvider")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        inputs = {i.name: i for i in self.session.get_inputs()}
        if set(inputs) != {"obs", "depth"}:
            raise ValueError("Expected Lite ONNX inputs obs and depth")
        for name, shape in (("obs", (1, 450)), ("depth", (1, 8, 18, 32))):
            actual = inputs[name]
            if actual.type != "tensor(float)" or len(actual.shape) != len(shape) or any(
                isinstance(a, int) and a != b for a, b in zip(actual.shape, shape)
            ):
                raise ValueError(f"Wrong Lite {name} input: {actual.shape}, expected {shape}")
        outputs = self.session.get_outputs()
        if len(outputs) != 1 or outputs[0].type != "tensor(float)":
            raise ValueError("Expected a single float32 Lite actions output")
        if len(outputs[0].shape) != 2 or any(isinstance(a, int) and a != b for a, b in zip(outputs[0].shape, (1, 26))):
            raise ValueError(f"Wrong Lite actions output: {outputs[0].shape}, expected (1,26)")
        self.output_name = outputs[0].name

    def __call__(self, proprio, depth):
        actions = self.session.run([self.output_name], {"obs": proprio, "depth": depth})[0]
        if actions.shape != (1, 26) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite Lite actions (1,26), got {actions.shape}")
        return actions[0].astype(np.float64)


class DepthCamera(shared.DepthCamera):
    def __init__(self, model, metadata, self_occlusion=True):
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.local_rays = shared.build_local_rays()
        self.geomid = np.empty(len(self.local_rays), dtype=np.int32)
        self.distance = np.empty(len(self.local_rays), dtype=np.float64)
        self.geom_group = shared.DEPTH_GEOM_GROUP.copy()
        self.geom_group[2] = int(self_occlusion)
        self.history = shared.HistoryBuffer(37, shared.DEPTH_SHAPE)
        self.with_normals = True
        self.local_pos = np.asarray(metadata["depth"]["mount_pos_m"], dtype=float)
        quat = np.asarray(metadata["depth"]["mount_rot_xyzw"], dtype=float)
        self.local_rot = np.empty((3, 3))
        mujoco.mju_quat2Mat(self.local_rot.ravel(), quat[[3, 0, 1, 2]])

    def pose(self, data):
        rotation = data.xmat[self.body_id].reshape(3, 3)
        return data.xpos[self.body_id] + rotation @ self.local_pos, rotation @ self.local_rot


class LiteSimulation(shared.JointPositionSimulation):
    def __init__(self, args, model, metadata, policy):
        self.args, self.model, self.names, self.policy = args, model, metadata["joint_names"], policy
        self.data = mujoco.MjData(model)
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.joint_ids = np.asarray([self._joint_id(n) for n in self.names])
        self.qpos_ids, self.qvel_ids = model.jnt_qposadr[self.joint_ids], model.jnt_dofadr[self.joint_ids]
        self.motor_ids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_motor") for n in self.names])
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1 or np.any(self.motor_ids < 0):
            raise ValueError("Lite requires one free base and one torque motor per action joint")
        self.root_qpos = model.jnt_qposadr[free[0]]
        self.q0, self.kp, self.kd, self.effort, self.speed, self.action_scale = (
            np.asarray(metadata[k], dtype=float) for k in
            ("default_joint_pos", "kp", "kd", "effort_limit", "velocity_limit", "action_scale"))
        self.initial_root_pos = np.asarray(metadata["initial_root_pos"], dtype=float)
        self.initial_root_rot = np.asarray(metadata.get("default_root_rot_xyzw", [0, 0, 0, 1]), dtype=float)[[3, 0, 1, 2]]
        self.camera = DepthCamera(model, metadata, not args.no_self_occlusion)
        self.last_action = np.zeros(26)
        self.target = self.q0.copy()
        self.histories = [shared.HistoryBuffer(5, (width,)) for width in (3, 6, 3, 26, 26, 26)]
        self.command = np.asarray([args.cmd_vx, args.cmd_vy, args.cmd_dyaw], dtype=float)
        self.heading = args.heading
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.root_qpos:self.root_qpos + 7] = [*self.initial_root_pos, *self.initial_root_rot]
        self.data.qpos[self.qpos_ids] = self.q0
        mujoco.mj_forward(self.model, self.data)
        self.last_action[:] = 0
        self.target[:] = self.q0
        for history in self.histories + [self.camera.history]:
            history.initialized = False
        self.command[:] = [self.args.cmd_vx, self.args.cmd_vy, self.args.cmd_dyaw]
        self.heading = self.args.heading


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", "--pt", type=Path, help="PT file or run directory; export ONNX first")
    source.add_argument("--onnx", "--model-dir", "--model_dir", type=Path, help="ONNX/PT file or exported/run directory")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--joint-order", type=Path)
    parser.add_argument("--terrain", choices=("flat", "stairs", "box"), default="flat")
    parser.add_argument("--step-height", type=float, default=0.20)
    parser.add_argument("--cmd-vx", "--cmd_vx", type=float, default=0.6)
    parser.add_argument("--cmd-vy", "--cmd_vy", type=float, default=0.0)
    parser.add_argument("--cmd-dyaw", "--cmd_dyaw", type=float, default=0.0)
    parser.add_argument("--heading", type=float)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument("--duration", "--sim_duration", type=float, default=600)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--follow-view", "--follow_view", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--no-self-occlusion", action="store_true")
    parser.add_argument("--save-frame", type=Path)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    if args.dt <= 0 or args.decimation < 1 or not math.isclose(args.dt * args.decimation, 0.02, abs_tol=1e-9):
        parser.error("Keep dt * decimation = 0.02 s to match the trained observation/depth histories")
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    if not -0.8 <= args.cmd_vx <= 1.5 or not -0.5 <= args.cmd_vy <= 0.5 or not -1 <= args.cmd_dyaw <= 1:
        parser.error("Use vx in [-0.8,1.5], vy in [-0.5,0.5], yaw rate in [-1,1]")
    if not 0 < args.step_height <= 0.3:
        parser.error("--step-height must be in (0,0.3]")
    if args.heading is not None and not math.isfinite(args.heading):
        parser.error("--heading must be finite")
    return args


def main():
    args = parse_args()
    policy_path = resolve_policy(args.checkpoint if args.checkpoint is not None else args.onnx)
    metadata = load_metadata(policy_path, args.urdf.expanduser().resolve(), args.joint_order)
    if args.export_only:
        print(f"[ONNX] {policy_path}")
        return
    policy = OnnxPolicy(policy_path, args.device)
    model = build_model(args.xml.expanduser().resolve(), args.urdf.expanduser().resolve(), metadata,
                        args.dt, args.terrain, args.step_height)
    simulation = LiteSimulation(args, model, metadata, policy)
    print(f"[POLICY] {policy_path}\n[MODEL] Lite 26 hinges, floating base, {model.ngeom} geoms, terrain={args.terrain}")
    print(f"[IO] obs=(1,450), depth=(1,8,18,32), actions=(1,26); dt={args.dt}, decimation={args.decimation}")
    print("[DEPTH] base_link camera, position=" + str(simulation.camera.local_pos.tolist()))
    print("[KEYS] Focus Keyboard Window: W/S forward/back, A/D strafe, Q/E turn, hold R accelerate, T reset.")
    context = nullcontext((None, None)) if args.headless else shared.interactive_viewer(simulation)
    minimum_height, start = float("inf"), time.perf_counter()
    with context as (viewer, keyboard):
        for step in range(int(math.ceil(args.duration / args.dt))):
            if viewer is not None and not viewer.is_alive:
                break
            if keyboard is not None and not keyboard.poll(simulation):
                break
            if step % args.decimation == 0:
                simulation.control_step()
                if keyboard is not None:
                    keyboard.draw_command(simulation)
            simulation.physics_step()
            minimum_height = min(minimum_height, simulation.data.qpos[simulation.root_qpos + 2])
            if viewer is not None:
                mujoco.mj_forward(model, simulation.data)
                shared.draw_camera(viewer, simulation)
                viewer.render()
        position = simulation.data.qpos[simulation.root_qpos:simulation.root_qpos + 3]
        print(f"[DONE] wall={time.perf_counter() - start:.2f}s, position={position.round(3)}, min_base_z={minimum_height:.3f}m")
    if args.save_frame is not None:
        shared.save_frame(args.save_frame, simulation)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
