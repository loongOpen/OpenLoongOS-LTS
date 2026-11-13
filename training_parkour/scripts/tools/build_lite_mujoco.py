#!/usr/bin/env python3
"""Build Lite's standalone MuJoCo scene from its training URDF.

Run from the repository root:
    python scripts/tools/build_lite_mujoco.py --check

The URDF remains the source of truth for link frames, inertias, meshes and
limits. Rotor inertias and the home pose are read from assets/lite.py without
importing Isaac Lab. Torque motors leave PD control to the sim2sim runtime.
Visual meshes use group 2; training collision proxies use group 3, which the
runtime hides from depth rays and rendering. URDF cylinders become capsules
with the same total extent, matching the training URDF importer.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


PACKAGE = Path(__file__).resolve().parents[2] / "source/lite_parkour/lite_parkour"
DEFAULT_URDF = PACKAGE / "data/Robots/Lite/lite_26dof.urdf"
DEFAULT_OUTPUT = DEFAULT_URDF.with_suffix(".xml")
DEFAULT_ASSET = PACKAGE / "assets/lite.py"


def numbers(values) -> str:
    return " ".join(f"{float(value):.16g}" for value in values)


def origin(element: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    if element is None:
        return np.zeros(3), np.zeros(3)
    return tuple(np.fromstring(element.get(key, "0 0 0"), sep=" ") for key in ("xyz", "rpy"))


def rpy_quaternion(rpy) -> np.ndarray:
    r, p, y = np.asarray(rpy) / 2
    cr, cp, cy = math.cos(r), math.cos(p), math.cos(y)
    sr, sp, sy = math.sin(r), math.sin(p), math.sin(y)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def rpy_matrix(rpy) -> np.ndarray:
    w, x, y, z = rpy_quaternion(rpy)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def training_parameters(path: Path):
    """Read literal configuration values without starting an Isaac application."""
    tree = ast.parse(path.read_text())
    parameters = None
    home, base_position = {}, (0.0, 0.0, 0.99)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_ACTUATORS" for t in node.targets):
            parameters = ast.literal_eval(node.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "InitialStateCfg":
            for keyword in node.keywords:
                if keyword.arg == "joint_pos":
                    home = ast.literal_eval(keyword.value)
                elif keyword.arg == "pos":
                    base_position = ast.literal_eval(keyword.value)
    if parameters is None:
        raise ValueError(f"Missing literal _ACTUATORS dictionary in {path}")
    return parameters, home, base_position


def joint_kind(name: str) -> str:
    return name.removeprefix("left_").removeprefix("right_").removesuffix("_joint")


def urdf_inertia(link: ET.Element):
    inertial = link.find("inertial")
    if inertial is None:
        return None
    xyz, rpy = origin(inertial.find("origin"))
    inertia = {key: float(value) for key, value in inertial.find("inertia").attrib.items()}
    tensor = np.array([[inertia["ixx"], inertia["ixy"], inertia["ixz"]],
                       [inertia["ixy"], inertia["iyy"], inertia["iyz"]],
                       [inertia["ixz"], inertia["iyz"], inertia["izz"]]])
    rotation = rpy_matrix(rpy)
    return float(inertial.find("mass").get("value")), xyz, rotation @ tensor @ rotation.T


def collision_attributes(shape: ET.Element) -> dict[str, str]:
    if shape.tag == "cylinder":
        radius, length = float(shape.get("radius")), float(shape.get("length"))
        if length <= 2 * radius:
            return dict(type="sphere", size=numbers([radius]))
        return dict(type="capsule", size=numbers([radius, length / 2 - radius]))
    if shape.tag == "box":
        return dict(type="box", size=numbers(np.fromstring(shape.get("size"), sep=" ") / 2))
    if shape.tag == "sphere":
        return dict(type="sphere", size=shape.get("radius"))
    raise ValueError(f"Unsupported Lite collision geometry: {shape.tag}")


def build_scene(urdf_path: Path, asset_path: Path, output_path: Path) -> ET.Element:
    urdf = ET.parse(urdf_path).getroot()
    parameters, home, base_position = training_parameters(asset_path)
    links = {link.get("name"): link for link in urdf.findall("link")}
    children = {name: [] for name in links}
    movable = []
    for joint in urdf.findall("joint"):
        children[joint.find("parent").get("link")].append(joint)
        if joint.get("type") != "fixed":
            if joint.get("type") != "revolute":
                raise ValueError(f"Expected fixed/revolute joints, got {joint.attrib}")
            movable.append(joint)
    if len(movable) != 26 or any("waist" in joint.get("name") for joint in movable):
        raise ValueError("Lite must have 26 active joints, including all six wrists, and no active waist")
    for side in ("left", "right"):
        for axis in ("roll", "pitch", "yaw"):
            if not any(joint.get("name") == f"{side}_wrist_{axis}_joint" for joint in movable):
                raise ValueError(f"Missing active {side} wrist {axis}")

    root = ET.Element("mujoco", model="lite_26dof")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", inertiafromgeom="false",
                  fusestatic="false", meshdir=".")
    ET.SubElement(root, "option", timestep="0.005", integrator="implicitfast", solver="Newton",
                  iterations="50", gravity="0 0 -9.81")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "quality", shadowsize="4096")
    ET.SubElement(visual, "map", znear="0.01")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1="0.6 0.75 0.9",
                  rgb2="0.85 0.9 0.96", width="512", height="512")
    ET.SubElement(asset, "texture", name="texplane", type="2d", builtin="checker",
                  rgb1="0.55 0.61 0.67", rgb2="0.4 0.46 0.52", width="512", height="512",
                  mark="cross", markrgb="0.85 0.88 0.92")
    ET.SubElement(asset, "material", name="matplane", texture="texplane", texrepeat="1 1",
                  texuniform="true", reflectance="0")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", name="key", directional="true", diffuse="0.7 0.7 0.7",
                  pos="1 -1 3", dir="-1 1 -3")
    ET.SubElement(world, "light", name="fill", directional="true", diffuse="0.4 0.4 0.4",
                  pos="-1 1 2", dir="1 -1 -2")
    ET.SubElement(world, "geom", name="ground", type="plane", size="0 0 1", material="matplane",
                  group="0", condim="3", friction="1 0.005 0.0001")

    traversal = []

    def add_mesh(shape: ET.Element, mesh_name: str) -> str:
        filename = shape.get("filename")
        if filename.startswith("package://"):
            raise ValueError(f"Resolve package mesh URI before building: {filename}")
        mesh_path = (urdf_path.parent / filename).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        relative_path = os.path.relpath(mesh_path, output_path.parent)
        for existing in asset.findall("mesh"):
            if existing.get("file") == relative_path and existing.get("scale", "1 1 1") == shape.get("scale", "1 1 1"):
                return existing.get("name")
        mesh = ET.SubElement(asset, "mesh", name=mesh_name, file=relative_path)
        if shape.get("scale"):
            mesh.set("scale", shape.get("scale"))
        return mesh_name

    def add_link(parent: ET.Element, name: str, parent_joint: ET.Element | None = None):
        xyz, rpy = origin(parent_joint.find("origin") if parent_joint is not None else None)
        body = ET.SubElement(parent, "body", name=name,
                             pos=numbers(base_position if parent_joint is None else xyz),
                             quat=numbers(rpy_quaternion(rpy)))
        if parent_joint is None:
            ET.SubElement(body, "freejoint", name="floating_base_joint")
        elif parent_joint.get("type") != "fixed":
            joint_name = parent_joint.get("name")
            limit = parent_joint.find("limit")
            ET.SubElement(body, "joint", name=joint_name, type="hinge", pos="0 0 0",
                          axis=parent_joint.find("axis").get("xyz"),
                          range=f"{limit.get('lower')} {limit.get('upper')}", damping="0", frictionloss="0",
                          armature=numbers([parameters[joint_kind(joint_name)][2]]))
            traversal.append(parent_joint)
        link = links[name]
        inertial = urdf_inertia(link)
        if inertial is not None:
            mass, xyz, tensor = inertial
            principal, axes = np.linalg.eigh(tensor)
            if np.linalg.det(axes) < 0:
                axes[:, -1] *= -1
            # Diagonalize explicitly: MuJoCo's fullinertia eigensolver rounds
            # tiny off-diagonal terms more than numpy's symmetric solver.
            quaternion = Rotation.from_matrix(axes).as_quat()[[3, 0, 1, 2]]
            ET.SubElement(body, "inertial", mass=numbers([mass]), pos=numbers(xyz),
                          diaginertia=numbers(principal), quat=numbers(quaternion))
        for index, urdf_visual in enumerate(link.findall("visual")):
            xyz, rpy = origin(urdf_visual.find("origin"))
            shape = next(iter(urdf_visual.find("geometry")))
            if shape.tag != "mesh":
                raise ValueError(f"Expected Lite visual mesh, got {shape.tag} on {name}")
            mesh_name = add_mesh(shape, f"{name}_visual_mesh_{index}")
            color = urdf_visual.find("material/color")
            ET.SubElement(body, "geom", name=f"{name}_visual_{index}", type="mesh", mesh=mesh_name,
                          pos=numbers(xyz), quat=numbers(rpy_quaternion(rpy)), group="2",
                          contype="0", conaffinity="0", rgba=color.get("rgba") if color is not None else "0.8 0.8 0.8 1")
        for index, collision in enumerate(link.findall("collision")):
            xyz, rpy = origin(collision.find("origin"))
            shape = next(iter(collision.find("geometry")))
            attributes = (dict(type="mesh", mesh=add_mesh(shape, f"{name}_collision_mesh_{index}"))
                          if shape.tag == "mesh" else collision_attributes(shape))
            ET.SubElement(body, "geom", name=f"{name}_{index}_collision", pos=numbers(xyz),
                          quat=numbers(rpy_quaternion(rpy)), group="3", contype="1", conaffinity="1",
                          condim="3", friction="1 0.005 0.0001", solref="0.01 1",
                          solimp="0.9 0.95 0.001", rgba="0.9 0.25 0.2 0.5", **attributes)
        for child in children[name]:
            add_link(body, child.find("child").get("link"), child)

    add_link(world, "base_link")
    if len(traversal) != len(movable):
        raise ValueError("URDF contains disconnected joints")
    actuator = ET.SubElement(root, "actuator")
    home_qpos = [*base_position, 1, 0, 0, 0]
    for joint in traversal:
        name = joint.get("name")
        effort = float(joint.find("limit").get("effort"))
        ET.SubElement(actuator, "motor", name=f"{name}_motor", joint=name, gear="1",
                      ctrllimited="true", ctrlrange=numbers([-effort, effort]),
                      forcelimited="true", forcerange=numbers([-effort, effort]))
        matches = [value for pattern, value in home.items() if re.fullmatch(pattern, name)]
        home_qpos.append(matches[-1] if matches else 0.0)
    ET.SubElement(ET.SubElement(root, "keyframe"), "key", name="home", qpos=numbers(home_qpos))
    ET.indent(root, space="  ")
    return root


def check_model(xml_path: Path, urdf_path: Path, samples: int = 64) -> dict:
    """Compare every body FK and physical inertia to independent URDF parsing."""
    import mujoco

    spec = importlib.util.spec_from_file_location("_lite_urdf_fk", Path(__file__).with_name("kinematics.py"))
    kinematics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kinematics)
    urdf = ET.parse(urdf_path).getroot()
    joints = [joint for joint in urdf.findall("joint") if joint.get("type") != "fixed"]
    names = [joint.get("name") for joint in joints]
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    if (model.nq, model.nv, model.nu) != (33, 32, 26):
        raise AssertionError(f"Wrong Lite dimensions: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    rng = np.random.default_rng(20261111)
    positions = np.column_stack([rng.uniform(float(j.find("limit").get("lower")),
                                           float(j.find("limit").get("upper")), samples) for j in joints])
    positions[0] = 0
    reference = kinematics.forward_kinematics(urdf, "base_link", dict(zip(names, positions.T)), samples)
    joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names])
    qpos_ids = model.jnt_qposadr[joint_ids]
    max_position, max_rotation, max_inertia, max_mass = 0.0, 0.0, 0.0, 0.0
    for sample, position in enumerate(positions):
        data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        data.qpos[qpos_ids] = position
        mujoco.mj_forward(model, data)
        for name, transforms in reference.items():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                raise AssertionError(f"Missing URDF body {name}")
            max_position = max(max_position, float(np.max(np.abs(data.xpos[body_id] - transforms[sample, :3, 3]))))
            max_rotation = max(max_rotation, float(np.max(np.abs(data.xmat[body_id].reshape(3, 3) - transforms[sample, :3, :3]))))
    for link in urdf.findall("link"):
        inertial = urdf_inertia(link)
        if inertial is None:
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link.get("name"))
        mass, xyz, tensor = inertial
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.body_iquat[body_id])
        rotation = rotation.reshape(3, 3)
        actual = rotation @ np.diag(model.body_inertia[body_id]) @ rotation.T
        max_inertia = max(max_inertia, float(np.max(np.abs(actual - tensor))))
        max_mass = max(max_mass, abs(float(model.body_mass[body_id]) - mass))
        np.testing.assert_allclose(model.body_ipos[body_id], xyz, atol=1e-12)
    for value, tolerance in ((max_position, 1e-10), (max_rotation, 1e-10), (max_inertia, 1e-10), (max_mass, 1e-12)):
        if value > tolerance:
            raise AssertionError(f"MuJoCo/URDF mismatch: {value:g} > {tolerance:g}")
    return dict(joints=len(names), bodies=len(reference), samples=samples,
                max_position_error_m=max_position, max_rotation_matrix_error=max_rotation,
                max_inertia_error_kg_m2=max_inertia, max_mass_error_kg=max_mass,
                nq=model.nq, nv=model.nv, nu=model.nu)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--asset-config", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Verify MuJoCo FK and inertias against the URDF")
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    output = args.output.expanduser().resolve()
    if output == args.urdf.expanduser().resolve():
        parser.error("Output must not overwrite the source URDF")
    root = build_scene(args.urdf.expanduser().resolve(), args.asset_config.expanduser().resolve(), output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    print(f"Generated {output}")
    if args.check:
        print(json.dumps(check_model(output, args.urdf, args.samples), indent=2))


if __name__ == "__main__":
    main()
