#!/usr/bin/env python3
"""Check an assembled bundle without starting any robot process."""
import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import re
import sys


def check(root):
    root = root.resolve()
    manifest = json.loads((root / "RELEASE.json").read_text())
    errors = []
    controller = manifest["components"].get("loong_gui_ctrl")
    if controller and controller.get("included", True):
        executable = root / controller["binary"]
        if not executable.is_file():
            errors.append(f"Missing controller executable: {controller['binary']}")
        elif executable.read_bytes()[:4] != b"\x7fELF" or not executable.stat().st_mode & 0o111:
            errors.append("Controller must be an executable Linux ELF file")
    inventory = json.loads((root / "FILE_HASHES.json").read_text())
    for name, expected in inventory.items():
        path = root / name
        if not path.is_file():
            errors.append(f"Missing: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"Modified: {name}")
    # A Git checkout and an installed training environment are valid user setups.
    ignored_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache",
                    "logs", "runtime_logs", "wandb", ".vscode"}
    paths = []
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in ignored_dirs and not (
            d == "log" and Path(parent) == root / "loong_gui_sim/runtime")]
        paths.extend(Path(parent) / n for n in dirs + files)
    for path in paths:
        if path.is_symlink() and root not in path.resolve().parents:
            errors.append(f"External link: {path.relative_to(root)}")
        if path.name == ".release_build":
            errors.append(f"Development artifact: {path.relative_to(root)}")
        if (path.name.startswith(("robot_gui_", "robot_control_gui"))
                and path.suffix in (".py", ".pyc", ".pyx", ".c", ".cpp")):
            errors.append(f"Controller source: {path.relative_to(root)}")
        if path.is_file():
            with path.open("rb") as stream:
                if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    errors.append(f"Git LFS pointer: {path.relative_to(root)}")
    sim = root / "loong_gui_sim"
    for path in sim.rglob("*"):
        if path.suffix.lower() in (".py", ".pyc", ".pyx", ".c", ".cpp", ".h", ".hpp", ".hxx", ".a", ".sh"):
            errors.append(f"Simulator source/build file: {path.relative_to(root)}")
    for name in ("loong_gui_sim", "runtime/bin/loong_share_sim_x64", "runtime/bin/loong_driver_x64",
                 "runtime/bin/loong_interface_x64", "runtime/bin/loong_locomotion_x64"):
        path = sim / name
        if not path.is_file() or not path.stat().st_mode & 0o111:
            errors.append(f"Missing simulator executable: {name}")
        elif path.open("rb").read(4) != b"\x7fELF":
            errors.append(f"Not a native executable: {name}")
    for name in ("native/libsim_camera_follow.so", "native/liblog_replay.so", "runtime/config/joint_map.ini"):
        if not (sim / name).is_file():
            errors.append(f"Missing simulator runtime file: {name}")
    config = configparser.ConfigParser(interpolation=None)
    text = (sim / "runtime/config/loco_plugins.ini").read_text(encoding="utf-8-sig")
    config.read_string("\n".join(line.split(";", 1)[0].split("#", 1)[0] for line in text.splitlines()))
    keys = set()
    for section in config.sections():
        library = config.get(section, "lib", fallback="").strip()
        key = config.getint(section, "key", fallback=-1)
        if key in keys or not 0 <= key <= 65535:
            errors.append(f"Invalid plugin key: {section}")
        keys.add(key)
        if not library or Path(library).name != library or not (sim / "runtime/module/nabo_locomotion" / f"{library}_x64.so").is_file():
            errors.append(f"Missing plugin library: {section}")
    if config.sections() != ["rc", "rl", "jntTest"]:
        errors.append("Expected only rc, rl and jntTest plugins")
    expected_libraries = {"libplan_recover_x64.so", "libplan_rl_x64.so", "libplan_jnt_test_x64.so"}
    if {p.name for p in (sim / "runtime/module/nabo_locomotion").glob("libplan*")} != expected_libraries:
        errors.append("Unexpected or missing plan libraries")
    if {p.name for p in (sim / "runtime/config").glob("plan_*.ini")} != {
            "plan_base.ini", "plan_rc.ini", "plan_rl.ini", "plan_jnt_sdk.ini", "plan_jnt_test.ini"}:
        errors.append("Unexpected or missing plan configurations")
    if {p.name for p in (sim / "runtime/model/onnx").iterdir()} != {"policy.onnx"}:
        errors.append("Runtime must contain only the uploaded policy")
    def array(name, key):
        text = (sim / "runtime/config" / name).read_text()
        values = []
        found = False
        for line in text.splitlines():
            line = line.split(";", 1)[0].split("#", 1)[0].strip()
            if not line:
                continue
            if found and ("=" in line or line.startswith("[")):
                break
            if not found:
                if "=" not in line or line.split("=", 1)[0].strip() != key:
                    continue
                found = True
                line = line.split("=", 1)[1]
            values.extend(float(v.strip()) for v in line.split(",") if v.strip())
        return values
    if array("driver.ini", "pdMode") != [3.] * 31:
        errors.append("Driver PD mode must match upstream main (3)")
    fixed_gains = {
        "jntKp": [130,20,20,20,20,20,20,130.23,20,20,20,20,20,20] + [20] * 5 + [280,240,100,280,50,50] * 2,
        "jntKd": [8.3,.1,.1,.1,.1,.1,.1] * 2 + [.1] * 5 + [14,12,5,14,5,5] * 2,
    }
    for gain, expected in fixed_gains.items():
        if array("driver_mujoco.ini", gain) != expected:
            errors.append(f"Simulator {gain} differs from original 26-axis MuJoCo PD")
    policy_to_bus = [*range(19, 31), *range(4), *range(7, 11)]
    mapping = array("plan_rl.ini", "idMap")
    if len(mapping) != 31 or [mapping[bus] for bus in policy_to_bus] != list(range(20)):
        errors.append("Incorrect policy joint mapping")
    metadata = json.loads((root / "models/policy.metadata.json").read_text())["metadata"]
    for field, key in (("default_joint_pos", "default_joint_angle"), ("action_scale", "scales_action"),
                       ("joint_stiffness", "jntKp"), ("joint_damping", "jntKd")):
        values = array("plan_rl.ini", key)
        if len(values) != 31 or [values[bus] for bus in policy_to_bus] != [float(x) for x in metadata[field].split(",")]:
            errors.append(f"RL parameters differ from model metadata: {key}")
    expected_map = [*range(12, 19), *range(19, 26), -1, -1, -1, -1, -1, *range(12)]
    for name in ("driver.ini", "driver_mujoco.ini"):
        if array(name, "idMap") != expected_map:
            errors.append(f"Incorrect LingLong1.4 mapping: {name}")
    if array("driver.ini", "activeDrvNumUser") != [26]:
        errors.append("Expected 26 active legacy simulation joints")
    if array("driver.ini", "useJntMotMap") != [0]:
        errors.append("Legacy simulation requires direct joint coordinates")
    driver_text = (sim / "runtime/config/driver.ini").read_text()
    if not re.search(r"^\s*simAnkleMap=none\s*$", driver_text, re.M):
        errors.append("Legacy simulation requires simAnkleMap=none")
    if (sim / "runtime/config/driver_mujoco.ini").read_bytes() != (sim / "runtime/config/driver_mujoco.ini.bak_26dof").read_bytes():
        errors.append("MuJoCo driver must match the original 26-axis backup byte for byte")
    if "untaggedPoseSpace=joint" not in (sim / "runtime/config/replay_log.ini").read_text():
        errors.append("Direct-joint logs must not be converted from motor angles")
    import xml.etree.ElementTree as ET
    scene = ET.parse(sim / "runtime/model/scene.xml").getroot()
    assert scene.find("include").get("file") == manifest["robot"]["model"]
    robot = ET.parse(sim / "runtime/model" / manifest["robot"]["model"]).getroot()
    joints = [j for j in robot.findall(".//body/joint") if j.get("type") != "free"]
    motors = robot.findall("actuator/motor")
    if len(joints) != 26 or [j.get("name") for j in joints] != [m.get("joint") for m in motors]:
        errors.append("LingLong1.4 joint / actuator count or order mismatch")
    if len(joints) == 26:
        def joint_name(name):
            return name[:-6] if name.endswith("_joint") else name
        names = [joint_name(joints[expected_map[bus]].get("name")) for bus in policy_to_bus]
        trained_names = [joint_name(name) for name in metadata["joint_names"].split(",")]
        if names != trained_names:
            errors.append("Policy and model anatomical joint names differ")
    for name in ("pyproject.toml", "uv.lock", "LICENSE", "src/mjlab/tasks/amp/rl/runner.py"):
        if not (root / "mjlab_amp_ovision" / name).is_file():
            errors.append(f"Missing training file: {name}")
    policy = root / manifest["components"]["policy"]["file"]
    if hashlib.sha256(policy.read_bytes()).hexdigest() != manifest["components"]["policy"]["sha256"]:
        errors.append("Policy differs from the supplied ONNX model")
    if hashlib.sha256((sim / "runtime/model/onnx/policy.onnx").read_bytes()).hexdigest() != manifest["components"]["policy"]["sha256"]:
        errors.append("Runtime policy differs from the supplied ONNX model")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"OK: {len(inventory)} hashes; simulator binary; LingLong original 26-DoF runtime; training source; ONNX model; no LFS pointers")


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    check(parser.parse_args().root)
