#!/usr/bin/env python3
"""Check an assembled bundle without starting any robot process."""
import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import re
import struct
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
    ignored_dirs = {".git", ".venv", ".external", "__pycache__", ".pytest_cache", ".ruff_cache",
                    "logs", "runtime_logs", "wandb", ".vscode"}
    paths = []
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.endswith(".egg-info") and not (
            d in {"build", "dist"} and Path(parent).name.startswith("training_")) and not (
            d in {"log", "run"} and Path(parent) == root / "loong_gui_sim/runtime")]
        paths.extend(Path(parent) / n for n in dirs + files)
    for path in paths:
        if path.is_symlink() and (not path.exists() or root not in path.resolve().parents):
            errors.append(f"Broken or external link: {path.relative_to(root)}")
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
    gui = manifest["components"]["loong_gui_sim"]
    public_sources = set(gui.get("public_runtime_sources", []))
    for name in public_sources:
        source_path = Path(name)
        if (source_path.is_absolute() or ".." in source_path.parts
                or not (name.startswith(("native/", "runtime/python/"))
                        or name == "setup_runtime.sh")):
            errors.append(f"Invalid public runtime source declaration: {name}")
        elif not (sim / source_path).is_file():
            errors.append(f"Missing public runtime source: {name}")
    # Use the same ignored-directory traversal as the inventory scan. Users may
    # install the optional policy runtime's .venv inside this tree.
    for path in (p for p in paths if sim in p.parents):
        relative = path.relative_to(sim)
        if "__pycache__" in relative.parts:
            continue
        # LGPL binding support code stays external so recipients can replace it.
        if (len(relative.parts) >= 3 and relative.parts[0] == "_internal"
                and relative.parts[1] in {"PySide6", "shiboken6"}):
            continue
        if relative.as_posix() in public_sources:
            continue
        if path.suffix.lower() in (".py", ".pyc", ".pyx", ".c", ".cpp", ".h", ".hpp", ".hxx", ".a", ".o", ".sh"):
            errors.append(f"Simulator source/build file: {path.relative_to(root)}")
    if gui.get("binding") == "PySide6":
        for name in ("PySide6/__init__.py", "shiboken6/__init__.py", "libxcb-cursor.so.0",
                     "PySide6/QtCore.abi3.so", "PySide6/Qt/lib/libQt6Core.so.6"):
            if not (sim / "_internal" / name).is_file():
                errors.append(f"Missing replaceable GUI library: {name}")
        for name in ("licenses/libxcb-cursor/copyright", "licenses/libxcb-cursor/SOURCE.json"):
            if not (sim / name).is_file():
                errors.append(f"Missing bundled cursor library notice: {name}")
        if hashlib.sha256((sim / "loong_gui_sim").read_bytes()).hexdigest() != gui["sha256"]:
            errors.append("GUI executable differs from release metadata")
        for item in json.loads((root / "third_party/sources/SOURCES.json").read_text()):
            source = root / "third_party/sources" / item["file"]
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]:
                errors.append(f"Missing or modified LGPL source: {item['file']}")
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
    if config.sections() != ["rc", "rl", "jntTest", "dance", "parkour"]:
        errors.append("Expected rc, rl, jntTest, dance and parkour SDK plans")
    for name, key, library in [("rl",4,"libplan_walking_sdk"),("dance",28,"libplan_dance_sdk"),("parkour",43,"libplan_parkour_sdk")]:
        if config.getint(name,"key",fallback=-1) != key or config.get(name,"lib",fallback="") != library:
            errors.append(f"Incorrect unified SDK plan: {name}")
    expected_libraries = {"libplan_recover_x64.so", "libplan_rl_x64.so", "libplan_jnt_test_x64.so", "libplan_walking_sdk_x64.so", "libplan_dance_sdk_x64.so", "libplan_parkour_sdk_x64.so"}
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
    if array("driver.ini", "pdMode") != [1.] * 31:
        errors.append("Unified SDK requires driver-side joint PD (pdMode=1)")
    for gain in ("jntKp", "jntKd"):
        if array("driver_mujoco.ini", gain) != [0.] * 31:
            errors.append(f"Unified SDK must not duplicate PD in MuJoCo: {gain}")
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
        errors.append("Expected 26 active motor axes")
    if array("driver.ini", "useJntMotMap") != [1]:
        errors.append("Closed-chain ankles require the real motor/joint map")
    driver_text = (sim / "runtime/config/driver.ini").read_text()
    if not re.search(r"^\s*simAnkleMap=closed_chain\s*$", driver_text, re.M):
        errors.append("Closed-chain motor feedback requires simAnkleMap=closed_chain")
    if "untaggedPoseSpace=joint" not in (sim / "runtime/config/replay_log.ini").read_text():
        errors.append("Physical actuator replay must not pre-convert motor coordinates")
    import xml.etree.ElementTree as ET
    scene = ET.parse(sim / "runtime/model/scene.xml").getroot()
    assert scene.find("include").get("file") == manifest["robot"]["model"]
    robot = ET.parse(sim / "runtime/model" / manifest["robot"]["model"]).getroot()
    joints = [j for j in robot.findall(".//body/joint") if j.get("type") != "free"]
    motors = robot.findall("actuator/motor")
    joint_by_name = {j.get("name"): j for j in joints}
    if len(joints) != 34 or len(motors) != 26:
        errors.append("Expected 26 actuators, 26 anatomical joints, 4 motor hinges and 4 passive balls")
    def anatomical(name):
        return name.replace("_ankle_motor_long", "_ankle_pitch").replace("_ankle_motor_short", "_ankle_roll").removesuffix("_joint")
    if len(motors) == 26:
        names = [anatomical(motors[expected_map[bus]].get("joint", "")) for bus in policy_to_bus]
        trained = [n.removesuffix("_joint") for n in metadata["joint_names"].split(",")]
        if names != trained:
            errors.append("Policy bus axes differ from closed-chain actuator mapping")
    for motor in motors:
        joint = joint_by_name.get(motor.get("joint"))
        if joint is None or joint.get("type", "hinge") != "hinge":
            errors.append("Motor must bind a scalar hinge")
    closures = {c.get("name"): c for c in robot.findall("equality/connect")}
    for side in ("left", "right"):
        for length in ("short", "long"):
            name = side + "_ankle_" + length
            c = closures.get(name + "_closure")
            if c is None or c.get("body1") != name + "_rod" or c.get("body2") != side + "_ankle_roll_link":
                errors.append("Missing physical linkage closure: " + name)
            ball = joint_by_name.get(name + "_ball")
            if ball is None or ball.get("type") != "ball":
                errors.append("Missing passive ball joint: " + name)
        for logical, mechanical in (("pitch", "long"), ("roll", "short")):
            if any(m.get("joint") == side + "_ankle_" + logical for m in motors):
                errors.append("Anatomical ankle must be passive; actuate motor shafts")
    map_ini = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    map_ini.read(sim / "runtime/config/joint_map.ini")
    for side in ("left", "right"):
        if map_ini.get("ankle", side + "LegMotorOrder", fallback="").replace(" ", "") != "1,0":
            errors.append("Expected long/short ankle motor ordering: " + side)
    for field in ("pitchLower", "pitchUpper", "rollLower", "rollUpper"):
        if not map_ini.has_option("limit", field):
            errors.append("Explicit model limits required in joint_map.ini: " + field)
    library = sim / "runtime/module/loong_driver_sdk/libjoint_map_x64.so"
    header = library.read_bytes()[:20]
    if header[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", header, 18)[0] != 62:
        errors.append("Closed-chain joint map must be an x86_64 ELF library")
    for name in ("pyproject.toml", "uv.lock", "LICENSE", "src/mjlab/tasks/amp/rl/runner.py"):
        if not (root / "training_walking" / name).is_file():
            errors.append(f"Missing training file: {name}")
    training_entries = {
        "training_parkour": ("pyproject.toml", "LICENSE", "scripts/train_lite.sh", "scripts/rsl_rl/train.py", "source/lite_parkour/lite_parkour/tasks/amp/lite_env_cfg.py"),
        "training_dance": ("pyproject.toml", "LICENSE", "train.sh", "src/mjlab/scripts/train.py", "src/mjlab/asset_zoo/motions/linglong14/xiao_speed075_modified.npz"),
    }
    for directory, required in training_entries.items():
        for name in required:
            if not (root / directory / name).is_file():
                errors.append(f"Missing training file: {directory}/{name}")
    profiles = gui.get("runtime_profiles", {})
    if profiles and set(profiles) != {"walking", "dance", "parkour"}:
        errors.append("GUI must declare walking, dance and parkour runtime profiles")
    for feature, profile in profiles.items():
        for name in profile.get("required_files", []):
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(f"Invalid {feature} runtime path: {name}")
                continue
            if not (sim / relative).is_file():
                errors.append(f"Missing {feature} runtime file: {name}")
        for name in profile.get("release_required_files", []):
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(f"Invalid {feature} release asset path: {name}")
            elif not (root / relative).is_file():
                errors.append(f"Missing {feature} release asset: {name}")
        library = profile.get("plan_library")
        if library and (sim / library).is_file():
            header = (sim / library).read_bytes()[:20]
            if (len(header) != 20 or header[:6] != b"\x7fELF\x02\x01"
                    or struct.unpack_from("<H", header, 16)[0] != 3
                    or struct.unpack_from("<H", header, 18)[0] != 62):
                errors.append(f"GUI {feature} plan must be an x86_64 ELF shared library")
    dance_assets = sim / "runtime/deploy/dance/ASSETS.json"
    if "dance" in profiles:
        if not dance_assets.is_file():
            errors.append("Missing supplied dance asset provenance")
        else:
            for item in json.loads(dance_assets.read_text())["files"]:
                relative = Path(item["file"])
                if relative.is_absolute() or ".." in relative.parts:
                    errors.append("Invalid supplied dance asset path")
                    continue
                asset = sim / relative
                if not asset.is_file() or hashlib.sha256(asset.read_bytes()).hexdigest() != item["sha256"]:
                    errors.append(f"Missing or modified supplied dance asset: {item['file']}")
    policy = root / manifest["components"]["policy"]["file"]
    if hashlib.sha256(policy.read_bytes()).hexdigest() != manifest["components"]["policy"]["sha256"]:
        errors.append("Policy differs from the supplied ONNX model")
    if hashlib.sha256((sim / "runtime/model/onnx/policy.onnx").read_bytes()).hexdigest() != manifest["components"]["policy"]["sha256"]:
        errors.append("Runtime policy differs from the supplied ONNX model")
    if errors:
        raise SystemExit("\n".join(errors))
    modes = "; GUI walking/dance/parkour runtime profiles" if profiles else ""
    print(f"OK: {len(inventory)} hashes; simulator binary; unified LingLong 26-axis runtime with physical closed-chain ankles; 3 training source trees; ONNX model{modes}; no LFS pointers")


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    check(parser.parse_args().root)
