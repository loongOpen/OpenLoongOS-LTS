"""Diagnose the selected Python environment without importing Isaac Sim/Kit.

Exit 0 means the requested dependencies and local data passed preflight. It is
not a simulation or learning test; use train_lite.sh --num_envs 2
--max_iterations 2 afterwards to exercise those paths.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import platform
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source/lite_parkour/lite_parkour"


def check_assets() -> dict:
    """Check the resource closure without importing the training package."""
    urdf = PACKAGE / "data/Robots/Lite/lite_26dof.urdf"
    motion = PACKAGE / "data/MotionData/Lite/amp/reference"
    manifest = json.loads((motion / "conversion_manifest.json").read_text())
    if hashlib.sha256(urdf.read_bytes()).hexdigest() != manifest["urdf_sha256"][urdf.name]:
        raise ValueError("Lite URDF hash differs from the reference motion manifest")
    tree = ET.parse(urdf).getroot()
    joints = [joint.get("name") for joint in tree.findall("joint") if joint.get("type") != "fixed"]
    if len(joints) != 26 or joints != manifest["joint_names"]:
        raise ValueError("The 26-joint URDF order differs from the motion manifest")
    paths = {urdf, urdf.parent / "lite_26dof.xml"}
    meshes = set()
    for mesh in tree.findall(".//mesh"):
        # The bundled URDF deliberately uses relative mesh paths.
        meshes.add(urdf.parent / mesh.get("filename"))
    paths.update(meshes)
    clips = {motion / (entry["name"] + ".pkl") for entry in manifest["clips"]}
    if clips != set(motion.glob("*.pkl")):
        raise ValueError("Reference clip files differ from the manifest")
    paths.update(clips)
    for path in sorted(paths):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty resource: {path.relative_to(ROOT)}")
        with path.open("rb") as stream:
            if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"Git LFS pointer instead of resource: {path.relative_to(ROOT)}; run git lfs pull")
    return {"clips": len(clips), "frames": sum(entry["frames"] for entry in manifest["clips"]),
            "dofs": len(joints), "meshes": len(meshes), "files": len(paths)}


def inspect_environment(training: bool = False) -> dict:
    report = {"python": sys.version.split()[0], "executable": sys.executable,
              "platform": platform.platform(), "mode": "training" if training else "sim2sim",
              "errors": []}
    errors = report["errors"]
    required = ["numpy", "torch", "yaml"]
    required += (["isaacsim", "isaaclab", "isaaclab_physx", "isaaclab_rl", "isaaclab_tasks", "lite_parkour",
                  "rsl_rl", "tensordict", "warp", "gymnasium", "joblib", "psutil", "tensorboard"]
                 if training else ["mujoco", "onnx", "onnxruntime"])
    for name in required:
        try:
            found = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            found = None
        if found is None:
            errors.append(f"Missing module: {name}")
    packages = ("isaacsim", "isaacsim-asset", "isaacsim-robot-setup", "isaaclab", "isaaclab_physx", "rsl-rl-lib", "torch", "mujoco", "onnxruntime",
                "tensordict", "warp-lang", "gymnasium", "joblib", "psutil")
    report["packages"] = {}
    for package in packages:
        try:
            report["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    if training:
        if sys.version_info[:2] != (3, 12):
            errors.append("Training requires Python 3.12")
        if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
            errors.append("The validated training platform is Linux x86_64")
        expected = json.loads((ROOT / "environment-tested.json").read_text())["packages"]
        expected.update({"isaacsim-asset": expected["isaacsim"], "isaacsim-robot-setup": expected["isaacsim"]})
        for package in ("isaacsim", "isaacsim-asset", "isaacsim-robot-setup", "isaaclab", "isaaclab_physx", "rsl-rl-lib", "torch",
                        "tensordict", "warp-lang", "gymnasium", "joblib", "psutil"):
            actual = report["packages"].get(package)
            if actual is None or actual.split("+")[0] != expected[package]:
                errors.append(f"Install {package}=={expected[package]} (found {actual or 'not installed'}); see README.md")
    try:
        import torch
        report["cuda_build"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if training:
            if torch.version.cuda not in ("12.8", "13.0"):
                errors.append(f"Install PyTorch 2.11 CUDA 12.8 or 13.0 wheels (found {torch.version.cuda})")
            if not torch.cuda.is_available():
                errors.append("A working NVIDIA CUDA GPU is required for training")
            else:
                # Availability alone does not catch incompatible driver/runtime wheels.
                probe = torch.ones(1, device="cuda")
                torch.cuda.synchronize()
                report["cuda_probe"] = bool(probe.isfinite().all().item())
                report["gpu"] = torch.cuda.get_device_name()
    except Exception as exc:
        errors.append(f"PyTorch/CUDA runtime check failed: {type(exc).__name__}: {exc}")
    try:
        report["assets"] = check_assets()
    except Exception as exc:
        errors.append(f"Local resources: {type(exc).__name__}: {exc}")
    report["status"] = "pass" if not errors else "fail"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--assets-only", action="store_true", help="Check bundled resources using only the Python standard library")
    parser.add_argument("--report", type=Path, help="Also write the JSON report to this path")
    args = parser.parse_args()
    if args.assets_only:
        try:
            report = {"status": "pass", "mode": "assets", "assets": check_assets(), "errors": []}
        except Exception as exc:
            report = {"status": "fail", "mode": "assets", "errors": [f"{type(exc).__name__}: {exc}"]}
    else:
        report = inspect_environment(args.training)
    output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output)
    print(output, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
