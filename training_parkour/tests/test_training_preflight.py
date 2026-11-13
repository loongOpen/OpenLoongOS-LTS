"""Resource closure checks that run without Isaac Sim or the training package."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_lite_preflight", ROOT / "scripts/check_environment.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def test_bundled_training_assets_are_complete():
    assert preflight.check_assets() == {"clips": 72, "frames": 7569, "dofs": 26, "meshes": 27, "files": 101}


def make_asset_fixture(tmp_path, monkeypatch):
    package = tmp_path / "source/lite_parkour/lite_parkour"
    robot = package / "data/Robots/Lite"
    motion = package / "data/MotionData/Lite/amp/reference"
    (robot / "meshes").mkdir(parents=True)
    motion.mkdir(parents=True)
    names = [f"joint_{i}" for i in range(26)]
    urdf = robot / "lite_26dof.urdf"
    urdf.write_text('<robot><mesh filename="meshes/body.STL"/>' +
                    ''.join(f'<joint name="{name}" type="revolute"/>' for name in names) + '</robot>')
    (robot / "lite_26dof.xml").write_text("<mujoco/>")
    (robot / "meshes/body.STL").write_bytes(b"mesh")
    (motion / "clip.pkl").write_bytes(b"clip")
    (motion / "conversion_manifest.json").write_text(json.dumps({
        "urdf_sha256": {urdf.name: hashlib.sha256(urdf.read_bytes()).hexdigest()},
        "joint_names": names, "clips": [{"name": "clip", "frames": 1}]}))
    monkeypatch.setattr(preflight, "PACKAGE", package)
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    return robot, motion


def test_missing_mesh_is_rejected(tmp_path, monkeypatch):
    robot, _ = make_asset_fixture(tmp_path, monkeypatch)
    (robot / "meshes/body.STL").unlink()
    with pytest.raises(ValueError, match="Missing or empty resource"):
        preflight.check_assets()


def test_lfs_motion_pointer_is_rejected(tmp_path, monkeypatch):
    _, motion = make_asset_fixture(tmp_path, monkeypatch)
    (motion / "clip.pkl").write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="Git LFS pointer"):
        preflight.check_assets()


def test_changed_urdf_is_rejected(tmp_path, monkeypatch):
    robot, _ = make_asset_fixture(tmp_path, monkeypatch)
    with (robot / "lite_26dof.urdf").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="URDF hash"):
        preflight.check_assets()


@pytest.mark.parametrize("mode", ["missing", "old_nonce", "missing_checkpoint", "stale_checkpoint", "bad_progress", "valid"])
def test_wrapper_requires_this_launchs_saved_checkpoint(tmp_path, mode):
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    for name in ("train_lite.sh", "validate_training_report.py"):
        shutil.copy2(ROOT / "scripts" / name, scripts / name)
    fake_python = tmp_path / "python"
    fake_python.write_text(f"#!{sys.executable}\n" + '''
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
args = sys.argv[1:]
if args == ["-"]:
    sys.stdin.read()
    raise SystemExit(0)
if "scripts/check_environment.py" in args:
    raise SystemExit(0)
if "scripts/rsl_rl/train.py" in args:
    mode = os.environ["FAKE_REPORT_MODE"]
    if mode == "missing":
        print("simulated Kit exception followed by zero exit", file=sys.stderr)
        raise SystemExit(0)
    report_path = Path(args[args.index("--completion_report")+1])
    nonce = args[args.index("--completion_nonce")+1]
    started = time.time_ns() - 1000000000
    checkpoint = report_path.parent / "model_6.pt"
    checkpoint.write_bytes(b"simulated saved checkpoint")
    if mode == "stale_checkpoint":
        os.utime(checkpoint, ns=(started-1000000000, started-1000000000))
    stat = checkpoint.stat()
    report = {"completed":True,"status":"completed","nonce":nonce,"log_dir":str(report_path.parent),
              "start_iteration":5,"final_iteration":6,"requested_iterations":2,"iterations_completed":2,
              "started_ns":started,"finished_ns":time.time_ns(),
              "checkpoints":[{"path":str(checkpoint),"size":stat.st_size,"mtime_ns":stat.st_mtime_ns,
                              "sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()}]}
    if mode == "old_nonce":report["nonce"] = "previous-launch"
    if mode == "bad_progress":report["final_iteration"] = 5
    if mode == "missing_checkpoint":checkpoint.unlink()
    report_path.write_text(json.dumps(report))
    raise SystemExit(0)
raise SystemExit(subprocess.run([sys.executable,*args]).returncode)
''')
    fake_python.chmod(0o755)
    result = subprocess.run(["bash", str(scripts / "train_lite.sh"), "--max_iterations", "2"],
                            cwd=tmp_path, env={**os.environ, "LITE_PYTHON": str(fake_python), "FAKE_REPORT_MODE": mode},
                            capture_output=True, text=True)
    assert result.returncode == (0 if mode == "valid" else 1), result.stdout + result.stderr
    logs = list((project / "logs/launches").glob("*/train.log"))
    assert len(logs) == 1
    if mode == "missing":
        assert "simulated Kit exception" in logs[0].read_text()
        assert "未生成完成报告" in result.stderr
