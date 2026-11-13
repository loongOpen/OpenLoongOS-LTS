"""Validate every bundled Lite reference clip against its URDF and manifest."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import joblib
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "source/lite_parkour/lite_parkour"
URDF = PACKAGE / "data/Robots/Lite/lite_26dof.urdf"
DATA = PACKAGE / "data/MotionData/Lite/amp/reference"

def validate():
    manifest = json.loads((DATA / "conversion_manifest.json").read_text())
    assert manifest["urdf_sha256"][URDF.name] == hashlib.sha256(URDF.read_bytes()).hexdigest()
    urdf = ET.parse(URDF).getroot()
    joints = [j for j in urdf.findall("joint") if j.get("type") != "fixed"]
    names = [j.get("name") for j in joints]
    assert len(names) == 26 and names == manifest["joint_names"]
    spec = importlib.util.spec_from_file_location("_lite_fk", Path(__file__).with_name("kinematics.py"))
    fk_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(fk_module)
    expected = {entry["name"] for entry in manifest["clips"]}
    assert expected == {p.stem for p in DATA.glob("*.pkl")}
    frames = 0
    for entry in manifest["clips"]:
        clip = joblib.load(DATA / (entry["name"] + ".pkl"))
        assert clip["joint_names"] == names and clip["key_body_names"] == manifest["key_body_names"]
        q = clip["dof_pos"]; assert q.shape == (entry["frames"], 26)
        for field in ("dof_pos", "root_pos", "root_rot", "key_body_pos"): assert np.isfinite(clip[field]).all()
        for i,joint in enumerate(joints):
            low, high = (float(joint.find("limit").get(k)) for k in ("lower", "upper"))
            assert q[:, i].min() >= low-1e-6 and q[:, i].max() <= high+1e-6
        np.testing.assert_allclose(np.linalg.norm(clip["root_rot"], axis=1), 1, atol=1e-6)
        ids = np.unique([0, len(q)//2, len(q)-1])
        fk = fk_module.forward_kinematics(urdf, "base_link", dict(zip(names, q[ids].T)), len(ids))
        rotation = Rotation.from_quat(clip["root_rot"][ids][:, [1,2,3,0]]).as_matrix()
        for i,name in enumerate(manifest["key_body_names"]):
            expected_pos = clip["root_pos"][ids] + np.einsum("nij,nj->ni", rotation, fk[name][:,:3,3])
            np.testing.assert_allclose(clip["key_body_pos"][ids,i], expected_pos, atol=1e-6)
        frames += len(q)
    return {"clips": len(expected), "frames": frames, "dofs": len(names), "key_bodies": len(manifest["key_body_names"])}

if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
