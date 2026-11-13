"""Numerical deployment tests; run using the minimal runtime environment."""
import ctypes
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation

from dance_runtime import ASSETS, DancePolicy, JOINTS


class DanceContractTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.model = ASSETS / "linglong14_mjlab_dance_29999.onnx"
    cls.motion = ASSETS / "linglong14_mjlab_xiao.refbin"
    raw = cls.motion.read_bytes()
    cls.header = struct.unpack("<8sIIIf",raw[:24])
    cls.reference = np.frombuffer(raw[24:],dtype="<f4").reshape(-1,35)
    cls.session = ort.InferenceSession(str(cls.model),providers=["CPUExecutionProvider"])
    cls.metadata = cls.session.get_modelmeta().custom_metadata_map
    cls.scale = np.fromstring(cls.metadata["action_scale"],sep=",",dtype=np.float32)

  def test_supplied_assets_agree(self):
    magic,frames,actions,dimension,dt = self.header
    self.assertEqual((magic,frames,actions,dimension),(b"BMJLAB01",1931,26,35))
    self.assertAlmostEqual(dt,.02,places=7)
    self.assertEqual(tuple(self.metadata["joint_names"].split(",")),JOINTS)
    self.assertEqual(self.metadata["deployment_contract"],"linglong14_mjlab_1610_26_v1")
    with np.load(ASSETS / "linglong14_mjlab_xiao_source.npz",allow_pickle=False) as source:
      self.assertEqual(tuple(source["joint_names"]),JOINTS)
      root = list(source["body_names"]).index("base_link")
      rotations = Rotation.from_quat(source["body_quat_w"][:,root,[1,2,3,0]])
      local = np.concatenate((rotations.inv().apply(source["body_lin_vel_w"][:,root]),
        rotations.inv().apply(source["body_ang_vel_w"][:,root]),
        rotations.inv().apply(np.tile([0,0,-1.],(1781,1))),source["joint_pos"]),axis=1)
      np.testing.assert_allclose(self.reference[:1781],local,atol=5e-7,rtol=1e-6)
      np.testing.assert_array_equal(self.reference[1781:,:6],np.zeros((150,6)))
      np.testing.assert_array_equal(self.reference[1781:,6:],np.tile(self.reference[1780,6:],(150,1)))

  def test_native_matches_python_onnx_and_reset(self):
    rng = np.random.default_rng(17)
    policy = DancePolicy(self.model,self.motion)
    history = []
    previous = np.zeros(26,np.float32)
    try:
      for frame in range(15):
        q = rng.normal(0,.2,26).astype(np.float32)
        dq = rng.normal(0,.1,26).astype(np.float32)
        quat = rng.normal(size=4).astype(np.float32)
        quat /= np.linalg.norm(quat)
        gyro = rng.normal(0,.1,3).astype(np.float32)
        gravity = Rotation.from_quat(quat[[1,2,3,0]]).inv().apply([0,0,-1.]).astype(np.float32)
        prop = np.concatenate((gravity,gyro,q,dq,previous))
        history = [prop.copy() for _ in range(10)] if not history else history[1:]+[prop]
        window = self.reference[np.clip(np.arange(frame-10,frame+11),0,len(self.reference)-1)].ravel()
        obs = np.concatenate((window,self.reference[frame],np.concatenate(history))).astype(np.float32)
        expected_action = self.session.run(["actions"],{"obs":obs[None]})[0][0]
        target = policy.step(q,dq,quat,gyro).copy()
        np.testing.assert_allclose(policy.action,expected_action,atol=2e-5,rtol=2e-5)
        np.testing.assert_allclose(target,self.reference[frame,9:]+self.scale*expected_action,atol=2e-5,rtol=2e-5)
        previous = expected_action.copy()
      policy.reset()
      a = policy.step(q,dq,quat,gyro).copy()
      policy.reset()
      b = policy.step(q,dq,quat,gyro).copy()
      np.testing.assert_array_equal(a,b)
    finally:
      policy.close()

  def test_invalid_reference_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      broken = Path(directory)/"broken.refbin"
      broken.write_bytes(self.motion.read_bytes()[:-4])
      with self.assertRaisesRegex(RuntimeError,"reference file length"):
        DancePolicy(self.model,broken)


if __name__ == "__main__":
  unittest.main()
