# SPDX-License-Identifier: MIT
"""Run with the GUI runtime Python; no Isaac Lab, torch or pytest required."""
import ctypes
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import onnxruntime as ort

GUI = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(GUI / "runtime/python"))
from native_policy import NativeParkourPolicy
from parkour_runtime import load_deployment,validate_gui_contract


class NativePlanContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GUI.parent / "training_parkour/checkpoints/lite/exported/policy_10200.onnx"
        cls.policy = NativeParkourPolicy(cls.model)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        cls.reference = ort.InferenceSession(str(cls.model),options,providers=["CPUExecutionProvider"])

    @classmethod
    def tearDownClass(cls):
        cls.policy.close()

    def test_random_native_inference_matches_python_ort(self):
        random = np.random.default_rng(20261112)
        for _ in range(100):
            obs = random.normal(0,.5,(1,450)).astype(np.float32)
            depth = random.uniform(0,1,(1,8,18,32)).astype(np.float32)
            actual = self.policy(obs,depth)
            expected = self.reference.run(None,{"obs":obs,"depth":depth})[0][0]
            np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)

    def test_nonfinite_input_is_rejected(self):
        obs = np.zeros((1,450),dtype=np.float32)
        obs[0,0] = np.nan
        with self.assertRaisesRegex(RuntimeError,"Non-finite"):
            self.policy(obs,np.zeros((1,8,18,32),dtype=np.float32))

    def test_wrong_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"Wrong Lite26 policy shapes"):
            self.policy(np.zeros((1,449)),np.zeros((1,8,18,32)))

    def test_bad_depth_does_not_write_any_action(self):
        obs = np.zeros(450,dtype=np.float32)
        depth = np.full(8*18*32,2,dtype=np.float32)
        output = np.full(26,123,dtype=np.float32)
        pointer = ctypes.POINTER(ctypes.c_float)
        status = self.policy.lib.loong_parkour_infer(self.policy.context,obs.ctypes.data_as(pointer),obs.size,
                                                    depth.ctypes.data_as(pointer),depth.size,
                                                    output.ctypes.data_as(pointer),output.size,
                                                    self.policy.error,len(self.policy.error))
        self.assertEqual(status,-1)
        np.testing.assert_array_equal(output,np.full(26,123,dtype=np.float32))
        self.assertIn(b"Depth must be",self.policy.error.value)

    def test_legacy_walking_model_is_rejected(self):
        walking = GUI.parent / "models/policy.onnx"
        if not walking.is_file():
            self.skipTest("No legacy walking model")
        with self.assertRaisesRegex(RuntimeError,"Expected exactly two inputs"):
            NativeParkourPolicy(walking)


class DeploymentSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deployment = load_deployment()
        cls.model = GUI.parent / "training_parkour/checkpoints/lite/exported/policy_10200.onnx"
        cls.metadata = cls.deployment.load_metadata(cls.model,cls.deployment.DEFAULT_URDF,None)

    def test_same_shape_with_different_history_is_rejected(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["obs_history"] = 1
        with self.assertRaisesRegex(ValueError,"obs_history"):
            validate_gui_contract(metadata,self.model,self.deployment)

    def test_same_shape_with_different_camera_fov_is_rejected(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["depth"]["hfov_deg"] = 100
        with self.assertRaisesRegex(ValueError,"hfov_deg"):
            validate_gui_contract(metadata,self.model,self.deployment)

    def test_missing_model_checksum_is_rejected(self):
        metadata = copy.deepcopy(self.metadata)
        del metadata["onnx_sha256"]
        with self.assertRaisesRegex(ValueError,"checksum"):
            validate_gui_contract(metadata,self.model,self.deployment)

    def test_negative_pd_gain_is_rejected(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["kp"][0] = -1
        with self.assertRaisesRegex(ValueError,"positive Lite deployment kp"):
            validate_gui_contract(metadata,self.model,self.deployment)


if __name__ == "__main__":
    unittest.main()
