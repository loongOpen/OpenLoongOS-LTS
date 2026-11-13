# SPDX-License-Identifier: MIT
"""Native view presentation must not alter the policy's physical experiment."""
from pathlib import Path
import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

from dance_runtime import make_model
import parkour_runtime as parkour
from native_scene_style import apply_native_style

PHYSICAL = (
    "body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat", "body_parentid",
    "jnt_qposadr", "jnt_dofadr", "jnt_type", "jnt_pos", "jnt_axis", "jnt_range", "jnt_limited",
    "dof_damping", "dof_frictionloss", "dof_armature", "dof_solref", "dof_solimp",
    "actuator_gainprm", "actuator_biasprm", "actuator_dynprm", "actuator_ctrlrange", "actuator_forcerange",
    "actuator_gear", "actuator_trntype", "actuator_trnid", "actuator_ctrllimited", "actuator_forcelimited",
    "geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity",
    "geom_friction", "geom_solref", "geom_solimp", "geom_condim", "geom_group", "geom_margin", "geom_gap",
    "mesh_vert", "mesh_face", "sensor_type", "sensor_objid", "sensor_adr", "sensor_dim",
)
OPTIONS = ("timestep", "integrator", "solver", "cone", "iterations", "ls_iterations", "gravity",
           "tolerance", "impratio", "density", "viscosity")


def assert_physics(test, before, after):
    for name in PHYSICAL:
        np.testing.assert_array_equal(getattr(before,name),getattr(after,name),err_msg=name)
    for name in OPTIONS:
        np.testing.assert_array_equal(getattr(before.opt,name),getattr(after.opt,name),err_msg="opt."+name)
    np.testing.assert_array_equal(before.geom_rgba[:,3],after.geom_rgba[:,3])


class NativeSceneStyleTests(unittest.TestCase):
    def test_dance_style_changes_only_visual_assets(self):
        before,after=make_model(),make_model(native_style=True)
        assert_physics(self,before,after)
        self.assertGreater(after.ntex,before.ntex)
        self.assertAlmostEqual(after.stat.extent,.8)
        np.testing.assert_array_equal(after.vis.headlight.ambient,np.array([.3,.3,.3],dtype=np.float32))

    def test_lite_style_preserves_dynamics_and_actual_depth(self):
        deploy=parkour.load_deployment()
        path=deploy.resolve_policy(parkour.DEFAULT_POLICY)
        metadata=deploy.load_metadata(path,deploy.DEFAULT_URDF,None)
        before=deploy.build_model(deploy.DEFAULT_XML,deploy.DEFAULT_URDF,metadata,.005,"box",.2)
        after=deploy.build_model(deploy.DEFAULT_XML,deploy.DEFAULT_URDF,metadata,.005,"box",.2)
        original_pixels=after.tex_data.copy()
        apply_native_style(after)
        assert_physics(self,before,after)
        self.assertTrue(np.any(after.tex_data!=original_pixels))
        args=SimpleNamespace(cmd_vx=.4,cmd_vy=0.,cmd_dyaw=0.,heading=None,no_self_occlusion=False)
        first=deploy.LiteSimulation(args,before,metadata,None)
        second=deploy.LiteSimulation(args,after,metadata,None)
        for dx,yaw in ((0,0),(3.5,.15),(5.2,-.2)):
            for sim in (first,second):
                sim.reset()
                sim.data.qpos[sim.root_qpos]+=dx
                quat=np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)])
                sim.data.qpos[sim.root_qpos+3:sim.root_qpos+7]=quat
                mujoco.mj_forward(sim.model,sim.data)
                sim.camera.capture(sim.model,sim.data)
            np.testing.assert_array_equal(first.camera.history.buffer[-1],second.camera.history.buffer[-1])
            np.testing.assert_array_equal(first.data.qacc,second.data.qacc)


if __name__=="__main__":
    unittest.main()
