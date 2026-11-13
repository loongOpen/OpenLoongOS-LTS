"""Lite with 26 movable joints: 12 leg and 14 arm joints, no waist.

Leg gains and reflected inertias follow HumanoidAmpLocomotion's lite_uneven
configuration; arm PD gains/inertias use the validated Lite actuator set.
Torque, speed and position limits follow the supplied Lite URDF. In particular,
wrists use 7 Nm and shoulder pitch/roll use 6.28 rad/s.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg

from lite_parkour.assets.urdf import spawn_nested_contact_urdf

LITE_URDF = Path(__file__).resolve().parents[1] / "data/Robots/Lite/lite_26dof.urdf"
_URDF = ET.parse(LITE_URDF).getroot()
_JOINTS = [joint for joint in _URDF.findall("joint") if joint.get("type") != "fixed"]
LITE_JOINT_ORDER = tuple(joint.get("name") for joint in _JOINTS)
LITE_KEY_BODIES = (
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link",
)

# kp, kd, reflected rotor inertia [kg m^2]. Limits come from the Lite URDF.
_ACTUATORS = {
    "hip_pitch": (280.0, 14.0, 0.27430387),
    "hip_roll": (240.0, 12.0, 0.07031016),
    "hip_yaw": (100.0, 5.0, 0.06247368),
    "knee": (280.0, 14.0, 0.07031016),
    "ankle_pitch": (50.0, 5.0, 0.09505512),
    "ankle_roll": (50.0, 5.0, 0.09505512),
    "shoulder_pitch": (50.0, 1.5, 0.033048),
    "shoulder_roll": (50.0, 1.5, 0.033048),
    "shoulder_yaw": (50.0, 1.5, 0.033048),
    "elbow": (50.0, 1.5, 0.033048),
    "wrist_roll": (50.0, 1.5, 0.033048),
    "wrist_pitch": (50.0, 1.5, 0.033048),
    "wrist_yaw": (50.0, 1.5, 0.033048),
}
LITE_ACTUATORS = {}
LITE_ACTION_SCALE = {}
for _joint in _JOINTS:
    _name = _joint.get("name")
    _kind = _name.removeprefix("left_").removeprefix("right_").removesuffix("_joint")
    _kp, _kd, _armature = _ACTUATORS[_kind]
    _effort = float(_joint.find("limit").get("effort"))
    _velocity = float(_joint.find("limit").get("velocity"))
    LITE_ACTUATORS[_name] = IdealPDActuatorCfg(
        joint_names_expr=[_name], effort_limit_sim=_effort, velocity_limit_sim=_velocity,
        stiffness=_kp, damping=_kd, armature=_armature,
    )
    # Quarter of the torque range the gain can command.
    LITE_ACTION_SCALE[_name] = 0.25 * _effort / _kp

# Resolve nested geometry paths after merging the empty fixed pelvis link.
# One target per visual avoids cross-environment pose indexing.
_PARENT_JOINT = {j.find("child").get("link"): j for j in _URDF.findall("joint")}


def lite_geometry_path(link):
    chain = [link]
    while chain[-1] != "base_link":
        chain.append(_PARENT_JOINT[chain[-1]].find("parent").get("link"))
    return "/".join(
        name for name in reversed(chain)
        if name == "base_link" or _PARENT_JOINT[name].get("type") != "fixed"
    )


LITE_DEPTH_VISUAL_PATHS = []
for _link in _URDF.findall("link"):
    if _link.find("visual") is None:
        continue
    _name = _owner = _link.get("name")
    while _owner in _PARENT_JOINT and _PARENT_JOINT[_owner].get("type") == "fixed":
        _owner = _PARENT_JOINT[_owner].find("parent").get("link")
    LITE_DEPTH_VISUAL_PATHS.append(f"{lite_geometry_path(_owner)}/{_name}")


def spawn_lite_urdf(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Reuse the existing cylinder conversion and nested contact activation."""
    return spawn_nested_contact_urdf(
        prim_path, cfg, translation=translation, orientation=orientation, **kwargs,
    )


LITE_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        func=spawn_lite_urdf, asset_path=str(LITE_URDF),
        fix_base=False, merge_fixed_joints=True, activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, linear_damping=0.0, angular_damping=0.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    # Slightly flexed stance; hip + knee + ankle pitch = 0 keeps soles level.
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.99),
        joint_pos={
            ".*_hip_pitch_joint": -0.2, ".*_knee_joint": 0.48, ".*_ankle_pitch_joint": -0.28,
            "left_shoulder_roll_joint": 0.25, "right_shoulder_roll_joint": -0.25,
            "left_shoulder_yaw_joint": -0.2, "right_shoulder_yaw_joint": 0.2,
            ".*_elbow_joint": 1.14,
            "left_wrist_roll_joint": -0.09, "right_wrist_roll_joint": 0.09,
            ".*_wrist_pitch_joint": -0.25,
            "left_wrist_yaw_joint": -0.08, "right_wrist_yaw_joint": 0.08,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9, actuators=LITE_ACTUATORS,
)
