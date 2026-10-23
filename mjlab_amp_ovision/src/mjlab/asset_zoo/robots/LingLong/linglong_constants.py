"""LingLong constants - full-size humanoid, 20 DoF (12 leg + 8 arm), no waist.

The XML is not edited; get_spec() names foot cylinders and IMU sensors for mjlab.
"""

from pathlib import Path

import mujoco
from mjlab import MJLAB_SRC_PATH
from dataclasses import replace

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec import disable_collision
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

LINGLONG_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "LingLong" / "linglong_20dof.xml"
)
assert LINGLONG_XML.exists(), f"XML not found: {LINGLONG_XML}"

_FOOT_BODY_TO_SIDE = {
  "left_ankle_roll_link": "left",
  "right_ankle_roll_link": "right",
}


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, LINGLONG_XML.parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(LINGLONG_XML))
  spec.assets = get_assets(spec.meshdir)
  # Remove XML-defined actuators — mjlab injects its own builtin actuators.
  for act in list(spec.actuators):
    spec.delete(act)

  # Training / sim2sim add their own terrain; drop the XML plane.
  for geom in list(spec.geoms):
    if geom.name == "ground":
      spec.delete(geom)
      break

  apply_urdf_collision_names(
    spec,
    _FOOT_BODY_TO_SIDE,
    extra_names=(
      ("left_knee_link", "left_knee_collision"),
      ("right_knee_link", "right_knee_collision"),
      ("base_link", "torso_collision"),
    ),
  )

  # velocity_env_cfg reads robot/imu_ang_vel and robot/imu_lin_vel.
  spec.sensor("angular-velocity").name = "imu_ang_vel"
  spec.sensor("linear-velocity").name = "imu_lin_vel"

  # velocity MDP angular_momentum reward reads scene["robot/root_angmom"].
  if not any(s.name == "root_angmom" for s in spec.sensors):
    spec.add_sensor(
      name="root_angmom",
      type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
      objtype=mujoco.mjtObj.mjOBJ_BODY,
      objname="base_link",
    )

  return spec


def apply_urdf_collision_names(
  spec: mujoco.MjSpec,
  foot_body_to_side: dict[str, str],
  extra_names: tuple[tuple[str, str], ...] = (),
) -> None:
  """Name every URDF ``<collision>`` primitive so ``FULL_COLLISION`` keeps it on.

  The MJCF copies URDF collision cylinders/boxes without names. CollisionCfg
  only matches ``*_collision``, and unnamed geoms used to be treated as visual
  leftovers (``contype=0``). Name them here, then disable remaining unnamed
  visual meshes.
  """
  for body_name, side in foot_body_to_side.items():
    body = spec.body(body_name)
    idx = 0
    for geom in body.geoms:
      if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER and geom.contype != 0:
        idx += 1
        geom.name = f"{side}_foot{idx}_collision"
    if idx != 5:
      raise RuntimeError(f"{body_name}: expected 5 foot cylinders, got {idx}")
    site_name = f"{side}_foot"
    if not any(site.name == site_name for site in body.sites):
      body.add_site(name=site_name, pos=[0.02, 0.0, -0.09], size=[0.005, 0.0, 0.0])

  for body_name, geom_name in extra_names:
    body = spec.body(body_name)
    for geom in body.geoms:
      if geom.contype != 0 and not geom.name:
        geom.name = geom_name
        break

  for body in spec.bodies:
    n = 0
    for geom in body.geoms:
      if geom.contype == 0 or geom.name:
        continue
      n += 1
      base = body.name.removesuffix("_link")
      geom.name = f"{base}_collision" if n == 1 else f"{base}{n}_collision"

  for geom in spec.geoms:
    if not geom.name:
      disable_collision(geom)


##
# Actuator config — PD gains / effort limits for the 20-DoF LingLong.
##

# Stiffness (Kp)
STIFFNESS_HIP_PITCH   = 280.0
STIFFNESS_HIP_ROLL    = 240.0
STIFFNESS_HIP_YAW     = 100.0
STIFFNESS_KNEE        = 280.0
STIFFNESS_ANKLE_PITCH =  50.0
STIFFNESS_ANKLE_ROLL  =  50.0
STIFFNESS_SHOULDER_PITCH = 50.0
STIFFNESS_SHOULDER_ROLL  = 50.0
STIFFNESS_SHOULDER_YAW   = 50.0
STIFFNESS_ELBOW          = 50.0

# Damping (Kd)
DAMPING_HIP_PITCH   = 14.0
DAMPING_HIP_ROLL    = 12.0
DAMPING_HIP_YAW     = 10.0
DAMPING_KNEE        = 14.0
DAMPING_ANKLE_PITCH =  5.0
DAMPING_ANKLE_ROLL  =  5.0
DAMPING_SHOULDER_PITCH = 0.2
DAMPING_SHOULDER_ROLL  = 1.5
DAMPING_SHOULDER_YAW   = 1.5
DAMPING_ELBOW          = 1.5

# Effort limits (from XML ctrlrange, symmetric).
EFFORT_HIP_PITCH   = 330.0
EFFORT_HIP_ROLL    = 150.0
EFFORT_HIP_YAW     = 130.0
EFFORT_KNEE        = 150.0
EFFORT_ANKLE_PITCH =  90.0
EFFORT_ANKLE_ROLL  =  90.0
EFFORT_SHOULDER_PITCH = 97.0
EFFORT_SHOULDER_ROLL  = 97.0
EFFORT_SHOULDER_YAW   = 27.0
EFFORT_ELBOW          = 27.0

ARMATURE = 0.003


def _leg_actuators(side: str) -> tuple[BuiltinPositionActuatorCfg, ...]:
  return (
    BuiltinPositionActuatorCfg(
      target_names_expr=(f"{side}_hip_pitch",),
      stiffness=STIFFNESS_HIP_PITCH, damping=DAMPING_HIP_PITCH,
      effort_limit=EFFORT_HIP_PITCH, armature=ARMATURE,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(f"{side}_hip_roll",),
      stiffness=STIFFNESS_HIP_ROLL, damping=DAMPING_HIP_ROLL,
      effort_limit=EFFORT_HIP_ROLL, armature=ARMATURE,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(f"{side}_hip_yaw",),
      stiffness=STIFFNESS_HIP_YAW, damping=DAMPING_HIP_YAW,
      effort_limit=EFFORT_HIP_YAW, armature=ARMATURE,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(f"{side}_knee",),
      stiffness=STIFFNESS_KNEE, damping=DAMPING_KNEE,
      effort_limit=EFFORT_KNEE, armature=ARMATURE,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(f"{side}_ankle_pitch",),
      stiffness=STIFFNESS_ANKLE_PITCH, damping=DAMPING_ANKLE_PITCH,
      effort_limit=EFFORT_ANKLE_PITCH, armature=ARMATURE,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(f"{side}_ankle_roll",),
      stiffness=STIFFNESS_ANKLE_ROLL, damping=DAMPING_ANKLE_ROLL,
      effort_limit=EFFORT_ANKLE_ROLL, armature=ARMATURE,
    ),
  )


def _arm_actuators(side: str) -> tuple[BuiltinPositionActuatorCfg, ...]:
  return (
    BuiltinPositionActuatorCfg(
      target_names_expr=(f"{side}_shoulder_pitch_joint",),
      stiffness=STIFFNESS_SHOULDER_PITCH, damping=DAMPING_SHOULDER_PITCH,
      effort_limit=EFFORT_SHOULDER_PITCH, armature=ARMATURE,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(f"{side}_shoulder_roll_joint",),
      stiffness=STIFFNESS_SHOULDER_ROLL, damping=DAMPING_SHOULDER_ROLL,
      effort_limit=EFFORT_SHOULDER_ROLL, armature=ARMATURE,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(f"{side}_shoulder_yaw_joint",),
      stiffness=STIFFNESS_SHOULDER_YAW, damping=DAMPING_SHOULDER_YAW,
      effort_limit=EFFORT_SHOULDER_YAW, armature=ARMATURE,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(f"{side}_elbow_joint",),
      stiffness=STIFFNESS_ELBOW, damping=DAMPING_ELBOW,
      effort_limit=EFFORT_ELBOW, armature=ARMATURE,
    ),
  )


##
# Initial state (keyframe).
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.99),  # XML base_link spawn height
  joint_pos={
    "left_hip_pitch":   -0.2,
    "left_hip_roll":     0.0,
    "left_hip_yaw":      0.0,
    "left_knee":         0.48,
    "left_ankle_pitch": -0.28,
    "left_ankle_roll":   0.0,
    "right_hip_pitch":  -0.2,
    "right_hip_roll":    0.0,
    "right_hip_yaw":     0.0,
    "right_knee":        0.48,
    "right_ankle_pitch": -0.28,
    "right_ankle_roll":  0.0,
    "left_shoulder_pitch_joint":  0.0,
    "left_shoulder_roll_joint":   0.28,
    "left_shoulder_yaw_joint":    0.0,
    "left_elbow_joint":           1.15,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": -0.28,
    "right_shoulder_yaw_joint":   0.0,
    "right_elbow_joint":          1.15,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config — 5 URDF sole cylinders per foot (visual meshes stay off).
##

_FOOT_COLLISION = r"^(left|right)_foot[1-5]_collision$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={
    _FOOT_COLLISION: 3,
    ".*_collision": 1,
  },
  priority={
    _FOOT_COLLISION: 1,
    ".*": 0,
  },
  friction={
    _FOOT_COLLISION: (0.8,),
  },
)

##
# Action scale — 0.5 * effort_limit / Kp per joint.
##

LINGLONG_ACTION_SCALE: dict[str, float] = {
  "left_hip_pitch":    0.294642857,
  "left_hip_roll":     0.15625,
  "left_hip_yaw":      0.216666667,
  "left_knee":         0.133928571,
  "left_ankle_pitch":  0.45,
  "left_ankle_roll":   0.45,
  "right_hip_pitch":   0.294642857,
  "right_hip_roll":    0.15625,
  "right_hip_yaw":     0.216666667,
  "right_knee":        0.133928571,
  "right_ankle_pitch": 0.45,
  "right_ankle_roll":  0.45,
  "left_shoulder_pitch_joint":  0.5,
  "left_shoulder_roll_joint":   0.5,
  "left_shoulder_yaw_joint":    0.5,
  "left_elbow_joint":           0.5,
  "right_shoulder_pitch_joint": 0.5,
  "right_shoulder_roll_joint":  0.5,
  "right_shoulder_yaw_joint":   0.5,
  "right_elbow_joint":          0.5,
}

##
# Articulation.
##


def _make_delayed(base_cfg: BuiltinPositionActuatorCfg) -> BuiltinPositionActuatorCfg:
  # mjlab 1.6: delay is on ActuatorCfg, not a DelayedActuator wrapper.
  return replace(
    base_cfg,
    delay_min_lag=0,
    delay_max_lag=4,
    delay_hold_prob=0.5,
    delay_update_period=50,
  )


LINGLONG_ARTICULATION = EntityArticulationInfoCfg(
  actuators=tuple(
    _make_delayed(a)
    for a in (
      *_leg_actuators("left"),
      *_leg_actuators("right"),
      *_arm_actuators("left"),
      *_arm_actuators("right"),
    )
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_linglong_robot_cfg() -> EntityCfg:
  """Get a fresh LingLong (20-DoF) robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=LINGLONG_ARTICULATION,
  )
