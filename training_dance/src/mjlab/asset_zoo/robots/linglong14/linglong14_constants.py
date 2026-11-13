"""26-DOF Linglong 1.4, using the IsaacLab workspace's exact URDF parameters.

The provided MJCF supplies the matching link tree and visuals. Collision
geometry is rebuilt from the actual training URDF (including cylinder->capsule
conversion), and mass/COM/full inertia are copied from that URDF.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from mjlab.actuator import BuiltinPositionActuatorCfg, IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

ASSET_DIR = Path(__file__).resolve().parent
LINGLONG14_XML = ASSET_DIR / "linglong.xml"
LINGLONG14_URDF = ASSET_DIR / "linglong.urdf"
LINGLONG14_JOINT_NAMES = tuple(
  [f"{s}_{j}_joint" for s in ("left", "right")
   for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
  + [f"{s}_{j}_joint" for s in ("left", "right")
     for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
)
LINGLONG14_BODY_NAMES = ("base_link",) + tuple(n.removesuffix("_joint") + "_link" for n in LINGLONG14_JOINT_NAMES)
LINGLONG14_TRACKED_BODIES = (
  "base_link", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
  "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
  "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
  "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)

# kp, kd, effort Nm, reflected rotor inertia kg*m², source velocity limit rad/s.
MOTOR_PARAMETERS = {
  "hip_pitch": (320., 18., 330., 476.222e-6 * 24**2, 32.),
  "hip_roll": (300., 18., 130., 488.265e-6 * 12**2, 20.),
  "hip_yaw": (180., 8., 130., 192.82e-6 * 18**2, 32.),
  "knee": (340., 20., 190., 488.265e-6 * 12**2, 20.),
  "ankle_pitch": (140., 8., 120., 146.69e-6 * 18**2, 20.),
  "ankle_roll": (140., 8., 120., 146.69e-6 * 18**2, 20.),
  "shoulder_pitch": (120., 10., 97., 5.4e-5 * 48**2, 6.28),
  "shoulder_roll": (120., 10., 97., 5.4e-5 * 48**2, 6.28),
  "shoulder_yaw": (40., 5., 27., 2e-5 * 40**2, 10.47),
  "elbow": (40., 5., 27., 2e-5 * 40**2, 10.47),
  "wrist_roll": (8., 1., 7., 1.8e-5 * 10**2, 20.94),
  "wrist_pitch": (8., 1., 7., 1.8e-5 * 10**2, 20.94),
  "wrist_yaw": (8., 1., 7., 1.8e-5 * 10**2, 20.94),
}
LINGLONG14_ACTION_SCALE = {
  f"{side}_{joint}_joint": (0.5 if joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
                             else 0.25 * values[2] / values[0])
  for side in ("left", "right") for joint, values in MOTOR_PARAMETERS.items()
}


def _origin(node):
  if node is None:
    return np.zeros(3), np.array([1., 0., 0., 0.])
  pos = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
  rpy = np.fromstring(node.get("rpy", "0 0 0"), sep=" ")
  return pos, Rotation.from_euler("xyz", rpy).as_quat()[[3, 0, 1, 2]]


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(LINGLONG14_XML))
  assets = {}
  update_assets(assets, ASSET_DIR / "meshes", spec.meshdir)
  spec.assets = assets
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  for light in list(spec.lights):
    spec.delete(light)
  # The old XML has extra scene geometry and simplified collision meshes.
  for geom in list(spec.geoms):
    if geom.type == mujoco.mjtGeom.mjGEOM_PLANE or geom.classname == "collision":
      spec.delete(geom)
    else:
      geom.contype = geom.conaffinity = 0
      geom.group = 2
  tree = ET.parse(LINGLONG14_URDF).getroot()
  for link in tree.findall("link"):
    body = spec.body(link.get("name"))
    inertial = link.find("inertial")
    if inertial is not None:
      body.mass = float(inertial.find("mass").get("value"))
      body.ipos, body.iquat = _origin(inertial.find("origin"))
      a = inertial.find("inertia")
      body.inertia = [0., 0., 0.]
      body.fullinertia = [float(a.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
    for i, collision in enumerate(link.findall("collision"), 1):
      pos, quat = _origin(collision.find("origin"))
      geometry = collision.find("geometry")
      kind = list(geometry)[0]
      name = f"{body.name}_collision{i}"
      if body.name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        name = f"{body.name.split('_')[0]}_foot{i}_collision"
      kwargs = dict(name=name, pos=pos, quat=quat, contype=1, conaffinity=1,
                    condim=3, group=3, friction=[1., .005, .0001], rgba=[.6, .6, .6, 0.])
      if kind.tag == "cylinder":
        # Source IsaacLab importer uses replace_cylinders_with_capsules=True.
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      size=[float(kind.get("radius")), float(kind.get("length")) / 2, 0], **kwargs)
      elif kind.tag == "box":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=np.fromstring(kind.get("size"), sep=" ") / 2, **kwargs)
      elif kind.tag == "mesh":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=Path(kind.get("filename")).name, **kwargs)
      else:
        raise ValueError(f"unsupported source collision {kind.tag}")
  for joint in tree.findall("joint"):
    j = spec.joint(joint.get("name"))
    d = joint.find("dynamics")
    j.damping = [float(d.get("damping", "0")) if d is not None else 0., 0., 0.]
    j.frictionloss = float(d.get("friction", "0")) if d is not None else 0.
  spec.sensor("base_link_site_vel").name = "imu_lin_vel"
  spec.add_sensor(name="imu_ang_vel", type=mujoco.mjtSensor.mjSENS_GYRO,
                  objtype=mujoco.mjtObj.mjOBJ_SITE, objname="base_link_site")
  return spec


def get_linglong14_robot_cfg(play: bool = False) -> EntityCfg:
  actuators = []
  for joint, (kp, kd, effort, armature, _) in MOTOR_PARAMETERS.items():
    # The old task uses explicit PD at ankles and implicit drives elsewhere.
    kind = IdealPdActuatorCfg if joint.startswith("ankle") else BuiltinPositionActuatorCfg
    actuators.append(kind(target_names_expr=(f".*_{joint}_joint",), stiffness=kp, damping=kd,
                          effort_limit=effort, armature=armature, frictionloss=0., viscous_damping=0.))
  return EntityCfg(
    spec_fn=get_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0., 0., .97), joint_pos={".*": 0.}, joint_vel={".*": 0.}),
    articulation=EntityArticulationInfoCfg(actuators=tuple(actuators), soft_joint_pos_limit_factor=.9),
    sort_actuators=True,
  )
