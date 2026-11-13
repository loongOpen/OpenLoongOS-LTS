#!/usr/bin/env python3
"""CPU MuJoCo + native ONNX dance deployment, without training dependencies.

The reference and robot follow training_dance's Linglong14 contract. The original
aarch64 Nabo plugin is separately preserved in runtime/deploy/dance; this runner
uses the portable native/dance_plan.cpp C API and never loads ARM code on x86.
"""
from __future__ import annotations

import argparse
import configparser
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
import xml.etree.ElementTree as ET

GUI = Path(__file__).resolve().parents[2]
RELEASE = GUI.parent
ASSETS = GUI / "runtime/model/dance"
ROBOT = RELEASE / "training_dance/src/mjlab/asset_zoo/robots/linglong14"
JOINTS = tuple(f"{side}_{joint}_joint" for side in ("left", "right")
              for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")) + tuple(
  f"{side}_{joint}_joint" for side in ("left", "right")
  for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"))
# Copied from training_dance/linglong14_constants.py (Apache-2.0 / Linglong source MIT).
# kp, kd, effort, armature. Ankles use explicit PD, other joints use native PD.
MOTORS = {
  "hip_pitch": (320.,18.,330.,476.222e-6*24**2),
  "hip_roll": (300.,18.,130.,488.265e-6*12**2),
  "hip_yaw": (180.,8.,130.,192.82e-6*18**2),
  "knee": (340.,20.,190.,488.265e-6*12**2),
  "ankle_pitch": (140.,8.,120.,146.69e-6*18**2),
  "ankle_roll": (140.,8.,120.,146.69e-6*18**2),
  "shoulder_pitch": (120.,10.,97.,5.4e-5*48**2),
  "shoulder_roll": (120.,10.,97.,5.4e-5*48**2),
  "shoulder_yaw": (40.,5.,27.,2e-5*40**2),
  "elbow": (40.,5.,27.,2e-5*40**2),
  "wrist_roll": (8.,1.,7.,1.8e-5*10**2),
  "wrist_pitch": (8.,1.,7.,1.8e-5*10**2),
  "wrist_yaw": (8.,1.,7.,1.8e-5*10**2),
}


def atomic_json(path, value):
  if path is None:
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
  temporary.replace(path)


def make_model(native_style=False):
  """Rebuild the same URDF inertia/collisions and actuator types as training."""
  import mujoco
  import numpy as np
  from scipy.spatial.transform import Rotation
  spec = mujoco.MjSpec.from_file(str(ROBOT / "linglong.xml"))
  spec.assets = {p.name: p.read_bytes() for p in (ROBOT / "meshes").iterdir() if p.is_file()}
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  for light in list(spec.lights):
    spec.delete(light)
  for geom in list(spec.geoms):
    if geom.type == mujoco.mjtGeom.mjGEOM_PLANE or geom.classname == "collision":
      spec.delete(geom)
    else:
      geom.contype = geom.conaffinity = 0
      geom.group = 2
  def origin(node):
    if node is None:
      return np.zeros(3), np.array([1.,0.,0.,0.])
    pos = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
    quat = Rotation.from_euler("xyz", np.fromstring(node.get("rpy", "0 0 0"), sep=" ")).as_quat()[[3,0,1,2]]
    return pos, quat
  tree = ET.parse(ROBOT / "linglong.urdf").getroot()
  for link in tree.findall("link"):
    body = spec.body(link.get("name"))
    inertial = link.find("inertial")
    if inertial is not None:
      body.mass = float(inertial.find("mass").get("value"))
      body.ipos, body.iquat = origin(inertial.find("origin"))
      inertia = inertial.find("inertia")
      body.inertia = [0.,0.,0.]
      body.fullinertia = [float(inertia.get(k)) for k in ("ixx","iyy","izz","ixy","ixz","iyz")]
    for index, collision in enumerate(link.findall("collision"), 1):
      pos, quat = origin(collision.find("origin"))
      kind = list(collision.find("geometry"))[0]
      name = f"{body.name}_collision{index}"
      if body.name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        name = f"{body.name.split('_')[0]}_foot{index}_collision"
      options = dict(name=name,pos=pos,quat=quat,contype=1,conaffinity=1,condim=3,group=3,
                     friction=[1.,.005,.0001],rgba=[.6,.6,.6,0.])
      if kind.tag == "cylinder":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE,size=[float(kind.get("radius")),float(kind.get("length"))/2,0.],**options)
      elif kind.tag == "box":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,size=np.fromstring(kind.get("size"),sep=" ")/2,**options)
      elif kind.tag == "mesh":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH,meshname=Path(kind.get("filename")).name,**options)
      else:
        raise ValueError(f"Unsupported collision {kind.tag}")
  for joint in tree.findall("joint"):
    item = spec.joint(joint.get("name"))
    item.damping = 0.
    item.frictionloss = 0.
  for name in JOINTS:
    short = name.split("_",1)[1].removesuffix("_joint")
    kp,kd,effort,armature = MOTORS[short]
    joint = spec.joint(name)
    joint.armature = armature
    actuator = spec.add_actuator(name=name,target=name)
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
    actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    actuator.gear[0] = 1.
    actuator.forcelimited = True
    actuator.forcerange = [-effort,effort]
    if short.startswith("ankle"):
      actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
      actuator.gainprm[0] = 1.
      actuator.ctrllimited = True
      actuator.ctrlrange = [-effort,effort]
    else:
      actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
      actuator.gainprm[0] = kp
      actuator.biasprm[1:3] = [-kp,-kd]
      actuator.ctrllimited = False
      actuator.inheritrange = 0.
  ground_options = {}
  if native_style:
    from native_scene_style import add_native_assets
    ground_options["material"] = add_native_assets(spec)
  spec.worldbody.add_geom(name="ground",type=mujoco.mjtGeom.mjGEOM_PLANE,size=[0,0,.1],
    rgba=[.34,.39,.44,1.],contype=1,conaffinity=1,friction=[1.,.005,.0001],**ground_options)
  spec.worldbody.add_light(pos=[0,-2,4],dir=[0,.4,-1],diffuse=[.8,.8,.8])
  spec.add_sensor(name="imu_ang_vel",type=mujoco.mjtSensor.mjSENS_GYRO,
    objtype=mujoco.mjtObj.mjOBJ_SITE,objname="base_link_site")
  model = spec.compile()
  model.opt.timestep = .005
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
  model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  model.opt.iterations = 10
  model.opt.ls_iterations = 20
  model.vis.headlight.ambient[:] = [.45,.45,.45]
  model.vis.headlight.diffuse[:] = [.7,.7,.7]
  if native_style:
    from native_scene_style import apply_native_style
    apply_native_style(model)
  return model


class DancePolicy:
  def __init__(self, model, reference, first=0, loop=False):
    import numpy as np
    arch = {"x86_64":"x64","amd64":"x64","aarch64":"a64","arm64":"a64"}.get(platform.machine().lower())
    if arch is None:
      raise RuntimeError(f"Unsupported CPU {platform.machine()}")
    library = GUI / f"native/libdance_plan_{arch}.so"
    self.lib = ctypes.CDLL(str(library))
    self.lib.dance_create.argtypes = [ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int,ctypes.c_int]
    self.lib.dance_create.restype = ctypes.c_void_p
    self.lib.dance_error.restype = ctypes.c_char_p
    self.lib.dance_destroy.argtypes = [ctypes.c_void_p]
    self.lib.dance_reset.argtypes = [ctypes.c_void_p]
    self.lib.dance_frame.argtypes = [ctypes.c_void_p]
    ptr = np.ctypeslib.ndpointer(dtype=np.float32,ndim=1,flags="C_CONTIGUOUS")
    self.lib.dance_step.argtypes = [ctypes.c_void_p,ptr,ptr,ptr,ptr,ptr,ptr]
    self.handle = self.lib.dance_create(os.fsencode(model),os.fsencode(reference),first,int(loop))
    if not self.handle:
      raise RuntimeError(self.lib.dance_error().decode())
    self.target = np.zeros(26,np.float32)
    self.action = np.zeros(26,np.float32)
  def step(self,q,dq,quat,gyro):
    import numpy as np
    args = [np.ascontiguousarray(x,dtype=np.float32) for x in (q,dq,quat,gyro)]
    if self.lib.dance_step(self.handle,*args,self.target,self.action):
      raise RuntimeError(self.lib.dance_error().decode())
    return self.target
  def reset(self):
    self.lib.dance_reset(self.handle)
  @property
  def frame(self):
    return self.lib.dance_frame(self.handle)
  def close(self):
    if self.handle:
      self.lib.dance_destroy(self.handle)
      self.handle = None


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--headless",action="store_true",help="No standalone viewer")
  parser.add_argument("--native-viewer",action="store_true",help="Interactive native MuJoCo/GLFW window for GUI embedding")
  parser.add_argument("--status-dir",type=Path,help="Native viewport/camera metadata directory (viewer.json)")
  parser.add_argument("--steps",type=int,default=0,help="Policy steps; 0 runs until stopped")
  parser.add_argument("--command-file",type=Path)
  parser.add_argument("--status-file",type=Path)
  parser.add_argument("--snapshot-dir",type=Path)
  parser.add_argument("--config",type=Path,default=GUI / "runtime/deploy/dance/plan_bm_taiji_mjlab.ini")
  parser.add_argument("--model",type=Path)
  parser.add_argument("--reference",type=Path)
  parser.add_argument("--motion",type=Path,default=ASSETS / "linglong14_mjlab_xiao_source.npz")
  parser.add_argument("--start-frame",type=int)
  parser.add_argument("--loop",action=argparse.BooleanOptionalAction,default=None)
  parser.add_argument("--fast",action="store_true",help="Run faster than wall clock for checks")
  args = parser.parse_args(argv)
  if args.native_viewer and (args.headless or args.snapshot_dir):
    parser.error("--native-viewer uses a real GLFW window; omit --headless/--snapshot-dir")
  if args.status_dir and args.status_file is None:
    args.status_file = args.status_dir / "status.json"
  config = configparser.ConfigParser(inline_comment_prefixes=(";","#"))
  config.read_string("[dance]\n"+args.config.read_text(encoding="utf-8"))
  cfg = config["dance"]
  if cfg.getint("obsDim")!=1610 or cfg.getint("num_actions")!=26 or cfg["jointOrder"]!="mujoco_linglong14" or abs(cfg.getfloat("policyDt")-.02)>1e-6:
    raise ValueError("Unsupported dance INI contract; expected 1610/26/MuJoCo/0.02s")
  def asset(name):
    value = Path(cfg[name])
    if value.is_absolute():
      return value
    local = args.config.parent / value
    return local if local.is_file() else ASSETS / value
  args.model = args.model or asset("onnxFile")
  args.reference = args.reference or asset("motionFile")
  args.start_frame = cfg.getint("startFrame",fallback=0) if args.start_frame is None else args.start_frame
  args.loop = cfg.getboolean("playbackLoop",fallback=False) if args.loop is None else args.loop
  if args.native_viewer:
    from render_probe import configure_native_viewer
    configure_native_viewer()
  elif args.snapshot_dir:
    from render_probe import configure_rendering
    configure_rendering()
  elif not os.environ.get("DISPLAY") and args.headless:
    os.environ.setdefault("MUJOCO_GL","egl")
  import mujoco
  import numpy as np
  from scipy.spatial.transform import Rotation
  model = make_model(native_style=args.native_viewer)
  data = mujoco.MjData(model)
  qids = np.array([model.joint(n).qposadr[0] for n in JOINTS])
  vids = np.array([model.joint(n).dofadr[0] for n in JOINTS])
  aids = np.array([model.actuator(n).id for n in JOINTS])
  gyro = model.sensor("imu_ang_vel").adr[0]
  kp,kd,limits = np.array([MOTORS[n.split("_",1)[1].removesuffix("_joint")][:3] for n in JOINTS]).T
  ankles = np.array(["_ankle_" in n for n in JOINTS])
  source = np.load(args.motion,allow_pickle=False)
  source_joint_ids = [list(source["joint_names"]).index(n) for n in JOINTS]
  root = list(source["body_names"]).index("base_link")
  if not 0 <= args.start_frame < len(source["joint_pos"]) + 150:
    raise ValueError("start frame is outside the reference")
  policy = DancePolicy(args.model,args.reference,args.start_frame,args.loop)
  def reset():
    mujoco.mj_resetData(model,data)
    frame = min(args.start_frame,len(source["joint_pos"])-1)
    quat = source["body_quat_w"][frame,root]
    data.qpos[:3] = source["body_pos_w"][frame,root]
    data.qpos[3:7] = quat
    data.qpos[qids] = source["joint_pos"][frame,source_joint_ids]
    if args.start_frame < len(source["joint_pos"]):
      data.qvel[:3] = source["body_lin_vel_w"][frame,root]
      data.qvel[3:6] = Rotation.from_quat(quat[[1,2,3,0]]).inv().apply(source["body_ang_vel_w"][frame,root])
      data.qvel[vids] = source["joint_vel"][frame,source_joint_ids]
    policy.reset()
    mujoco.mj_forward(model,data)
  reset()
  renderer = None
  viewer = None
  if args.snapshot_dir:
    args.snapshot_dir.mkdir(parents=True,exist_ok=True)
    model.vis.global_.offwidth = 640
    model.vis.global_.offheight = 480
    renderer = mujoco.Renderer(model,height=480,width=640)
  elif not args.headless:
    from native_view import NativePolicyView
    viewer = NativePolicyView(model,data,status_dir=args.status_dir,
                              distance=3.5,azimuth=150,elevation=-20,
                              lookat_offset=(0,0,0))
  camera = mujoco.MjvCamera()
  camera.distance,camera.azimuth,camera.elevation = 3.5,135,-15
  stopping = False
  def stop(_signum,_frame):
    nonlocal stopping
    stopping = True
  signal.signal(signal.SIGTERM,stop)
  signal.signal(signal.SIGINT,stop)
  steps = 0
  reset_id = None
  enabled = True
  started = time.monotonic()
  next_step = started
  next_physics = started
  minimum_height = float(data.qpos[2])
  last_status = 0.
  snapshot = None
  image_sha256 = None
  state = {}
  command = {}
  try:
    while not stopping and (args.steps<=0 or steps<args.steps):
      if viewer and not viewer.is_running():
        break
      if args.command_file and args.command_file.exists():
        try:
          command = json.loads(args.command_file.read_text(encoding="utf-8"))
          enabled = bool(command.get("enabled",True))
          incoming_reset = command.get("reset_id",0)
          if reset_id is not None and incoming_reset != reset_id:
            reset()
            minimum_height = float(data.qpos[2])
          reset_id = incoming_reset
          if command.get("stop",False):
            break
        except (OSError,ValueError):
          pass  # Atomic command replacement may race with the first read.
      if enabled:
        target = policy.step(data.qpos[qids],data.qvel[vids],data.qpos[3:7],data.sensordata[gyro:gyro+3])
        for _ in range(4):
          control = target.astype(float).copy()
          torque = np.clip(kp*(target-data.qpos[qids])-kd*data.qvel[vids],-limits,limits)
          control[ankles] = torque[ankles]
          data.ctrl[aids] = control
          mujoco.mj_step(model,data)
          if viewer:
            viewer.update(data.qpos[:3],command,sim_time=float(data.time),enabled=enabled)
            if not args.fast:
              next_physics += .005
              time.sleep(max(0,next_physics-time.monotonic()))
              if time.monotonic()-next_physics > .1:
                next_physics = time.monotonic()
        steps += 1
        minimum_height = min(minimum_height,float(data.qpos[2]))
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
          raise RuntimeError("non-finite MuJoCo state")
      now = time.monotonic()
      if now-last_status>=.1 or (args.steps>0 and steps>=args.steps):
        if renderer:
          from PIL import Image
          camera.lookat[:] = data.qpos[:3]
          renderer.update_scene(data,camera=camera)
          frame = renderer.render()
          snapshot = args.snapshot_dir / "snapshot.png"
          temporary = args.snapshot_dir / "snapshot.tmp.png"
          Image.fromarray(frame).save(temporary)
          image_sha256 = hashlib.sha256(temporary.read_bytes()).hexdigest()
          temporary.replace(snapshot)
        state = dict(mode="dance",running=True,pid=os.getpid(),enabled=enabled,steps=steps,
          frame=policy.frame,sim_time=float(data.time),base_height=float(data.qpos[2]),
          minimum_base_height=minimum_height,fallen=bool(data.qpos[2]<.45),
          snapshot=str(snapshot) if snapshot else None,wall_time=now-started,
          image="snapshot.png" if snapshot else None,image_sha256=image_sha256,
          render_backend="glfw" if viewer else os.environ.get("MUJOCO_GL", "glfw") if renderer else None,
          native_viewer=bool(viewer),
          native_library="libdance_plan",contract="linglong14_mjlab_1610_26_v1")
        atomic_json(args.status_file,state)
        last_status = now
      if viewer:
        if not enabled:
          viewer.update(data.qpos[:3],command,sim_time=float(data.time),enabled=False,state="paused")
          time.sleep(.005)
          next_physics = time.monotonic()
      else:
        next_step += .02
        if not args.fast or not enabled:
          time.sleep(max(.001,next_step-time.monotonic()))
        elif time.monotonic()-next_step>1.:
          next_step = time.monotonic()
  finally:
    state.update(running=False,steps=steps,frame=policy.frame,sim_time=float(data.time),
                 base_height=float(data.qpos[2]),minimum_base_height=minimum_height)
    atomic_json(args.status_file,state)
    policy.close()
    source.close()
    if renderer:
      renderer.close()
    if viewer:
      viewer.close()
  print(json.dumps(state,ensure_ascii=False),flush=True)
  return 0


if __name__ == "__main__":
  try:
    sys.exit(main())
  except Exception as exc:
    print(f"Dance runtime failed: {exc}",file=sys.stderr,flush=True)
    sys.exit(1)
