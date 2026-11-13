#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""GUI backend: trained Lite26 scene, native ONNX plan and actual camera depth.

The legacy LingLong/Nabo simulator has a different robot bus. This process runs
the exact Lite26 model and deployment helpers supplied with training_parkour.
No Isaac Lab, PyTorch, pygame or Qt is required. The optional RGB snapshot path
requires MuJoCo EGL/OSMesa or a desktop OpenGL context.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from native_policy import NativeParkourPolicy

GUI_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = GUI_ROOT.parent
TRAINING_ROOT = RELEASE_ROOT / "training_parkour"
DEFAULT_POLICY = TRAINING_ROOT / "checkpoints/lite/exported/policy_10200.onnx"


def load_deployment():
    path = TRAINING_ROOT / "scripts/sim2sim_mujoco_lite.py"
    spec = importlib.util.spec_from_file_location("_gui_lite_deployment",path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import the supplied Lite26 deployment helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_gui_contract(metadata,policy_path,deployment):
    """Reject same-shaped policies whose sensor/control semantics do not match."""
    expected = {"robot":"lite_26dof","root_link":"base_link","policy_dt":.02,
                "training_physics_dt":.005,"obs_history":5,
                "onnx_inputs":{"obs":[1,450],"depth":[1,8,18,32]}}
    for key,value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Unsupported Lite GUI deployment field {key}: {metadata.get(key)!r}")
    if metadata.get("onnx_sha256") != deployment.sha256(policy_path):
        raise ValueError("Matching ONNX checksum is required in GUI deployment metadata")
    if metadata.get("urdf_sha256") != deployment.sha256(deployment.DEFAULT_URDF):
        raise ValueError("Matching Lite26 URDF checksum is required in GUI deployment metadata")
    expected_depth = {"image_wh":[64,36],"hfov_deg":89.51,"vfov_deg":58.29,
                      "value":"distance_to_image_plane","sees_robot_itself":True,
                      "crop_up_down_left_right":[18,0,16,16],
                      "blur":{"kernel":3,"sigma":1.0,"padding":"reflect"},
                      "depth_range_m":[0.0,2.5],"update_period_s":.02,
                      "history_length":37,"history_skip":5,"output_frames":8,
                      "output_order":"oldest to newest"}
    for key,value in expected_depth.items():
        if metadata["depth"].get(key) != value:
            raise ValueError(f"Unsupported Lite GUI depth field {key}: {metadata['depth'].get(key)!r}")
    start = 0
    expected_slices = {}
    for name,width in (("base_ang_vel",3),("root_local_rot_tan_norm",6),("velocity_commands",3),
                       ("joint_pos",26),("joint_vel",26),("last_action",26)):
        expected_slices[name] = {"start":start,"end":start+width*5,"dim_per_frame":width}
        start += width*5
    if metadata.get("obs_slices") != expected_slices:
        raise ValueError("Unsupported Lite GUI observation term/history layout")
    for key in ("kp","effort_limit","velocity_limit","action_scale"):
        if np.any(np.asarray(metadata[key]) <= 0):
            raise ValueError(f"Expected positive Lite deployment {key}")
    for key in ("kd","armature"):
        if np.any(np.asarray(metadata[key]) < 0):
            raise ValueError(f"Expected nonnegative Lite deployment {key}")


def atomic_bytes(path: Path, value: bytes):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def atomic_json(path: Path, value):
    atomic_bytes(path,(json.dumps(value,ensure_ascii=False,allow_nan=False) + "\n").encode())


class Commands:
    def __init__(self,path,enabled=True):
        self.path,self.enabled = path,enabled
        self.stamp,self.reset_id = None,None
        self.reset_count = 0
        self.error = None
        self.viewer_command = {}

    def poll(self,simulation):
        if self.path is None or not self.path.is_file():
            return
        stamp = self.path.stat().st_mtime_ns
        if stamp == self.stamp:
            return
        self.stamp = stamp
        try:
            command = json.loads(self.path.read_text())
            if not isinstance(command,dict):
                raise ValueError("Command must be an object")
            self.viewer_command = command
            velocity = [command.get(k,float(v)) for k,v in zip(("vx","vy","yaw_rate"),simulation.command)]
            if any(isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) for x in velocity):
                raise ValueError("Command velocity must be finite numbers")
            if not (-.8 <= velocity[0] <= 1.5 and -.5 <= velocity[1] <= .5 and -1 <= velocity[2] <= 1):
                raise ValueError("Command outside vx[-.8,1.5],vy[-.5,.5],yaw_rate[-1,1]")
            enabled = command.get("enabled",self.enabled)
            if not isinstance(enabled,bool):
                raise ValueError("enabled must be a boolean")
            reset_id = command.get("reset_id",self.reset_id)
            if reset_id is not None and (not isinstance(reset_id,int) or isinstance(reset_id,bool)):
                raise ValueError("reset_id must be an integer")
            if self.reset_id is not None and reset_id != self.reset_id:
                simulation.reset()
                self.reset_count += 1
            self.reset_id = reset_id
            simulation.command[:] = velocity
            simulation.heading = None
            self.enabled,self.error = enabled,None
        except (OSError,ValueError,TypeError) as failure:
            self.enabled,self.error = False,str(failure)
            print(f"[COMMAND ERROR] Paused: {failure}",file=sys.stderr,flush=True)


def publish_depth(directory,simulation,frame_id,commands,state="running",failure=None):
    if directory is None:
        return
    # This is the exact newest camera frame used in depth[1,8,18,32].
    frame = simulation.camera.history.buffer[-1]
    if not simulation.camera.history.initialized:
        return
    metres = np.clip(frame,0,1) * simulation.camera.history.buffer.dtype.type(2.5)
    pixels = np.rint((1-np.clip(frame,0,1))*255).astype(np.uint8)
    image = f"P5\n{pixels.shape[1]} {pixels.shape[0]}\n255\n".encode() + pixels.tobytes()
    atomic_bytes(directory / "depth.pgm",image)
    # Full precision transport is available for numerical inspection/replay.
    atomic_bytes(directory / "depth.f32",np.asarray(metres,dtype="<f4").tobytes())
    root = simulation.data.qpos[simulation.root_qpos:simulation.root_qpos+3]
    atomic_json(directory / "depth.json",{
        "schema_version":1,"source":"mujoco_rays","robot":"lite_26dof",
        "width":32,"height":18,"near_m":0.0,"far_m":2.5,
        "min_m":float(metres.min()),"max_m":float(metres.max()),
        "mean_m":float(metres.mean()),"frame_id":frame_id,
        "monotonic_ns":time.monotonic_ns(),"sim_time":float(simulation.data.time),
        "normalization":"depth_m/2.5","image":"depth.pgm","depth_float32_le":"depth.f32",
        "image_sha256":hashlib.sha256(image).hexdigest(),
        "depth_float32_sha256":hashlib.sha256(np.asarray(metres,dtype="<f4").tobytes()).hexdigest(),
        "depth_sim_time":getattr(simulation,"depth_sim_time",0.0),
        "display":"near_white_far_black","history_lags":[35,30,25,20,15,10,5,0],
        "mount_link":"base_link","mount_pos_m":simulation.camera.local_pos.tolist(),
        "state":state,"enabled":commands.enabled,"error":failure or commands.error,
        "base_position_m":root.tolist(),"command":simulation.command.tolist(),
        "policy_calls":simulation.policy.calls,"action_finite":bool(np.isfinite(simulation.last_action).all()),
        "reset_count":commands.reset_count,
        "native_library":str(simulation.policy.library_path.name),
    })


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx",type=Path,default=DEFAULT_POLICY)
    parser.add_argument("--terrain",choices=("flat","stairs","box"),default="box")
    parser.add_argument("--step-height",type=float,default=.20)
    parser.add_argument("--duration",type=float,default=600)
    parser.add_argument("--cmd-vx",type=float,default=.6)
    parser.add_argument("--cmd-vy",type=float,default=0)
    parser.add_argument("--cmd-dyaw",type=float,default=0)
    parser.add_argument("--depth-dir",type=Path)
    parser.add_argument("--snapshot-dir",type=Path,help="Actual MuJoCo RGB scene.ppm at 5 Hz; renderer is probed automatically")
    parser.add_argument("--native-viewer",action="store_true",help="Interactive native MuJoCo/GLFW window for GUI embedding")
    parser.add_argument("--status-dir",type=Path,help="Native viewport/camera metadata directory (viewer.json)")
    parser.add_argument("--command-file",type=Path,help="Atomic JSON {vx,vy,yaw_rate,reset_id,enabled}")
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--no-realtime",action="store_true")
    parser.add_argument("--start-paused",action="store_true")
    parser.add_argument("--self-test",action="store_true",help="Compare native vs Python ORT on actual sensor inputs, run 1 s headlessly")
    args = parser.parse_args()
    if args.native_viewer and (args.headless or args.snapshot_dir):
        parser.error("--native-viewer uses a real GLFW window; omit --headless/--snapshot-dir")
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    if not 0 < args.step_height <= .3:
        parser.error("--step-height must be in (0,.3]")
    if not (-.8 <= args.cmd_vx <= 1.5 and -.5 <= args.cmd_vy <= .5 and -1 <= args.cmd_dyaw <= 1):
        parser.error("Invalid velocity command bounds")
    if args.self_test:
        args.duration,args.headless,args.no_realtime,args.start_paused = 1.0,True,True,False
    return args


def main():
    global mujoco
    args = parse_args()
    if args.native_viewer:
        from render_probe import configure_native_viewer
        configure_native_viewer()
    elif args.snapshot_dir:
        from render_probe import configure_rendering
        configure_rendering()
    import mujoco
    deployment = load_deployment()
    policy_path = deployment.resolve_policy(args.onnx)
    if not policy_path.with_suffix(".json").is_file():
        raise FileNotFoundError(f"GUI deployment requires matching metadata: {policy_path.with_suffix('.json')}; export your PT with the supplied Lite exporter")
    metadata = deployment.load_metadata(policy_path,deployment.DEFAULT_URDF,None)
    validate_gui_contract(metadata,policy_path,deployment)
    policy = NativeParkourPolicy(policy_path)
    renderer = None
    simulation = None
    commands = Commands(args.command_file,not args.start_paused)
    try:
        model = deployment.build_model(deployment.DEFAULT_XML,deployment.DEFAULT_URDF,metadata,.005,args.terrain,args.step_height)
        if args.native_viewer:
            from native_scene_style import apply_native_style
            apply_native_style(model)
        simulation_args = SimpleNamespace(cmd_vx=args.cmd_vx,cmd_vy=args.cmd_vy,cmd_dyaw=args.cmd_dyaw,
                                           heading=None,no_self_occlusion=False)
        simulation = deployment.LiteSimulation(simulation_args,model,metadata,policy)
        initial_position = simulation.data.qpos[simulation.root_qpos:simulation.root_qpos+3].copy()
        boxes = []
        stairs_course = None
        stair_geoms = []
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,geom_id) or ""
            if name.startswith("box_obstacle_"):
                center,half_size = model.geom_pos[geom_id].copy(),model.geom_size[geom_id].copy()
                boxes.append({"name":name,"center_m":center.tolist(),"half_size_m":half_size.tolist(),
                              "near_x_m":float(center[0]-half_size[0]),"far_x_m":float(center[0]+half_size[0]),
                              "collision_enabled":bool(model.geom_contype[geom_id] and model.geom_conaffinity[geom_id]),
                              "root_samples_over_footprint":0,"max_root_lateral_offset_m":0.0})
            elif name.startswith("stairs_") and not name.endswith("_tread"):
                stair_geoms.append(geom_id)
        if stair_geoms:
            stairs_course = {"step_height_m":args.step_height,
                             "near_x_m":float(min(model.geom_pos[i,0]-model.geom_size[i,0] for i in stair_geoms)),
                             "far_x_m":float(max(model.geom_pos[i,0]+model.geom_size[i,0] for i in stair_geoms)),
                             "half_width_m":1.0,"root_samples_on_course":0,"max_abs_root_y_m":0.0}
        reference = deployment.OnnxPolicy(policy_path) if args.self_test else None
        print(f"[PLAN] {policy.library_path}\n[CONTRACT] {policy.contract}\n[POLICY] {policy_path}\n[ROBOT] Lite26, terrain={args.terrain}, dt=.005, policy_dt=.02",flush=True)
        # Initial depth is sensed from the actual spawn before paused mode.
        simulation.camera.capture(model,simulation.data)
        if args.snapshot_dir:
            # Preview quality changes only RGB rendering, never ray depth/physics.
            model.vis.quality.shadowsize = 512
            model.vis.quality.offsamples = 0
            renderer = mujoco.Renderer(model,height=240,width=320)
        if args.headless:
            context = nullcontext(None)
        else:
            from native_view import NativePolicyView
            context = NativePolicyView(model,simulation.data,status_dir=args.status_dir,
                                       distance=3.5,azimuth=150,elevation=-20,
                                       lookat_offset=(0,0,.2))
        frame_id,step,comparison_count,max_error = 0,0,0,0.0
        policy_step,handled_reset = 0,0
        finish_state = "finished"
        minimum_height = float(simulation.data.qpos[simulation.root_qpos+2])
        started = time.monotonic()
        last_depth,last_rgb,last_rgb_sim = -1.0,-1.0,-1.0
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.distance,camera.azimuth,camera.elevation = 3.5,135,-20
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
        with context as viewer:
            while step < math.ceil(args.duration/.005):
                wall_start = time.monotonic()
                if viewer is not None and not viewer.is_running():
                    break
                commands.poll(simulation)
                if commands.reset_count != handled_reset:
                    policy_step = 0
                    handled_reset = commands.reset_count
                    simulation.camera.capture(model,simulation.data)
                    simulation.depth_sim_time = 0.0
                    last_rgb_sim,last_depth = -1.0,-1.0
                if commands.enabled:
                    if policy_step % 4 == 0:
                        # Do not capture an extra frame: the same observation feeds
                        # native inference and the comparison/display transport.
                        obs,depth = simulation.observation()
                        simulation.depth_sim_time = float(simulation.data.time)
                        action = policy(obs,depth)
                        if reference is not None:
                            expected = reference(obs,depth)
                            difference = float(np.max(np.abs(action-expected)))
                            max_error = max(max_error,difference)
                            if not np.allclose(action,expected,rtol=2e-5,atol=2e-5):
                                raise AssertionError(f"Native/Python ORT mismatch: {difference}")
                            comparison_count += 1
                        simulation.last_action[:] = action
                        simulation.target = simulation.q0 + simulation.action_scale * action
                        frame_id += 1
                    simulation.physics_step()
                    minimum_height = min(minimum_height,float(simulation.data.qpos[simulation.root_qpos+2]))
                    step += 1
                    policy_step += 1
                    root_position = simulation.data.qpos[simulation.root_qpos:simulation.root_qpos+3]
                    for box in boxes:
                        if box["near_x_m"] <= root_position[0] <= box["far_x_m"]:
                            box["root_samples_over_footprint"] += 1
                            box["max_root_lateral_offset_m"] = max(box["max_root_lateral_offset_m"],abs(float(root_position[1])-box["center_m"][1]))
                    if stairs_course and stairs_course["near_x_m"] <= root_position[0] <= stairs_course["far_x_m"]:
                        stairs_course["root_samples_on_course"] += 1
                        stairs_course["max_abs_root_y_m"] = max(stairs_course["max_abs_root_y_m"],abs(float(root_position[1])))
                    # A finite solver state alone does not mean successful walking.
                    # Stop a fallen rollout and expose it as a failed scene status.
                    torso_up_z = float(simulation.data.xmat[simulation.base_id].reshape(3,3)[2,2])
                    if float(simulation.data.qpos[simulation.root_qpos+2]) < .55 or torso_up_z < .4:
                        finish_state = "fallen"
                        print("[FALLEN] Rollout stopped: base below .55 m or torso up-Z below .4",file=sys.stderr,flush=True)
                        break
                now = time.monotonic()
                if now-last_depth >= .1:
                    publish_depth(args.depth_dir,simulation,frame_id,commands,"running" if commands.enabled else "paused")
                    last_depth = now
                root = simulation.data.xpos[simulation.base_id]
                if viewer is not None:
                    viewer.update(root,commands.viewer_command,sim_time=float(simulation.data.time),
                                  enabled=commands.enabled,state="running" if commands.enabled else "paused")
                if renderer is not None and now-last_rgb >= .2 and (float(simulation.data.time)-last_rgb_sim >= .2 or last_rgb_sim < 0):
                    camera.lookat[:] = root + np.array([0,0,.2])
                    renderer.update_scene(simulation.data,camera=camera,scene_option=option)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                    rgb = renderer.render()
                    scene_image = f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode()+rgb.tobytes()
                    atomic_bytes(args.snapshot_dir / "scene.ppm",scene_image)
                    atomic_json(args.snapshot_dir / "scene.json",{"frame_id":frame_id,"sim_time":float(simulation.data.time),"monotonic_ns":time.monotonic_ns(),"source":"mujoco_rgb","robot":"lite_26dof","terrain":args.terrain,"image":"scene.ppm","image_sha256":hashlib.sha256(scene_image).hexdigest(),"render_backend":os.environ.get("MUJOCO_GL","glfw")})
                    last_rgb = time.monotonic()
                    last_rgb_sim = float(simulation.data.time)
                if not args.no_realtime or not commands.enabled:
                    time.sleep(max(0,.005-(time.monotonic()-wall_start)))
        publish_depth(args.depth_dir,simulation,frame_id,commands,finish_state)
        mujoco.mj_forward(model,simulation.data)
        feet = {side:simulation.data.xpos[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f"{side}_ankle_roll_link")].tolist() for side in ("left","right")}
        for box in boxes:
            box["both_feet_past_far_edge_with_0p3m_margin"] = all(foot[0]-.3 > box["far_x_m"] for foot in feet.values())
            box["root_remained_over_footprint"] = bool(box["root_samples_over_footprint"] and box["max_root_lateral_offset_m"] < box["half_size_m"][1]-.2)
        if stairs_course:
            stairs_course["both_feet_past_far_edge_with_0p3m_margin"] = all(foot[0]-.3 > stairs_course["far_x_m"] for foot in feet.values())
        result = {"state":finish_state,"robot":"lite_26dof","terrain":args.terrain,
                  "physics_steps":step,"policy_calls":policy.calls,"sim_time":float(simulation.data.time),
                  "wall_seconds":time.monotonic()-started,"minimum_base_z_m":minimum_height,
                  "finite":bool(np.isfinite(simulation.data.qpos).all() and np.isfinite(simulation.data.qvel).all()),
                  "base_position_m":simulation.data.qpos[simulation.root_qpos:simulation.root_qpos+3].tolist(),
                  "initial_base_position_m":initial_position.tolist(),"ankle_positions_m":feet,"box_obstacles":boxes,
                  "stairs_course":stairs_course,"reset_count":commands.reset_count,
                  "native_python_comparisons":comparison_count,"max_action_abs_error":max_error}
        print("[DONE] "+json.dumps(result,allow_nan=False),flush=True)
        if args.depth_dir:
            atomic_json(args.depth_dir / "result.json",result)
        if args.self_test and (comparison_count != 50 or not result["finite"]):
            raise AssertionError("Expected 50 native/Python comparisons and finite simulation")
        if finish_state == "fallen":
            return 2
    except Exception as failure:
        if simulation is not None:
            publish_depth(args.depth_dir,simulation,0,commands,"error",str(failure))
        raise
    finally:
        if renderer is not None:
            renderer.close()
        policy.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
