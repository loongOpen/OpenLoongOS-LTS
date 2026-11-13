#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe real MuJoCo RGB/depth rendering in isolated backend processes."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
import time

BACKENDS = ("glfw", "egl", "osmesa")


def candidate_backends(backend=None, environment=None):
    environment = os.environ if environment is None else environment
    requested = backend or environment.get("LOONG_RUNTIME_GL", "auto")
    if requested != "auto":
        if requested not in BACKENDS:
            raise ValueError("渲染后端必须为 auto、glfw、egl 或 osmesa")
        return [requested]
    # MUJOCO_GL is set separately in each child. An inherited value from a
    # different Python environment must not turn automatic mode into EGL-only.
    return ["glfw", "egl"] if environment.get("DISPLAY") else ["egl", "glfw"]


def _render_child(check_policies=True):
    cleanup_errors = []
    old_hook = sys.unraisablehook
    sys.unraisablehook = lambda value: cleanup_errors.append(
        "".join(traceback.format_exception(value.exc_type, value.exc_value, value.exc_traceback)))
    renderer = None
    phase = "environment"
    result = {}
    try:
        import numpy as np
        import scipy
        import PIL.Image
        import onnxruntime
        if check_policies:
            phase = "policies"
            from check_runtime import check_policies as verify_policies
            policies = verify_policies()
        else:
            policies = None
        phase = "render"
        import mujoco
        model = mujoco.MjModel.from_xml_string('''<mujoco>
          <worldbody><light pos="0 0 3"/>
            <geom type="plane" size="2 2 .1" rgba=".4 .5 .6 1"/>
            <body pos="0 0 .3"><geom type="sphere" size=".2" rgba="1 .2 .1 1"/></body>
            <camera name="probe" pos="0 -2 1" xyaxes="1 0 0 0 .5 1"/>
          </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=16, width=16)
        renderer.update_scene(data, camera="probe")
        rgb = renderer.render()
        if rgb.shape != (16, 16, 3) or not np.isfinite(rgb).all() or not np.any(rgb):
            raise RuntimeError("MuJoCo RGB 测试未生成有效图像")
        renderer.enable_depth_rendering()
        depth = renderer.render()
        if depth.shape != (16, 16) or not np.isfinite(depth).all():
            raise RuntimeError("MuJoCo 深度测试未生成有限距离")
        renderer.close()
        renderer = None
        result = {"ok": True, "status": "ok", "backend": os.environ["MUJOCO_GL"],
                  "python": sys.version.split()[0], "executable": sys.executable,
                  "mujoco": mujoco.__version__, "rgb_shape": list(rgb.shape),
                  "depth_shape": list(depth.shape), "depth_finite": True,
                  "policies": policies, "error": None}
    except Exception as failure:
        result = {"ok": False, "status": "error", "backend": os.environ.get("MUJOCO_GL"),
                  "phase": phase, "error": f"{type(failure).__name__}: {failure}",
                  "traceback": traceback.format_exc()}
    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception as failure:
                cleanup_errors.append(f"{type(failure).__name__}: {failure}")
        # MuJoCo versions differ in partially initialized Renderer/GLContext
        # destructors. Preserve those diagnostics in JSON, rather than mixing
        # unraisable exception spam with the real initialization failure.
        gc.collect()
        sys.unraisablehook = old_hook
    if cleanup_errors:
        result["cleanup_errors"] = cleanup_errors
    return result


def _native_child(check_policies=True):
    viewer = None
    phase = "environment"
    try:
        configure_native_viewer()
        import mujoco
        from native_view import NativePolicyView
        policies = None
        if check_policies:
            phase = "policies"
            from check_runtime import check_policies as verify_policies
            policies = verify_policies()
        phase = "render"
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <light pos="0 0 3"/><geom type="plane" size="2 2 .1"/>
          <body pos="0 0 .3"><geom type="sphere" size=".2"/></body>
        </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        viewer = NativePolicyView(model,data)
        # Let the native rendering thread complete its first frame before the
        # bounded probe closes it. NativePolicyView joins its render thread.
        for _ in range(6):
            viewer.update([0,0,.3],sim_time=0,enabled=False)
            time.sleep(1/60)
        viewport = viewer.metadata()["viewport_size"]
        if not viewer.is_running() or min(viewport) <= 0:
            raise RuntimeError("原生 MuJoCo GLFW 窗口未就绪")
        return {"ok":True,"status":"ok","backend":"glfw","native_viewer":True,
                "python":sys.version.split()[0],"executable":sys.executable,
                "mujoco":mujoco.__version__,"viewport_size":viewport,
                "policies":policies,"error":None}
    except Exception as failure:
        return {"ok":False,"status":"error","backend":"glfw","native_viewer":True,
                "phase":phase,"error":f"{type(failure).__name__}: {failure}",
                "traceback":traceback.format_exc()}
    finally:
        if viewer is not None:
            viewer.close()


def _run_backend(backend, timeout=15.0, check_policies=True, native_viewer=False):
    environment = dict(os.environ)
    environment["MUJOCO_GL"] = backend
    command = [sys.executable, str(Path(__file__).resolve()), "--child", "--backend", backend]
    if native_viewer:
        command.append("--native-viewer")
    if not check_policies:
        command.append("--render-only")
    child = subprocess.Popen(command, env=environment, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    previous = {}
    def stop(signum, frame):
        child.terminate()
        raise SystemExit(128 + signum)
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, stop)
    try:
        try:
            stdout, stderr = child.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            child.kill()
            stdout, stderr = child.communicate()
            return {"ok": False, "backend": backend, "phase": "render",
                    "error": f"{backend} 渲染检测超时（{timeout:g} 秒）", "stderr": stderr[-4000:],
                    "pid": child.pid}
        try:
            record = json.loads(stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            record = {"ok": False, "phase": "render",
                      "error": f"{backend} 检测进程退出 {child.returncode}，未生成有效结果",
                      "stdout": stdout[-2000:]}
        record["backend"] = backend
        record["returncode"] = child.returncode
        record["pid"] = child.pid
        if stderr.strip():
            record["stderr"] = stderr[-4000:]
        if child.returncode:
            record["ok"] = False
            record["status"] = "error"
            if not record.get("error"):
                record["error"] = f"{backend} 检测进程异常退出 {child.returncode}"
        return record
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def probe_rendering(backend=None, timeout=15.0, check_policies=True, run_backend=None):
    attempts = []
    run_backend = _run_backend if run_backend is None else run_backend
    for candidate in candidate_backends(backend):
        record = run_backend(candidate, timeout=timeout, check_policies=check_policies)
        attempts.append(record)
        if record.get("ok"):
            return {"ok": True, "status": "ok", "backend": candidate, "attempts": attempts,
                    "error": None}
        if record.get("phase") in {"environment", "policies"}:
            break  # A different GL backend cannot repair missing packages/models.
    detail = "；".join(f"{entry['backend']}: {entry.get('error', '渲染失败')}" for entry in attempts)
    return {"ok": False, "status": "error", "backend": None, "attempts": attempts,
            "error": "运行环境或渲染检测未通过。" + detail +
            "。请使用本包 setup_runtime.sh 创建的独立 Python；桌面需可用的 X11/OpenGL，"
            "无桌面运行需可用的 EGL 驱动。"}


def configure_rendering():
    """Select before importing MuJoCo for direct CLI runs with RGB snapshots."""
    inherited = os.environ.get("MUJOCO_GL")
    if inherited and inherited != "auto" and not os.environ.get("LOONG_RUNTIME_GL"):
        if inherited not in BACKENDS:
            raise ValueError(f"Unsupported MUJOCO_GL for RGB rendering: {inherited}")
        return inherited
    result = probe_rendering(check_policies=False)
    if not result["ok"]:
        raise RuntimeError(result["error"])
    os.environ["MUJOCO_GL"] = result["backend"]
    return result["backend"]


def configure_native_viewer():
    """Native window embedding uses the same desktop GLFW path as walking."""
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("原生 MuJoCo 视口需要 X11/XWayland DISPLAY；无桌面时使用 headless 调试入口")
    requested = os.environ.get("LOONG_RUNTIME_GL","auto").strip().lower()
    if requested not in ("auto","glfw"):
        raise RuntimeError("原生 MuJoCo 视口需要 GLFW；请将 LOONG_RUNTIME_GL 设为 auto 或 glfw")
    os.environ["MUJOCO_GL"] = "glfw"
    # GLFW may otherwise select its Wayland backend in a mixed desktop session;
    # the GUI embeds an X11 child window and therefore needs the X11 backend.
    os.environ["GLFW_PLATFORM"] = "x11"
    os.environ["PYGLFW_LIBRARY_VARIANT"] = "x11"
    return "glfw"


def probe_native_viewer(timeout=15.0, check_policies=True):
    try:
        configure_native_viewer()
    except Exception as failure:
        return {"ok":False,"status":"error","native_viewer":True,"backend":None,
                "attempts":[],"error":str(failure)}
    record = _run_backend("glfw",timeout=timeout,check_policies=check_policies,native_viewer=True)
    return {"ok":bool(record.get("ok")),"status":"ok" if record.get("ok") else "error",
            "native_viewer":True,"backend":"glfw" if record.get("ok") else None,
            "attempts":[record],"error":None if record.get("ok") else record.get("error","原生视口检测失败")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--auto", action="store_true", help="try desktop GLFW first, EGL first without DISPLAY")
    selection.add_argument("--backend", choices=BACKENDS + ("auto",))
    parser.add_argument("--json", action="store_true", help="emit a single JSON result line")
    parser.add_argument("--timeout", type=float, default=15.0, help="seconds per isolated backend")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--render-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--native-viewer", action="store_true", help="probe an actual interactive GLFW/MuJoCo window for X11 embedding")
    args = parser.parse_args()
    if not 0 < args.timeout <= 60:
        parser.error("--timeout must be in (0,60]")
    if args.child:
        os.environ["MUJOCO_GL"] = args.backend
        result = _native_child(check_policies=not args.render_only) if args.native_viewer else _render_child(check_policies=not args.render_only)
    elif args.native_viewer:
        result = probe_native_viewer(args.timeout,check_policies=not args.render_only)
    else:
        result = probe_rendering(args.backend, args.timeout, check_policies=not args.render_only)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
