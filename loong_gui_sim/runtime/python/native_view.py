#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""A native, interactive MuJoCo viewport shared by the policy runtimes.

The passive viewer owns its GLFW/OpenGL rendering thread. This module only
copies simulation state at a 60 Hz target and controls the free camera under
the viewer lock; it never renders a low-resolution image for the GUI to scale.
"""
from __future__ import annotations

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import threading
import time


def _vector3(value):
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise ValueError("Native camera root position must contain three finite values")
    return result


def apply_native_visual_style(model):
    """Use the walking viewport's lighting without touching physics or sensing."""
    model.vis.headlight.diffuse[:] = [.6, .6, .6]
    model.vis.headlight.ambient[:] = [.3, .3, .3]
    model.vis.headlight.specular[:] = [0., 0., 0.]
    model.vis.rgba.haze[:] = [.15, .25, .35, 1.]
    model.vis.quality.shadowsize = 4096


class NativePolicyView:
    """A full-resolution GLFW window that an X11 GUI can embed by process PID."""

    def __init__(self, model, data, status_dir=None, follow_camera=True,
                 distance=3.5, azimuth=150., elevation=-20.,
                 lookat_offset=(0., 0., .2), viewer_factory=None,
                 clock=time.monotonic, apply_style=True):
        self._clock = clock
        self._started = clock()
        self._next_sync = self._started
        self._next_publish = self._started
        self._sync_period = 1. / 60.
        self._sync_count = 0
        self._closed = False
        self._viewer = None
        self._render_threads = ()
        self._last_position = None
        self._camera_reset_id = 0
        self._camera_resets = 0
        self.follow_camera = bool(follow_camera)
        self._defaults = (float(distance), float(azimuth), float(elevation))
        self._offset = _vector3(lookat_offset)
        self._status_dir = Path(status_dir) if status_dir is not None else None
        self._last_state = {"sim_time": float(data.time), "enabled": True,
                            "state": "starting"}
        self._last_metadata = {}
        if not all(math.isfinite(item) for item in self._defaults) or distance <= 0:
            raise ValueError("Native camera defaults must be finite and distance positive")
        if apply_style:
            apply_native_visual_style(model)
        if viewer_factory is None:
            if not os.environ.get("DISPLAY"):
                raise RuntimeError("原生 MuJoCo 视口需要可访问的 X11/XWayland DISPLAY")
            from mujoco import viewer
            viewer_factory = viewer.launch_passive
        existing_threads = set(threading.enumerate())
        self._viewer = viewer_factory(model, data, show_left_ui=False, show_right_ui=False)
        # launch_passive starts a Python daemon thread running the native GLFW
        # loop. Join that exact newly created thread on close before Python's
        # process finalizers unload MuJoCo/OpenGL underneath it.
        self._render_threads = tuple(thread for thread in threading.enumerate()
                                     if thread not in existing_threads
                                     and thread is not threading.current_thread())
        initial = (data.qpos[:3] if len(data.qpos) >= 3 else model.stat.center)
        self._last_position = _vector3(initial)
        with self.lock():
            self._reset_camera(self._last_position)
            # Collision shapes remain in the physical model and depth sensor;
            # only the native viewer's debug-geometry visibility is disabled.
            self._viewer.opt.geomgroup[3] = 0
        self._publish(force=True)

    @property
    def handle(self):
        return self._viewer

    def lock(self):
        return self._viewer.lock() if self._viewer is not None and not self._closed else nullcontext()

    def is_running(self):
        return bool(not self._closed and self._viewer is not None and self._viewer.is_running())

    def _reset_camera(self, root):
        camera = self._viewer.cam
        camera.type = 0  # mjCAMERA_FREE; reset only, never override user orbit per frame.
        camera.fixedcamid = -1
        camera.distance, camera.azimuth, camera.elevation = self._defaults
        camera.lookat[:] = [position + offset for position, offset in zip(root, self._offset)]
        self._camera_resets += 1

    def update(self, root_position, command=None, sim_time=None, enabled=True, state="running"):
        """Sync the actual viewer while preserving native mouse orbit and zoom.

        A follow camera translates by the root displacement, so mouse panning
        keeps its relative offset too. Re-enabling follow recentres lookat;
        only an explicit reset_camera counter change resets distance/angles.
        """
        if not self.is_running():
            return False
        root = _vector3(root_position)
        command = {} if command is None else command
        follow = command.get("follow_camera", self.follow_camera)
        reset_id = command.get("reset_camera", self._camera_reset_id)
        if type(follow) is not bool:
            raise ValueError("follow_camera must be a boolean")
        if type(reset_id) is not int:
            raise ValueError("reset_camera must be an integer")
        reenabled = follow and not self.follow_camera
        reset = reset_id != self._camera_reset_id
        with self.lock():
            camera = self._viewer.cam
            if reset:
                self._reset_camera(root)
            elif reenabled:
                camera.lookat[:] = [position + offset for position, offset in zip(root, self._offset)]
            elif follow:
                camera.lookat[:] = [float(target) + position - previous
                                    for target, position, previous in
                                    zip(camera.lookat, root, self._last_position)]
        self.follow_camera = follow
        self._camera_reset_id = reset_id
        self._last_position = root
        self._last_state = {"sim_time": float(sim_time) if sim_time is not None else None,
                            "enabled": bool(enabled), "state": str(state)}
        now = self._clock()
        if now + 1e-9 >= self._next_sync:
            self._viewer.sync()
            self._sync_count += 1
            # Accumulate deadlines: 200 Hz physics gives alternating 15/20 ms
            # sync gaps averaging 60 Hz, rather than a fixed 20 ms / 50 Hz cap.
            missed = max(1, math.floor((now - self._next_sync) / self._sync_period) + 1)
            self._next_sync += missed * self._sync_period
        self._publish(now=now)
        return self.is_running()

    def metadata(self):
        if self._closed:
            return dict(self._last_metadata, running=False, state="closed")
        with self.lock():
            camera = self._viewer.cam
            viewport = self._viewer.viewport
            width = int(viewport.width) if viewport is not None else 0
            height = int(viewport.height) if viewport is not None else 0
            camera_state = {"lookat": [float(value) for value in camera.lookat],
                            "distance": float(camera.distance),
                            "azimuth": float(camera.azimuth),
                            "elevation": float(camera.elevation)}
        self._last_metadata = {
            "schema_version": 1, "native_viewer": True, "backend": "glfw",
            "pid": os.getpid(), "running": self.is_running(),
            "viewport": {"width": width, "height": height},
            "viewport_size": [width, height], "framebuffer_size": [width, height],
            "camera": camera_state, "follow_camera": self.follow_camera,
            "camera_resets": self._camera_resets, "sync_count": self._sync_count,
            "sync_target_hz": 60, "renderer": "mujoco_native_glfw",
            "monotonic_ns": time.monotonic_ns(), **self._last_state,
        }
        return dict(self._last_metadata)

    def _publish(self, now=None, force=False):
        if self._status_dir is None:
            return
        now = self._clock() if now is None else now
        if not force and now < self._next_publish:
            return
        self._status_dir.mkdir(parents=True, exist_ok=True)
        path = self._status_dir / "viewer.json"
        temporary = path.with_name("viewer.json.tmp")
        temporary.write_text(json.dumps(self.metadata(), ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(path)
        self._next_publish = now + .1

    def close(self):
        if self._closed:
            return
        self.metadata()
        self._viewer.close()
        self._closed = True
        deadline = time.monotonic() + 2.
        for thread in self._render_threads:
            thread.join(timeout=max(0., deadline - time.monotonic()))
        alive = [thread.name for thread in self._render_threads if thread.is_alive()]
        if alive:
            raise RuntimeError("MuJoCo native rendering thread did not stop: " + ", ".join(alive))
        self._publish(force=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
