# SPDX-License-Identifier: MIT
"""Camera/transport regressions for the real native viewer controller."""
from contextlib import nullcontext
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import threading
import unittest

from native_view import NativePolicyView


class Handle:
    def __init__(self):
        self.cam = SimpleNamespace(type=0, fixedcamid=-1, distance=1., azimuth=0.,
                                   elevation=0., lookat=[0., 0., 0.])
        self.opt = SimpleNamespace(geomgroup=[1] * 6)
        self.viewport = SimpleNamespace(width=1512, height=824)
        self.open = True
        self.syncs = 0

    def is_running(self):
        return self.open

    def lock(self):
        return nullcontext()

    def sync(self):
        self.syncs += 1

    def close(self):
        self.open = False


class NativeViewTests(unittest.TestCase):
    def make_view(self, status_dir=None):
        handle = Handle()
        time = SimpleNamespace(now=0.)
        view = NativePolicyView(SimpleNamespace(), SimpleNamespace(qpos=[0, 0, 1], time=0.),
                                status_dir=status_dir, viewer_factory=lambda *a, **k: handle,
                                clock=lambda: time.now, apply_style=False)
        self.addCleanup(view.close)
        return view, handle, time

    def test_follow_preserves_native_orbit_zoom_and_pan(self):
        view, handle, clock = self.make_view()
        handle.cam.distance, handle.cam.azimuth, handle.cam.elevation = 1.7, 44., -37.
        handle.cam.lookat[:] = [8., 9., 10.]
        view.update([1, 2, 1])
        self.assertEqual(handle.cam.lookat, [9., 11., 10.])
        self.assertEqual((handle.cam.distance, handle.cam.azimuth, handle.cam.elevation), (1.7, 44., -37.))
        view.update([3, 4, 1], {"follow_camera": False})
        self.assertEqual(handle.cam.lookat, [9., 11., 10.])
        view.update([3, 4, 1], {"follow_camera": True})
        self.assertEqual(handle.cam.lookat, [3., 4., 1.2])
        self.assertEqual(handle.cam.distance, 1.7)

    def test_explicit_camera_reset_happens_once(self):
        view, handle, clock = self.make_view()
        handle.cam.distance = .6
        view.update([0, 0, 1], {"reset_camera": 1})
        self.assertEqual(handle.cam.distance, 3.5)
        handle.cam.distance = 2.3
        view.update([0, 0, 1], {"reset_camera": 1})
        self.assertEqual(handle.cam.distance, 2.3)

    def test_sync_deadline_does_not_drop_to_fifty_hz(self):
        view, handle, clock = self.make_view()
        for step in range(201):
            clock.now = step * .005
            view.update([0, 0, 1])
        self.assertIn(handle.syncs, (60, 61))
        self.assertEqual(view.metadata()["sync_count"], handle.syncs)

    def test_native_framebuffer_metadata_and_closed_state(self):
        with TemporaryDirectory() as directory:
            view, handle, clock = self.make_view(Path(directory))
            view.update([0, 0, 1], sim_time=.02, enabled=False, state="paused")
            clock.now = .2
            view.update([0, 0, 1], sim_time=.02, enabled=False, state="paused")
            record = json.loads((Path(directory) / "viewer.json").read_text())
            self.assertEqual(record["framebuffer_size"], [1512, 824])
            self.assertTrue(record["native_viewer"])
            self.assertEqual(record["state"], "paused")
            view.close()
            self.assertFalse(json.loads((Path(directory) / "viewer.json").read_text())["running"])

    def test_bad_camera_command_is_rejected(self):
        view, handle, clock = self.make_view()
        for command in ({"follow_camera": "yes"}, {"reset_camera": True}):
            with self.assertRaises(ValueError):
                view.update([0, 0, 1], command)

    def test_close_joins_the_new_native_rendering_thread(self):
        stopped = threading.Event()
        handle = Handle()
        def factory(*args, **kwargs):
            thread = threading.Thread(target=stopped.wait, daemon=True)
            thread.start()
            handle.close = lambda: (setattr(handle, "open", False), stopped.set())
            return handle
        view = NativePolicyView(SimpleNamespace(), SimpleNamespace(qpos=[0, 0, 1], time=0.),
                                viewer_factory=factory, apply_style=False)
        threads = view._render_threads
        self.assertEqual(len(threads), 1)
        view.close()
        self.assertFalse(any(thread.is_alive() for thread in threads))


if __name__ == "__main__":
    unittest.main()
