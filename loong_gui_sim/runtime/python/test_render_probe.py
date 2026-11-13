"""Regression checks for renderer selection after a backend cannot initialize."""
import os
import unittest
from unittest.mock import patch

from render_probe import candidate_backends, probe_rendering


class RenderProbeTests(unittest.TestCase):
    def test_desktop_and_headless_order(self):
        self.assertEqual(candidate_backends(environment={"DISPLAY": ":1", "MUJOCO_GL": "egl"}), ["glfw", "egl"])
        self.assertEqual(candidate_backends(environment={}), ["egl", "glfw"])

    def test_egl_render_failure_can_use_glfw(self):
        calls = []
        def run(backend, **kwargs):
            calls.append(backend)
            return {"ok": backend == "glfw", "backend": backend, "phase": "render", "error": "EGL display unavailable"}
        with patch.dict(os.environ, {}, clear=True):
            result = probe_rendering(run_backend=run)
        self.assertTrue(result["ok"])
        self.assertEqual(result["backend"], "glfw")
        self.assertEqual(calls, ["egl", "glfw"])

    def test_glfw_render_failure_can_use_egl(self):
        def run(backend, **kwargs):
            return {"ok": backend == "egl", "backend": backend, "phase": "render", "error": "X11 unavailable"}
        with patch.dict(os.environ, {"DISPLAY": ":invalid"}, clear=True):
            result = probe_rendering(run_backend=run)
        self.assertEqual(result["backend"], "egl")
        self.assertEqual(len(result["attempts"]), 2)

    def test_explicit_backend_does_not_silently_change(self):
        calls = []
        def run(backend, **kwargs):
            calls.append(backend)
            return {"ok": False, "backend": backend, "phase": "render", "error": "bad driver"}
        with patch.dict(os.environ, {"DISPLAY": ":1", "LOONG_RUNTIME_GL": "egl"}, clear=True):
            result = probe_rendering(run_backend=run)
        self.assertFalse(result["ok"])
        self.assertEqual(calls, ["egl"])

    def test_missing_runtime_does_not_repeat_as_driver_failures(self):
        calls = []
        def run(backend, **kwargs):
            calls.append(backend)
            return {"ok": False, "backend": backend, "phase": "environment", "error": "No module named numpy"}
        with patch.dict(os.environ, {"DISPLAY": ":1"}, clear=True):
            result = probe_rendering(run_backend=run)
        self.assertEqual(calls, ["glfw"])
        self.assertIn("No module named numpy", result["error"])


if __name__ == "__main__":
    unittest.main()
