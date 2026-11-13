# SPDX-License-Identifier: MIT
"""ctypes binding for the real native Lite26 dual-input ONNX plan."""
from __future__ import annotations

import ctypes
from pathlib import Path
import platform

import numpy as np


def ort_library_path() -> Path:
    import onnxruntime
    directory = Path(onnxruntime.__file__).resolve().parent / "capi"
    libraries = sorted(directory.glob("libonnxruntime.so*"))
    if not libraries:
        raise RuntimeError(f"No ONNX Runtime C library in {directory}; install onnxruntime >= 1.21")
    return libraries[-1]


class NativeParkourPolicy:
    """One native context, owned by the simulation thread; no Python ORT fallback."""

    def __init__(self, model: Path, library: Path | None = None):
        machine = platform.machine().lower()
        suffix = {"x86_64": "x64", "aarch64": "a64", "arm64": "a64"}.get(machine)
        if suffix is None:
            raise RuntimeError(f"Unsupported native parkour platform: {machine}")
        self.library_path = library or Path(__file__).resolve().parents[1] / "module/parkour" / f"libplan_parkour_{suffix}.so"
        if not self.library_path.is_file():
            raise FileNotFoundError(f"Missing native plan {self.library_path}; run loong_gui_sim/native/build_parkour_plan.sh")
        self.lib = ctypes.CDLL(str(self.library_path))
        self.lib.loong_parkour_abi_version.restype = ctypes.c_uint
        if self.lib.loong_parkour_abi_version() != 1:
            raise RuntimeError("Unsupported parkour native ABI")
        self.lib.loong_parkour_contract.restype = ctypes.c_char_p
        self.contract = self.lib.loong_parkour_contract().decode()
        self.lib.loong_parkour_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.loong_parkour_create.restype = ctypes.c_void_p
        float_pointer = ctypes.POINTER(ctypes.c_float)
        self.lib.loong_parkour_infer.argtypes = [ctypes.c_void_p, float_pointer, ctypes.c_size_t,
                                                float_pointer, ctypes.c_size_t, float_pointer, ctypes.c_size_t,
                                                ctypes.c_void_p, ctypes.c_size_t]
        self.lib.loong_parkour_infer.restype = ctypes.c_int
        self.lib.loong_parkour_destroy.argtypes = [ctypes.c_void_p]
        self.lib.loong_parkour_destroy.restype = None
        self.error = ctypes.create_string_buffer(2048)
        self.context = self.lib.loong_parkour_create(str(model.resolve()).encode(), str(ort_library_path()).encode(),
                                                    self.error, len(self.error))
        if not self.context:
            raise RuntimeError(self.error.value.decode(errors="replace"))
        self.calls = 0

    def __call__(self, proprio, depth):
        proprio = np.ascontiguousarray(proprio, dtype=np.float32)
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        if proprio.shape != (1,450) or depth.shape != (1,8,18,32):
            raise ValueError(f"Wrong Lite26 policy shapes: {proprio.shape}, {depth.shape}")
        actions = np.empty(26, dtype=np.float32)
        pointer = ctypes.POINTER(ctypes.c_float)
        status = self.lib.loong_parkour_infer(self.context, proprio.ctypes.data_as(pointer), proprio.size,
                                             depth.ctypes.data_as(pointer), depth.size,
                                             actions.ctypes.data_as(pointer), actions.size,
                                             self.error, len(self.error))
        if status:
            raise RuntimeError(self.error.value.decode(errors="replace"))
        self.calls += 1
        return actions.astype(np.float64)

    def close(self):
        if getattr(self,"context",None):
            self.lib.loong_parkour_destroy(self.context)
            self.context = None

    def __del__(self):
        self.close()
