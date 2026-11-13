#!/usr/bin/env bash
# Use the activated Isaac Lab Python, or set LITE_PYTHON to its absolute path.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
cd -- "$project_root"
if [[ -n "${LITE_PYTHON:-}" ]]; then
    python_bin="$LITE_PYTHON"
elif [[ -x "$project_root/.venv/bin/python" ]]; then
    python_bin="$project_root/.venv/bin/python"
else
    python_bin=python
fi

if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf '未找到 Python：%s。请激活 README.md 指定的 Python 3.12 训练环境，或设置 LITE_PYTHON。\n' "$python_bin" >&2
    exit 1
fi

export PYTHONPATH="$project_root/source/lite_parkour${PYTHONPATH:+:$PYTHONPATH}"
# Fail before importing Kit when an old Isaac Lab or a CPU-only torch is active.
# A report is retained for a new machine's environment setup and diagnostics.
"$python_bin" -B scripts/check_environment.py --training --report logs/environment-check.json

# Kit can change the loader search paths after torch is imported. Make the
# matching pip NVRTC builtins discoverable (CUDA 13 wheels use nvidia/cu13/lib).
nvrtc_lib_dir="$("$python_bin" - <<'PY'
from pathlib import Path
import sys
import torch

if torch.version.cuda:
    major = torch.version.cuda.split(".")[0]
    candidates = [Path(p) / suffix for p in sys.path if p
                  for suffix in (f"nvidia/cu{major}/lib", "nvidia/cuda_nvrtc/lib")]
    for candidate in candidates:
        if any(candidate.glob(f"libnvrtc-builtins.so.{major}.*")):
            print(candidate)
            break
PY
)"
if [[ -n "$nvrtc_lib_dir" ]]; then
    export LD_LIBRARY_PATH="$nvrtc_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMNI_KIT_ACCEPT_EULA=YES
# NVRTC rejects long UTF-8 temporary directory paths. Keep its scratch path
# short even when the checkout lives under a long Chinese directory name.
kernel_tmp_dir="$(mktemp -d /tmp/loong-lite.XXXXXXXX)"
export TMPDIR="$kernel_tmp_dir"
trap 'rm -rf -- "$kernel_tmp_dir"' EXIT
mkdir -p -- "$project_root/logs/launches"
completion_dir="$(mktemp -d "$project_root/logs/launches/train.XXXXXX")"
completion_report="$completion_dir/completed.json"
completion_nonce="${completion_dir##*/}"
launch_log="$completion_dir/train.log"
[[ ! -e "$completion_report" ]]

# Local materials avoid downloading textures when training without a GUI.
if "$python_bin" -u scripts/rsl_rl/train.py \
    --task Lite-AMP-Depth-v0 --headless \
    env.scene.terrain.visual_material=null \
    env.scene.sky_light.spawn.texture_file=null \
    "$@" --completion_report "$completion_report" --completion_nonce "$completion_nonce" 2>&1 | tee "$launch_log"; then
    # Some Kit shutdown paths return zero after a Python exception. A fresh
    # report written after learn()/save/close is required in addition to exit 0.
    if [[ ! -s "$completion_report" ]]; then
        printf '训练未完成：未生成完成报告；日志：%s；报告：%s\n' "$launch_log" "$completion_report" >&2
        exit 1
    fi
    if ! "$python_bin" -B scripts/validate_training_report.py "$completion_report" "$completion_nonce" 2>&1 | tee -a "$launch_log"; then
        printf '训练完成校验失败；日志：%s；报告：%s\n' "$launch_log" "$completion_report" >&2
        exit 1
    fi
else
    result="$?"
    printf '训练进程失败；日志：%s；报告：%s\n' "$launch_log" "$completion_report" >&2
    exit "$result"
fi
