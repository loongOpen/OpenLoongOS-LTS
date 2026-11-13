#!/usr/bin/env bash
# Sourced by train.sh/play.sh after switching to this bundle's directory.
# Explicit selection also supports a separately managed compatible environment.
if [[ -n "${MJLAB_PYTHON:-}" ]]; then
  python_bin="$MJLAB_PYTHON"
elif [[ -n "${PYTHON:-}" ]]; then
  python_bin="$PYTHON"
elif [[ -x .venv/bin/python ]]; then
  python_bin="$PWD/.venv/bin/python"
else
  python_bin="python3"
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
  printf '找不到 Python：%s。请先运行 bash install.sh，或设置 MJLAB_PYTHON。\n' "$python_bin" >&2
  exit 1
fi
# Walking and dance both use the mjlab package name. Resolve this bundle first
# when a caller deliberately selects a shared environment.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if ! "$python_bin" -c 'import torch, mujoco, mujoco_warp, warp, rsl_rl, tyro' >/dev/null 2>&1; then
  printf '舞蹈运行依赖不完整。请运行 bash install.sh，或用 MJLAB_PYTHON 指定按 requirements.txt 配置的环境。\n' >&2
  exit 1
fi
