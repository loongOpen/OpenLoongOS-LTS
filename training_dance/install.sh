#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python_bin="${PYTHON:-python3}"
package_index="${MJLAB_PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
torch_index="${MJLAB_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
if [[ ! -x .venv/bin/python ]]; then
  "$python_bin" -m venv .venv
fi
.venv/bin/python - <<'PY'
import sys
if not (3, 12) <= sys.version_info[:2] < (3, 14):
    raise SystemExit("Use Python 3.12 or 3.13 (set PYTHON=python3.13 before the first install).")
PY

# A reused venv may have been created by uv without pip. Bootstrap it before
# using the documented pip-based installer.
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  .venv/bin/python -m ensurepip --upgrade
fi

# Fetch only the CUDA-specific torch wheel here. Resolving all dependencies
# against the PyTorch index can redirect NVIDIA wheels to pypi.nvidia.com.
.venv/bin/python -m pip install --timeout 120 --retries 5 \
  --index-url "$torch_index" --no-deps 'torch==2.9.0+cu128'
# Resolve CUDA runtime packages and all other dependencies through the regular
# package index. The installed +cu128 torch satisfies requirements.txt's pin.
.venv/bin/python -m pip install --timeout 120 --retries 5 \
  --index-url "$package_index" 'setuptools==80.10.2' -r requirements.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m pip check
.venv/bin/python - <<'PY'
import torch
import mujoco
print(f"Runtime ready: torch={torch.__version__}, CUDA={torch.version.cuda}, MuJoCo={mujoco.__version__}")
PY
printf '\nInstalled. Run: bash play.sh\n'
