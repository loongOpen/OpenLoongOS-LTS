#!/usr/bin/env bash
# Install only the CPU policy/MuJoCo environment, independently of training.
set -euo pipefail
gui_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_project="$gui_root/runtime/python"
if [[ "${1:-}" == "--help" ]]; then
  printf 'Usage: %s\nCreates runtime/python/.venv with the locked CPU dependencies and loads all three ONNX policies plus both native plans.\nNeeds uv and a usable graphics driver for interactive simulation.\n' "$0"
  exit 0
fi
if [[ $# -ne 0 ]]; then printf 'Unknown argument. Use --help.\n' >&2; exit 2; fi
command -v uv >/dev/null || { printf 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/\n' >&2; exit 1; }
uv sync --project "$runtime_project" --locked --python "${LOONG_RUNTIME_PYTHON_VERSION:-3.12}"
"$runtime_project/.venv/bin/python" "$runtime_project/check_runtime.py" --policies
printf 'Runtime Python: %s/.venv/bin/python\n' "$runtime_project"
