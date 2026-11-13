#!/usr/bin/env bash
# Install only into this project's own Python 3.12 environment.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
venv_dir="${LITE_VENV_DIR:-$project_root/.venv}"
lab_dir="${LITE_ISAACLAB_DIR:-$project_root/.external/IsaacLab}"
cd -- "$project_root"

for command_name in uv git; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        printf '请先安装 %s，再执行本脚本。\n' "$command_name" >&2
        exit 1
    fi
done
if [[ ! -x "$venv_dir/bin/python" ]]; then
    uv venv --python 3.12 "$venv_dir"
fi
python_bin="$venv_dir/bin/python"
"$python_bin" -c 'import platform,sys; assert sys.version_info[:2] == (3,12), "需要 Python 3.12"; assert platform.system() == "Linux" and platform.machine() == "x86_64", "需要 Linux x86_64"'
if [[ ! -d "$lab_dir/.git" ]]; then
    if [[ -e "$lab_dir" ]]; then
        printf '目标目录已存在且不是 Git 仓库：%s\n' "$lab_dir" >&2
        exit 1
    fi
    mkdir -p -- "$(dirname -- "$lab_dir")"
    git clone --branch v3.0.0-beta2 --depth 1 https://github.com/isaac-sim/IsaacLab.git "$lab_dir"
fi
"$python_bin" -B scripts/prepare_isaaclab.py "$lab_dir"

# Avoid upstream extras pinning Sim 5.1/RSL-RL 5.0.1 and its CLI's Torch 2.10
# reinstall. Resolve all declared dependencies together against Sim 6's pins.
editable_args=()
for module_name in isaaclab isaaclab_ppisp isaaclab_assets isaaclab_contrib \
    isaaclab_experimental isaaclab_newton isaaclab_ov isaaclab_ovphysx \
    isaaclab_physx isaaclab_rl isaaclab_tasks isaaclab_tasks_experimental \
    isaaclab_visualizers; do
    editable_args+=(-e "$lab_dir/source/$module_name")
done
uv pip install --python "$python_bin" \
    --index-url "${LITE_PYPI_INDEX_URL:-https://pypi.org/simple}" \
    -r requirements-training-cu128.lock \
    "${editable_args[@]}" -e '.[train,sim,test]'
uv pip check --python "$python_bin"
"$python_bin" -B scripts/check_environment.py --training --report logs/environment-check.json
printf '安装与环境预检完成。短训练：LITE_PYTHON=%q bash scripts/train_lite.sh --num_envs 2 --max_iterations 2 --run_name setup_smoke --device cuda:0 agent.device=cuda:0 agent.save_interval=1\n' "$python_bin"
