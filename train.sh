#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! command -v uv >/dev/null 2>&1; then
    printf '未找到 uv，请按 README.md 安装训练依赖。\n' >&2
    exit 1
fi
cd "$BUNDLE_DIR/mjlab_amp_ovision"
exec uv run --locked --extra cu128 train "$@"
