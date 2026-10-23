#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$BUNDLE_DIR/loong_gui_sim/loong_gui_sim" "$@"
