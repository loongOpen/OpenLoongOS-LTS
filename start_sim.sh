#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_PRELOAD="$BUNDLE_DIR/loong_gui_sim/native/libsim_depth_bridge.so${LD_PRELOAD:+:$LD_PRELOAD}"
exec "$BUNDLE_DIR/loong_gui_sim/loong_gui_sim" "$@"
