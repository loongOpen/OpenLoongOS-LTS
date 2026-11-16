#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--help" ]]; then
    cat <<'HELP'
用法：./start_openloong_gui.sh [GUI 参数]
默认使用 VirtualGL GPU 渲染，直接启动项目内的 GUI。
--local  本机图形桌面直接启动，不加载 VirtualGL。
QT_QPA_PLATFORM=offscreen 时自动使用离屏模式，不要求 DISPLAY 或 VirtualGL。
依赖：可用的图形桌面、NVIDIA 驱动及 VirtualGL。
可通过 VIRTUALGL_HOME 指定 VirtualGL 安装目录，VGL_DISPLAY 选择 GPU（默认 egl0）。
./start_openloong_gui.sh --check-renderer  只查询渲染器，不启动 GUI 或控制链。
HELP
    exit 0
fi
local_mode=0
if [[ "${1:-}" == "--local" ]]; then
    local_mode=1
    shift
fi
if [[ "${QT_QPA_PLATFORM:-}" == "offscreen" && "${1:-}" != "--check-renderer" ]]; then
    local_mode=1
fi
if [[ "$local_mode" == 1 && "${1:-}" == "--check-renderer" ]]; then
    printf '%s\n' '--check-renderer 用于 VirtualGL 检查，请勿与 --local 同时使用。' >&2
    exit 2
fi
export PYTHONNOUSERSITE=1
if [[ "$local_mode" == 0 ]]; then
    if [[ -z "${DISPLAY:-}" ]]; then
        printf '未找到 DISPLAY。请在远程图形桌面的终端中运行此脚本。\n' >&2
        exit 1
    fi
    vgl_root=""
    if [[ -n "${VIRTUALGL_HOME:-}" ]]; then
        candidates=("$VIRTUALGL_HOME")
    else
        candidates=("$BUNDLE_DIR/tools/virtualgl" "/opt/VirtualGL" "$HOME/.local/opt/virtualgl-3.1.5")
    fi
    for candidate in "${candidates[@]}"; do
        if [[ -r "$candidate/lib/libdlfaker.so" && -r "$candidate/lib/libvglfaker.so" ]]; then
            vgl_root="$candidate"
            break
        fi
    done
    if [[ -z "$vgl_root" ]]; then
        printf '未找到 VirtualGL。安装后设置 VIRTUALGL_HOME=/安装目录；详见 docs/quickstart.md 的远程桌面说明。\n' >&2
        exit 1
    fi
    export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
    export PYTHONNOUSERSITE=1
    export VGL_DISPLAY="${VGL_DISPLAY:-egl0}"
    export VGL_COMPRESS="${VGL_COMPRESS:-proxy}"
    unset LIBGL_ALWAYS_SOFTWARE
    export LD_PRELOAD="$vgl_root/lib/libdlfaker.so:$vgl_root/lib/libvglfaker.so${LD_PRELOAD:+:$LD_PRELOAD}"
    if [[ "${1:-}" == "--check-renderer" ]]; then
        if [[ ! -x "$vgl_root/bin/glxinfo" ]]; then
            printf '该 VirtualGL 安装中没有 bin/glxinfo，无法查询渲染器。\n' >&2
            exit 1
        fi
        exec "$vgl_root/bin/glxinfo" -B
    fi
fi
GUI_BIN="$BUNDLE_DIR/loong_gui_sim/loong_gui_sim"
DEPTH_BRIDGE="$BUNDLE_DIR/loong_gui_sim/native/libsim_depth_bridge.so"
if [[ ! -x "$GUI_BIN" || ! -r "$DEPTH_BRIDGE" ]]; then
    printf 'GUI 或深度图组件缺失，请在项目目录运行 python3 -B check_release.py 检查文件完整性。\n' >&2
    exit 1
fi
export LD_PRELOAD="$DEPTH_BRIDGE${LD_PRELOAD:+:$LD_PRELOAD}"
exec "$GUI_BIN" "$@"
