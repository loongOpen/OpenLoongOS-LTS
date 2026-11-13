#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${1:---help}" in
    -h|--help)
        cat <<'HELP'
用法：./train.sh <walking|dance|parkour> [训练参数]

  walking  行走（默认 Mjlab-Amp-Flat-LingLong；也可先指定其它 walking task）
  dance    跳舞（Mjlab-Tracking-Flat-LingLong14-Xiao）
  parkour  视觉越障（Lite-AMP-Depth-v0；使用已激活的 Isaac Lab Python）

示例：
  ./train.sh walking --env.scene.num-envs 1024
  ./train.sh dance --device cuda:0 --env.scene.num-envs 1024
  LITE_PYTHON=/path/to/python ./train.sh parkour --num_envs 4096

保留旧用法：./train.sh Mjlab-Amp-Flat-LingLong [参数]
三套环境分别安装，详见各 training_*/README.md。不会交叉安装或升级依赖。
HELP
        exit 0
        ;;
    dance)
        shift
        exec bash "$BUNDLE_DIR/training_dance/train.sh" "$@"
        ;;
    parkour)
        shift
        exec bash "$BUNDLE_DIR/training_parkour/scripts/train_lite.sh" "$@"
        ;;
    walking)
        shift
        ;;
    Mjlab-*)
        ;;
    *)
        printf '未知训练功能：%s。运行 ./train.sh --help 查看入口。\n' "$1" >&2
        exit 2
        ;;
esac
if ! command -v uv >/dev/null 2>&1; then
    printf '未找到 uv，请按 training_walking/README.md 安装行走训练依赖。\n' >&2
    exit 1
fi
cd "$BUNDLE_DIR/training_walking"
if [[ $# -eq 0 || "$1" == -* ]]; then
    set -- Mjlab-Amp-Flat-LingLong "$@"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec uv run --locked --extra cu128 train "$@"
