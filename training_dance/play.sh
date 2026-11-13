#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
checkpoint="checkpoints/dance_demo.pt"
if [[ $# -gt 0 && "$1" != --* ]]; then
  checkpoint="$1"
  shift
fi
source ./runtime_python.sh
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
exec "$python_bin" -m mjlab.scripts.play Mjlab-Tracking-Flat-LingLong14-Xiao --checkpoint-file "$checkpoint" "$@"
