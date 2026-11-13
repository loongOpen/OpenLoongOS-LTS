#!/usr/bin/env bash
set -euo pipefail
native_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
arch="$(uname -m)"
case "$arch" in
  x86_64) suffix=x64 ;;
  aarch64|arm64) suffix=a64 ;;
  *) echo "Unsupported parkour plan architecture: $arch" >&2; exit 2 ;;
esac
output_dir="$native_dir/../runtime/module/parkour"
mkdir -p "$output_dir"
"${CXX:-g++}" -std=c++17 -O2 -fPIC -shared -Wall -Wextra -Werror \
  "$native_dir/parkour_plan.cpp" -ldl -o "$output_dir/libplan_parkour_${suffix}.so"
echo "$output_dir/libplan_parkour_${suffix}.so"
