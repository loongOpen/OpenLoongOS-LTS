#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
case "$(uname -m)" in x86_64|amd64) arch=x64;; aarch64|arm64) arch=a64;; *) echo "Unsupported CPU architecture" >&2; exit 1;; esac
libdir="../runtime/third_party/lib_lin_${arch}"
test -f vendor/onnxruntime/onnxruntime_c_api.h
test -f "$libdir/libonnxruntime.so"
"${CXX:-c++}" -std=c++17 -O2 -fPIC -shared -Wall -Wextra dance_plan.cpp \
  -L"$libdir" -lonnxruntime -Wl,-z,defs -Wl,-rpath,'$ORIGIN/../runtime/third_party/lib_lin_'"$arch" \
  -o "libdance_plan_${arch}.so"
echo "Built native/libdance_plan_${arch}.so"
