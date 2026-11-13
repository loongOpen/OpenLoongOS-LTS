#!/usr/bin/env bash
set -euo pipefail
native=$(cd "$(dirname "$0")" && pwd)
runtime="$native/../runtime"
eigen_include=${EIGEN_INCLUDE_DIR:-/usr/include/eigen3}
[[ $(uname -m) == x86_64 ]] || { echo 'This bundle targets Linux x86_64.' >&2; exit 1; }
[[ -f "$eigen_include/Eigen/Dense" ]] || { echo 'Install libeigen3-dev (Eigen 3.4), or set EIGEN_INCLUDE_DIR.' >&2; exit 1; }
cd "$native/sdk"
for item in walking dance parkour; do
  dance=0; walk=0
  [[ "$item" == dance ]] && dance=1
  [[ "$item" == walking ]] && walk=1
  "${CXX:-g++}" -std=c++17 -O2 -fPIC -shared -DDefSim -DPOLICY_DANCE="$dance" -DPOLICY_WALKING="$walk" \
    -I"$native/vendor/quill" -I"$native/vendor/loong_sdk" -I"$eigen_include" sdk_policy_plan.cpp \
    -L"$runtime/module/nabo_locomotion" -Wl,--no-as-needed -lnabo_x64 -ldl \
    -Wl,-z,defs -Wl,-rpath,'$ORIGIN' -o "$runtime/module/nabo_locomotion/libplan_${item}_sdk_x64.so"
done
"${CXX:-g++}" -std=c++17 -O2 -fPIC -shared walking_plan.cpp -I"$native/vendor/onnxruntime" \
  -L"$runtime/third_party/lib_lin_x64" -lonnxruntime -Wl,-z,defs \
  -Wl,-rpath,'$ORIGIN/../runtime/third_party/lib_lin_x64' -o "$native/libwalking_plan_x64.so"
"${CXX:-g++}" -std=c++17 -O2 -fPIC -shared -I"$native/vendor" sim_depth_bridge.cpp \
  -L"$runtime/third_party/lib_lin_x64" -lmujoco -ldl -Wl,-z,defs \
  -Wl,-rpath,'$ORIGIN/../runtime/third_party/lib_lin_x64' -o "$native/libsim_depth_bridge.so"
echo 'Built SDK plans and same-world depth sensor for MuJoCo 3.1.1.'
