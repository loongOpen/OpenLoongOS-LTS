// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
struct alignas(8) DepthFrame {
  uint64_t sequence;
  uint32_t magic, version;
  double simulation_time, monotonic_time;
  float depth[18*32];
  double root_position[3], root_quaternion[4];
};
constexpr uint32_t depth_magic=0x4c445031;
