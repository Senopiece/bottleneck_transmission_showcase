#pragma once

#include <cstdint>

namespace receiver {

constexpr int kPacketBits = 5;
constexpr int kMessageWidth = 8;
constexpr int kMessageHeight = 8;
constexpr int kMessageBits = kMessageWidth * kMessageHeight;

struct PacketLlrs {
  int64_t packet_index = 0;
  float llr[kPacketBits] = {};
};

}  // namespace receiver
