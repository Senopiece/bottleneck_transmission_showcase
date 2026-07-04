#pragma once

#include <cstdint>

namespace receiver {

struct Point2f {
  float x = 0.0f;
  float y = 0.0f;
};

struct Pose {
  Point2f center;
  float angle_rad = 0.0f;
  float scale_px = 1.0f;
};

}  // namespace receiver
