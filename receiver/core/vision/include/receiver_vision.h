#pragma once

#include <cstdint>
#include "receiver_common.h"

namespace receiver {

constexpr int kLedCount = 5;

struct FrameView {
  int width = 0;
  int height = 0;
  int y_stride = 0;
  const uint8_t* y = nullptr;
};

enum class TrackerMode : int {
  kAcquire = 0,
  kTrack = 1,
};

struct TrackerResult {
  bool hit = false;
  TrackerMode mode = TrackerMode::kAcquire;
  float confidence = 0.0f;
  Pose pose;
  Point2f square_center;
  Point2f triangle_center;
  Point2f led_centers[kLedCount];
  float led_scores[kLedCount] = {};
};

class Tracker {
 public:
  Tracker();
  ~Tracker();

  TrackerResult Process(const FrameView& frame, int64_t timestamp_ns);
  void Reset();
};

}  // namespace receiver
