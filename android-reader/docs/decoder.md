# LED Marker Decoder

The decoder assumes the user initially places the physical pattern near the centered guide pattern. The guide is visual only: once tracking is running, the pose can move anywhere in the camera frame as long as the whole pattern still fits inside the image.

## Pose

The optimized pose is:

- `cx`, `cy`: center of the whole pattern, halfway between square and triangle marker centers.
- `angle`: pattern axis angle around `cx/cy`.
- `logDistance`: log of the distance between square center and triangle center.

`logDistance` is used instead of raw distance so scale updates are multiplicative and bounded.

When tracking is continuous, the previous accepted pose is used as the seed. Otherwise, the seed is the ideal centered pose with pattern width close to the visual guide.

## Optimization

The current implementation uses a fixed small coordinate-ascent schedule, not a full image scan:

1. Evaluate the current pose.
2. For each step size, test `+/- x`, `+/- y`, `+/- angle`, and `+/- logDistance`.
3. Move to the best candidate if it improves the score.
4. Repeat for a fixed decreasing step schedule.

If a step level does not improve a pose that already passes the acceptance gates, refinement stops early. This avoids spending fine-grained steps when the seed was already good.

There are two schedules:

- acquire: wider steps from the ideal centered guide pose;
- tracking: smaller steps from the previous accepted pose, so the tracker does not drift aggressively onto nearby UI/text.

The accepted `Fit` keeps the final score breakdown, so acceptance does not resample the same pose a second time.

## Score

The score is a weighted sum:

- square marker template match
- triangle marker template match

Square and triangle templates are evaluated in normalized marker coordinates. They use positive interior samples and negative outside samples. The square template requires filled corners; the triangle template requires a filled taper and base with empty upper side regions.

The score is intentionally geometric and local. It does not run Canny/Sobel contours or global connected-component analysis.

## Acceptance

A pattern is emitted only when:

- final score is above the acquire/tracking score gate
- square and triangle component scores pass individual minimum gates

Otherwise `null` is emitted and tracking is reset after repeated misses.

The five LEDs are decoded only after the marker pose is accepted. They do not contribute to pose scoring.

## Packet Clock

Bits are first converted to a 5-bit packet with hysteresis:

- off -> on at score `1.0`
- on -> off below score `0.6`

The packet clock has four phases:

1. Wait for a stable `00000` pattern while the user is aimed at the stopped device.
2. Treat those stable zeros as the first preamble packet `00000`.
3. Wait for the remaining preamble packets: `01010`, `10101`, `11111`.
4. Use the measured preamble deltas as the packet period and emit one packet per period.

The visible decode progress starts on `01010`, not on idle zeros. Progress includes the three visible preamble packets plus the twelve payload packets.

At high rates the camera exposure can make the last all-on preamble appear early while the previous transition appears late. A long/short preamble interval pair around `166ms/100ms` is treated as an exposure-overlap case and mapped back to a fast stream period near `125ms` instead of being accepted as a slower `146ms` stream.

If marker detection is lost during the active payload phase, the clock does not immediately abort. It continues ticking by time and emits erasure packets for symbol windows that could not be sampled. Camera/app stream interruption is still treated as a hard failure.

During the active phase the emitted packet is not sampled from a single frame at the period boundary. The decoder accumulates LED scores from the middle of each symbol window and emits soft packets:

- confident on -> `1`
- confident off -> `0`
- ambiguous transition/noisy bit -> `?`

Each non-erased bit also carries a reliability score into the message decoder. This avoids turning exposure tails and transition frames into hard confident errors.

At fast periods, a single high peak is not enough to produce a hard `1` unless the weighted average for the symbol is also high. This makes the decoder prefer `?` over a confident false `1` when the camera sees a short exposure tail from the previous bright symbol.

Fast streams also use a stricter average threshold for hard `1` decisions. A bit that only reaches the normal threshold but not the fast threshold is treated as ambiguous, because the logs show those mid-level samples are usually timing/exposure overlap after a previous bright packet.

If a whole symbol period is skipped, or no frame lands inside the middle sampling window, the clock emits an erasure instead of fabricating a packet from a boundary frame.

## Message Codeword

The virtual device sends a 6x6 binary image as 36 payload bits. The transmitted message is a systematic sparse parity codeword:

- 36 payload bits
- 24 sparse parity bits
- 60 total bits, sent as 12 packets of 5 bits after the four-packet preamble

Each parity bit checks 8 payload bits. The parity matrix is fixed and balanced so every payload bit participates in 5-6 checks. This is still a lightweight systematic codeword, but it is less likely to accept a phase-shifted payload as another valid message than the older 4-input parity layout.

The Android side uses a soft normalized min-sum LDPC pass first. A decoded codeword is accepted only if it stays close to the actually observed hard bits; correcting erasures is fine, but flipping reliable known bits is treated as evidence of a timing/phase error and the decode is rejected. The pass has a fixed iteration budget and is still cheap compared with camera tracking.

If belief propagation fails, the decoder fills erasures from single-unknown parity checks. It then tries an exact GF(2) linear solve for the remaining erasures and only accepts a unique solution that does not flip known bits. The older weighted flip fallback is now restricted to very small erasure cases and is not allowed to flip known bits. If the input has too many erasures, too many fully missing packets, or requires changing observed bits, no image is emitted.
