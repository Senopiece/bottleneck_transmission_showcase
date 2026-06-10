# LED Marker Decoder

The decoder assumes the user manually places the physical pattern near the center of the reader ROI. The algorithm only corrects small placement errors.

## Pose

The optimized pose is:

- `cx`, `cy`: center of the whole pattern, halfway between square and triangle marker centers.
- `angle`: pattern axis angle around `cx/cy`.
- `logDistance`: log of the distance between square center and triangle center.

`logDistance` is used instead of raw distance so scale updates are multiplicative and bounded.

When tracking is continuous, the previous accepted pose is used as the seed. Otherwise, the seed is the ideal centered pose in the ROI with pattern width close to the expected user framing.

## Optimization

The current implementation uses a fixed small coordinate-ascent schedule, not a full image scan:

1. Evaluate the current pose.
2. For each step size, test `+/- x`, `+/- y`, `+/- angle`, and `+/- logDistance`.
3. Move to the best candidate if it improves the score.
4. Repeat for a fixed decreasing step schedule.

If a step level does not improve a pose that already passes the acceptance gates, refinement stops early. This avoids spending fine-grained steps when the seed was already good.

There are two schedules:

- acquire: wider steps from the ideal ROI-centered pose;
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

- final score is above `MIN_ACCEPT_SCORE`
- square and triangle component scores pass individual minimum gates

Otherwise `null` is emitted and tracking is reset after repeated misses.

The five LEDs are decoded only after the marker pose is accepted. They do not contribute to pose scoring.

## Bit Decoding

Bits are decoded from the five expected LED slots after the pose is accepted.

The decoder computes five per-slot on-scores and compares each score with the fixed `BIT_ABSOLUTE_ON_THRESHOLD`.
