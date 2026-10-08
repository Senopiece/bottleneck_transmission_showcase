# Neural Vision Extractor

This app now uses neural models only for the vision extractor:

```text
camera frame
  -> broad pose ascent with tracker_acquire.onnx
  -> precise pose refinement/verification with tracker_precise.onnx
  -> accepted marker pose
  -> led_reader.onnx
  -> 5 soft LED scores
  -> existing preamble / clock / weighted sampling / BP decoder
```

The protocol layer is intentionally unchanged. Preamble detection, rate snapping,
symbol windows, erasures, online BP/fountain recovery, and result UI still live in
the existing Android receiver code.

## Assets

Model assets:

```text
app/src/main/assets/tracker_acquire.onnx
app/src/main/assets/tracker_precise.onnx
app/src/main/assets/led_reader.onnx
```

They are copied from:

```text
receiver/models/tracker_likelihood/tracker_likelihood_acquire_v005.onnx
receiver/models/tracker_likelihood/tracker_likelihood_precise_v005.onnx
receiver/models/led_reader/led_reader_crop_v003_gate.onnx
```

Android inference uses ONNX Runtime Android.

## Tracker Model

Both trackers score one full, geometrically fixed marker pose. Acquire input is
`N x 2 x 24 x 64`; precise input is `N x 2 x 36 x 96`. Both output broad and
sharp logits.

```text
input:  patch, float32, N x 2 x H x W
output: likelihood_logits, float32, N x 2
```

Patch channels:

1. normalized luma: per-patch mean/std, clipped to `[-3, 3] / 3`;
2. edge magnitude: finite-difference gradient normalized by p95.

The CNN head preserves its final spatial grid. Global average pooling is not
used: it made endpoint locations too translation-invariant and caused broad pose
optima. Acquire uses the broad `64 x 24` model. Every acquired candidate must
then pass high-resolution sharp refinement/verification. Tracking uses the same
precise model.

The pose has exactly four degrees of freedom:

- center x;
- center y;
- angle around marker center;
- marker square-to-triangle distance.

The square, five LEDs, and triangle have fixed relative geometry. They are not
independently scaled or rotated.

## Pose Ascent

`LedFrameDecoder` runs a fixed-budget coordinate ascent:

- acquire mode starts from the centered guide pose;
- tracking mode scores both previous pose and a bounded constant-velocity
  prediction, then keeps the better seed;
- each step tests +/- x, +/- y, +/- angle, +/- log-distance;
- all eight neighbors of one step are evaluated in one batched ONNX call;
- each scale can take two improving coordinate moves before the step shrinks;
- score is `sigmoid` of the acquire or tracking likelihood logit;
- acquire and tracking use separate score thresholds.

No pose smoothing filter is used. Motion prediction only seeds likelihood
ascent, and cannot override a better previous-pose score.

## LED Model

The LED model scores each LED independently:

```text
input 1: led_crop, float32, 5 x 3 x 28 x 28
input 2: detector_likelihood, float32, 5
output: gated logits, float32, 5
```

Crop channels:

1. normalized luma;
2. blue dominance;
3. edge magnitude.

The model contains a learned positive gate over `detector_likelihood`. Low marker
confidence reduces LED logit magnitude before the protocol decoder sees it.

Android maps logits to the legacy packet score scale:

```text
score = 0.54 + sigmoid(logit) * 0.64
```

That keeps the existing packet sampler thresholds and BP code compatible.

## Debugging

Debug builds can show:

- tracker hit/miss;
- acquire/tracking mode;
- tracker score;
- per-LED scores;
- total decoder timing;
- tracker patch preparation and ONNX timing, including call/candidate counts;
- LED patch preparation and ONNX timing.

Release builds should keep debug overlays/logging behind compile-time flags.

## Updating Models

1. Train/export in `receiver/training`.
2. Render overlays in `receiver/tools` and inspect good/bad videos.
3. Copy ONNX files into `android-reader/app/src/main/assets`.
4. Build:

```powershell
cd android-reader
.\gradlew.bat :app:compileDebugKotlin
```

## Android Port Notes

The Android extractor must stay numerically close to the Python feature
extraction:

- same patch sizes;
- same local marker coordinates;
- same luma normalization;
- same edge normalization;
- same LED crop size.

If tracker quality diverges between overlays and phone, first compare canonical
patch dumps rather than tuning thresholds.

## Room For Improvement

- Add a local pose-update head. It could support two or three recurrent
  corrections instead of several coordinate-ascent neighborhoods, but must be
  accepted from end-to-end pose and false-positive metrics rather than regression
  loss alone.
- Benchmark static int8 quantization of both spatial ReLU trackers.
- Benchmark CPU thread counts and NNAPI after batching; tiny unbatched calls often
  make accelerator dispatch slower rather than faster.
- Add Android/Python canonical patch parity tests from saved frames.
- Expand manually verified pose labels beyond weak pseudo-labels.
- Train the LED gate with more low-confidence tracked examples and downstream BP
  loss, not only per-crop weak labels.
- Try int8 quantization or TFLite/NNAPI only after the floating ONNX baseline is
  stable on-device.
