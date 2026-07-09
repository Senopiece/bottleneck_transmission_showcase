# Neural Vision Extractor

This app now uses neural models only for the vision extractor:

```text
camera frame
  -> pose hypothesis ascent
  -> tracker_likelihood.onnx
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
app/src/main/assets/tracker_likelihood.onnx
app/src/main/assets/led_reader.onnx
```

They are copied from:

```text
receiver/models/tracker_likelihood/tracker_likelihood_fast_v003.onnx
receiver/models/led_reader/led_reader_crop_v003_gate.onnx
```

Android inference uses ONNX Runtime Android.

## Tracker Model

The tracker is a black-box likelihood function for one full marker pose:

```text
input:  patch, float32, 1 x 2 x 36 x 96
output: likelihood_logit, float32, 1
```

Patch channels:

1. normalized luma: per-patch mean/std, clipped to `[-3, 3] / 3`;
2. edge magnitude: finite-difference gradient normalized by p95.

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
- tracking mode starts from the previous accepted pose;
- each step tests +/- x, +/- y, +/- angle, +/- log-distance;
- score is `sigmoid(tracker_likelihood_logit)`;
- acquire and tracking use separate score thresholds.

This is deliberately simple and bounded. The neural model makes the likelihood
less sensitive to material glare and lighting, while the ascent remains cheap.

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
- decoder timing.

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

- Batch pose candidates into one tracker ONNX call per ascent step. The current
  implementation is simpler and easier to verify, but one call per candidate has
  avoidable runtime overhead.
- Add Android/Python canonical patch parity tests from saved frames.
- Calibrate acquire/tracking thresholds from good/bad validation score distributions.
- Train the LED gate with more low-confidence tracked examples and downstream BP
  loss, not only per-crop weak labels.
- Try int8 quantization or TFLite/NNAPI only after the floating ONNX baseline is
  stable on-device.
