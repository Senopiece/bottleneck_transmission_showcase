# Receiver

Android-first research workspace for the optical receiver.

The current production direction is:

1. Train tiny PyTorch models on desktop.
2. Export them to ONNX.
3. Run the same canonical patch extraction and likelihood ascent in Android.
4. Feed 5 LED soft scores into the existing Android clock sampler and online BP/fountain decoder.

There is no active C++ portability layer in this tree now. If iOS becomes real later,
the intended path is a native iOS implementation that reuses the same model contracts
and canonical geometry, not shared KMP/C++ code.

## Layout

- `datasets/raw/videos/good`: videos that contain real transfers.
- `datasets/raw/videos/bad`: videos without a transfer, used to punish false positives.
- `datasets/raw/expected`: expected messages used for weak supervision/evaluation.
- `datasets/derived/overlays`: rendered tracking/LED prediction videos.
- `datasets/derived/metrics`: CSV/JSON experiment metrics.
- `models/tracker_likelihood`: current tracker likelihood checkpoint and ONNX export.
- `models/led_reader`: current one-LED scorer checkpoint and ONNX export.
- `models/neural_bp`: placeholder for future neural BP models.
- `training`: PyTorch training code, own `uv` project.
- `tools`: dataset runners, renderers, and evaluators, own `uv` project.

## Current Models

Tracker likelihood:

- acquire input: `N x 2 x 24 x 64`;
- precise verify/tracking input: `N x 2 x 36 x 96`;
- channels: normalized luma and normalized edge magnitude;
- output: broad and sharp logits for `(x, y, angle, scale)`;
- spatial head retains canonical endpoint layout instead of averaging it away.

LED reader:

- input: five independent canonical LED crops, shape `5 x 3 x 28 x 28`;
- second input: detector likelihood for the marker pose;
- output: five gated logits, one per LED crop.

The Android app stores the exported models in:

```text
android-reader/app/src/main/assets/tracker_acquire.onnx
android-reader/app/src/main/assets/tracker_precise.onnx
android-reader/app/src/main/assets/led_reader.onnx
```

## Runtime Pipeline

For every camera frame:

1. The Android decoder builds an initial pose hypothesis:
   - previous pose in tracking mode;
   - centered guide pose in acquire mode.
2. Small-model ascent finds a broad acquire candidate.
3. High-resolution spatial likelihood refines and verifies it.
4. Tracking scores previous and bounded velocity-predicted seeds, then runs
   precise local ascent.
5. Accepted pose yields five canonical LED crops.
6. The LED ONNX model emits soft scores.
7. Existing Android code handles preamble detection, clock inference, weighted symbol sampling, and BP/fountain message recovery.

## How To Run

Install Python environments:

```powershell
cd receiver/tools
uv sync

cd ..\training
uv sync
```

Train/export tracker:

```powershell
cd receiver/training
uv run python -m tracker_likelihood.train --patch-width 64 --patch-height 24 --out ..\models\tracker_likelihood\tracker_likelihood_acquire_v005.pt
uv run python -m tracker_likelihood.train --patch-width 96 --patch-height 36 --out ..\models\tracker_likelihood\tracker_likelihood_precise_v005.pt
```

Train/export LED reader:

```powershell
cd receiver/training
uv run python -m led_reader.train --out ..\models\led_reader\led_reader_crop_v003_gate.pt
uv run python -m led_reader.export_onnx --checkpoint ..\models\led_reader\led_reader_crop_v003_gate.pt --out ..\models\led_reader\led_reader_crop_v003_gate.onnx
```

Render tracker overlays:

```powershell
cd receiver/tools
uv run python -m run_tracker_dataset.run_tracker_dataset --backend neural --acquire-model ..\models\tracker_likelihood\tracker_likelihood_acquire_v005.onnx --precise-model ..\models\tracker_likelihood\tracker_likelihood_precise_v005.onnx
```

Render LED reader overlays:

```powershell
cd receiver/tools
uv run python -m evaluate.eval_led_reader --tracker-model ..\models\tracker_likelihood\tracker_likelihood_fast_v004.onnx --model ..\models\led_reader\led_reader_crop_v003_gate.onnx --out ..\datasets\derived\overlays\led_reader_v003_gate
```

Copy models into Android:

```powershell
Copy-Item receiver\models\tracker_likelihood\tracker_likelihood_acquire_v005.onnx android-reader\app\src\main\assets\tracker_acquire.onnx -Force
Copy-Item receiver\models\tracker_likelihood\tracker_likelihood_precise_v005.onnx android-reader\app\src\main\assets\tracker_precise.onnx -Force
Copy-Item receiver\models\led_reader\led_reader_crop_v003_gate.onnx android-reader\app\src\main\assets\led_reader.onnx -Force
```

Build Android:

```powershell
cd android-reader
.\gradlew.bat :app:compileDebugKotlin
```

## Room For Improvement

- LED gate: current gate is trained mostly on high-confidence weak labels, so it mainly scales confidence down but does not improve hard LED accuracy. Better training needs more low-confidence tracked examples and explicit downstream BP loss.
- Tracker threshold calibration: thresholds are currently chosen from experiment behavior. A proper calibration run should choose acquire/track thresholds from good/bad score distributions.
- Android/Python parity: the Android extractor mirrors Python feature extraction, but a parity test over recorded frames would catch channel normalization drift.
- Quantization: the ONNX models are small, but int8/NNAPI/TFLite experiments may lower mobile CPU cost.
- Neural BP: future work can train learned BP updates or damping schedules from packet/factor traces. Keep this separate from the current vision work until the classical BP baseline is stable.
