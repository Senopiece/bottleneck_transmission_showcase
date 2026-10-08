# Tracker Likelihood Training

Tracker training is intentionally separate from LED reading. It does not use the
known transmitted message or LED sequence as supervision.

The model is a tiny pose-hypothesis scorer:

- acquire input: canonical pose patch, shape `2 x 24 x 64`;
- precise input: canonical pose patch, shape `2 x 36 x 96`;
- channel 0: locally normalized luma;
- channel 1: normalized edge magnitude;
- output: broad acquire and sharp tracking likelihood logits for
  `(pos_x, pos_y, angle, scale)`.

Runtime design:

1. The acquirer/tracker starts from a pose hypothesis.
2. The frame is warped into the canonical patch for that hypothesis.
3. The CNN scores the patch.
4. A fixed-budget coordinate ascent searches nearby `(x, y, angle, scale)`.

The spatial head keeps the final feature grid instead of global-average pooling.
This makes the score sensitive to exact square/LED/triangle placement. Android
uses the small model for acquisition and the larger model for final verification
and tracking.

## Train

```powershell
cd receiver/training
uv sync
uv run python -m tracker_likelihood.train --patch-width 64 --patch-height 24 --out ..\models\tracker_likelihood\tracker_likelihood_acquire_v005.pt
uv run python -m tracker_likelihood.train --patch-width 96 --patch-height 36 --out ..\models\tracker_likelihood\tracker_likelihood_precise_v005.pt
```

Current checkpoints:

```text
receiver/models/tracker_likelihood/tracker_likelihood_acquire_v005.pt
receiver/models/tracker_likelihood/tracker_likelihood_precise_v005.pt
```

## Export

```powershell
cd receiver/training
uv run python -m tracker_likelihood.export_onnx --checkpoint ..\models\tracker_likelihood\tracker_likelihood_acquire_v005.pt --out ..\models\tracker_likelihood\tracker_likelihood_acquire_v005.onnx
uv run python -m tracker_likelihood.export_onnx --checkpoint ..\models\tracker_likelihood\tracker_likelihood_precise_v005.pt --out ..\models\tracker_likelihood\tracker_likelihood_precise_v005.onnx
```

Current ONNX files:

```text
receiver/models/tracker_likelihood/tracker_likelihood_acquire_v005.onnx
receiver/models/tracker_likelihood/tracker_likelihood_precise_v005.onnx
```

## Data Sources

Positive tracker samples:

- synthetic canonical marker patches with controlled pose jitter;
- pseudo-positive patches found in `datasets/raw/videos/good`.

Negative tracker samples:

- random crops from all videos;
- pose-shaped local maxima mined by coordinate ascent from
  `datasets/raw/videos/bad`, including `bad6` text/icon material;
- wrong poses over real patterns, including LED-as-endpoint decoys.

This is still weak supervision, not final manual labeling. The output overlays
must be inspected. Bad pseudo-labels should be fixed by improving the mining
rules or adding manual labels around failure cases.

## Android Port Contract

Android must reproduce the same canonical patch:

- local marker patch width: `1.35` marker-line units;
- local marker patch height: `0.46` marker-line units;
- output tensor: `NCHW`, either `N x 2 x 24 x 64` or
  `N x 2 x 36 x 96`;
- luma normalization: per-patch mean/std, clipped to `[-3, 3] / 3`;
- edge channel: finite-difference gradient magnitude normalized by p95.
