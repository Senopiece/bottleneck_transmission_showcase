# Tracker Likelihood Training

Tracker training is intentionally separate from LED reading. It does not use the
known transmitted message or LED sequence as supervision.

The model is a tiny pose-hypothesis scorer:

- input: canonical pose patch, shape `2 x 36 x 96`;
- channel 0: locally normalized luma;
- channel 1: normalized edge magnitude;
- output: one likelihood logit for `(pos_x, pos_y, angle, scale)`.

Runtime design:

1. The acquirer/tracker starts from a pose hypothesis.
2. The frame is warped into the canonical patch for that hypothesis.
3. The CNN scores the patch.
4. A fixed-budget coordinate ascent searches nearby `(x, y, angle, scale)`.

This avoids full-frame neural inference and keeps mobile cost predictable.

## Train

```powershell
cd receiver/training
uv sync
uv run python -m tracker_likelihood.train
```

The current checkpoint is written to:

```text
receiver/models/tracker_likelihood/tracker_likelihood_fast_v003.pt
```

## Export

```powershell
cd receiver/training
uv run python -m tracker_likelihood.export_onnx
```

The current ONNX file is written to:

```text
receiver/models/tracker_likelihood/tracker_likelihood_fast_v003.onnx
```

## Data Sources

Positive tracker samples:

- synthetic canonical marker patches with controlled pose jitter;
- pseudo-positive patches found in `datasets/raw/videos/good`.

Negative tracker samples:

- random crops from all videos;
- hard false-positive candidates mined from `datasets/raw/videos/bad`.

This is still weak supervision, not final manual labeling. The output overlays
must be inspected. Bad pseudo-labels should be fixed by improving the mining
rules or adding manual labels around failure cases.

## Android Port Contract

Android must reproduce the same canonical patch:

- local marker patch width: `1.35` marker-line units;
- local marker patch height: `0.46` marker-line units;
- output tensor: `NCHW`, `1 x 2 x 36 x 96`;
- luma normalization: per-patch mean/std, clipped to `[-3, 3] / 3`;
- edge channel: finite-difference gradient magnitude normalized by p95.
