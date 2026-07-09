# LED Reader Training

Shared one-LED crop CNN.

The current model has two inputs:

- `led_crop`: canonical one-LED crop.
- `detector_likelihood`: tracker/pattern likelihood for the pose that produced the crop.

The CNN predicts a raw LED logit. A small learnable evidence gate maps
`detector_likelihood` to a positive multiplier and scales the raw logit magnitude.
This does not change the ON/OFF sign by itself; it suppresses weak tracker evidence
before the soft decoder sees it.

Pipeline:

1. Run the current tracker on good videos.
2. Estimate stream start by matching tracked LED score time series against the known preamble + fountain stream.
3. Use only central symbol-window frames as weak labels.
4. Crop each expected LED from the canonical marker patch.
5. Train one shared CNN that emits one gated logit per LED crop.

The model intentionally has no LED index embedding. All five LEDs use the same scorer.

Current gated model:

```powershell
cd receiver/training
uv run python -m led_reader.train --include good1 good2 good3 good4 good5 good6 good7 good8 good10 good11 good12 good13 --max-frames 180 --stride 5 --synthetic-samples 1800 --bad-negative-samples 120 --epochs 8 --batch-size 256 --out ..\models\led_reader\led_reader_crop_v003_gate.pt
uv run python -m led_reader.export_onnx --checkpoint ..\models\led_reader\led_reader_crop_v003_gate.pt --out ..\models\led_reader\led_reader_crop_v003_gate.onnx
```

Observed v003 gated validation:

- accuracy: `0.9841`
- raw accuracy before gate: `0.9841`
- balanced accuracy after threshold sweep: `0.9886`
- gate mean / p05: `0.84 / 0.84`
- margin p05 after gate: `1.70`
- raw margin p05 before gate: `2.02`

Interpretation: on this mostly high-confidence weak-label dataset, the gate does not
improve hard per-frame LED accuracy. It intentionally lowers confidence magnitude,
which should help downstream soft decoding when tracking is blurry or weak.

Render predictions:

```powershell
cd receiver/tools
uv run python -m evaluate.eval_led_reader --tracker-model ..\models\tracker_likelihood\tracker_likelihood_fast_v003.onnx --model ..\models\led_reader\led_reader_crop_v003_gate.onnx --out ..\datasets\derived\overlays\led_reader_v003_gate --include good5 --max-frames 180 --stride 3
```

## Android Port Contract

Android crops each LED from the accepted marker pose and feeds:

- `led_crop`: `N x 3 x 28 x 28`, where `N = 5`;
- channel 0: per-crop normalized luma;
- channel 1: blue dominance, clipped to `[-1, 1]`;
- channel 2: normalized edge magnitude;
- `detector_likelihood`: `N`, the tracker likelihood copied for each LED.

The ONNX output is a gated logit. Android maps it back to the legacy score scale
used by `PacketClockDecoder`:

```text
score = 0.54 + sigmoid(logit) * 0.64
```

This keeps the existing preamble, weighted sampling, and BP decoder unchanged.
