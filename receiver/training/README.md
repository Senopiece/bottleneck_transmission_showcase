# Training

PyTorch training code for the Android receiver models.

This directory is its own `uv` project. Run all training/export commands from
here so they use the pinned environment and editable `../tools` package:

```powershell
cd receiver/training
uv sync
```

Current active modules:

- `tracker_likelihood`: small CNN that scores a canonical marker pose hypothesis.
- `led_reader`: shared one-LED CNN that emits gated LED logits.
- `neural_bp`: placeholder for future learned BP work.

Train/export tracker:

```powershell
uv run python -m tracker_likelihood.train --out ..\models\tracker_likelihood\tracker_likelihood_fast_v003.pt
uv run python -m tracker_likelihood.export_onnx --checkpoint ..\models\tracker_likelihood\tracker_likelihood_fast_v003.pt --out ..\models\tracker_likelihood\tracker_likelihood_fast_v003.onnx
```

Train/export LED reader:

```powershell
uv run python -m led_reader.train --out ..\models\led_reader\led_reader_crop_v003_gate.pt
uv run python -m led_reader.export_onnx --checkpoint ..\models\led_reader\led_reader_crop_v003_gate.pt --out ..\models\led_reader\led_reader_crop_v003_gate.onnx
```

Use each module's `runs/` directory for local experiment outputs; those are
gitignored.
