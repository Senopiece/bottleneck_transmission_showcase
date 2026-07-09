# LED Reader Models

Current model:

- `led_reader_crop_v003_gate.pt`: PyTorch checkpoint.
- `led_reader_crop_v003_gate.onnx`: Android/Python inference export.
- `led_reader_crop_v003_gate.json`: validation metrics.

The model is shared per LED crop. It emits one gated logit per crop. Android
maps the logits to the existing packet score scale before clock sampling and BP.
