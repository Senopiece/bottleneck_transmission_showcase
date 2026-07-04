# Receiver

Shared receiver workspace for the optical channel.

It is split into:

- `core/`: future portable C++ vision and codec cores.
- `datasets/`: raw videos/logs/expected messages and generated labels/traces/metrics.
- `models/`: exported inference models.
- `training/`: tracker, LED reader, and future neural BP training code.
- `tools/`: Python/uv tooling for dataset runs, overlays, trace export, and evaluation.

Python is split into two local `uv` projects:

- `tools/` owns shared dataset/render/eval code as the `receiver-tools` package.
- `training/` owns training entry points and depends on `receiver-tools` via an
  editable local path.
