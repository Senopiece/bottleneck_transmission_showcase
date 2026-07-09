# Datasets

Dataset area for raw recordings and generated artifacts.

Raw videos live under `raw/videos/good` and `raw/videos/bad`.
Generated artifacts live under `derived`:

- `overlays`: rendered videos with tracker/reader predictions.
- `metrics`: CSV/JSON reports from dataset runs and evaluations.

Intermediate crop caches and pseudo-label dumps are intentionally not kept. They
are regenerated directly from raw videos by the training/evaluation scripts.
