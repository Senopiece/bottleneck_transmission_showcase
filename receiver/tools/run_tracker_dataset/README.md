# Run Tracker Dataset

Runs the acquire/track loop over all raw videos, then writes:

- per-video CSV metrics;
- overlay videos;
- a summary CSV.

The active backend is `cv`: a Python/OpenCV harness that uses the exported ONNX
tracker likelihood model and the same fixed-pose marker geometry as Android.

Example:

```powershell
cd receiver/tools
uv run python -m run_tracker_dataset.run_tracker_dataset --backend cv --tracker-model ..\models\tracker_likelihood\tracker_likelihood_fast_v003.onnx --overlay-out ..\datasets\derived\overlays\tracker_v003 --metrics-out ..\datasets\derived\metrics\tracker_v003
```

Use `--include good5 bad3` for focused iteration.
