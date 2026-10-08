# Run Tracker Dataset

Runs the acquire/track loop over all raw videos, then writes:

- per-video CSV metrics;
- overlay videos;
- a summary CSV.

Use `neural` to mirror Android's acquire, precise verification, velocity seed,
and tracking ascent without classical CV fallback.

Example:

```powershell
cd receiver/tools
uv run python -m run_tracker_dataset.run_tracker_dataset --backend neural --acquire-model ..\models\tracker_likelihood\tracker_likelihood_acquire_v005.onnx --precise-model ..\models\tracker_likelihood\tracker_likelihood_precise_v005.onnx --overlay-out ..\datasets\derived\overlays\neural_v005 --metrics-out ..\datasets\derived\metrics\neural_v005
```

Use `--include good5 bad3` for focused iteration.
