# Receiver Tools

Python tooling managed by `uv`.

Main commands:

```powershell
cd receiver/tools
uv sync
uv run python -m run_tracker_dataset.run_tracker_dataset --backend cv
```

Useful tools:

- `run_tracker_dataset`: render tracker overlays and metrics over raw videos.
- `evaluate.eval_led_reader`: render LED reader predictions over tracked videos.
- `evaluate.eval_tracker`: tracker metric experiments.
- `export_traces`: future trace export hooks for neural BP work.
