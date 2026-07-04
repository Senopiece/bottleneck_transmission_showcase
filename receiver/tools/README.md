# Receiver Tools

Python tooling managed by `uv`.

Main commands:

```powershell
cd receiver/tools
uv sync
uv run python run_tracker_dataset/run_tracker_dataset.py
```

The current tracker backend is a stub. It is intentionally shaped so the future C++ acquire/track loop can be called from the same runner without changing dataset/overlay code.
