# Training

Training code for tracker likelihood, LED reader, and future neural BP.

This directory is its own `uv` project. Run training from here so the scripts
use the pinned local environment and the editable `../tools` package:

```powershell
cd receiver/training
uv sync
uv run python run_bootstrap_alternating.py
```

Use `runs/` for local experiment outputs; those are gitignored.
