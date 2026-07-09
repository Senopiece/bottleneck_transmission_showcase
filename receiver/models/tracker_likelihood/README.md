# Tracker Likelihood Models

Current model:

- `tracker_likelihood_fast_v003.pt`: PyTorch checkpoint.
- `tracker_likelihood_fast_v003.onnx`: Android/Python inference export.
- `tracker_likelihood_fast_v003.json`: validation metrics.

The model scores canonical marker patches. Pose correction is performed by the
external fixed-budget ascent loop, not by the network output.
