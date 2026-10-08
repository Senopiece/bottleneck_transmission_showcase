# Tracker Likelihood Models

Current models:

- `tracker_likelihood_acquire_v005.*`: `64 x 24` broad acquisition scorer.
- `tracker_likelihood_precise_v005.*`: `96 x 36` precise verifier/tracker.

Both use ReLU depthwise-separable backbones and spatial likelihood heads. V4 is
retained as an experiment baseline.

The model scores canonical marker patches. Pose correction is performed by the
external fixed-budget ascent loop, not by the network output.
