# Neural BP Training

Placeholder for future neural BP work.

This is intentionally not part of the current Android path.

Potential future direction:

1. Export packet LLR traces, factor graph structure, BP messages, and final decode outcome.
2. Train learned damping/update rules or a small neural residual over classical BP.
3. Keep the graph/fountain code deterministic and use the neural part only as an
   update policy, not as an opaque message decoder.

Expected inputs are not raw videos. They should be traces produced after vision:

- symbol timestamps;
- 5-bit packet LLRs;
- erasure markers;
- LDGM/fountain row ids;
- BP posterior snapshots;
- success/failure labels.

Main risks:

- self-reinforcement from training on the decoder's own wrong posterior;
- memory growth if traces include full per-edge history;
- mismatch between desktop traces and Android online fixed-budget BP.

This should only be revisited after the classical BP + neural vision baseline is
stable and well logged.
