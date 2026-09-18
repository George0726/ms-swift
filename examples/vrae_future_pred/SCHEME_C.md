# Scheme C: interleave predicted-latent training

Run `bash examples/vrae_future_pred/sft_c.sh` with the same dataset, latent statistics,
GPU and optimizer arguments as your existing training command. The existing `sft.sh`
entry point remains unchanged for direct / one-shot / teacher-forcing comparisons.

This entry point explicitly overrides inherited settings with:

- `FUTURE_PREDICTOR_MODE=interleave`
- `PREDICTION_MODE=autoregressive`
- `INTERLEAVE_FEEDBACK=latent`
- `ROLLOUT_WEIGHT=1.0`
- `RESIDUAL_FEEDBACK=1`

Data flow: video/text context + last observed latent -> Qwen -> predicted delta ->
predicted latent -> pool/project -> next Qwen call. Ground-truth future latents only
enter the loss. All horizons are supervised. Existing model.py implements this path;
no architecture or checkpoint tensor changes are needed.

`ROLLOUT_STEPS` defaults to `LATENT_CHUNKS` (10). `BPTT_DEPTH` defaults to the rollout
length, so later losses backpropagate through earlier predictions. A smaller BPTT_DEPTH
truncates gradients, but still feeds predicted latents. It is not full cross-step BPTT.
The current implementation recomputes the growing sequence without a KV cache.
Full 10-step training can be substantially more expensive and may exceed GPU memory.

Start with a short run, per-device batch 1, for example:

```bash
PY=/path/to/env/bin/python \
SWIFT_ROOT=/path/to/ms-swift \
VPDATA_LATENT_ROOT=/path/to/cache \
LATENT_STATS_PATH=/path/to/stats.pt \
CUDA_VISIBLE_DEVICES=0 GLOBAL_BATCH=1 PER_DEVICE=1 \
ROLLOUT_STEPS=3 BPTT_DEPTH=3 \
bash examples/vrae_future_pred/sft_c.sh \
  --max_steps 2 --output_dir output/interleave_c_smoke
```

For the full horizon, use `ROLLOUT_STEPS=10 BPTT_DEPTH=10` only after checking memory.
A three-horizon loss cannot be compared directly with a ten-horizon loss.
Training logs must show `rollout_weight=1` and `loss_rollout`; this is the decisive
check, not the experiment name. Evaluation already uses predicted feedback.

For mixed training, use the original entry point with
`FUTURE_PREDICTOR_MODE=interleave PREDICTION_MODE=autoregressive ROLLOUT_WEIGHT=0.2`.
This computes two branches and weights their losses; it is not scheduled sampling.
For teacher forcing use weight 0. The C entry point intentionally forces weight 1.

CPU regression tests use a small stub backbone and the real predictor/autograd. They
check sequential feedback, target independence, and final-step gradient propagation
to the first predicted delta, with truncated BPTT as a negative control. They do not
establish real Qwen GPU memory requirements or improved dataset convergence.
