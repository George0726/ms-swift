# Variable-duration direct prediction

Supported: direct + one_shot. Each chunk keeps its original temporal duration and
spatial geometry. LATENT_CHUNKS is the maximum learned horizon, not a per-call constant.
No parameter shapes change; existing checkpoints remain loadable with the same maximum.
This does not provide extrapolation beyond the trained maximum or video decoding.

## Inference

```python
model.eval()
out = model(inputs_embeds=context_embeds, attention_mask=context_mask,
            num_future_chunks=4)
# out.z_pred: [B, 4, H*W, D]
```

K must be a Python integer between 1 and LATENT_CHUNKS. The model creates exactly
K*H*W future queries and calls Qwen once. Omitting K uses the target's length when
provided, otherwise LATENT_CHUNKS. With targets, an explicit K selects their first K
chunks and cannot exceed their length. This changes future duration, not input-video
sampling. Convert seconds into chunks using the actual latent cache temporal stride
and sampling rate; do not assume a fixed number of seconds from the chunk index alone.

## Training

Use the same paths, GPUs and optimizer arguments as the existing direct run, replacing
`sft.sh` with `sft_direct_variable.sh`. This sets direct + one_shot and exports
DIRECT_TRAIN_MIN_CHUNKS=1 by default. During training with targets and no explicit K,
each forward samples K uniformly from [DIRECT_TRAIN_MIN_CHUNKS, target_length].
All samples in that forward share K. Loss uses only those target chunks. Temporal
loss is zero for K=1. Evaluation never randomly samples K.

Set DIRECT_TRAIN_MIN_CHUNKS=0 to disable random sampling. With the original sft.sh,
the default remains zero, preserving full-horizon training. The setting is saved in
vrae_predictor config and can be overridden through the loader environment setting.
Logs include num_future_chunks. Different horizons have different difficulty; use
fixed full-horizon evaluation for comparisons. Uniform K sampling supervises late
horizons less frequently than early ones, so compare against full-horizon training.

## Dataset / batching

A JSONL row may optionally contain `num_future_chunks: 4`; the template carries this
through encode, collate and the model hook. All rows within a batch must request the
same K and have the same target tensor shape. Use length-grouped batches or batch
size 1 for variable-sized cached targets. Mixed-length padded target batches are not
implemented and raise an error rather than training against padding. Existing full
length caches need no changes for random-prefix training.

The runtime argument is deliberately rejected for other predictor/prediction modes;
resampler, slice, interleave and paired-residual behavior stays unchanged.

## Checks

CPU stub-backbone tests cover K=1/2/3, exact query count and one backbone call, prefix
losses, target-free inference, invalid lengths, random training / full-length eval,
used/unused time-embedding gradients, and template parameter propagation. Real Qwen
GPU inference, memory consumption and accuracy have not been tested locally.
