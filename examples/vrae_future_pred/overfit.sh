#!/usr/bin/env bash
# The convergence gate before any real run: memorize a handful of samples.
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash examples/vrae_future_pred/overfit.sh
#
# What it proves, and why it is worth the GPU hours: a smoke test only shows that
# gradients flow, which is also true when the resampler is frozen at its random init and
# LoRA alone is slowly learning to predict the mean. On a handful of samples with
# regularization off, a correctly wired model has more than enough capacity to drive the
# loss to ~0. A plateau well above 0 is the model, not the data -- and finding that out
# on 8 samples costs minutes instead of a full training run.
#
# Reading the result (the terms are logged separately, see template.py):
#
#   loss ~ 1.19, loss_cos ~ 1.0    predicting a zero vector: nothing is learning.
#                                  Check that resampler/latent_head are in
#                                  adapter_model.safetensors (verify.py
#                                  --check-checkpoint) -- this is what
#                                  --trainable_parameters instead of --modules_to_save
#                                  looks like.
#   loss_cos -> 0, loss_mse high   direction learned, magnitude not. Suspect the latent
#                                  statistics: refit LATENT_STATS_PATH on this cache's
#                                  train split, not on a smoke cache.
#   loss < 0.1                     gate passed, go to the real run.
#   plateau at 0.3-0.6             capacity or optimization. Try resampler_depth 6-8 /
#                                  resampler_dim 1280, or a higher LR, before blaming
#                                  the data.
#
# PREDICTION_MODE=autoregressive reads differently at the start: the residual rollout is
# initialised as the identity, so it opens at the *persistence* loss (~0.57 on trunk0
# train samples), not at ~1.19. "Nothing is learning" therefore looks like a plateau at
# ~0.57 here, and 1.19 would instead mean the residual path is not wired up at all.
# The gate itself is unchanged: 8 samples with regularization off must still reach < 0.1.
set -euo pipefail

SWIFT_ROOT=${SWIFT_ROOT:-/data1/qirui/ms-swift}
PY=${PY:-/home/node-user/anaconda3/envs/vrae_swift/bin/python}
# The cache whose train split the overfit samples are drawn from.
SOURCE_ROOT=${SOURCE_ROOT:-/data1/qirui/Datasets/vpdata_future_latents/trunk0}
NUM_SAMPLES=${NUM_SAMPLES:-8}
OVERFIT_ROOT=${OVERFIT_ROOT:-${SOURCE_ROOT}/_overfit${NUM_SAMPLES}}

SOURCE_JSONL="${SOURCE_ROOT}/swift_jsonl/train.jsonl"
if [ ! -f "${SOURCE_JSONL}" ]; then
    echo "missing ${SOURCE_JSONL}; build it first:" >&2
    echo "  PYTHONPATH=${SWIFT_ROOT} ${PY} examples/vrae_future_pred/dataset.py \\" >&2
    echo "      --cache-root ${SOURCE_ROOT} --build" >&2
    exit 1
fi

# The jsonl rows carry absolute paths to the clip and the target latent, so an overfit
# "cache root" needs nothing but the jsonl -- no copies of the 9 MB latents. dataset.py
# registers <root>/swift_jsonl/{train,val}.jsonl, which is all VPDATA_LATENT_ROOT has to
# provide. No val.jsonl on purpose: there is nothing to generalize to here.
mkdir -p "${OVERFIT_ROOT}/swift_jsonl"
head -n "${NUM_SAMPLES}" "${SOURCE_JSONL}" > "${OVERFIT_ROOT}/swift_jsonl/train.jsonl"
rm -f "${OVERFIT_ROOT}/swift_jsonl/val.jsonl"
echo "overfitting on $(wc -l < "${OVERFIT_ROOT}/swift_jsonl/train.jsonl") samples from ${SOURCE_JSONL}"

# Keep one sample per rank per step: with fewer rows than ranks the sampler pads by
# repeating, and with max_steps beyond what the sampler yields the trainer stops at step 0
# without failing (exit code 0, train_loss 0.0 -- easy to mistake for a finished run).
export GLOBAL_BATCH=${GLOBAL_BATCH:-${NUM_SAMPLES}}

# Regularization off, schedule flat, LR up. Every one of these defaults is correct for a
# real run and wrong for this test:
#   --lora_dropout 0        the default 0.05 actively fights memorization
#   --weight_decay 0        the default 0.1 pulls the fresh resampler back toward zero
#   --lr_scheduler_type constant + --warmup_ratio 0
#                           cosine decay to zero makes the tail of any run look like a
#                           plateau, which is the exact signal this test reads
#   --learning_rate 3e-4    the resampler is training from scratch, not adapting
#   --split_dataset_ratio 0 do not carve a val split out of 8 samples
VPDATA_LATENT_ROOT="${OVERFIT_ROOT}" \
LATENT_STATS_PATH=${LATENT_STATS_PATH:-${SWIFT_ROOT}/ckpts/latent_stats/vjepa_vpdata_40future.pt} \
    bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sft.sh" \
    --max_steps "${MAX_STEPS:-1500}" \
    --learning_rate "${LR:-3e-4}" \
    --lr_scheduler_type constant \
    --warmup_ratio 0 \
    --weight_decay 0 \
    --lora_dropout 0 \
    --split_dataset_ratio 0 \
    --logging_steps 25 \
    --save_steps "${MAX_STEPS:-1500}" \
    --save_total_limit 1 \
    --output_dir "${OUTPUT_DIR:-output/_overfit${NUM_SAMPLES}}" \
    "$@"
