#!/usr/bin/env bash
# Train Qwen3.5 -> V-RAE latent future prediction.
#
#   sh examples/vrae_future_pred/sft.sh
#
# Prerequisites:
#   1. Latent cache built WITH context clips:
#        python /data1/qirui/V-RAE/scripts/build_vpdata_latents.py --context-clip
#   2. jsonl splits:
#        python examples/vrae_future_pred/dataset.py --cache-root "$VPDATA_LATENT_ROOT" --build
#   3. Latent statistics:
#        python examples/vrae_future_pred/latent_stats.py \
#            --cache-root "$VPDATA_LATENT_ROOT" --out "$LATENT_STATS_PATH"
set -euo pipefail

SWIFT_ROOT=${SWIFT_ROOT:-/data1/qirui/ms-swift}
PY=${PY:-/home/node-user/anaconda3/envs/vrae_swift/bin/python}
# sdpa because flash_attn is not installed in this env; set ATTN_IMPL=flash_attn after
# installing a wheel built for torch 2.12 / cu126 if the speedup is worth it.
ATTN_IMPL=${ATTN_IMPL:-sdpa}

export PYTHONPATH=${SWIFT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export VPDATA_LATENT_ROOT=${VPDATA_LATENT_ROOT:-/data1/qirui/Datasets/vpdata_future_latents/trunk0}
export LATENT_STATS_PATH=${LATENT_STATS_PATH:-${SWIFT_ROOT}/ckpts/latent_stats/vjepa_vpdata_40future.pt}

# Latent geometry; must match the cache. 40 target frames / 4 = 10 chunks,
# 256x448 / patch 16 = 16x28 spatial grid, D_z = V-JEPA hidden size.
#
# UPPERCASE is load-bearing: get_env_args (swift/utils/utils.py:268) reads only
# `os.getenv(name.upper())`, so the lowercase spellings this script used to export were
# silently ignored -- and the failure was invisible because the hardcoded defaults in
# Qwen35VRAELoader happened to equal them. Exporting `latent_chunks=20` changed nothing.
# The lowercase forms are still accepted as input here, for convenience.
export LATENT_CHUNKS=${LATENT_CHUNKS:-${latent_chunks:-10}}
export LATENT_HEIGHT=${LATENT_HEIGHT:-${latent_height:-16}}
export LATENT_WIDTH=${LATENT_WIDTH:-${latent_width:-28}}
export LATENT_DIM=${LATENT_DIM:-${latent_dim:-1024}}

# Predictor shape. RESAMPLER_SELF_ATTN=0 is the weak-resampler ablation: queries stop
# attending to each other, so joint reasoning over future positions can only happen
# inside Qwen. Pair it with a larger LoRA rank to move capacity there deliberately.
export RESAMPLER_DIM=${RESAMPLER_DIM:-${resampler_dim:-1024}}
export RESAMPLER_DEPTH=${RESAMPLER_DEPTH:-${resampler_depth:-4}}
export RESAMPLER_HEADS=${RESAMPLER_HEADS:-${resampler_heads:-16}}
export RESAMPLER_SELF_ATTN=${RESAMPLER_SELF_ATTN:-${resampler_self_attn:-1}}

# Loss weights from section 8 of the design doc.
export LAMBDA_MSE=${LAMBDA_MSE:-${lambda_mse:-0.1}}
export LAMBDA_TEMPORAL=${LAMBDA_TEMPORAL:-${lambda_temporal:-0.1}}

# Frame count and resolution are pinned per sample through chat_template_kwargs in
# dataset.py (nframes / resized_height / resized_width), which is what
# qwen_vl_utils.fetch_video actually reads. Do not try to set them with env vars here --
# NFRAMES and friends belong to other templates and silently do nothing on this path.

# ---- Multi-GPU ------------------------------------------------------------------
# swift/cli/main.py re-execs the whole command under `torch.distributed.run` as soon as
# NPROC_PER_NODE is set (use_torchrun / get_torchrun_args), and get_default_device_map()
# in swift/model/utils.py then places each rank on cuda:LOCAL_RANK. Nothing in model.py
# or template.py has to know about ranks.
#
# Plain DDP is the right level here: the 9B backbone is 18 GB in bf16 and only LoRA plus
# the resampler and latent head train (~159 M params, ~1.3 GB of Adam state), so there is
# nothing for ZeRO to shard. Add --deepspeed zero2 only if a larger batch OOMs.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_GPUS=$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")
export NPROC_PER_NODE=${NPROC_PER_NODE:-${NUM_GPUS}}
# torchrun binds this; concurrent runs on one node need different ports.
export MASTER_PORT=${MASTER_PORT:-29511}

# Hold the effective batch (per_device x accumulation x ranks) fixed as the GPU count
# changes, so the learning rate stays comparable between a 1-GPU debug run and an 8-GPU
# one. Without this, going 1 -> 8 GPUs at gradient_accumulation_steps 16 silently turns
# a batch of 16 into 128.
GLOBAL_BATCH=${GLOBAL_BATCH:-16}
# PER_DEVICE has to be in the divisor: without it, raising the per-device batch silently
# multiplies the effective batch instead of trading accumulation for parallelism, and the
# learning rate is then calibrated for a batch you did not ask for.
PER_DEVICE=${PER_DEVICE:-1}
GRAD_ACCUM=$((GLOBAL_BATCH / (NPROC_PER_NODE * PER_DEVICE)))
[ "${GRAD_ACCUM}" -lt 1 ] && GRAD_ACCUM=1
echo "ranks ${NPROC_PER_NODE} x per_device ${PER_DEVICE} x accum ${GRAD_ACCUM} = batch $((NPROC_PER_NODE * PER_DEVICE * GRAD_ACCUM))"

cd "${SWIFT_ROOT}"

"${PY}" -m swift.cli.main sft \
    --external_plugins examples/vrae_future_pred/dataset.py \
                       examples/vrae_future_pred/template.py \
                       examples/vrae_future_pred/model.py \
    --model /GPFS/ComfyUI_models/LLM/Qwen3.5-9B \
    --model_type qwen3_5_vrae \
    --template qwen3_5_vrae \
    --dataset vpdata_future_pred \
    --remove_unused_columns false \
    --tuner_type lora \
    --target_modules all-linear \
    --lora_rank 32 \
    --lora_alpha 64 \
    --freeze_vit true \
    --modules_to_save resampler latent_head \
    --torch_dtype bfloat16 \
    --attn_impl "${ATTN_IMPL:-sdpa}" \
    --per_device_train_batch_size "${PER_DEVICE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --gradient_checkpointing true \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --warmup_ratio 0.05 \
    --split_dataset_ratio 0 \
    --logging_steps 5 \
    --save_steps 200 \
    --save_total_limit 3 \
    --dataloader_num_workers "${DATALOADER_WORKERS:-4}" \
    --max_length 8192 \
    --output_dir output/vrae_future_pred \
    "$@"

# Notes on flags that are load-bearing rather than cosmetic:
#
# --remove_unused_columns false
#     DatasetLoader._load_dataset_path ends with remove_useless_columns(), which keeps
#     only RowPreprocessor.standard_keys. Without this, target_latent_path is dropped
#     and the model silently trains on nothing. (This is the dataset-loading switch,
#     not the HF TrainingArguments one that sft_args.py already disables.)
#
# no --loss_type
#     The model returns its own loss. Setting --loss_type makes the trainer pop
#     `labels` and take the custom-loss branch, which recomputes cross entropy.
#
# --split_dataset_ratio 0
#     The val split is a separate jsonl split by source video; letting swift carve a
#     random slice off the train split instead would put clips of one video on both
#     sides. To evaluate, pass the *registered* val name --
#     `--val_dataset vpdata_future_pred_val --eval_strategy steps --eval_steps 200` --
#     and not the jsonl path: a bare path routes through AutoPreprocessor, which drops
#     target_latent_path, and eval then scores nothing.
#
# --modules_to_save resampler latent_head   (NOT --trainable_parameters)
#     These two modules are new and must train in full alongside LoRA. Two reasons the
#     obvious flag does not work:
#       * swift/pipelines/train/tuner.py calls activate_parameters() only in the
#         `tuner_type == 'full'` branch, so --trainable_parameters is ignored outright
#         under LoRA;
#       * activate_parameters matches with name.startswith(...), and after PEFT wrapping
#         the parameters are named base_model.model.resampler.*, so the bare prefix
#         `resampler` would not match even if it were reached.
#     Getting this wrong is silent: training runs, LoRA learns, and the resampler stays
#     at its random init -- the loss then plateaus at the value for predicting zero
#     (~1.20 here) instead of approaching 0. Check for `resampler` keys in
#     checkpoint-*/adapter_model.safetensors to confirm it is actually being trained.
#
# Multi-GPU notes (checked against this repo, not assumed):
#
# gradient sync
#     Qwen35VRAETemplate.compute_sft_loss calls `model(**inputs)`, and the `model` it
#     receives from Seq2SeqTrainer.compute_loss is the DDP-wrapped module, so the
#     backward does reduce across ranks. Calling an unwrapped module there would train
#     8 independent copies with no error.
#
# no --ddp_find_unused_parameters needed
#     Every trainable parameter is on the loss path: LoRA only attaches under
#     `qwen.model.language_model` (get_multimodal_target_regex with freeze_vit true, and
#     find_all_linears skips lm_head), and hidden_states[-1] depends on all 32 layers.
#     Qwen3.5-9B is dense, so there are no unrouted experts either. If a future change
#     does leave a trainable parameter off the path, DDP fails at step 1 with "Expected
#     to have finished reduction in the prior iteration"; pass
#     --ddp_find_unused_parameters true then, rather than pre-emptively (it costs a full
#     graph traversal per step).
#
# keep --freeze_vit true
#     Template.pre_forward_hook runs _post_encode (which encodes the video through the
#     vision tower) as a forward pre-hook, so it executes inside the DDP forward and
#     would sync -- but only as long as the ViT is frozen is that irrelevant. Unfreezing
#     it under DDP is a change to validate, not to assume.
#
# use_reentrant
#     swift/trainers/mixin.py::_fix_gradient_checkpointing already forces
#     use_reentrant=False whenever is_dist() and neither deepspeed nor FSDP is on, which
#     is what DDP + gradient checkpointing needs. Nothing to pass.
