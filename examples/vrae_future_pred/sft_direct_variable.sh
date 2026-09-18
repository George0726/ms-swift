#!/usr/bin/env bash
# LATENT_CHUNKS remains the learned maximum; each training forward samples a prefix.
set -euo pipefail
PLUGIN_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export FUTURE_PREDICTOR_MODE=direct
export PREDICTION_MODE=one_shot
export DIRECT_TRAIN_MIN_CHUNKS=${DIRECT_TRAIN_MIN_CHUNKS:-1}
echo "Variable direct: train min=${DIRECT_TRAIN_MIN_CHUNKS}; maximum=LATENT_CHUNKS; eval uses target length"
exec bash "${PLUGIN_DIR}/sft.sh" "$@"
