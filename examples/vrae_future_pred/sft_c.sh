#!/usr/bin/env bash
# Scheme C: predicted latent feedback throughout training.
set -euo pipefail
PLUGIN_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export FUTURE_PREDICTOR_MODE=interleave
export PREDICTION_MODE=autoregressive
export INTERLEAVE_FEEDBACK=latent
export ROLLOUT_WEIGHT=1.0
export RESIDUAL_FEEDBACK=1
export ROLLOUT_STEPS=${ROLLOUT_STEPS:-${LATENT_CHUNKS:-10}}
export BPTT_DEPTH=${BPTT_DEPTH:-${ROLLOUT_STEPS}}
echo "Scheme C: predicted latent feedback; steps=${ROLLOUT_STEPS}, BPTT=${BPTT_DEPTH}, weight=${ROLLOUT_WEIGHT}"
exec bash "${PLUGIN_DIR}/sft.sh" "$@"
