#!/usr/bin/env bash
# Single-task SFT or DFT cell from the main table: train, then the new task and IFEval.
#
# Usage:
#   GPU=0 bash scripts/run_sft.sh qwen tooluse sft
#   GPU=0 bash scripts/run_sft.sh internlm medical dft
#
# Epochs and batch are the cells behind the main-table numbers, not a shared default.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
# shellcheck source=lib_trsd.sh
source "${ROOT}/scripts/lib_trsd.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 3 ]]; then
  echo "Usage: GPU=0 bash scripts/run_sft.sh <qwen|internlm> <tooluse|science|medical> <sft|dft>" >&2
  exit 2
fi

MODEL_TAG="$1"
TASK="$2"
LOSS="$3"

trsd_activate
trsd_defaults

case "${MODEL_TAG}:${TASK}:${LOSS}" in
  qwen:tooluse:sft)       EPOCHS=6; NUM_PROMPTS=16 ;;
  qwen:science:sft)       EPOCHS=2; NUM_PROMPTS=32 ;;
  qwen:medical:sft)       EPOCHS=2; NUM_PROMPTS=16 ;;
  qwen:tooluse:dft)       EPOCHS=2; NUM_PROMPTS=32 ;;
  qwen:science:dft)       EPOCHS=2; NUM_PROMPTS=32 ;;
  qwen:medical:dft)       EPOCHS=6; NUM_PROMPTS=64 ;;
  internlm:tooluse:sft)   EPOCHS=6; NUM_PROMPTS=16 ;;
  internlm:science:sft)   EPOCHS=2; NUM_PROMPTS=32 ;;
  internlm:medical:sft)   EPOCHS=2; NUM_PROMPTS=32 ;;
  internlm:tooluse:dft)   EPOCHS=6; NUM_PROMPTS=16 ;;
  internlm:science:dft)   EPOCHS=2; NUM_PROMPTS=16 ;;
  internlm:medical:dft)   EPOCHS=4; NUM_PROMPTS=64 ;;
  *)
    echo "ERROR: no main-table cell for ${MODEL_TAG} ${TASK} ${LOSS}" >&2
    exit 1
    ;;
esac

INIT="$(trsd_base_model "${MODEL_TAG}")"
OUT="outputs/${MODEL_TAG}_${TASK}_${LOSS}"

echo "=== single ${MODEL_TAG} ${TASK} ${LOSS} ==="
echo "init=${INIT}"
echo "out=${OUT}"
echo "epochs=${EPOCHS} lr=${LR} batch=${NUM_PROMPTS} gpu=${GPU}"

trsd_sft_train "${TASK}" "${INIT}" "${OUT}" "${LOSS}"
trsd_eval_tasks "${OUT}" single "${TASK}"
echo "=== DONE ${OUT} ==="
