#!/usr/bin/env bash
# One single-task cell from the main table: train, then the new task and IFEval.
#
# Usage:
#   GPU=0 bash scripts/run_single.sh qwen tooluse forward trsd
#   GPU=0 bash scripts/run_single.sh internlm medical reverse sdft
#
# model:    qwen | internlm
# task:     tooluse | science | medical
# geometry: forward | reverse | hellinger
# method:   trsd | sdft
#
# TRSD rho / clip are the main-table cells
# (Qwen: good_pick matched to Table 1; InternLM: T1 / S1 / M1).
# SDFT is the matched tau=0 run and ignores rho / clip.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
# shellcheck source=lib_trsd.sh
source "${ROOT}/scripts/lib_trsd.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 4 ]]; then
  echo "Usage: GPU=0 bash scripts/run_single.sh <qwen|internlm> <tooluse|science|medical> <forward|reverse|hellinger> <trsd|sdft>" >&2
  exit 2
fi

MODEL_TAG="$1"
TASK="$2"
GEOM="$3"
METHOD="$4"

trsd_activate
trsd_defaults

case "${MODEL_TAG}:${TASK}:${GEOM}" in
  qwen:tooluse:forward)     RHO=0.05; CMIN=0.05; CMAX=1.00 ;;
  qwen:science:forward)     RHO=0.10; CMIN=0.15; CMAX=0.80 ;;
  qwen:medical:forward)     RHO=0.10; CMIN=0.10; CMAX=0.85 ;;
  qwen:tooluse:reverse)     RHO=0.05; CMIN=0.05; CMAX=0.80 ;;
  qwen:science:reverse)     RHO=0.20; CMIN=0.20; CMAX=0.85 ;;
  qwen:medical:reverse)     RHO=0.05; CMIN=0.05; CMAX=1.00 ;;
  qwen:tooluse:hellinger)   RHO=0.15; CMIN=0.00; CMAX=0.85 ;;
  qwen:science:hellinger)   RHO=0.25; CMIN=0.10; CMAX=0.80 ;;
  qwen:medical:hellinger)   RHO=0.10; CMIN=0.15; CMAX=0.85 ;;
  internlm:tooluse:forward)   RHO=0.25; CMIN=0.10; CMAX=0.80 ;;
  internlm:science:forward)   RHO=0.20; CMIN=0.10; CMAX=0.80 ;;
  internlm:medical:forward)   RHO=0.05; CMIN=0.05; CMAX=1.00 ;;
  internlm:tooluse:reverse)   RHO=0.05; CMIN=0.05; CMAX=0.80 ;;
  internlm:science:reverse)   RHO=0.30; CMIN=0.05; CMAX=0.75 ;;
  internlm:medical:reverse)   RHO=0.20; CMIN=0.10; CMAX=0.80 ;;
  internlm:tooluse:hellinger) RHO=0.05; CMIN=0.05; CMAX=1.00 ;;
  internlm:science:hellinger) RHO=0.10; CMIN=0.10; CMAX=0.80 ;;
  internlm:medical:hellinger) RHO=0.20; CMIN=0.10; CMAX=0.80 ;;
  *)
    echo "ERROR: no main-table cell for ${MODEL_TAG} ${TASK} ${GEOM}" >&2
    exit 1
    ;;
esac

INIT="$(trsd_base_model "${MODEL_TAG}")"
OUT="outputs/${MODEL_TAG}_${TASK}_${GEOM}_${METHOD}"
trsd_fill_method_args "${GEOM}" "${METHOD}" "${RHO}" "${CMIN}" "${CMAX}"

echo "=== single ${MODEL_TAG} ${TASK} ${GEOM} ${METHOD} ==="
echo "init=${INIT}"
echo "out=${OUT}"
echo "epochs=${EPOCHS} lr=${LR} batch=${NUM_PROMPTS} gpu=${GPU}"
if [[ "${METHOD}" == "trsd" ]]; then
  echo "rho=${RHO} clip=[${CMIN},${CMAX}]"
fi

trsd_train "${TASK}" "${INIT}" "${OUT}"
trsd_eval_tasks "${OUT}" single "${TASK}"
echo "=== DONE ${OUT} ==="
