#!/usr/bin/env bash
# Sequential Tool -> Science -> Medical for one geometry.
# TRSD uses the bold trajectory in the cross-task appendix, not the
# single-task main-table cell (those differ on some stages).
# SDFT is the matched tau=0 chain from the base model.
#
# After each stage: the new task, the tasks already learned, and IFEval.
#
# Usage:
#   GPU=0 bash scripts/run_cross.sh qwen forward trsd
#   GPU=0 bash scripts/run_cross.sh internlm hellinger sdft
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
# shellcheck source=lib_trsd.sh
source "${ROOT}/scripts/lib_trsd.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 3 ]]; then
  echo "Usage: GPU=0 bash scripts/run_cross.sh <qwen|internlm> <forward|reverse|hellinger> <trsd|sdft>" >&2
  exit 2
fi

MODEL_TAG="$1"
GEOM="$2"
METHOD="$3"

trsd_activate
trsd_defaults

# Paper bold rows. IDs are the appendix cross-grid labels.
case "${MODEL_TAG}:${GEOM}" in
  qwen:forward)
    TRAJ="T1 x S2 x M1"
    T_RHO=0.05; T_CMIN=0.05; T_CMAX=1.00
    S_RHO=0.05; S_CMIN=0.20; S_CMAX=0.80
    M_RHO=0.10; M_CMIN=0.10; M_CMAX=0.85
    ;;
  qwen:reverse)
    TRAJ="T2 x S1 x M2"
    T_RHO=0.05; T_CMIN=0.05; T_CMAX=0.80
    S_RHO=0.20; S_CMIN=0.20; S_CMAX=0.85
    M_RHO=0.20; M_CMIN=0.10; M_CMAX=1.00
    ;;
  qwen:hellinger)
    TRAJ="T1 x S2 x M2"
    T_RHO=0.15; T_CMIN=0.00; T_CMAX=0.85
    S_RHO=0.25; S_CMIN=0.05; S_CMAX=1.00
    M_RHO=0.20; M_CMIN=0.05; M_CMAX=0.80
    ;;
  internlm:forward)
    TRAJ="T1 x S1 x M2"
    T_RHO=0.25; T_CMIN=0.10; T_CMAX=0.80
    S_RHO=0.20; S_CMIN=0.10; S_CMAX=0.80
    M_RHO=0.05; M_CMIN=0.10; M_CMAX=1.00
    ;;
  internlm:reverse)
    TRAJ="T2 x S2 x M1"
    T_RHO=0.05; T_CMIN=0.00; T_CMAX=0.75
    S_RHO=0.30; S_CMIN=0.05; S_CMAX=0.80
    M_RHO=0.20; M_CMIN=0.10; M_CMAX=0.80
    ;;
  internlm:hellinger)
    TRAJ="T1 x S1 x M2"
    T_RHO=0.05; T_CMIN=0.05; T_CMAX=1.00
    S_RHO=0.10; S_CMIN=0.10; S_CMAX=0.80
    M_RHO=0.10; M_CMIN=0.10; M_CMAX=0.80
    ;;
  *)
    echo "ERROR: no cross trajectory for ${MODEL_TAG} ${GEOM}" >&2
    exit 1
    ;;
esac

BASE="$(trsd_base_model "${MODEL_TAG}")"
ROOT_OUT="outputs/${MODEL_TAG}_cross_${GEOM}_${METHOD}"
TOOL_OUT="${ROOT_OUT}/tool"
SCI_OUT="${ROOT_OUT}/science"
MED_OUT="${ROOT_OUT}/medical"

echo "=== cross ${MODEL_TAG} ${GEOM} ${METHOD} trajectory ${TRAJ} ==="
echo "base=${BASE}"
echo "epochs=${EPOCHS} lr=${LR} batch=${NUM_PROMPTS} gpu=${GPU}"

echo "--- stage tool (${TRAJ%% x*}) rho=${T_RHO} clip=[${T_CMIN},${T_CMAX}] ---"
trsd_fill_method_args "${GEOM}" "${METHOD}" "${T_RHO}" "${T_CMIN}" "${T_CMAX}"
trsd_train tooluse "${BASE}" "${TOOL_OUT}"
trsd_eval_tasks "${TOOL_OUT}" cross tooluse

echo "--- stage science rho=${S_RHO} clip=[${S_CMIN},${S_CMAX}] ---"
trsd_fill_method_args "${GEOM}" "${METHOD}" "${S_RHO}" "${S_CMIN}" "${S_CMAX}"
trsd_train science "${TOOL_OUT}" "${SCI_OUT}"
trsd_eval_tasks "${SCI_OUT}" cross tooluse science

echo "--- stage medical rho=${M_RHO} clip=[${M_CMIN},${M_CMAX}] ---"
trsd_fill_method_args "${GEOM}" "${METHOD}" "${M_RHO}" "${M_CMIN}" "${M_CMAX}"
trsd_train medical "${SCI_OUT}" "${MED_OUT}"
trsd_eval_tasks "${MED_OUT}" cross tooluse science medical

echo "=== DONE ${ROOT_OUT} ==="
