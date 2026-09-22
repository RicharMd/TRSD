#!/usr/bin/env bash
# Qwen2.5-7B-Instruct · Medical · Reverse KL · TRSD
# Main-table TRSD cell: rho=0.05, clip [0.05, 1.00].
# Train, then eval the new task and IFEval.
#   GPU=0 bash scripts/single/qwen_medical_reverse_trsd.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_single.sh" qwen medical reverse trsd "$@"
