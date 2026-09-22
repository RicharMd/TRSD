#!/usr/bin/env bash
# Qwen2.5-7B-Instruct · Medical · Forward KL · TRSD
# Main-table TRSD cell: rho=0.10, clip [0.10, 0.85].
# Train, then eval the new task and IFEval.
#   GPU=0 bash scripts/single/qwen_medical_forward_trsd.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_single.sh" qwen medical forward trsd "$@"
