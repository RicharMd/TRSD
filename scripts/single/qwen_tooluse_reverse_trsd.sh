#!/usr/bin/env bash
# Qwen2.5-7B-Instruct · Tool Use · Reverse KL · TRSD
# Main-table TRSD cell: rho=0.05, clip [0.05, 0.80].
# Train, then eval the new task and IFEval.
#   GPU=0 bash scripts/single/qwen_tooluse_reverse_trsd.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_single.sh" qwen tooluse reverse trsd "$@"
