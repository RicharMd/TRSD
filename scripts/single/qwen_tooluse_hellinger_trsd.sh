#!/usr/bin/env bash
# Qwen2.5-7B-Instruct · Tool Use · Squared Hellinger · TRSD
# Main-table TRSD cell: rho=0.15, clip [0.00, 0.85].
# Train, then eval the new task and IFEval.
#   GPU=0 bash scripts/single/qwen_tooluse_hellinger_trsd.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_single.sh" qwen tooluse hellinger trsd "$@"
