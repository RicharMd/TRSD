#!/usr/bin/env bash
# InternLM2.5-7B-Chat · Tool Use · Squared Hellinger · TRSD
# Main-table TRSD cell: rho=0.05, clip [0.05, 1.00].
# Train, then eval the new task and IFEval.
#   GPU=0 bash scripts/single/internlm_tooluse_hellinger_trsd.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_single.sh" internlm tooluse hellinger trsd "$@"
