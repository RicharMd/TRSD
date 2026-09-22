#!/usr/bin/env bash
# Qwen2.5-7B-Instruct · Science Q&A · Squared Hellinger · TRSD
# Main-table TRSD cell: rho=0.25, clip [0.10, 0.80].
# Train, then eval the new task and IFEval.
#   GPU=0 bash scripts/single/qwen_science_hellinger_trsd.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_single.sh" qwen science hellinger trsd "$@"
