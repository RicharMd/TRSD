#!/usr/bin/env bash
# InternLM2.5-7B-Chat · Science Q&A · Forward KL · TRSD
# Main-table TRSD cell: rho=0.20, clip [0.10, 0.80].
# Train, then eval the new task and IFEval.
#   GPU=0 bash scripts/single/internlm_science_forward_trsd.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_single.sh" internlm science forward trsd "$@"
