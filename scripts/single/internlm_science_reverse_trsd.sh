#!/usr/bin/env bash
# InternLM2.5-7B-Chat · Science Q&A · Reverse KL · TRSD
# Main-table TRSD cell: rho=0.30, clip [0.05, 0.75].
# Train, then eval the new task and IFEval.
#   GPU=0 bash scripts/single/internlm_science_reverse_trsd.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_single.sh" internlm science reverse trsd "$@"
