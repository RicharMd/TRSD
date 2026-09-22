#!/usr/bin/env bash
# InternLM2.5-7B-Chat · Science Q&A · Squared Hellinger · TRSD
# Main-table TRSD cell: rho=0.10, clip [0.10, 0.80].
# Train, then eval the new task and IFEval.
#   GPU=0 bash scripts/single/internlm_science_hellinger_trsd.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_single.sh" internlm science hellinger trsd "$@"
