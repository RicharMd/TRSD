#!/usr/bin/env bash
# InternLM2.5-7B-Chat · Medical · Reverse KL · SDFT
# Matched SDFT (tau=0). Ignores the TRSD rho/clip.
# Train, then eval the new task and IFEval.
#   GPU=0 bash scripts/single/internlm_medical_reverse_sdft.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_single.sh" internlm medical reverse sdft "$@"
