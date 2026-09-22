#!/usr/bin/env bash
# Qwen2.5-7B-Instruct · Science Q&A · Forward KL · SDFT
# Matched SDFT (tau=0). Ignores the TRSD rho/clip.
# Train, then eval the new task and IFEval.
#   GPU=0 bash scripts/single/qwen_science_forward_sdft.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_single.sh" qwen science forward sdft "$@"
