#!/usr/bin/env bash
# Qwen2.5-7B-Instruct · Science Q&A · Squared Hellinger · SDFT
# Matched SDFT (tau=0). Ignores the TRSD rho/clip.
# Train, then eval the new task and IFEval.
#   GPU=0 bash scripts/single/qwen_science_hellinger_sdft.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_single.sh" qwen science hellinger sdft "$@"
