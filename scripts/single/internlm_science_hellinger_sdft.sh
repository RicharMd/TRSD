#!/usr/bin/env bash
# InternLM2.5-7B-Chat · Science Q&A · Squared Hellinger · SDFT
# Matched SDFT (tau=0). Ignores the TRSD rho/clip.
# Train, then eval the new task and IFEval.
#   GPU=0 bash scripts/single/internlm_science_hellinger_sdft.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_single.sh" internlm science hellinger sdft "$@"
