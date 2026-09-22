#!/usr/bin/env bash
# InternLM2.5-7B-Chat · sequential Tool -> Science -> Medical · Reverse KL · TRSD
# T2 x S2 x M1  (tool 0.05/[0.00,0.75], science 0.30/[0.05,0.80], medical 0.20/[0.10,0.80])
#   GPU=0 bash scripts/cross/internlm_reverse_trsd.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_cross.sh" internlm reverse trsd "$@"
