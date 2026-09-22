#!/usr/bin/env bash
# InternLM2.5-7B-Chat · sequential Tool -> Science -> Medical · Forward KL · TRSD
# T1 x S1 x M2  (tool 0.25/[0.10,0.80], science 0.20/[0.10,0.80], medical 0.05/[0.10,1.00])
#   GPU=0 bash scripts/cross/internlm_forward_trsd.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_cross.sh" internlm forward trsd "$@"
