#!/usr/bin/env bash
# Qwen2.5-7B-Instruct · sequential Tool -> Science -> Medical · Forward KL · TRSD
# T1 x S2 x M1  (tool 0.05/[0.05,1.00], science 0.05/[0.20,0.80], medical 0.10/[0.10,0.85])
#   GPU=0 bash scripts/cross/qwen_forward_trsd.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_cross.sh" qwen forward trsd "$@"
