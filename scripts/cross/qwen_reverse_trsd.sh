#!/usr/bin/env bash
# Qwen2.5-7B-Instruct · sequential Tool -> Science -> Medical · Reverse KL · TRSD
# T2 x S1 x M2  (tool 0.05/[0.05,0.80], science 0.20/[0.20,0.85], medical 0.20/[0.10,1.00])
#   GPU=0 bash scripts/cross/qwen_reverse_trsd.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_cross.sh" qwen reverse trsd "$@"
