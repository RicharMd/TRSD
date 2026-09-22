#!/usr/bin/env bash
# Qwen2.5-7B-Instruct · sequential Tool -> Science -> Medical · Squared Hellinger · TRSD
# T1 x S2 x M2  (tool 0.15/[0.00,0.85], science 0.25/[0.05,1.00], medical 0.20/[0.05,0.80])
#   GPU=0 bash scripts/cross/qwen_hellinger_trsd.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_cross.sh" qwen hellinger trsd "$@"
