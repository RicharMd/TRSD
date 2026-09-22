#!/usr/bin/env bash
# Qwen2.5-7B-Instruct · sequential Tool -> Science -> Medical · Reverse KL · SDFT
# Matched SDFT tau=0 chain, same stage order, no trust-region budget.
#   GPU=0 bash scripts/cross/qwen_reverse_sdft.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_cross.sh" qwen reverse sdft "$@"
