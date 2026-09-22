#!/usr/bin/env bash
# InternLM2.5-7B-Chat · sequential Tool -> Science -> Medical · Squared Hellinger · TRSD
# T1 x S1 x M2  (tool 0.05/[0.05,1.00], science 0.10/[0.10,0.80], medical 0.10/[0.10,0.80])
#   GPU=0 bash scripts/cross/internlm_hellinger_trsd.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_cross.sh" internlm hellinger trsd "$@"
