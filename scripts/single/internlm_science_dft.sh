#!/usr/bin/env bash
# InternLM2.5-7B-Chat · Science Q&A · DFT
# Main-table cell. Train, then eval the new task and IFEval.
#   GPU=0 bash scripts/single/internlm_science_dft.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_sft.sh" internlm science dft "$@"
