#!/usr/bin/env bash
# Qwen2.5-7B-Instruct · Medical · DFT
# Main-table cell. Train, then eval the new task and IFEval.
#   GPU=0 bash scripts/single/qwen_medical_dft.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_sft.sh" qwen medical dft "$@"
