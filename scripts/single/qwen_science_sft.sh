#!/usr/bin/env bash
# Qwen2.5-7B-Instruct · Science Q&A · SFT
# Main-table cell. Train, then eval the new task and IFEval.
#   GPU=0 bash scripts/single/qwen_science_sft.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/run_sft.sh" qwen science sft "$@"
