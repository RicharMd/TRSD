#!/usr/bin/env bash
# Shared train / eval helpers.
# Flags follow the paper runs (later scripts, probe off):
#   forward TRSD  -> budget_log_ratio_var_relative, alpha 0
#   reverse TRSD  -> budget_chi2_relative, alpha 1, chi2 topk 0, rPE alpha 0.1
#   hellinger TRSD -> budget_hellinger_relative + proximal_selection hellinger
#   SDFT          -> proximal_teacher_tau 0, no online-tau flags
set -Eeuo pipefail

trsd_root() {
  cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd
}

trsd_activate() {
  export WANDB_MODE="${WANDB_MODE:-offline}"
  export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
  export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
  unset PYTORCH_CUDA_ALLOC_CONF || true
  unset PYTORCH_ALLOC_CONF || true
  if [[ -n "${CONDA_ENV:-trsd}" ]] && command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-trsd}"
  fi
}

trsd_defaults() {
  : "${GPU:=0}"
  : "${MODEL_QWEN:=model/Qwen2.5-7B-Instruct}"
  : "${MODEL_INTERNLM:=model/InternLM2.5-7B-Chat}"
  : "${VERIFIER_PATH:=model/medical_o1_verifier_3B}"
  : "${TRAIN_VLLM_UTIL:=0.3}"
  : "${EVAL_VLLM_UTIL:=0.8}"
  : "${EPOCHS:=2}"
  : "${NUM_PROMPTS:=32}"
  : "${LR:=5e-5}"
  : "${SKIP_IF_DONE:=1}"
  : "${IFEVAL_BACKEND:=vllm}"
  : "${IFEVAL_BATCH_SIZE:=auto}"
  : "${IFEVAL_MAX_MODEL_LEN:=4096}"
  : "${CHI2_REL_ALPHA:=0.1}"
  export CUDA_VISIBLE_DEVICES="${GPU}"
  : "${MASTER_PORT:=$((29600 + GPU))}"
}

trsd_base_model() {
  case "$1" in
    qwen) echo "${MODEL_QWEN}" ;;
    internlm) echo "${MODEL_INTERNLM}" ;;
    *) echo "ERROR: model must be qwen or internlm (got $1)" >&2; return 1 ;;
  esac
}

trsd_ckpt_ok() {
  local d="$1"
  [[ -f "${d}/config.json" ]] || return 1
  [[ -f "${d}/model.safetensors" || -f "${d}/model.safetensors.index.json" ]] \
    || compgen -G "${d}/model-"*.safetensors >/dev/null
}

trsd_eval_done() {
  [[ -f "$1/$2/eval_results.json" ]]
}

trsd_ifeval_done() {
  [[ -d "$1/lm_eval" ]] && find "$1/lm_eval" -name 'results_*.json' -print -quit | grep -q .
}

# Fill TRAIN_EXTRA for one geometry.
# method: trsd | sdft
# geom:   forward | reverse | hellinger
trsd_fill_method_args() {
  local geom="$1" method="$2" rho="${3:-}" cmin="${4:-}" cmax="${5:-}"
  TRAIN_EXTRA=()
  if [[ "${method}" == "sdft" ]]; then
    case "${geom}" in
      forward)
        TRAIN_EXTRA=(
          --distillation_divergence kl
          --alpha 0
          --proximal_teacher_tau 0
        )
        ;;
      reverse)
        TRAIN_EXTRA=(
          --distillation_divergence kl
          --alpha 1
          --proximal_teacher_tau 0
        )
        ;;
      hellinger)
        TRAIN_EXTRA=(
          --distillation_divergence hellinger
          --proximal_selection hellinger
          --alpha 0
          --proximal_teacher_tau 0
        )
        ;;
      *) echo "ERROR: unknown geometry ${geom}" >&2; return 1 ;;
    esac
    return 0
  fi
  if [[ "${method}" != "trsd" ]]; then
    echo "ERROR: method must be trsd or sdft (got ${method})" >&2
    return 1
  fi
  case "${geom}" in
    forward)
      TRAIN_EXTRA=(
        --distillation_divergence kl
        --alpha 0
        --online_sample_tau_mode budget_log_ratio_var_relative
        --online_sample_tau_k_bar_ema_decay 0
        --online_sample_tau_budget_alpha "${rho}"
        --online_sample_tau_clip_min "${cmin}"
        --online_sample_tau_clip_max "${cmax}"
      )
      ;;
    reverse)
      TRAIN_EXTRA=(
        --distillation_divergence kl
        --alpha 1
        --online_sample_tau_mode budget_chi2_relative
        --online_sample_tau_k_bar_ema_decay 0
        --online_sample_tau_chi2_topk 0
        --online_sample_tau_chi2_relative_alpha "${CHI2_REL_ALPHA}"
        --online_sample_tau_budget_alpha "${rho}"
        --online_sample_tau_clip_min "${cmin}"
        --online_sample_tau_clip_max "${cmax}"
      )
      ;;
    hellinger)
      TRAIN_EXTRA=(
        --distillation_divergence hellinger
        --proximal_selection hellinger
        --alpha 0
        --online_sample_tau_mode budget_hellinger_relative
        --online_sample_tau_k_bar_ema_decay 0
        --online_sample_tau_budget_alpha "${rho}"
        --online_sample_tau_clip_min "${cmin}"
        --online_sample_tau_clip_max "${cmax}"
      )
      ;;
    *) echo "ERROR: unknown geometry ${geom}" >&2; return 1 ;;
  esac
}

trsd_sft_train() {
  local dataset="$1" init_ckpt="$2" out="$3" loss_type="$4"
  if [[ "${SKIP_IF_DONE}" == "1" ]] && trsd_ckpt_ok "${out}"; then
    echo "=== SKIP train ${out} (weights present) ==="
    return 0
  fi
  mkdir -p "${out}"
  echo "=== TRAIN ${loss_type} ${dataset} -> ${out} ==="
  echo "init=${init_ckpt} epochs=${EPOCHS} batch=${NUM_PROMPTS} lr=${LR}"
  CUDA_VISIBLE_DEVICES="${GPU}" python main_sft.py \
    --dataset_name "${dataset}" \
    --model_name "${init_ckpt}" \
    --learning_rate "${LR}" \
    --num_train_epochs "${EPOCHS}" \
    --num_prompts_per_batch "${NUM_PROMPTS}" \
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --loss_type "${loss_type}" \
    --save_only_final \
    --report_to none \
    --output_dir "${out}" \
    2>&1 | tee "${out}/train.log"
  trsd_ckpt_ok "${out}"
}

trsd_train() {
  local dataset="$1" init_ckpt="$2" out="$3"
  shift 3
  if [[ "${SKIP_IF_DONE}" == "1" ]] && trsd_ckpt_ok "${out}"; then
    echo "=== SKIP train ${out} (weights present) ==="
    return 0
  fi
  mkdir -p "${out}"
  echo "=== TRAIN ${dataset} -> ${out} ==="
  echo "init=${init_ckpt}"
  echo "cmd: python main.py --dataset_name ${dataset} --model_name ${init_ckpt} ${TRAIN_EXTRA[*]} --output_dir ${out}"
  CUDA_VISIBLE_DEVICES="${GPU}" MASTER_PORT="${MASTER_PORT}" python main.py \
    --dataset_name "${dataset}" \
    --model_name "${init_ckpt}" \
    --learning_rate "${LR}" \
    --num_train_epochs "${EPOCHS}" \
    --num_prompts_per_batch "${NUM_PROMPTS}" \
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --vllm_gpu_memory_utilization "${TRAIN_VLLM_UTIL}" \
    --save_only_final \
    "${TRAIN_EXTRA[@]}" \
    --output_dir "${out}" \
    --sample_metrics_path "${out}/sample_metrics.jsonl" \
    2>&1 | tee "${out}/train.log"
  trsd_ckpt_ok "${out}"
}

trsd_eval_one() {
  local ckpt="$1" dataset="$2" subdir="$3"
  local script max_new
  case "${dataset}" in
    tooluse) script=eval_tooluse.py; max_new=1024 ;;
    science) script=eval_science.py; max_new=2048 ;;
    medical) script=eval_medical.py; max_new=1024 ;;
    *) echo "ERROR: bad dataset ${dataset}" >&2; return 1 ;;
  esac
  if [[ "${SKIP_IF_DONE}" == "1" ]] && trsd_eval_done "${ckpt}" "${subdir}"; then
    echo "=== SKIP eval ${ckpt}/${subdir} ==="
    return 0
  fi
  mkdir -p "${ckpt}/${subdir}"
  local -a extra=()
  if [[ "${dataset}" == "medical" ]]; then
    extra=(--verifier_path "${VERIFIER_PATH}")
  fi
  echo "=== EVAL ${dataset} -> ${ckpt}/${subdir} ==="
  CUDA_VISIBLE_DEVICES="${GPU}" python "${script}" \
    --model_path "${ckpt}" \
    --output_dir "${ckpt}/${subdir}" \
    --max_new_tokens "${max_new}" \
    --temperature 0.0 \
    --vllm_gpu_memory_utilization "${EVAL_VLLM_UTIL}" \
    "${extra[@]}" \
    2>&1 | tee "${ckpt}/${subdir}.log"
  trsd_eval_done "${ckpt}" "${subdir}"
}

trsd_ifeval() {
  local ckpt="$1"
  if [[ "${SKIP_IF_DONE}" == "1" ]] && trsd_ifeval_done "${ckpt}"; then
    echo "=== SKIP IFEval ${ckpt}/lm_eval ==="
    return 0
  fi
  mkdir -p "${ckpt}/lm_eval"
  echo "=== EVAL IFEval -> ${ckpt}/lm_eval ==="
  if [[ "${IFEVAL_BACKEND}" == "vllm" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" lm_eval \
      --model vllm \
      --model_args "pretrained=${ckpt},trust_remote_code=True,dtype=bfloat16,gpu_memory_utilization=${EVAL_VLLM_UTIL},max_model_len=${IFEVAL_MAX_MODEL_LEN}" \
      --tasks ifeval \
      --batch_size "${IFEVAL_BATCH_SIZE}" \
      --output_path "${ckpt}/lm_eval" \
      --confirm_run_unsafe_code \
      2>&1 | tee "${ckpt}/lm_eval.log"
  else
    CUDA_VISIBLE_DEVICES="${GPU}" lm_eval \
      --model hf \
      --model_args "pretrained=${ckpt},trust_remote_code=True,dtype=bfloat16" \
      --tasks ifeval \
      --batch_size "${IFEVAL_BATCH_SIZE}" \
      --output_path "${ckpt}/lm_eval" \
      --confirm_run_unsafe_code \
      2>&1 | tee "${ckpt}/lm_eval.log"
  fi
  trsd_ifeval_done "${ckpt}"
}

# Last dataset is the new task. Earlier ones are tasks already learned.
# Their results go to eval_<task>_after_cross. IFEval is always run.
trsd_eval_tasks() {
  local ckpt="$1" mode="$2"
  shift 2
  local current=""
  if [[ "$#" -gt 0 ]]; then
    current="${!#}"
  fi
  local dataset subdir
  for dataset in "$@"; do
    if [[ "${mode}" == "cross" && "${dataset}" != "${current}" ]]; then
      subdir="eval_${dataset}_after_cross"
    else
      subdir="eval_${dataset}"
    fi
    trsd_eval_one "${ckpt}" "${dataset}" "${subdir}"
  done
  trsd_ifeval "${ckpt}"
}
