from distil_trainer_new import DistilTrainer
from distil_config import DistilConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset, load_dataset, load_from_disk
from string import Template
import argparse
import torch.distributed as dist
import os
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Distil Trainer")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, help="Number of prompts per batch")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01, help="Reference model mixup alpha")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--dataset_name", type=str, default="tooluse", help="Dataset name", choices=["tooluse", "science", "math", "medical"])
    parser.add_argument("--math_data_path", type=str, default="../self-distillation-analysis/data/math/train.parquet", help="Path to DAPO-Math parquet data")
    parser.add_argument("--math_max_samples", type=int, default=8192, help="Maximum math samples to use; set <=0 to use all")
    parser.add_argument("--max_prompt_length", type=int, default=1024, help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=1024, help="Maximum completion length")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.22, help="vLLM GPU memory utilization")
    parser.add_argument("--save_steps", type=int, default=100, help="Checkpoint save interval")
    parser.add_argument(
        "--save_only_final",
        action="store_true",
        help="No intermediate checkpoints; save model+tokenizer once to output_dir after training.",
    )
    parser.add_argument("--save_optimizer_state", action="store_true", help="Also save optimizer/scheduler/RNG state for resuming")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="KL mixture: 0=forward KL, 1=reverse KL, 0.5=generalized Jensen–Shannon (e.g. align with MS SDPO appendix)",
    )
    parser.add_argument(
        "--distillation_divergence",
        type=str,
        default="kl",
        choices=["kl", "hellinger"],
        help=(
            "Training divergence D_f. 'kl' uses --alpha (0=forward KL, 1=reverse KL). "
            "'hellinger' uses Squared Hellinger H^2(p,q)=1-sum_v sqrt(p q). "
            "Baseline SDFT with Hellinger: --distillation_divergence hellinger --proximal_teacher_tau 0."
        ),
    )
    parser.add_argument(
        "--proximal_selection",
        type=str,
        default="auto",
        choices=["auto", "forward_kl", "reverse_kl", "hellinger"],
        help=(
            "Proximal teacher path for q_tau. auto: geometric if --alpha 0, arithmetic if --alpha 1. "
            "hellinger: √-mix u=(1-tau)sqrt(q_o)+tau sqrt(p). "
            "budget_hellinger* forces hellinger selection + hellinger D_f."
        ),
    )
    parser.add_argument(
        "--proximal_teacher_tau",
        type=float,
        default=0.0,
        help=(
            "Proximal teacher weight tau in [0, 1]: "
            "0=original teacher, 1=current-policy endpoint. "
            "q_tau closed form follows --proximal_selection (or --alpha when selection=auto)."
        ),
    )
    parser.add_argument("--probe_metrics_path", type=str, default=None, help="JSONL path for per-step probe metrics.")
    parser.add_argument("--probe_every_steps", type=int, default=0, help="Compute fixed new/old probes every N steps.")
    parser.add_argument("--probe_new_path", type=str, default="data/tooluse_data/eval_data", help="Path to tool-use probe dataset.")
    parser.add_argument("--probe_new_max_samples", type=int, default=32, help="Number of tool-use examples for new probe.")
    parser.add_argument("--probe_old_every_steps", type=int, default=0, help="Deprecated alias for --probe_every_steps.")
    parser.add_argument("--probe_old_max_samples", type=int, default=32, help="Number of IFEval prompts for old probe.")
    parser.add_argument("--probe_old_topk", type=int, default=64, help="Top-K support for old probe KL.")
    parser.add_argument("--sample_tau_map_path", type=str, default=None, help="JSONL map from original sample index to tau.")
    parser.add_argument(
        "--teacher_template_map_path",
        type=str,
        default=None,
        help=(
            "Optional JSONL from select_best_teacher_template_map.py: per original dataset index, "
            "use the rendered teacher_prompt as ref-teacher wrapper (tooluse only)."
        ),
    )
    parser.add_argument("--sample_metrics_path", type=str, default=None, help="JSONL path for per-sample training signals.")
    parser.add_argument(
        "--online_sample_tau_mode",
        type=str,
        default=None,
        choices=[
            None,
            "budget_forward_kl",
            "budget_forward_kl_exp",
            "budget_forward_kl_relative",
            "budget_forward_kl_relative_exp",
            "budget_forward_kl_relative_exp_range",
            "budget_forward_kl_relative_token",
            "budget_forward_kl_relative_exp_token",
            "budget_forward_kl_relative_exp_range_token",
            "budget_forward_kl_relative_entropy",
            "budget_forward_kl_relative_entropy_token",
            "budget_chi2",
            "budget_chi2_exp",
            "budget_chi2_relative",
            "budget_chi2_relative_exp",
            "budget_chi2_relative_exp_range",
            "budget_chi2_relative_token",
            "budget_chi2_relative_exp_token",
            "budget_chi2_relative_exp_range_token",
            "budget_hellinger",
            "budget_hellinger_exp",
            "budget_hellinger_relative",
            "budget_hellinger_relative_exp",
            "budget_hellinger_relative_exp_range",
            "budget_hellinger_relative_token",
            "budget_hellinger_relative_exp_token",
            "budget_hellinger_relative_exp_range_token",
            "budget_log_ratio_var",
            "budget_log_ratio_var_exp",
            "budget_log_ratio_var_relative",
            "budget_log_ratio_var_relative_exp",
            "budget_log_ratio_var_relative_exp_range",
            "budget_log_ratio_var_relative_token",
            "budget_log_ratio_var_relative_exp_token",
            "budget_log_ratio_var_relative_exp_range_token",
            "softmax_router_relative",
            "softmax_router_relative_token",
            "softmax_router_relative_entropy",
            "softmax_router_relative_entropy_token",
        ],
        help=(
            "Online tau rule. budget_forward_kl uses fixed C with K_i=KL(q_o||p): "
            "tau=clip(max(0,1-sqrt(C/K_i)),...). budget_forward_kl_exp uses "
            "tau=clip(tau_max*exp(-sqrt(C/K_i)),...) with tau_max=clip_max. "
            "budget_forward_kl_relative* uses 1 - sqrt(alpha * K_bar / K_i); "
            "budget_forward_kl_relative_exp* uses tau_max*exp(-sqrt(alpha * K_bar / K_i)); "
            "budget_forward_kl_relative_exp_range* uses "
            "tau_min + (tau_max - tau_min)*exp(-sqrt(alpha * K_bar / K_i)) with tau_min/max=clip_min/max. "
            "budget_chi2* (requires --alpha 1, reverse closed-loop) uses Pearson chi2 peer; "
            "default engineering peer is relative-Pearson (rPE) via "
            "--online_sample_tau_chi2_relative_alpha (default 0.1; set 0 for bare E_p[(q_o/p-1)^2]); "
            "optional top-K(p)+tail via --online_sample_tau_chi2_topk (64; <=0 = full vocab); "
            "budget_chi2_relative* uses 1 - sqrt(budget_alpha * chi2_bar / chi2_i) "
            "(same sqrt path as var_rel; budget_alpha is --online_sample_tau_budget_alpha); "
            "fixed-C budget_chi2* uses chi2_i/2 in sqrt(C/chi2). "
            "budget_hellinger* (requires --distillation_divergence hellinger) uses peer_H=b-a^2 "
            "along the √-mix path with standard H^2; relative: 1-sqrt(budget_alpha * peer_H_bar / peer_H_i); "
            "fixed-C uses peer_H/2 in sqrt(C/·) so C is in H^2 units (same as var/chi2). "
            "budget_log_ratio_var* uses V_i=Var_p(log(q_o/p)); fixed-C modes use D_i≈V_i/2 in sqrt(C/D). "
            "Relative var modes use V_bar/V_i (the 1/2 cancels in the ratio). "
            "budget_forward_kl_relative_entropy* is the same with H_i=H(p), H_bar=batch mean. "
            "softmax_router_relative* uses softmax on K/(gamma*K_bar); "
            "softmax_router_relative_entropy* uses H/(gamma*H_bar). "
            "Token variants recompute per-token signal at loss time."
        ),
    )
    parser.add_argument("--online_sample_tau_budget_c", type=float, default=0.0, help="Budget C for online sample tau.")
    parser.add_argument(
        "--online_sample_tau_budget_alpha",
        type=float,
        default=0.0,
        help=(
            "Alpha for relative online tau: sqrt-budget numerator for budget_forward_kl_relative*, "
            "budget_log_ratio_var_relative*, and budget_chi2_relative* (with --alpha 1 for chi2); "
            "router temperature for softmax_router_relative*."
        ),
    )
    parser.add_argument(
        "--online_sample_tau_k0_path",
        type=str,
        default=None,
        help=(
            "Optional K0 JSONL (kl_qo_to_p0 per index) for dual-anchor relative tau: "
            "forward/var sqrt modes use tau = 1 - sqrt(alpha * (Kbar_0/K_i,0)^gamma * (Kbar_t/K_i)); "
            "chi2 relative modes use the same dual-anchor ratio on chi2_i."
        ),
    )
    parser.add_argument(
        "--online_sample_tau_k0_gamma",
        type=float,
        default=1.0,
        help="Exponent gamma on (Kbar_0_global/K_i,0) in dual-anchor relative tau.",
    )
    parser.add_argument(
        "--online_sample_tau_router_gamma",
        type=float,
        default=1.0,
        help=(
            "Router Kbar scale gamma for softmax_router_relative*: "
            "tau uses K/(gamma*K_bar) inside the alpha power; gamma=1 recovers the original router."
        ),
    )
    parser.add_argument("--online_sample_tau_clip_min", type=float, default=0.0, help="Minimum online sample tau.")
    parser.add_argument("--online_sample_tau_clip_max", type=float, default=1.0, help="Maximum online sample tau.")
    parser.add_argument(
        "--online_sample_tau_chi2_topk",
        type=int,
        default=64,
        help=(
            "Top-K(p) + tail bucket for budget_chi2* peer support (default 64). "
            "Set <=0 for full-vocabulary. Applied before chi2/rPE on the buckets; "
            "does not replace --online_sample_tau_chi2_relative_alpha."
        ),
    )
    parser.add_argument(
        "--online_sample_tau_chi2_relative_alpha",
        type=float,
        default=0.1,
        help=(
            "Relativity mix a for relative-Pearson chi2 peer (RuLSIF-style): "
            "mix=a*q_o+(1-a)*p, peer=E_p[(q_o/mix-1)^2]. Default 0.1. "
            "Set 0 for bare chi^2. NOT --alpha and NOT --online_sample_tau_budget_alpha."
        ),
    )
    parser.add_argument(
        "--online_sample_tau_k_bar_ema_decay",
        type=float,
        default=0.99,
        help=(
            "EMA decay for batch K_bar/H_bar in relative/router online tau modes "
            "(0 disables EMA and uses the raw batch mean)."
        ),
    )
    parser.add_argument(
        "--batch_kl_filter_keep_frac",
        type=float,
        default=None,
        help=(
            "Keep lowest-KL(q_o||p) fraction for distillation loss (see --batch_kl_filter_level). "
            "Requires --alpha 0, tau=0, full-vocab."
        ),
    )
    parser.add_argument(
        "--batch_kl_filter_level",
        type=str,
        default="sample",
        choices=["sample", "token"],
        help=(
            "batch_kl_filter granularity: 'sample' masks whole sequences on the generation batch; "
            "'token' masks high-KL completion tokens within each sequence."
        ),
    )
    parser.add_argument(
        "--batch_kl_filter_drop_weight",
        type=float,
        default=0.0,
        help=(
            "Loss weight for filtered-out samples/tokens (kept units stay at 1.0). "
            "0 = hard mask (default); e.g. 0.2 or 0.4 = soft down-weight."
        ),
    )
    parser.add_argument(
        "--effective_tau_metrics_path",
        type=str,
        default=None,
        help="JSONL path for token-level effective tau diagnostics in global-direction runs.",
    )
    parser.add_argument(
        "--disable_vllm_importance_sampling_correction",
        action="store_true",
        help="Do not reweight loss when vLLM sampling logprobs differ from HF forward (ablation; default is on)",
    )
    parser.add_argument(
        "--soft_teacher_directions_pt",
        type=str,
        default=None,
        help="Unit directions .pt for teacher prompt embedding offset (prepare_soft_teacher_prompt_directions.py).",
    )
    parser.add_argument(
        "--soft_teacher_vector",
        type=str,
        default=None,
        help="Vector key in .pt, e.g. orig_minus_student (w<0 moves teacher toward question-only prompt).",
    )
    parser.add_argument(
        "--soft_teacher_w",
        type=float,
        default=0.0,
        help="Scale on soft_teacher_vector added to teacher prompt inputs_embeds only.",
    )
    parser.add_argument(
        "--soft_teacher_row_directions_pt",
        type=str,
        default=None,
        help="Optional .pt with row_directions [num_rows, hidden_size] for cluster-wise soft-teacher directions.",
    )
    parser.add_argument(
        "--soft_teacher_embed_mix_alpha",
        type=float,
        default=None,
        help="If set, prompt embeddings use alpha * original_embedding + (1-alpha) * direction instead of additive w.",
    )
    return parser.parse_args()

def _maybe_parse_json(value):
    if isinstance(value, str):
        return json.loads(value)
    return value

def _first_user_content(messages) -> str:
    messages = _maybe_parse_json(messages)
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    for message in messages:
        if message.get("role") == "user":
            return message["content"]
    return messages[-1]["content"]

def _ground_truth_answer(reward_model) -> str:
    reward_model = _maybe_parse_json(reward_model)
    return str(reward_model["ground_truth"])

DEFAULT_TOOLUSE_TEACHER_TEMPLATE = Template(
    """
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
"""
)


def load_teacher_template_map(path: str | None) -> dict[int, dict] | None:
    if not path:
        return None
    mapping: dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            index = int(row["index"])
            teacher_prompt = row.get("teacher_prompt")
            if not teacher_prompt:
                raise KeyError(
                    f"teacher_template_map index={index} missing non-empty teacher_prompt in {path}"
                )
            mapping[index] = row
    if not mapping:
        raise ValueError(f"teacher_template_map is empty: {path}")
    return mapping


def _default_tooluse_teacher_content(example: dict) -> str:
    return DEFAULT_TOOLUSE_TEACHER_TEMPLATE.substitute(
        orig_content=example["prompt"],
        output_text="\n".join(example["golden_response"]),
    )


def load_tooluse_dataset(
    seed: int = 42,
    teacher_template_map: dict[int, dict] | None = None,
) -> Dataset:
    """Load and prepare tooluse dataset with formatted prompts."""
    train_dir = "data/tooluse_data/train_data"
    train_dataset = load_from_disk(train_dir)

    if teacher_template_map is not None:
        missing = [
            idx for idx in range(len(train_dataset)) if idx not in teacher_template_map
        ]
        if missing:
            raise KeyError(
                "teacher_template_map is missing "
                f"{len(missing)} / {len(train_dataset)} indices; "
                f"first missing={missing[:5]}"
            )
        print(
            f"Using per-sample teacher prompts for {len(teacher_template_map)} tooluse examples",
            flush=True,
        )

    def format_example(example, idx):
        if teacher_template_map is not None:
            teacher_content = teacher_template_map[idx]["teacher_prompt"]
        else:
            teacher_content = _default_tooluse_teacher_content(example)

        row = {
            "prompt": [{"role": "user", "content": example["prompt"]}],
            "teacher_prompt": [{"role": "user", "content": teacher_content}],
            "soft_teacher_row_id": idx,
        }
        if teacher_template_map is not None:
            map_row = teacher_template_map[idx]
            row["teacher_template_id"] = int(map_row["template_id"])
            row["teacher_template_name"] = str(map_row.get("template_name", ""))
        return row

    train_dataset = train_dataset.map(format_example, with_indices=True, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    return train_dataset, None


def load_sample_tau_map(path: str | None) -> dict[int, dict] | None:
    if not path:
        return None
    mapping: dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            mapping[int(row["index"])] = row
    return mapping


def add_sample_tau_fields(dataset: Dataset, sample_tau_map: dict[int, dict] | None) -> Dataset:
    if not sample_tau_map:
        return dataset

    def add_fields(example):
        row_id = int(example["soft_teacher_row_id"])
        if row_id not in sample_tau_map:
            raise KeyError(f"sample_tau_map does not contain original row index {row_id}")
        row = sample_tau_map[row_id]
        example["sample_tau"] = float(row["tau"])
        example["sample_avg64"] = float(row.get("avg_score", -1.0))
        example["sample_uncertainty"] = float(row.get("uncertainty", -1.0))
        return example

    return dataset.map(add_fields)


def load_science_dataset(seed=42) -> Dataset:
    """Load and prepare science dataset with formatted prompts."""
    path = 'data/science_data/train_data'
    print(f"Loading science dataset from {path}")
    dataset = load_from_disk(path)

    def format_example(example, idx):
        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": example["messages"],
            "teacher_prompt": [
                example["messages"][0],
                {'role': 'user', 'content': teacher_prompt.substitute(
                    orig_content=example['messages'][1]['content'],
                    output_text=example['output_text']
                )},
            ],
            "soft_teacher_row_id": idx,
        }

    dataset = dataset.map(format_example, with_indices=True, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    print(f"Loaded {len(dataset)} training examples")
    return dataset, None


def load_medical_dataset(seed=42) -> Dataset:
    """Load and prepare medical dataset (science-compatible on-disk schema)."""
    path = "data/medical_data/train_data"
    print(f"Loading medical dataset from {path}")
    dataset = load_from_disk(path)

    def format_example(example, idx):
        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        orig_content = _first_user_content(example["messages"])
        return {
            "prompt": example["messages"],
            "teacher_prompt": [
                example["messages"][0],
                {"role": "user", "content": teacher_prompt.substitute(
                    orig_content=orig_content,
                    output_text=example["output_text"],
                )},
            ],
            "soft_teacher_row_id": idx,
        }

    dataset = dataset.map(format_example, with_indices=True, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    print(f"Loaded {len(dataset)} medical training examples")
    return dataset, None


def load_math_dataset(data_path: str, seed=42, max_samples=1024) -> Dataset:
    """Load DAPO-Math data and use the ground-truth answer as teacher context."""
    print(f"Loading math dataset from {data_path}")
    dataset = load_dataset("parquet", data_files=data_path, split="train")
    dataset = dataset.shuffle(seed=seed)
    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def format_example(example, idx):
        orig_content = _first_user_content(example["prompt"])
        answer = _ground_truth_answer(example["reward_model"])
        output_text = f"\\boxed{{{answer}}}"
        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": [{"role": "user", "content": orig_content}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(orig_content=orig_content, output_text=output_text)}],
            "soft_teacher_row_id": idx,
        }

    dataset = dataset.map(format_example, with_indices=True, remove_columns=dataset.column_names)
    print(f"Loaded {len(dataset)} math training examples")
    return dataset, None


if __name__ == "__main__":
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    teacher_template_map = load_teacher_template_map(args.teacher_template_map_path)
    if args.teacher_template_map_path and args.dataset_name != "tooluse":
        raise ValueError("--teacher_template_map_path is only supported for --dataset_name tooluse")
    if args.dataset_name == "tooluse":
        dataset, _ = load_tooluse_dataset(args.seed, teacher_template_map)
    elif args.dataset_name == "science":
        dataset, _ = load_science_dataset(args.seed)
    elif args.dataset_name == "math":
        dataset, _ = load_math_dataset(args.math_data_path, args.seed, args.math_max_samples)
    elif args.dataset_name == "medical":
        dataset, _ = load_medical_dataset(args.seed)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")
    sample_tau_map = load_sample_tau_map(args.sample_tau_map_path)
    dataset = add_sample_tau_fields(dataset, sample_tau_map)

    config = DistilConfig(
        seed=args.seed,
        use_vllm = True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1, 
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=True, 
        learning_rate = args.learning_rate,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        logging_steps = 1,
        bf16 = True,
        fp16 = False,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = args.num_prompts_per_batch,
        max_prompt_length = args.max_prompt_length,
        max_completion_length = args.max_completion_length,
        num_train_epochs = args.num_train_epochs,
        num_iterations = 1,
        num_generations = 1,
        save_strategy = "no" if args.save_only_final else "steps",
        save_steps = args.save_steps,
        save_only_model = not args.save_optimizer_state,
        max_grad_norm = 1,
        report_to = "wandb",
        output_dir = args.output_dir,
        log_completions = False, # True for debugging
        sync_ref_model = True,
        ref_model_sync_steps = 1,
        ref_model_mixup_alpha = args.ref_model_mixup_alpha,
        alpha=args.alpha,
        distillation_divergence=args.distillation_divergence,
        proximal_selection=args.proximal_selection,
        proximal_teacher_tau=args.proximal_teacher_tau,
        sample_tau_map_path=args.sample_tau_map_path,
        sample_metrics_path=args.sample_metrics_path,
        online_sample_tau_mode=args.online_sample_tau_mode,
        online_sample_tau_budget_c=args.online_sample_tau_budget_c,
        online_sample_tau_budget_alpha=args.online_sample_tau_budget_alpha,
        online_sample_tau_clip_min=args.online_sample_tau_clip_min,
        online_sample_tau_clip_max=args.online_sample_tau_clip_max,
        online_sample_tau_k_bar_ema_decay=args.online_sample_tau_k_bar_ema_decay,
        online_sample_tau_chi2_topk=args.online_sample_tau_chi2_topk,
        online_sample_tau_chi2_relative_alpha=args.online_sample_tau_chi2_relative_alpha,
        online_sample_tau_k0_path=args.online_sample_tau_k0_path,
        online_sample_tau_k0_gamma=args.online_sample_tau_k0_gamma,
        online_sample_tau_router_gamma=args.online_sample_tau_router_gamma,
        batch_kl_filter_keep_frac=args.batch_kl_filter_keep_frac,
        batch_kl_filter_level=args.batch_kl_filter_level,
        batch_kl_filter_drop_weight=args.batch_kl_filter_drop_weight,
        effective_tau_metrics_path=args.effective_tau_metrics_path,
        probe_metrics_path=args.probe_metrics_path,
        probe_every_steps=args.probe_every_steps,
        probe_new_path=args.probe_new_path,
        probe_new_max_samples=args.probe_new_max_samples,
        probe_old_every_steps=args.probe_old_every_steps,
        probe_old_max_samples=args.probe_old_max_samples,
        probe_old_topk=args.probe_old_topk,
        vllm_importance_sampling_correction=not args.disable_vllm_importance_sampling_correction,
        num_loss_tokens_to_skip = 3,
        soft_teacher_directions_pt=args.soft_teacher_directions_pt,
        soft_teacher_vector=args.soft_teacher_vector,
        soft_teacher_w=args.soft_teacher_w,
        soft_teacher_row_directions_pt=args.soft_teacher_row_directions_pt,
        soft_teacher_embed_mix_alpha=args.soft_teacher_embed_mix_alpha,
    )
    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    if args.save_only_final:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved final model + tokenizer to {args.output_dir}")
