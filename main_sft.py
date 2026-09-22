"""SFT baseline training entry point (aligned with main.py distillation setup).

Difference from distillation: loss = demo CE on student prompt p_0 (one-hot forward KL),
not on-policy KL(q_o || p_theta).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_config import SFTConfig
from sft_trainer import SFTDataCollator, SFTDataset, SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT baseline trainer (demo CE under p_0).")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, help="Gradient accumulation steps.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="model/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="tooluse",
        choices=["tooluse", "science", "medical"],
    )
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument(
        "--save_only_final",
        action="store_true",
        help="No intermediate checkpoints; save model+tokenizer once after training.",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Do not write any model/tokenizer checkpoints (smoke/debug).",
    )
    parser.add_argument("--save_optimizer_state", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--max_train_samples", type=int, default=0, help=">0: subsample for smoke tests.")
    parser.add_argument("--max_steps", type=int, default=-1, help=">0: override num_train_epochs with fixed steps.")
    parser.add_argument(
        "--loss_type",
        type=str,
        default="sft",
        choices=["sft", "dft"],
        help="sft=token CE; dft=Wu et al. 2025b DFT (CE × sg(π_θ)).",
    )
    return parser.parse_args()


def _golden_from_tool(example: dict) -> str:
    value = example["golden_response"]
    return "\n".join(value) if isinstance(value, list) else str(value)


def load_tooluse_sft_dataset(seed: int = 42) -> Dataset:
    dataset = load_from_disk("data/tooluse_data/train_data")

    def format_example(example, idx):
        return {
            "prompt": [{"role": "user", "content": example["prompt"]}],
            "golden_text": _golden_from_tool(example),
            "soft_teacher_row_id": idx,
        }

    dataset = dataset.map(format_example, with_indices=True, remove_columns=dataset.column_names)
    return dataset.shuffle(seed=seed)


def load_science_sft_dataset(seed: int = 42) -> Dataset:
    dataset = load_from_disk("data/science_data/train_data")

    def format_example(example, idx):
        return {
            "prompt": example["messages"],
            "golden_text": str(example["output_text"]),
            "soft_teacher_row_id": idx,
        }

    dataset = dataset.map(format_example, with_indices=True, remove_columns=dataset.column_names)
    return dataset.shuffle(seed=seed)


def load_medical_sft_dataset(seed: int = 42) -> Dataset:
    dataset = load_from_disk("data/medical_data/train_data")

    def format_example(example, idx):
        return {
            "prompt": example["messages"],
            "golden_text": str(example["output_text"]),
            "soft_teacher_row_id": idx,
        }

    dataset = dataset.map(format_example, with_indices=True, remove_columns=dataset.column_names)
    return dataset.shuffle(seed=seed)


def load_sft_dataset(dataset_name: str, seed: int) -> Dataset:
    if dataset_name == "tooluse":
        dataset = load_tooluse_sft_dataset(seed)
    elif dataset_name == "science":
        dataset = load_science_sft_dataset(seed)
    elif dataset_name == "medical":
        dataset = load_medical_sft_dataset(seed)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    print(f"Loaded {len(dataset)} SFT training examples ({dataset_name})", flush=True)
    return dataset


def main() -> None:
    args = parse_args()
    dataset = load_sft_dataset(args.dataset_name, args.seed)
    if args.max_train_samples > 0:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))
        print(f"Subsampled to {len(dataset)} rows for smoke/debug.", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = SFTConfig(
        seed=args.seed,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_prompts_per_batch,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_loss_tokens_to_skip=3,
        temperature=1.0,
        loss_type=args.loss_type,
        num_train_epochs=args.num_train_epochs,
        save_strategy="no" if (args.save_only_final or args.no_save) else "steps",
        save_steps=args.save_steps,
        save_only_model=not args.save_optimizer_state,
        max_grad_norm=1.0,
        report_to=args.report_to,
        output_dir=args.output_dir,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
    )

    train_rows = [dataset[i] for i in range(len(dataset))]
    train_dataset = SFTDataset(
        train_rows,
        tokenizer,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
    )
    data_collator = SFTDataCollator(
        tokenizer,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    if args.no_save:
        print("Training finished (no_save: skipped checkpoint write).", flush=True)
    elif args.save_only_final:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved final model + tokenizer to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
