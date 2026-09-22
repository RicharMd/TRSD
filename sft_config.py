"""Training config for SFT baseline (aligned with DistilConfig hyperparams where applicable)."""

from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class SFTConfig(TrainingArguments):
    max_prompt_length: int = field(
        default=1024,
        metadata={"help": "Max student prompt tokens (left-truncate overflow)."},
    )
    max_completion_length: int = field(
        default=1024,
        metadata={"help": "Max demo completion tokens."},
    )
    num_loss_tokens_to_skip: int = field(
        default=3,
        metadata={"help": "Skip first N completion tokens in CE (matches distillation)."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Softmax temperature for CE (distillation default 1.0)."},
    )
    # sft: standard token CE. dft: Wu et al. 2025b Dynamic Fine-Tuning —
    # token CE × sg(π_θ(y*_t | ...)) (offline SFT with token-level reweight).
    loss_type: str = field(
        default="sft",
        metadata={"help": "Completion loss: 'sft' or 'dft'."},
    )
