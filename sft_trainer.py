"""SFT baseline trainer: demo teacher-forcing CE under student prompt p_0.

Aligned with DistilTrainer where possible:
  - same student prompt / golden demo tokenization
  - num_loss_tokens_to_skip=3 on completion tokens
  - per-sample token-mean CE, then batch mean
  - no vLLM, no ref_model, no on-policy sampling
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, Trainer

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from train_sft_ce_utils import build_sft_input_ids

from sft_config import SFTConfig


class SFTDataset(Dataset):
    """HF dataset rows must contain `prompt` (messages) and `golden_text`."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_prompt_length: int,
        max_completion_length: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "prompt": row["prompt"],
            "golden_text": row["golden_text"],
            "soft_teacher_row_id": row.get("soft_teacher_row_id", index),
        }


class SFTDataCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_prompt_length: int,
        max_completion_length: int,
        pad_to_multiple_of: int | None = 8,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.pad_to_multiple_of = pad_to_multiple_of
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        packed_rows: list[tuple[torch.Tensor, int]] = []
        for feature in features:
            packed = build_sft_input_ids(
                self.tokenizer,
                feature["prompt"],
                feature["golden_text"],
                max_prompt_length=self.max_prompt_length,
                max_completion_length=self.max_completion_length,
            )
            if packed is None:
                continue
            input_ids, prompt_len, _ = packed
            packed_rows.append((input_ids, prompt_len))

        if not packed_rows:
            raise ValueError("SFT batch has no valid tokenized rows (all empty/truncated).")

        max_len = max(row[0].numel() for row in packed_rows)
        if self.pad_to_multiple_of:
            rem = max_len % self.pad_to_multiple_of
            if rem:
                max_len += self.pad_to_multiple_of - rem

        pad_id = self.tokenizer.pad_token_id
        batch_size = len(packed_rows)
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        prompt_lens = torch.zeros(batch_size, dtype=torch.long)

        for i, (ids, prompt_len) in enumerate(packed_rows):
            seq_len = ids.numel()
            input_ids[i, :seq_len] = ids
            attention_mask[i, :seq_len] = 1
            prompt_lens[i] = prompt_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_lens": prompt_lens,
        }


class SFTTrainer(Trainer):
    def __init__(
        self,
        *args,
        train_dataset: SFTDataset | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        data_collator: SFTDataCollator | None = None,
        **kwargs,
    ) -> None:
        if data_collator is None and processing_class is not None:
            args_cfg = kwargs.get("args")
            if isinstance(args_cfg, SFTConfig):
                data_collator = SFTDataCollator(
                    processing_class,
                    max_prompt_length=args_cfg.max_prompt_length,
                    max_completion_length=args_cfg.max_completion_length,
                )
        super().__init__(
            *args,
            train_dataset=train_dataset,
            processing_class=processing_class,
            data_collator=data_collator,
            **kwargs,
        )
        if not isinstance(self.args, SFTConfig):
            raise TypeError("SFTTrainer requires SFTConfig.")

    @staticmethod
    def _completion_ce_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lens: torch.Tensor,
        *,
        num_loss_tokens_to_skip: int,
        temperature: float,
        loss_type: str = "sft",
    ) -> torch.Tensor:
        logits = logits.float() / temperature
        token_ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        ).view(labels.shape)

        token_index = torch.arange(labels.size(1), device=labels.device).unsqueeze(0) + 1
        prompt_lens = prompt_lens.unsqueeze(1)
        pad_mask = attention_mask[:, 1:] > 0
        completion_mask = token_index >= prompt_lens
        skip_mask = token_index >= (prompt_lens + num_loss_tokens_to_skip)
        loss_mask = pad_mask & completion_mask & skip_mask

        # DFT (Wu et al., 2025b): L = -Σ sg(π_θ(y*_t)) log π_θ(y*_t)
        # ≡ token_ce * detach(p(y*_t)). Keep SFT path unchanged when loss_type=sft.
        if loss_type == "dft":
            probs = torch.softmax(logits, dim=-1)
            token_prob = probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            token_ce = token_ce * token_prob.detach()
        elif loss_type != "sft":
            raise ValueError(f"Unknown loss_type={loss_type!r}; expected 'sft' or 'dft'.")

        per_sample_loss = (token_ce * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
        return per_sample_loss.mean()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )
        logits = outputs.logits[:, :-1, :]
        labels = inputs["input_ids"][:, 1:]
        loss = self._completion_ce_loss(
            logits,
            labels,
            inputs["attention_mask"],
            inputs["prompt_lens"],
            num_loss_tokens_to_skip=self.args.num_loss_tokens_to_skip,
            temperature=self.args.temperature,
            loss_type=getattr(self.args, "loss_type", "sft"),
        )
        return (loss, outputs) if return_outputs else loss
