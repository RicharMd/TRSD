# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import json
import os
import textwrap
from collections import defaultdict, deque
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union

import datasets
import torch
import torch.utils.data
import transformers
from accelerate import logging
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_flash_attn_2_available, is_peft_available, is_rich_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template, prepare_multimodal_messages
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_liger_kernel_available, is_vllm_available
from trl.models import prepare_deepspeed, prepare_fsdp, prepare_peft_model, unwrap_model_for_generation
from trl.models.utils import _ForwardRedirection
from trl.trainer.base_trainer import BaseTrainer
from distil_config import DistilConfig
from accelerate.state import AcceleratorState
from trl.trainer.utils import (
    RepeatSampler,
    disable_dropout_in_model,
    ensure_master_addr_port,
    entropy_from_logits,
    identity,
    nanmax,
    nanmin,
    nanstd,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
    shuffle_sequence_dict,
    split_pixel_values_by_grid,
    split_tensor_dict,
    unsplit_pixel_values_by_grid,
)
from torch.nn.functional import log_softmax, kl_div

# Online tau modes:
#   K_i = mean-token KL(q_o || p_theta)
#   V_i = mean-token Var_p(log(q_o/p))
#   chi2_i = mean-token E_p[(q_o/p - 1)^2]  (reverse closed-loop Step-1 scalar)
#   peer_H,i = mean-token Hellinger Step-1 peer (b - a^2); NOT entropy H_i
#   H_i = mean-token entropy H(p_theta)  (metric key online_sample_tau_h*)
ONLINE_SAMPLE_TAU_KL_MODES = frozenset(
    {
        "budget_forward_kl",
        "budget_forward_kl_exp",
        "budget_forward_kl_relative",
        "budget_forward_kl_relative_exp",
        "budget_forward_kl_relative_exp_range",
        "budget_forward_kl_relative_token",
        "budget_forward_kl_relative_exp_token",
        "budget_forward_kl_relative_exp_range_token",
        "softmax_router_relative",
        "softmax_router_relative_token",
    }
)
ONLINE_SAMPLE_TAU_CHI2_MODES = frozenset(
    {
        "budget_chi2",
        "budget_chi2_exp",
        "budget_chi2_relative",
        "budget_chi2_relative_exp",
        "budget_chi2_relative_exp_range",
        "budget_chi2_relative_token",
        "budget_chi2_relative_exp_token",
        "budget_chi2_relative_exp_range_token",
    }
)
ONLINE_SAMPLE_TAU_HELLINGER_MODES = frozenset(
    {
        "budget_hellinger",
        "budget_hellinger_exp",
        "budget_hellinger_relative",
        "budget_hellinger_relative_exp",
        "budget_hellinger_relative_exp_range",
        "budget_hellinger_relative_token",
        "budget_hellinger_relative_exp_token",
        "budget_hellinger_relative_exp_range_token",
    }
)
ONLINE_SAMPLE_TAU_VAR_MODES = frozenset(
    {
        "budget_log_ratio_var",
        "budget_log_ratio_var_exp",
        "budget_log_ratio_var_relative",
        "budget_log_ratio_var_relative_exp",
        "budget_log_ratio_var_relative_exp_range",
        "budget_log_ratio_var_relative_token",
        "budget_log_ratio_var_relative_exp_token",
        "budget_log_ratio_var_relative_exp_range_token",
    }
)
ONLINE_SAMPLE_TAU_FIXED_C_MODES = frozenset(
    {
        "budget_forward_kl",
        "budget_forward_kl_exp",
        "budget_log_ratio_var",
        "budget_log_ratio_var_exp",
        "budget_chi2",
        "budget_chi2_exp",
        "budget_hellinger",
        "budget_hellinger_exp",
    }
)
ONLINE_SAMPLE_TAU_RELATIVE_BUDGET_SQRT_MODES = frozenset(
    {
        "budget_forward_kl_relative",
        "budget_forward_kl_relative_token",
        "budget_log_ratio_var_relative",
        "budget_log_ratio_var_relative_token",
        "budget_chi2_relative",
        "budget_chi2_relative_token",
        "budget_hellinger_relative",
        "budget_hellinger_relative_token",
    }
)
ONLINE_SAMPLE_TAU_RELATIVE_BUDGET_EXP_MODES = frozenset(
    {
        "budget_forward_kl_relative_exp",
        "budget_forward_kl_relative_exp_token",
        "budget_log_ratio_var_relative_exp",
        "budget_log_ratio_var_relative_exp_token",
        "budget_chi2_relative_exp",
        "budget_chi2_relative_exp_token",
        "budget_hellinger_relative_exp",
        "budget_hellinger_relative_exp_token",
    }
)
ONLINE_SAMPLE_TAU_RELATIVE_BUDGET_EXP_RANGE_MODES = frozenset(
    {
        "budget_forward_kl_relative_exp_range",
        "budget_forward_kl_relative_exp_range_token",
        "budget_log_ratio_var_relative_exp_range",
        "budget_log_ratio_var_relative_exp_range_token",
        "budget_chi2_relative_exp_range",
        "budget_chi2_relative_exp_range_token",
        "budget_hellinger_relative_exp_range",
        "budget_hellinger_relative_exp_range_token",
    }
)
ONLINE_SAMPLE_TAU_RELATIVE_BUDGET_MODES = (
    ONLINE_SAMPLE_TAU_RELATIVE_BUDGET_SQRT_MODES
    | ONLINE_SAMPLE_TAU_RELATIVE_BUDGET_EXP_MODES
    | ONLINE_SAMPLE_TAU_RELATIVE_BUDGET_EXP_RANGE_MODES
)
ONLINE_SAMPLE_TAU_ENTROPY_MODES = frozenset(
    {
        "budget_forward_kl_relative_entropy",
        "budget_forward_kl_relative_entropy_token",
        "softmax_router_relative_entropy",
        "softmax_router_relative_entropy_token",
    }
)
ONLINE_SAMPLE_TAU_ALL_MODES = (
    ONLINE_SAMPLE_TAU_KL_MODES
    | ONLINE_SAMPLE_TAU_VAR_MODES
    | ONLINE_SAMPLE_TAU_CHI2_MODES
    | ONLINE_SAMPLE_TAU_HELLINGER_MODES
    | ONLINE_SAMPLE_TAU_ENTROPY_MODES
)
ONLINE_SAMPLE_TAU_PEER_CACHE_MODES = frozenset(
    mode for mode in ONLINE_SAMPLE_TAU_ALL_MODES if mode not in ONLINE_SAMPLE_TAU_FIXED_C_MODES
)


def _prepare_distillation_log_probs(topk_log_probs: torch.Tensor, add_tail: bool) -> torch.Tensor:
    """
    Build a proper discrete distribution over K or K+1 buckets from per-token top-K log-probs
    (each log p_i is under the full softmax), aligned with verl `compute_self_distillation_loss`.
    Shape: [..., K] -> [..., K] or [..., K+1].
    """
    if add_tail:
        log_s = torch.logsumexp(topk_log_probs, dim=-1, keepdim=True)
        log_s = torch.clamp(log_s, max=-1e-7)
        tail_log = torch.log(-torch.expm1(log_s))
        return torch.cat([topk_log_probs, tail_log], dim=-1)
    log_z = torch.logsumexp(topk_log_probs, dim=-1, keepdim=True)
    return topk_log_probs - log_z


def _pearson_chi_square_from_probs(
    policy_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    chi2_relative_alpha: float = 0.0,
) -> torch.Tensor:
    """chi^2(p; q) peer on bucket probabilities that sum to 1.

    Bare (chi2_relative_alpha <= 0):
        E_p[(q/p - 1)^2]
    Relative Pearson / rPE (0 < chi2_relative_alpha < 1), direction-matched to χ²(q||p):
        mix = a*q + (1-a)*p,  r = q/mix (<= 1/a),  E_p[(r - 1)^2]
        a -> 0 recovers bare chi^2. This a is NOT budget alpha.
    """
    a = float(chi2_relative_alpha)
    if a <= 0.0:
        ratio = teacher_probs / policy_probs.clamp(min=1e-12)
        return (policy_probs * (ratio - 1.0).pow(2)).sum(dim=-1)
    if a >= 1.0:
        raise ValueError(f"chi2_relative_alpha must be in [0, 1), got {a}")
    mix = a * teacher_probs + (1.0 - a) * policy_probs
    r = teacher_probs / mix.clamp(min=1e-12)
    return (policy_probs * (r - 1.0).pow(2)).sum(dim=-1)


def per_token_pearson_chi_square(
    policy_logps: torch.Tensor,
    teacher_logps: torch.Tensor,
    chi2_topk: int = 64,
    chi2_relative_alpha: float = 0.0,
) -> torch.Tensor:
    """
    Per-position Pearson chi^2(p; q_o) peer signal.

    Data flow (unchanged order):
      1) optional top-K(p) + tail bucket (chi2_topk > 0)
      2) chi^2 / relative-Pearson on the resulting simplex buckets

    Set chi2_topk <= 0 for full vocabulary. Set chi2_relative_alpha in (0, 1) for rPE
    (trainer default is 0.1); 0 keeps bare E_p[(q/p-1)^2].
    """
    policy_probs = policy_logps.float().exp()
    teacher_probs = teacher_logps.float().exp()
    vocab = policy_probs.size(-1)
    if chi2_topk <= 0 or chi2_topk >= vocab:
        return _pearson_chi_square_from_probs(
            policy_probs, teacher_probs, chi2_relative_alpha=chi2_relative_alpha
        )

    k = min(int(chi2_topk), vocab)
    topk_probs, topk_idx = torch.topk(policy_probs, k, dim=-1)
    topk_teacher = torch.gather(teacher_probs, dim=-1, index=topk_idx)
    p_tail = (1.0 - topk_probs.sum(dim=-1)).clamp(min=0.0)
    q_tail = (1.0 - topk_teacher.sum(dim=-1)).clamp(min=0.0)
    policy_buckets = torch.cat([topk_probs, p_tail.unsqueeze(-1)], dim=-1)
    teacher_buckets = torch.cat([topk_teacher, q_tail.unsqueeze(-1)], dim=-1)
    return _pearson_chi_square_from_probs(
        policy_buckets, teacher_buckets, chi2_relative_alpha=chi2_relative_alpha
    )


if is_peft_available():
    from peft import PeftConfig, PeftModel

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb


logger = logging.get_logger(__name__)


def load_k0_kl_map(path: str, require_status_ok: bool = True) -> dict[int, float]:
    """Load per-dataset-index K_{i,0} = kl_qo_to_p0 from compute_tool_train_k0_rollout_kl.py."""
    mapping: dict[int, float] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if require_status_ok and row.get("status") != "ok":
                continue
            kl_value = row.get("kl_qo_to_p0")
            if kl_value is None:
                continue
            mapping[int(row["index"])] = float(kl_value)
    if not mapping:
        raise ValueError(f"No valid kl_qo_to_p0 rows found in {path}")
    return mapping


class MemoryEfficientSyncRefModelCallback(TrainerCallback):
    """
    Memory-efficient callback to synchronize the model with a reference model.
    
    Unlike the default SyncRefModelCallback, this version iterates through parameters
    one at a time instead of gathering all parameters at once. This reduces peak memory
    usage from O(full_model_size) to O(single_param_size), making it feasible to sync
    large models with DeepSpeed ZeRO-3.
    """

    def __init__(
        self,
        ref_model: Union[PreTrainedModel, nn.Module],
        accelerator: Optional[Any],
    ):
        self.accelerator = accelerator
        self.ref_model = ref_model

    @staticmethod
    def _sync_param(model_param, ref_param, alpha):
        """Sync a single parameter: ref = alpha * model + (1 - alpha) * ref"""
        ref_param.data.mul_(1.0 - alpha).add_(model_param.data, alpha=alpha)

    @staticmethod
    def sync_target_model_memory_efficient(model, target_model, alpha):
        """
        Sync target_model to track model, gathering one parameter at a time.
        
        This is O(1) in memory overhead instead of O(N) where N is model size.
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin
        is_zero3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        
        if is_zero3:
            import deepspeed
            
            # Iterate through parameters one at a time
            for (name, model_param), (_, ref_param) in zip(
                model.named_parameters(), target_model.named_parameters()
            ):
                # Gather only this pair of parameters
                with deepspeed.zero.GatheredParameters(
                    [model_param, ref_param], modifier_rank=0
                ):
                    if deepspeed.comm.get_rank() == 0:
                        MemoryEfficientSyncRefModelCallback._sync_param(
                            model_param, ref_param, alpha
                        )
        else:
            # Non-ZeRO-3: just iterate normally
            for model_param, ref_param in zip(model.parameters(), target_model.parameters()):
                MemoryEfficientSyncRefModelCallback._sync_param(model_param, ref_param, alpha)

    def on_step_end(self, args, state, control, **kwargs):
        model: PreTrainedModel = kwargs["model"]

        if self.ref_model is not None and state.global_step % args.ref_model_sync_steps == 0:
            if self.accelerator:
                model = self.accelerator.unwrap_model(model)
            self.sync_target_model_memory_efficient(model, self.ref_model, args.ref_model_mixup_alpha)

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class DistilTrainer(BaseTrainer):
    """
    Trainer for the Self-Distillation method. 

    Example:

    ```python
    from datasets import load_dataset
    from trl import DistilTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")


    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]


    trainer = DistilTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. Custom reward
                  functions can also return `None` when the reward is not applicable to those samples. This is useful
                  for multi-task training where different reward functions apply to different types of samples. When a
                  reward function returns `None` for a sample, that reward function is excluded from the reward
                  calculation for that sample. For more details, see [Using a custom reward
                  function](#using-a-custom-reward-function).

                  The trainer's state is also passed to the reward function. The trainer's state is an instance of
                  [`~transformers.TrainerState`] and can be accessed by accessing the `trainer_state` argument to the
                  reward function's signature.
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`DistilConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "distil"]
    _name = "Distil"

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        ref_model: Union[str, PreTrainedModel],
        args: Optional[DistilConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = DistilConfig(f"{model_name}-Distil")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            dtype = model_init_kwargs.get("dtype")
            if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
                pass  # dtype is already a torch.dtype or "auto" or None
            elif isinstance(dtype, str):  # it's a str, but not "auto"
                dtype = getattr(torch, dtype)
                model_init_kwargs["dtype"] = dtype
            else:
                raise ValueError(
                    "Invalid `dtype` passed to `DistilConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            model = architecture.from_pretrained(model_id, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `DistilConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        # Some models (SmolVLM/Idefics3) don't support `logits_to_keep` argument and error out if we pass it
        # Inspect the forward method before we wrap the model with PEFT
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        if peft_config is not None or (is_peft_available() and isinstance(model, PeftModel)):
            model = prepare_peft_model(model, peft_config, args)

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model.config._name_or_path, truncation_side="left")

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_transformers_paged = args.use_transformers_paged
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.vllm_importance_sampling_correction = args.vllm_importance_sampling_correction
        self.vllm_importance_sampling_cap = args.vllm_importance_sampling_cap
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.importance_sampling_level = args.importance_sampling_level
        self.mask_truncated_completions = args.mask_truncated_completions
        self.top_entropy_quantile = args.top_entropy_quantile
        self.num_loss_tokens_to_skip = args.num_loss_tokens_to_skip

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in DistilTrainer. Please use a standard dataset instead."
            )

        # Multi-step
        self.num_iterations = args.num_iterations
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO-like algorithms, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in Distil
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            # In Trainer, `training_step` scales the loss by `gradient_accumulation_steps` only if `compute_loss_func`
            # is None. For DAPO, loss scaling instead depends on the total number of completions tokens across the
            # global accumulated batch. To control scaling ourselves, we must disable Trainer’s built-in scaling. The
            # simplest (though a bit hacky) way is to set `compute_loss_func` to any non-None value, which bypasses
            # that behavior without rewriting `training_step`.
            compute_loss_func="non-None value to disable scaling",
        )

        # Reference model
        self.beta = args.beta
        self.alpha = args.alpha
        self.proximal_teacher_tau = float(getattr(args, "proximal_teacher_tau", 0.0) or 0.0)
        if not 0.0 <= self.proximal_teacher_tau <= 1.0:
            raise ValueError(f"proximal_teacher_tau must be in [0, 1], got {self.proximal_teacher_tau}")
        self.distillation_divergence = str(getattr(args, "distillation_divergence", "kl") or "kl")
        if self.distillation_divergence not in {"kl", "hellinger"}:
            raise ValueError(
                f"distillation_divergence must be 'kl' or 'hellinger', got {self.distillation_divergence!r}"
            )
        self.proximal_selection = str(getattr(args, "proximal_selection", "auto") or "auto")
        if self.proximal_selection not in {"auto", "forward_kl", "reverse_kl", "hellinger"}:
            raise ValueError(
                "proximal_selection must be one of auto|forward_kl|reverse_kl|hellinger, got "
                f"{self.proximal_selection!r}"
            )
        self.sample_tau_map_path = getattr(args, "sample_tau_map_path", None)
        self.uses_sample_tau = bool(self.sample_tau_map_path)
        self.online_sample_tau_mode = getattr(args, "online_sample_tau_mode", None)
        self.uses_online_sample_tau = self.online_sample_tau_mode is not None
        self.online_sample_tau_budget_c = float(getattr(args, "online_sample_tau_budget_c", 0.0) or 0.0)
        self.online_sample_tau_budget_alpha = float(getattr(args, "online_sample_tau_budget_alpha", 0.0) or 0.0)
        self.online_sample_tau_clip_min = float(getattr(args, "online_sample_tau_clip_min", 0.0) or 0.0)
        self.online_sample_tau_clip_max = float(getattr(args, "online_sample_tau_clip_max", 1.0) or 1.0)
        self.online_sample_tau_k_bar_ema_decay = float(
            getattr(args, "online_sample_tau_k_bar_ema_decay", 0.99) or 0.0
        )
        self._online_sample_tau_k_bar_ema: Optional[float] = None
        self.online_sample_tau_k0_path = getattr(args, "online_sample_tau_k0_path", None)
        self.online_sample_tau_k0_gamma = float(getattr(args, "online_sample_tau_k0_gamma", 1.0) or 1.0)
        self.online_sample_tau_router_gamma = float(
            getattr(args, "online_sample_tau_router_gamma", 1.0) or 1.0
        )
        self.k0_kl_map: Optional[dict[int, float]] = None
        self.k0_kl_global_mean: Optional[float] = None
        batch_kl_filter_keep_frac = getattr(args, "batch_kl_filter_keep_frac", None)
        self.batch_kl_filter_keep_frac = (
            float(batch_kl_filter_keep_frac) if batch_kl_filter_keep_frac is not None else None
        )
        self.batch_kl_filter_level = getattr(args, "batch_kl_filter_level", "sample") or "sample"
        self.batch_kl_filter_drop_weight = float(getattr(args, "batch_kl_filter_drop_weight", 0.0) or 0.0)
        if self.online_sample_tau_mode not in (None, *ONLINE_SAMPLE_TAU_ALL_MODES):
            raise ValueError(f"Unsupported online_sample_tau_mode={self.online_sample_tau_mode!r}")
        if self.online_sample_tau_k0_path and self.online_sample_tau_mode not in {
            "budget_forward_kl_relative",
            "budget_forward_kl_relative_exp",
            "budget_forward_kl_relative_exp_range",
            "budget_log_ratio_var_relative",
            "budget_log_ratio_var_relative_exp",
            "budget_log_ratio_var_relative_exp_range",
            "budget_chi2_relative",
            "budget_chi2_relative_exp",
            "budget_chi2_relative_exp_range",
        }:
            raise ValueError(
                "online_sample_tau_k0_path requires online_sample_tau_mode="
                "budget_forward_kl_relative, budget_forward_kl_relative_exp, "
                "budget_forward_kl_relative_exp_range, budget_log_ratio_var_relative, "
                "budget_log_ratio_var_relative_exp, budget_log_ratio_var_relative_exp_range, "
                "budget_chi2_relative, budget_chi2_relative_exp, or budget_chi2_relative_exp_range."
            )
        if self.online_sample_tau_k0_path:
            self.k0_kl_map = load_k0_kl_map(self.online_sample_tau_k0_path)
            self.k0_kl_global_mean = sum(self.k0_kl_map.values()) / len(self.k0_kl_map)
            logger.info(
                "Loaded %d offline K0 anchors from %s (global Kbar_0=%.6f, k0_gamma=%.4f)",
                len(self.k0_kl_map),
                self.online_sample_tau_k0_path,
                self.k0_kl_global_mean,
                self.online_sample_tau_k0_gamma,
            )
        if self.uses_sample_tau and self.uses_online_sample_tau:
            raise ValueError("Use either sample_tau_map_path or online_sample_tau_mode, not both.")
        if self.uses_online_sample_tau and self.proximal_teacher_tau != 0.0:
            raise ValueError("Use either proximal_teacher_tau or online_sample_tau_mode, not both.")
        if self.uses_online_sample_tau:
            if (
                self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_FIXED_C_MODES
                and self.online_sample_tau_budget_c <= 0.0
            ):
                raise ValueError(
                    "online_sample_tau_budget_c must be > 0 for fixed-C online tau modes "
                    "(budget_forward_kl*, budget_log_ratio_var*)."
                )
            if (
                self.online_sample_tau_mode
                in (
                    ONLINE_SAMPLE_TAU_PEER_CACHE_MODES
                    - {"budget_forward_kl"}
                )
                and self.online_sample_tau_budget_alpha <= 0.0
            ):
                raise ValueError(
                    "online_sample_tau_budget_alpha must be > 0 for relative / router / entropy online tau modes."
                )
            if self.online_sample_tau_mode in {
                "softmax_router_relative",
                "softmax_router_relative_token",
                "softmax_router_relative_entropy",
                "softmax_router_relative_entropy_token",
            } and self.online_sample_tau_router_gamma <= 0.0:
                raise ValueError(
                    "online_sample_tau_router_gamma must be > 0 for softmax_router_relative modes."
                )
            if not 0.0 <= self.online_sample_tau_clip_min <= self.online_sample_tau_clip_max <= 1.0:
                raise ValueError(
                    "online_sample_tau_clip_min/max must satisfy 0 <= min <= max <= 1, got "
                    f"{self.online_sample_tau_clip_min}, {self.online_sample_tau_clip_max}"
                )
            if self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_PEER_CACHE_MODES:
                if self.online_sample_tau_k_bar_ema_decay < 0.0:
                    raise ValueError(
                        "online_sample_tau_k_bar_ema_decay must be >= 0, got "
                        f"{self.online_sample_tau_k_bar_ema_decay}"
                    )
                if self.online_sample_tau_k_bar_ema_decay >= 1.0:
                    raise ValueError(
                        "online_sample_tau_k_bar_ema_decay must be < 1 when EMA is enabled, got "
                        f"{self.online_sample_tau_k_bar_ema_decay}"
                    )
        if self.uses_sample_tau and self.alpha != 0.0:
            raise ValueError(
                "sample_tau_map_path currently supports alpha=0 forward-KL training only."
            )
        if self.uses_online_sample_tau:
            if self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_CHI2_MODES:
                if self.alpha < 1.0 - 1e-9:
                    raise ValueError(
                        "budget_chi2* requires alpha=1 (reverse-KL arithmetic teacher; "
                        "closed-loop KL(p||q) budget with chi-square peer signal)."
                    )
                if self.proximal_selection not in {"auto", "reverse_kl"}:
                    raise ValueError("budget_chi2* requires proximal_selection=auto|reverse_kl.")
                if self.distillation_divergence != "kl":
                    raise ValueError("budget_chi2* requires distillation_divergence=kl.")
            elif self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_HELLINGER_MODES:
                if self.proximal_selection not in {"auto", "hellinger"}:
                    raise ValueError(
                        "budget_hellinger* requires proximal_selection=auto|hellinger "
                        "(√-mix proximal path + peer_H budget)."
                    )
                if self.distillation_divergence != "hellinger":
                    raise ValueError(
                        "budget_hellinger* requires distillation_divergence=hellinger "
                        "(closed-loop: D_s and D_f both Squared Hellinger)."
                    )
                # Force Hellinger selection even when proximal_selection=auto.
                self.proximal_selection = "hellinger"
            elif self.alpha != 0.0:
                raise ValueError(
                    "online_sample_tau_mode currently supports alpha=0 forward-KL training only "
                    "(except budget_chi2* which requires alpha=1, "
                    "and budget_hellinger* which uses distillation_divergence=hellinger)."
                )
            elif self.proximal_selection not in {"auto", "forward_kl"}:
                raise ValueError(
                    "forward/var online tau modes require proximal_selection=auto|forward_kl."
                )
        if self.proximal_teacher_tau != 0.0:
            sel = self._resolve_proximal_selection()
            if sel == "hellinger":
                if self.distillation_divergence != "hellinger":
                    raise ValueError(
                        "proximal_selection=hellinger with nonzero tau requires "
                        "distillation_divergence=hellinger."
                    )
            elif self.alpha not in (0.0, 1.0):
                raise ValueError(
                    "dataset-level proximal_teacher_tau with KL selection supports "
                    "alpha=0 (forward) and alpha=1 (reverse) only."
                )
        if self.distillation_divergence == "hellinger" and self.alpha not in (0.0, 1.0):
            # Hellinger loss ignores JS-style alpha mixes; keep alpha at an endpoint for clarity.
            raise ValueError(
                "distillation_divergence=hellinger ignores --alpha mixes; set --alpha 0 or 1 "
                "(unused for the Hellinger loss itself)."
            )
        self.distillation_topk = getattr(args, "distillation_topk", None)
        self.distillation_add_tail = getattr(args, "distillation_add_tail", True)
        if (
            self.proximal_teacher_tau != 0.0
            or self.uses_sample_tau
            or self.uses_online_sample_tau
        ) and self.distillation_topk is not None:
            raise ValueError("proximal teacher tau currently requires full-vocabulary distillation_topk=None.")
        if self.batch_kl_filter_keep_frac is not None:
            if not 0.0 < self.batch_kl_filter_keep_frac <= 1.0:
                raise ValueError(
                    f"batch_kl_filter_keep_frac must be in (0, 1], got {self.batch_kl_filter_keep_frac}"
                )
            if self.alpha != 0.0:
                raise ValueError("batch_kl_filter_keep_frac requires forward KL (--alpha 0).")
            if self.proximal_teacher_tau != 0.0:
                raise ValueError("batch_kl_filter_keep_frac requires proximal_teacher_tau=0.")
            if self.uses_online_sample_tau:
                raise ValueError("batch_kl_filter_keep_frac is incompatible with online_sample_tau_mode.")
            if self.uses_sample_tau:
                raise ValueError("batch_kl_filter_keep_frac is incompatible with sample_tau_map_path.")
            if self.distillation_topk is not None and self.distillation_topk > 0:
                raise ValueError(
                    "batch_kl_filter_keep_frac requires full-vocabulary distillation (no distillation_topk)."
                )
            if self.batch_kl_filter_level not in {"sample", "token"}:
                raise ValueError(
                    f"batch_kl_filter_level must be 'sample' or 'token', got {self.batch_kl_filter_level!r}"
                )
            if not 0.0 <= self.batch_kl_filter_drop_weight <= 1.0:
                raise ValueError(
                    f"batch_kl_filter_drop_weight must be in [0, 1], got {self.batch_kl_filter_drop_weight}"
                )
            drop_desc = (
                "hard mask (weight 0)"
                if self.batch_kl_filter_drop_weight == 0.0
                else f"soft weight {self.batch_kl_filter_drop_weight:.4g} on filtered units"
            )
            if self.batch_kl_filter_level == "sample":
                logger.info(
                    "Batch KL filter (sample): keep lowest %.1f%% of rows by mean-token KL(q_o||p) "
                    "within each generation batch; dropped rows use %s.",
                    100.0 * self.batch_kl_filter_keep_frac,
                    drop_desc,
                )
            else:
                logger.info(
                    "Batch KL filter (token): keep lowest %.1f%% of loss tokens by per-token KL(q_o||p) "
                    "within each sequence (cached on generation batch); dropped tokens use %s.",
                    100.0 * self.batch_kl_filter_keep_frac,
                    drop_desc,
                )
        probe_every_steps = int(getattr(args, "probe_every_steps", 0) or 0)
        old_every_steps = int(getattr(args, "probe_old_every_steps", 0) or 0)
        self.probe_every_steps = probe_every_steps or old_every_steps
        self.probe_new_path = getattr(args, "probe_new_path", "data/tooluse_data/eval_data")
        self.probe_new_max_samples = int(getattr(args, "probe_new_max_samples", 32) or 32)
        self.probe_old_max_samples = int(getattr(args, "probe_old_max_samples", 32) or 32)
        self.probe_old_topk = int(getattr(args, "probe_old_topk", 64) or 64)
        self.online_sample_tau_chi2_topk = int(getattr(args, "online_sample_tau_chi2_topk", 64))
        # RuLSIF-style relativity for chi2 peer (default 0.1). Distinct from budget alpha / --alpha.
        self.online_sample_tau_chi2_relative_alpha = float(
            getattr(args, "online_sample_tau_chi2_relative_alpha", 0.1) or 0.0
        )
        if not 0.0 <= self.online_sample_tau_chi2_relative_alpha < 1.0:
            raise ValueError(
                "online_sample_tau_chi2_relative_alpha must be in [0, 1), got "
                f"{self.online_sample_tau_chi2_relative_alpha}"
            )
        probe_metrics_path = getattr(args, "probe_metrics_path", None)
        self.probe_metrics_path = probe_metrics_path or os.path.join(args.output_dir, "probe_metrics.jsonl")
        sample_metrics_path = getattr(args, "sample_metrics_path", None)
        self.sample_metrics_path = sample_metrics_path
        self.effective_tau_metrics_path = getattr(args, "effective_tau_metrics_path", None)
        self._new_probe_cache: Optional[list[tuple[str, str]]] = None
        self._old_probe_cache: Optional[dict[str, torch.Tensor]] = None
        self._last_probe_step: Optional[int] = None
        self.generate_from_teacher = args.generate_from_teacher
        self.soft_teacher_w = float(getattr(args, "soft_teacher_w", 0.0) or 0.0)
        self.soft_teacher_embed_mix_alpha = getattr(args, "soft_teacher_embed_mix_alpha", None)
        if self.soft_teacher_embed_mix_alpha is not None:
            self.soft_teacher_embed_mix_alpha = float(self.soft_teacher_embed_mix_alpha)
            if not 0.0 <= self.soft_teacher_embed_mix_alpha <= 1.0:
                raise ValueError(
                    "soft_teacher_embed_mix_alpha must be in [0, 1], "
                    f"got {self.soft_teacher_embed_mix_alpha}"
                )
        self.soft_teacher_v: Optional[torch.Tensor] = None
        self.soft_teacher_row_directions: Optional[torch.Tensor] = None
        row_soft_pt = getattr(args, "soft_teacher_row_directions_pt", None)
        if row_soft_pt:
            if self.soft_teacher_w == 0.0:
                logger.warning(
                    "soft_teacher_row_directions_pt is set but soft_teacher_w=0; no embedding offset applied."
                )
            else:
                blob = torch.load(row_soft_pt, map_location="cpu", weights_only=False)
                row_directions = blob.get("row_directions")
                if row_directions is None:
                    raise KeyError(f"{row_soft_pt} does not contain row_directions")
                self.soft_teacher_row_directions = row_directions.float().cpu()
                logger.info(
                    "Soft teacher prompt: row_directions shape=%s w=%s from %s",
                    tuple(self.soft_teacher_row_directions.shape),
                    self.soft_teacher_w,
                    row_soft_pt,
                )
        soft_pt = getattr(args, "soft_teacher_directions_pt", None)
        soft_vec = getattr(args, "soft_teacher_vector", None)
        if soft_pt and soft_vec and self.soft_teacher_row_directions is None:
            if self.soft_teacher_w == 0.0:
                logger.warning(
                    "soft_teacher_directions_pt and soft_teacher_vector are set but soft_teacher_w=0; "
                    "no embedding offset applied."
                )
            else:
                blob = torch.load(soft_pt, map_location="cpu", weights_only=True)
                vectors = blob.get("vectors", blob)
                if soft_vec not in vectors:
                    raise KeyError(
                        f"soft_teacher_vector={soft_vec!r} not in {soft_pt}; "
                        f"have {sorted(vectors.keys())}"
                    )
                self.soft_teacher_v = vectors[soft_vec].float().cpu()
                logger.info(
                    "Soft teacher prompt: vector=%s w=%s from %s (prompt tokens only)",
                    soft_vec,
                    self.soft_teacher_w,
                    soft_pt,
                )
        elif (soft_pt or soft_vec) and self.soft_teacher_w != 0.0 and self.soft_teacher_row_directions is None:
            raise ValueError(
                "soft_teacher_w is non-zero but soft_teacher_directions_pt or soft_teacher_vector is missing."
            )
        if ref_model is not None:
            # If a reference model is provided, use it
            self.ref_model = ref_model
        elif self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # For deepspeed, fsdp or non-distributed models, create a reference model from scratch
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            self.ref_model = architecture.from_pretrained(model_id, **model_init_kwargs)

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # Keep logs sized to the generation batch to record only outputs from the latest model update.
        self._logs = {
            "images": deque(maxlen=args.generation_batch_size),
            "prompt": deque(maxlen=args.generation_batch_size),
            "completion": deque(maxlen=args.generation_batch_size),
            "rewards": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            "advantages": deque(maxlen=args.generation_batch_size),
        }

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install trl[vllm]` to use it."
                )

            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    if args.vllm_server_base_url is not None:
                        base_url = args.vllm_server_base_url
                    else:
                        base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                    self.vllm_client = VLLMClient(base_url=base_url, connection_timeout=args.vllm_server_timeout)
                    self.vllm_client.init_communicator(device=torch.cuda.current_device())

            elif self.vllm_mode == "colocate":
                # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have
                # the same number of ranks
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                # Ensure distributed rendezvous variables are set without colliding across concurrent runs
                ensure_master_addr_port()

                if self.max_prompt_length is not None and self.max_completion_length is not None:
                    max_model_len = self.max_prompt_length + self.max_completion_length
                else:
                    max_model_len = None
                # Use teacher model for vLLM when generate_from_teacher=True
                vllm_model_path = ref_model.name_or_path if self.generate_from_teacher and ref_model is not None else model.name_or_path
                logger.info(f"[DEBUG] Initializing vLLM with model: {vllm_model_path}, generate_from_teacher={self.generate_from_teacher}")
                self.llm = LLM(
                    model=vllm_model_path,
                    trust_remote_code=True,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.vllm_tensor_parallel_size
                    * self.args.steps_per_generation,
                    max_model_len=max_model_len,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768) - thinking there's not enough memory
                    max_num_batched_tokens=4096,
                    model_impl=self.args.vllm_model_impl,
                    enable_sleep_mode=self.args.vllm_enable_sleep_mode,
                    # Important so temperature scaling/logit tweaking affects the TIS log probs
                    logprobs_mode="processed_logprobs",
                )
                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)
            else:
                raise ValueError(f"vllm_mode must be either 'server' or 'colocate', got '{self.vllm_mode}'.")

            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            generation_kwargs = {
                "max_new_tokens": self.max_completion_length,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "cache_implementation": args.cache_implementation,
            }
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            self.generation_config = GenerationConfig(**generation_kwargs)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(MemoryEfficientSyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In DistilTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt",
                "teacher_prompt",
                "soft_teacher_row_id",
                "sample_tau",
                "sample_avg64",
                "sample_uncertainty",
                "image",
                "images",
            ]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification. As a result, some parts of the method aren't relevant to Distil, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )

            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    @profiling_decorator
    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
    ):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model

        # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # For Qwen models:
        if image_grid_thw is not None and pixel_values is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        # For Gemma, SmolVLM2, LLaVa-Next etc.:
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        # For SmolVLM2
        if pixel_attention_mask is not None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask
        # For LLaVa-Next
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes

        # Only add logits_to_keep if the model supports it
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

        last_hidden_state = unwrapped_model.model(**model_inputs).last_hidden_state
        # Exclude the last value: it corresponds to the next token pred
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
        last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    def get_high_entropy_mask(self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.

        Args:
            entropies (`torch.Tensor`):
                Tensor of shape (batch_size, seq_len) with per-token entropy values.
            mask (`torch.Tensor`):
                Binary mask of the same shape as `entropies`, where `1` indicates valid tokens and `0` padding.
            threshold (`float`):
                Quantile threshold between `0.0` and `1.0` to select high-entropy tokens.

        Returns:
            `torch.Tensor`:
                Boolean mask of shape (batch_size, seq_len), where `True` indicates tokens with entropy >= threshold
                and `False` otherwise.
        """
        local = entropies[mask.bool()].float()

        # Use a negative pad_value as a sentinel because entropy values are always >= 0.
        # This guarantees that the sentinel cannot collide with any real entropy value.
        pad_value = -1e9

        # Pad across processes so that every rank has the same tensor length
        padded = self.accelerator.pad_across_processes(local, dim=0, pad_index=pad_value)
        gathered = self.accelerator.gather(padded)

        # Drop sentinel values (safe because no entropy can be negative)
        gathered = gathered[gathered != pad_value]

        if gathered.numel() == 0:
            return torch.zeros_like(entropies, dtype=torch.bool)

        entropy_threshold = torch.quantile(gathered, threshold)
        masked_entropies = entropies * mask.float()
        entropy_mask = masked_entropies >= entropy_threshold
        return entropy_mask & mask.bool()  # ensure padding tokens are always masked out

    @profiling_decorator
    def _build_forward_model_inputs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        embed_perturb_mask: Optional[torch.Tensor] = None,
        embed_perturb_vectors: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Build model inputs; optionally add w*v on inputs_embeds where embed_perturb_mask==1."""
        if embed_perturb_mask is None or self.soft_teacher_w == 0.0:
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        if embed_perturb_vectors is None and self.soft_teacher_v is None:
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        unwrapped = self.accelerator.unwrap_model(model)
        emb_layer = unwrapped.get_input_embeddings()
        emb = emb_layer(input_ids)
        emb_f = emb.float()
        if embed_perturb_vectors is not None:
            v = embed_perturb_vectors.to(device=emb_f.device, dtype=emb_f.dtype).unsqueeze(1)
        else:
            v = self.soft_teacher_v.to(device=emb_f.device, dtype=emb_f.dtype).view(1, 1, -1)
        mask = embed_perturb_mask.to(device=emb_f.device, dtype=emb_f.dtype).unsqueeze(-1)
        if self.soft_teacher_embed_mix_alpha is not None:
            alpha = float(self.soft_teacher_embed_mix_alpha)
            mixed = alpha * emb_f + (1.0 - alpha) * v
            emb_in = (emb_f * (1.0 - mask) + mixed * mask).to(dtype=emb.dtype)
        else:
            emb_in = (emb_f + float(self.soft_teacher_w) * v * mask).to(dtype=emb.dtype)
        return {"inputs_embeds": emb_in, "attention_mask": attention_mask}

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
        compute_all_logps=True,
        return_energy=False,
        embed_perturb_mask: Optional[torch.Tensor] = None,
        embed_perturb_vectors: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Compute log-probs and (optionally) entropies for each token."""
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_selected_logps = []
        all_logps = []
        all_entropies = []
        all_negative_energies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            perturb_batch = None
            perturb_vectors_batch = None
            if embed_perturb_mask is not None:
                perturb_batch = embed_perturb_mask[start : start + batch_size]
            if embed_perturb_vectors is not None:
                perturb_vectors_batch = embed_perturb_vectors[start : start + batch_size]
            model_inputs = self._build_forward_model_inputs(
                model, input_ids_batch, attention_mask_batch, perturb_batch, perturb_vectors_batch
            )
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]

            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

            logits = model(**model_inputs).logits
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            if return_energy:
                all_negative_energies.append(torch.logsumexp(logits.float(), dim=-1))
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            selected_logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            if compute_all_logps:
                logps = log_softmax(logits, dim=-1)
            else:
                logps = None
            all_selected_logps.append(selected_logps)
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        selected_logps = torch.cat(all_selected_logps, dim=0)
        if compute_all_logps:
            logps = torch.cat(all_logps, dim=0)
        else:
            logps = None
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        negative_energies = torch.cat(all_negative_energies, dim=0) if return_energy else None
        if return_energy:
            return selected_logps, logps, entropies, negative_energies
        return selected_logps, logps, entropies

    @torch.no_grad()
    def _get_teacher_topk_distillation_tensors(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        distill_topk: int,
        batch_size=None,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
        embed_perturb_mask: Optional[torch.Tensor] = None,
        embed_perturb_vectors: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-only forward: top-K log p, indices, and sampled-token log p (no full-vocab log_softmax)."""
        batch_size = batch_size or input_ids.size(0)
        chunks_logps: list[torch.Tensor] = []
        chunks_idx: list[torch.Tensor] = []
        chunks_teacher_sel: list[torch.Tensor] = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            perturb_batch = None
            perturb_vectors_batch = None
            if embed_perturb_mask is not None:
                perturb_batch = embed_perturb_mask[start : start + batch_size]
            if embed_perturb_vectors is not None:
                perturb_vectors_batch = embed_perturb_vectors[start : start + batch_size]
            model_inputs = self._build_forward_model_inputs(
                model, input_ids_batch, attention_mask_batch, perturb_batch, perturb_vectors_batch
            )
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]

            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False

            logits = model(**model_inputs).logits
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.temperature

            vocab = logits.size(-1)
            k = min(int(distill_topk), vocab)
            topk_vals, topk_idx = torch.topk(logits, k, dim=-1)
            log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
            teacher_topk_logps = topk_vals - log_z
            completion_ids = input_ids_batch[:, -logits_to_keep:]
            teacher_selected_logps = selective_log_softmax(logits, completion_ids)
            chunks_logps.append(teacher_topk_logps)
            chunks_idx.append(topk_idx)
            chunks_teacher_sel.append(teacher_selected_logps)

        return (
            torch.cat(chunks_logps, dim=0),
            torch.cat(chunks_idx, dim=0),
            torch.cat(chunks_teacher_sel, dim=0),
        )

    def _get_student_logps_entropy_distill_topk(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        teacher_topk_indices: torch.Tensor,
        batch_size=None,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Student forward: sampled-token logps, entropy, and log p on teacher's top-K vocab ids."""
        batch_size = batch_size or input_ids.size(0)
        assert teacher_topk_indices.size(0) == input_ids.size(0)
        all_selected_logps: list[torch.Tensor] = []
        all_student_topk: list[torch.Tensor] = []
        all_entropies: list[torch.Tensor] = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]

            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False

            logits = model(**model_inputs).logits
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            selected_logps = selective_log_softmax(logits, completion_ids)

            idx_batch = teacher_topk_indices[start : start + batch_size].to(device=logits.device, dtype=torch.long)
            log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
            gathered = torch.gather(logits, dim=-1, index=idx_batch)
            student_topk_logps = gathered - log_z

            current_policy_all_logps = all_logps.detach()
            with torch.no_grad():
                entropies = entropy_from_logits(logits)

            all_selected_logps.append(selected_logps)
            all_student_topk.append(student_topk_logps)
            all_entropies.append(entropies)

        return (
            torch.cat(all_selected_logps, dim=0),
            torch.cat(all_student_topk, dim=0),
            torch.cat(all_entropies, dim=0),
        )

    def _fix_param_name_to_vllm(self, name, extra_prefixes: Optional[list[str]] = None):
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    def _load_old_probe_prompts(self) -> list[str]:
        dataset = datasets.load_dataset("google/IFEval", split="train")
        max_samples = min(self.probe_old_max_samples, len(dataset))
        prompts = []
        for example in dataset.select(range(max_samples)):
            messages = [{"role": "user", "content": example["prompt"]}]
            prompts.append(
                self.processing_class.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return prompts

    @staticmethod
    def _format_tool_probe_target(example: dict[str, Any]) -> str:
        if "golden_response" in example and example["golden_response"] is not None:
            value = example["golden_response"]
            return "\n".join(value) if isinstance(value, list) else str(value)
        if "golden_answer" in example and example["golden_answer"] is not None:
            lines = []
            for item in example["golden_answer"]:
                action = item.get("Action", "")
                action_input = item.get("Action_Input", "")
                lines.append(f"Action: {action}")
                lines.append(f"Action Input: {action_input}")
            return "\n".join(lines)
        if "output_text" in example and example["output_text"] is not None:
            return str(example["output_text"])
        if "answer" in example and example["answer"] is not None:
            answer = str(example["answer"]).strip()
            return f"<answer>\n{answer}\n</answer>"
        raise KeyError("Probe example has no golden_response, golden_answer, output_text, or answer field.")

    def _load_new_probe_examples(self) -> list[tuple[str, str]]:
        dataset = datasets.load_from_disk(self.probe_new_path)
        max_samples = min(self.probe_new_max_samples, len(dataset))
        examples = []
        for example in dataset.select(range(max_samples)):
            prompt = example["prompt"]
            messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
            prompt_text = self.processing_class.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            target_text = self._format_tool_probe_target(example)
            if self.eos_token_id is not None and self.processing_class.eos_token is not None:
                target_text = target_text + self.processing_class.eos_token
            examples.append((prompt_text, target_text))
        return examples

    @torch.no_grad()
    def _compute_new_probe_loss(self, model) -> Optional[float]:
        if self.probe_every_steps <= 0:
            return None
        if self._new_probe_cache is None:
            self._new_probe_cache = self._load_new_probe_examples()

        model_was_training = model.training
        model.eval()
        losses: list[torch.Tensor] = []
        for prompt_text, target_text in self._new_probe_cache:
            prompt_ids = self.processing_class(
                prompt_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]
            full_ids = self.processing_class(
                prompt_text + target_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]
            if full_ids.numel() < 2 or full_ids.numel() <= prompt_ids.numel():
                continue
            max_len = self.max_prompt_length + self.max_completion_length
            if full_ids.numel() > max_len:
                overflow = full_ids.numel() - max_len
                full_ids = full_ids[overflow:]
                label_start = max(1, prompt_ids.numel() - overflow)
            else:
                label_start = prompt_ids.numel()
            input_ids = full_ids.unsqueeze(0).to(self.accelerator.device)
            attention_mask = torch.ones_like(input_ids)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits[:, :-1, :] / self.temperature
            labels = input_ids[:, 1:]
            token_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1),
                reduction="none",
            ).view_as(labels)
            positions = torch.arange(labels.size(1), device=labels.device)
            target_mask = positions >= max(0, label_start - 1)
            if target_mask.any():
                losses.append(token_loss[:, target_mask].mean().detach())
        if model_was_training:
            model.train()
        if not losses:
            return None
        loss = torch.stack(losses).mean()
        return self.accelerator.gather(loss).nanmean().item()

    @torch.no_grad()
    def _old_probe_next_token_topk(self, model, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        model_was_training = model.training
        model.eval()
        all_logps: list[torch.Tensor] = []
        all_indices: list[torch.Tensor] = []
        batch_size = max(1, self.args.per_device_train_batch_size)
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            inputs = self.processing_class(
                text=batch_prompts,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                max_length=self.max_prompt_length,
                truncation=True,
                add_special_tokens=False,
            )
            inputs = super()._prepare_inputs(inputs)
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False)
            logits = outputs.logits[:, -1, :] / self.temperature
            logps = log_softmax(logits, dim=-1)
            k = min(self.probe_old_topk, logps.size(-1))
            topk_logps, topk_indices = torch.topk(logps, k, dim=-1)
            all_logps.append(topk_logps.detach().cpu())
            all_indices.append(topk_indices.detach().cpu())
        if model_was_training:
            model.train()
        return torch.cat(all_logps, dim=0), torch.cat(all_indices, dim=0)

    @torch.no_grad()
    def _compute_old_probe_loss(self, model) -> Optional[float]:
        if self.probe_every_steps <= 0:
            return None
        step = int(self.state.global_step)
        if step % self.probe_every_steps != 0:
            return None
        if self._old_probe_cache is None:
            prompts = self._load_old_probe_prompts()
            init_logps, init_indices = self._old_probe_next_token_topk(model, prompts)
            self._old_probe_cache = {
                "prompts": prompts,
                "init_logps": init_logps,
                "init_indices": init_indices,
            }
        prompts = self._old_probe_cache["prompts"]
        init_logps = self._old_probe_cache["init_logps"]
        init_indices = self._old_probe_cache["init_indices"]

        model_was_training = model.training
        model.eval()
        losses: list[torch.Tensor] = []
        batch_size = max(1, self.args.per_device_train_batch_size)
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            batch_init_logps = init_logps[start : start + batch_size].to(self.accelerator.device)
            batch_indices = init_indices[start : start + batch_size].to(self.accelerator.device)
            inputs = self.processing_class(
                text=batch_prompts,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                max_length=self.max_prompt_length,
                truncation=True,
                add_special_tokens=False,
            )
            inputs = super()._prepare_inputs(inputs)
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False)
            logits = outputs.logits[:, -1, :] / self.temperature
            current_logps = log_softmax(logits, dim=-1)
            current_on_init_support = torch.gather(current_logps, dim=-1, index=batch_indices)
            kl = (torch.exp(batch_init_logps.float()) * (batch_init_logps.float() - current_on_init_support.float())).sum(dim=-1)
            losses.append(kl.detach())
        if model_was_training:
            model.train()
        loss = torch.cat(losses).mean()
        return self.accelerator.gather(loss).nanmean().item()

    def _sync_fsdp1_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        # For FSDP1, we need to recurse into children and also use summon_full_params
        if visited is None:
            visited = set()
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp1_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    full_name = self._fix_param_name_to_vllm(full_name, extra_prefixes=["_fsdp_wrapped_module."])

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
        # For FSDP2, module.state_dict() already covers all parameters, so no need for recursion
        for name, param in module.state_dict().items():
            if param.is_cpu:
                param = param.to(torch.device("cuda"))
            param = param.full_tensor()

            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                self.vllm_client.update_named_param(name, param)
            elif self.vllm_mode == "colocate":
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights([(name, param)])

    @profiling_decorator
    def _move_model_to_vllm(self):
        # Select which model to sync to vLLM: teacher (ref_model) or student (model)
        # When generate_from_teacher=True, sync the teacher model since vLLM was initialized with teacher weights
        model_to_sync = self.ref_model if self.generate_from_teacher else self.model
        
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model):
            if self.generate_from_teacher:
                raise ValueError("PEFT model handling only applies when syncing student model (teacher is typically not PEFT)")
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                    fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                    if fsdp_version == 1:
                        self._sync_fsdp1_params_to_vllm(
                            self.model
                        )  # use memory-efficient post-order traversal for FSDP
                    elif fsdp_version == 2:
                        self._sync_fsdp2_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                if fsdp_version == 1:
                    self._sync_fsdp1_params_to_vllm(model_to_sync)  # use memory-efficient post-order traversal for FSDP
                elif fsdp_version == 2:
                    self._sync_fsdp2_params_to_vllm(model_to_sync)
            else:
                for name, param in model_to_sync.named_parameters():
                    name = self._fix_param_name_to_vllm(name)
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

    def _slice_multimodal_forward_kwargs(
        self,
        forward_kwargs: dict[str, Any],
        start: int,
        end: int,
    ) -> dict[str, Any]:
        """Slice optional multimodal fields for a [start:end) row range."""
        sliced = {k: v for k, v in forward_kwargs.items() if v is not None}
        pixel_values = sliced.get("pixel_values")
        if pixel_values is None:
            return sliced
        image_grid_thw = sliced.get("image_grid_thw")
        num_images = sliced.get("num_images")
        if image_grid_thw is not None:
            rows_per_image = image_grid_thw.prod(dim=-1)
            rows_per_sample = torch.split(rows_per_image, num_images)
            rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
            cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
            row_start, row_end = cum_rows[start].item(), cum_rows[end].item()
            sliced["pixel_values"] = pixel_values[row_start:row_end]
            cum_imgs = torch.tensor([0] + num_images).cumsum(0)
            sliced["image_grid_thw"] = image_grid_thw[cum_imgs[start] : cum_imgs[end]]
        else:
            sliced["pixel_values"] = pixel_values[start:end]
        if sliced.get("pixel_attention_mask") is not None:
            sliced["pixel_attention_mask"] = sliced["pixel_attention_mask"][start:end]
        if sliced.get("image_sizes") is not None:
            sliced["image_sizes"] = sliced["image_sizes"][start:end]
        if sliced.get("token_type_ids") is not None:
            sliced["token_type_ids"] = sliced["token_type_ids"][start:end]
        if num_images is not None:
            sliced["num_images"] = num_images[start:end]
        return sliced

    def _loss_completion_mask(self, completion_mask: torch.Tensor) -> torch.Tensor:
        """Completion mask for distillation loss (respects num_loss_tokens_to_skip)."""
        loss_completion_mask = completion_mask
        if self.num_loss_tokens_to_skip > 0:
            batch_size, seq_len = completion_mask.shape
            token_positions = torch.arange(seq_len, device=completion_mask.device).unsqueeze(0).expand(
                batch_size, -1
            )
            skip_mask = (token_positions >= self.num_loss_tokens_to_skip).int()
            loss_completion_mask = completion_mask * skip_mask
        return loss_completion_mask

    @torch.no_grad()
    def _compute_per_token_kl_qo_to_policy(
        self,
        model,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        teacher_prompt_ids: torch.Tensor,
        teacher_prompt_mask: torch.Tensor,
        forward_kwargs: dict[str, Any],
        soft_teacher_row_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-token KL(q_o || p_theta) summed over vocab; shape [batch, completion_len]."""
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        teacher_input_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        multimodal_kw = dict(
            pixel_values=forward_kwargs.get("pixel_values"),
            image_grid_thw=forward_kwargs.get("image_grid_thw"),
            num_images=forward_kwargs.get("num_images"),
            pixel_attention_mask=forward_kwargs.get("pixel_attention_mask"),
            image_sizes=forward_kwargs.get("image_sizes"),
            token_type_ids=forward_kwargs.get("token_type_ids"),
        )

        teacher_soft_mask = None
        teacher_soft_vectors = None
        if self.soft_teacher_w != 0.0 and (
            self.soft_teacher_v is not None or self.soft_teacher_row_directions is not None
        ):
            teacher_soft_mask = torch.cat(
                [
                    teacher_prompt_mask,
                    torch.zeros_like(completion_mask, dtype=teacher_prompt_mask.dtype),
                ],
                dim=1,
            )
        if self.soft_teacher_row_directions is not None and self.soft_teacher_w != 0.0:
            if soft_teacher_row_ids is None:
                raise KeyError("soft_teacher_row_ids required for row-wise soft teacher KL stats")
            teacher_soft_vectors = self.soft_teacher_row_directions[soft_teacher_row_ids].to(
                device=teacher_prompt_mask.device
            )

        per_token_kl_rows: list[torch.Tensor] = []
        for start in range(input_ids.size(0)):
            end = start + 1
            chunk_multimodal_kw = self._slice_multimodal_forward_kwargs(multimodal_kw, start, end)
            _, student_all_logps, _ = self._get_per_token_logps_and_entropies(
                model,
                input_ids[start:end],
                attention_mask[start:end],
                logits_to_keep,
                batch_size=1,
                compute_entropy=False,
                return_energy=False,
                **chunk_multimodal_kw,
            )
            _, teacher_all_logps, _ = self._get_per_token_logps_and_entropies(
                self.ref_model,
                teacher_input_ids[start:end],
                teacher_attention_mask[start:end],
                logits_to_keep,
                batch_size=1,
                compute_entropy=False,
                return_energy=False,
                embed_perturb_mask=teacher_soft_mask[start:end] if teacher_soft_mask is not None else None,
                embed_perturb_vectors=teacher_soft_vectors[start:end] if teacher_soft_vectors is not None else None,
                **chunk_multimodal_kw,
            )
            student_logps = student_all_logps.detach().float()
            teacher_logps = teacher_all_logps.float()
            per_token_kl_rows.append(
                (teacher_logps.exp() * (teacher_logps - student_logps)).sum(dim=-1)
            )
            del student_all_logps, teacher_all_logps, student_logps, teacher_logps

        return torch.cat(per_token_kl_rows, dim=0)

    @torch.no_grad()
    def _compute_mean_token_kl_qo_to_policy(
        self,
        model,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        teacher_prompt_ids: torch.Tensor,
        teacher_prompt_mask: torch.Tensor,
        forward_kwargs: dict[str, Any],
        soft_teacher_row_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-sample mean-token KL(q_o || p_theta) on a full generation batch."""
        loss_completion_mask = self._loss_completion_mask(completion_mask)
        teacher_policy_kl = self._compute_per_token_kl_qo_to_policy(
            model,
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            teacher_prompt_ids,
            teacher_prompt_mask,
            forward_kwargs,
            soft_teacher_row_ids=soft_teacher_row_ids,
        )
        return (
            teacher_policy_kl * loss_completion_mask.float()
        ).sum(dim=-1) / loss_completion_mask.sum(dim=-1).clamp(min=1.0)

    @staticmethod
    def _batch_kl_filter_token_keep_mask(
        per_token_kl: torch.Tensor,
        token_mask: torch.Tensor,
        keep_frac: float,
    ) -> torch.Tensor:
        """Per-sequence: keep lowest-keep_frac fraction of valid tokens by KL(q_o || p)."""
        batch_size, _ = per_token_kl.shape
        keep_mask = torch.zeros_like(per_token_kl, dtype=torch.bool)
        for row in range(batch_size):
            valid = token_mask[row].bool()
            valid_count = int(valid.sum().item())
            if valid_count <= 0:
                continue
            if keep_frac >= 1.0:
                keep_mask[row, valid] = True
                continue
            k_keep = max(1, int(round(valid_count * keep_frac)))
            row_kl = per_token_kl[row][valid]
            order = row_kl.float().argsort(stable=True)
            kept = torch.zeros(valid_count, dtype=torch.bool, device=per_token_kl.device)
            kept[order[:k_keep]] = True
            keep_mask[row, valid] = kept
        return keep_mask

    @staticmethod
    def _per_token_log_ratio_variance(
        policy_logps: torch.Tensor,
        teacher_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Per-position Var_p(log(q_o/p)) over the vocabulary simplex."""
        log_ratio = teacher_logps.float() - policy_logps.float()
        policy_probs = policy_logps.float().exp()
        mean_log_ratio = (policy_probs * log_ratio).sum(dim=-1, keepdim=True)
        return (policy_probs * (log_ratio - mean_log_ratio).pow(2)).sum(dim=-1)

    def _per_token_pearson_chi_square(
        self,
        policy_logps: torch.Tensor,
        teacher_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Per-position chi^2 peer; top-K then optional rPE (see chi2_topk / chi2_relative_alpha)."""
        return per_token_pearson_chi_square(
            policy_logps,
            teacher_logps,
            chi2_topk=self.online_sample_tau_chi2_topk,
            chi2_relative_alpha=self.online_sample_tau_chi2_relative_alpha,
        )

    @staticmethod
    def _per_token_hellinger_peer(
        policy_logps: torch.Tensor,
        teacher_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Per-position Squared-Hellinger Step-1 peer peer_H = b - a^2.

        With Delta = sqrt(q_o) - sqrt(p), a = sum sqrt(p) Delta, b = sum Delta^2:
          peer_H = b - a^2.
        For standard H^2 = (1/2)||√q-√p||^2 along the √-mix path,
          G(tau) ≈ (1/2) peer_H (1-tau)^2  (same template as var / chi2).
        """
        policy_probs = policy_logps.float().exp().clamp(min=1e-12)
        teacher_probs = teacher_logps.float().exp().clamp(min=1e-12)
        sqrt_p = policy_probs.sqrt()
        sqrt_q = teacher_probs.sqrt()
        delta = sqrt_q - sqrt_p
        a = (sqrt_p * delta).sum(dim=-1)  # = sum sqrt(p q_o) - 1
        b = (delta * delta).sum(dim=-1)
        return (b - a.pow(2)).clamp(min=0.0)

    def _resolve_proximal_selection(self) -> str:
        """Resolve proximal path: auto follows --alpha endpoints; hellinger is explicit."""
        if self.proximal_selection == "hellinger":
            return "hellinger"
        if self.proximal_selection == "forward_kl":
            return "forward_kl"
        if self.proximal_selection == "reverse_kl":
            return "reverse_kl"
        # auto
        if self.alpha >= 1.0 - 1e-9:
            return "reverse_kl"
        return "forward_kl"

    @torch.no_grad()
    def _compute_mean_token_log_ratio_variance(
        self,
        model,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        teacher_prompt_ids: torch.Tensor,
        teacher_prompt_mask: torch.Tensor,
        forward_kwargs: dict[str, Any],
        soft_teacher_row_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-sample mean-token Var_p(log(q_o/p)) on a full generation batch."""
        loss_completion_mask = completion_mask
        if self.num_loss_tokens_to_skip > 0:
            batch_size, seq_len = completion_mask.shape
            token_positions = torch.arange(seq_len, device=completion_mask.device).unsqueeze(0).expand(batch_size, -1)
            skip_mask = (token_positions >= self.num_loss_tokens_to_skip).int()
            loss_completion_mask = completion_mask * skip_mask

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        teacher_input_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        multimodal_kw = dict(
            pixel_values=forward_kwargs.get("pixel_values"),
            image_grid_thw=forward_kwargs.get("image_grid_thw"),
            num_images=forward_kwargs.get("num_images"),
            pixel_attention_mask=forward_kwargs.get("pixel_attention_mask"),
            image_sizes=forward_kwargs.get("image_sizes"),
            token_type_ids=forward_kwargs.get("token_type_ids"),
        )

        teacher_soft_mask = None
        teacher_soft_vectors = None
        if self.soft_teacher_w != 0.0 and (
            self.soft_teacher_v is not None or self.soft_teacher_row_directions is not None
        ):
            teacher_soft_mask = torch.cat(
                [
                    teacher_prompt_mask,
                    torch.zeros_like(completion_mask, dtype=teacher_prompt_mask.dtype),
                ],
                dim=1,
            )
        if self.soft_teacher_row_directions is not None and self.soft_teacher_w != 0.0:
            if soft_teacher_row_ids is None:
                raise KeyError("soft_teacher_row_ids required for row-wise soft teacher variance stats")
            teacher_soft_vectors = self.soft_teacher_row_directions[soft_teacher_row_ids].to(
                device=teacher_prompt_mask.device
            )

        per_sample_var: list[torch.Tensor] = []
        for start in range(input_ids.size(0)):
            end = start + 1
            chunk_multimodal_kw = self._slice_multimodal_forward_kwargs(multimodal_kw, start, end)
            _, student_all_logps, _ = self._get_per_token_logps_and_entropies(
                model,
                input_ids[start:end],
                attention_mask[start:end],
                logits_to_keep,
                batch_size=1,
                compute_entropy=False,
                return_energy=False,
                **chunk_multimodal_kw,
            )
            _, teacher_all_logps, _ = self._get_per_token_logps_and_entropies(
                self.ref_model,
                teacher_input_ids[start:end],
                teacher_attention_mask[start:end],
                logits_to_keep,
                batch_size=1,
                compute_entropy=False,
                return_energy=False,
                embed_perturb_mask=teacher_soft_mask[start:end] if teacher_soft_mask is not None else None,
                embed_perturb_vectors=teacher_soft_vectors[start:end] if teacher_soft_vectors is not None else None,
                **chunk_multimodal_kw,
            )
            per_sample_var.append(
                self._per_token_log_ratio_variance(student_all_logps.detach(), teacher_all_logps)
            )
            del student_all_logps, teacher_all_logps

        log_ratio_variance = torch.cat(per_sample_var, dim=0)
        return (
            log_ratio_variance.float() * loss_completion_mask.float()
        ).sum(dim=-1) / loss_completion_mask.sum(dim=-1).clamp(min=1.0)

    @torch.no_grad()
    def _compute_mean_token_pearson_chi_square(
        self,
        model,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        teacher_prompt_ids: torch.Tensor,
        teacher_prompt_mask: torch.Tensor,
        forward_kwargs: dict[str, Any],
        soft_teacher_row_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-sample mean-token chi^2(p; q_o) on a full generation batch."""
        loss_completion_mask = completion_mask
        if self.num_loss_tokens_to_skip > 0:
            batch_size, seq_len = completion_mask.shape
            token_positions = torch.arange(seq_len, device=completion_mask.device).unsqueeze(0).expand(batch_size, -1)
            skip_mask = (token_positions >= self.num_loss_tokens_to_skip).int()
            loss_completion_mask = completion_mask * skip_mask

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        teacher_input_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        multimodal_kw = dict(
            pixel_values=forward_kwargs.get("pixel_values"),
            image_grid_thw=forward_kwargs.get("image_grid_thw"),
            num_images=forward_kwargs.get("num_images"),
            pixel_attention_mask=forward_kwargs.get("pixel_attention_mask"),
            image_sizes=forward_kwargs.get("image_sizes"),
            token_type_ids=forward_kwargs.get("token_type_ids"),
        )

        teacher_soft_mask = None
        teacher_soft_vectors = None
        if self.soft_teacher_w != 0.0 and (
            self.soft_teacher_v is not None or self.soft_teacher_row_directions is not None
        ):
            teacher_soft_mask = torch.cat(
                [
                    teacher_prompt_mask,
                    torch.zeros_like(completion_mask, dtype=teacher_prompt_mask.dtype),
                ],
                dim=1,
            )
        if self.soft_teacher_row_directions is not None and self.soft_teacher_w != 0.0:
            if soft_teacher_row_ids is None:
                raise KeyError("soft_teacher_row_ids required for row-wise soft teacher chi-square stats")
            teacher_soft_vectors = self.soft_teacher_row_directions[soft_teacher_row_ids].to(
                device=teacher_prompt_mask.device
            )

        per_sample_chi2: list[torch.Tensor] = []
        for start in range(input_ids.size(0)):
            end = start + 1
            chunk_multimodal_kw = self._slice_multimodal_forward_kwargs(multimodal_kw, start, end)
            _, student_all_logps, _ = self._get_per_token_logps_and_entropies(
                model,
                input_ids[start:end],
                attention_mask[start:end],
                logits_to_keep,
                batch_size=1,
                compute_entropy=False,
                return_energy=False,
                **chunk_multimodal_kw,
            )
            _, teacher_all_logps, _ = self._get_per_token_logps_and_entropies(
                self.ref_model,
                teacher_input_ids[start:end],
                teacher_attention_mask[start:end],
                logits_to_keep,
                batch_size=1,
                compute_entropy=False,
                return_energy=False,
                embed_perturb_mask=teacher_soft_mask[start:end] if teacher_soft_mask is not None else None,
                embed_perturb_vectors=teacher_soft_vectors[start:end] if teacher_soft_vectors is not None else None,
                **chunk_multimodal_kw,
            )
            per_sample_chi2.append(
                self._per_token_pearson_chi_square(student_all_logps.detach(), teacher_all_logps)
            )
            del student_all_logps, teacher_all_logps

        pearson_chi_square = torch.cat(per_sample_chi2, dim=0)
        return (
            pearson_chi_square.float() * loss_completion_mask.float()
        ).sum(dim=-1) / loss_completion_mask.sum(dim=-1).clamp(min=1.0)

    @torch.no_grad()
    def _compute_mean_token_hellinger_peer(
        self,
        model,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        teacher_prompt_ids: torch.Tensor,
        teacher_prompt_mask: torch.Tensor,
        forward_kwargs: dict[str, Any],
        soft_teacher_row_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-sample mean-token Hellinger peer_H on a full generation batch."""
        loss_completion_mask = completion_mask
        if self.num_loss_tokens_to_skip > 0:
            batch_size, seq_len = completion_mask.shape
            token_positions = torch.arange(seq_len, device=completion_mask.device).unsqueeze(0).expand(batch_size, -1)
            skip_mask = (token_positions >= self.num_loss_tokens_to_skip).int()
            loss_completion_mask = completion_mask * skip_mask

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        teacher_input_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        multimodal_kw = dict(
            pixel_values=forward_kwargs.get("pixel_values"),
            image_grid_thw=forward_kwargs.get("image_grid_thw"),
            num_images=forward_kwargs.get("num_images"),
            pixel_attention_mask=forward_kwargs.get("pixel_attention_mask"),
            image_sizes=forward_kwargs.get("image_sizes"),
            token_type_ids=forward_kwargs.get("token_type_ids"),
        )

        teacher_soft_mask = None
        teacher_soft_vectors = None
        if self.soft_teacher_w != 0.0 and (
            self.soft_teacher_v is not None or self.soft_teacher_row_directions is not None
        ):
            teacher_soft_mask = torch.cat(
                [
                    teacher_prompt_mask,
                    torch.zeros_like(completion_mask, dtype=teacher_prompt_mask.dtype),
                ],
                dim=1,
            )
        if self.soft_teacher_row_directions is not None and self.soft_teacher_w != 0.0:
            if soft_teacher_row_ids is None:
                raise KeyError("soft_teacher_row_ids required for row-wise soft teacher hellinger stats")
            teacher_soft_vectors = self.soft_teacher_row_directions[soft_teacher_row_ids].to(
                device=teacher_prompt_mask.device
            )

        per_sample_peer: list[torch.Tensor] = []
        for start in range(input_ids.size(0)):
            end = start + 1
            chunk_multimodal_kw = self._slice_multimodal_forward_kwargs(multimodal_kw, start, end)
            _, student_all_logps, _ = self._get_per_token_logps_and_entropies(
                model,
                input_ids[start:end],
                attention_mask[start:end],
                logits_to_keep,
                batch_size=1,
                compute_entropy=False,
                return_energy=False,
                **chunk_multimodal_kw,
            )
            _, teacher_all_logps, _ = self._get_per_token_logps_and_entropies(
                self.ref_model,
                teacher_input_ids[start:end],
                teacher_attention_mask[start:end],
                logits_to_keep,
                batch_size=1,
                compute_entropy=False,
                return_energy=False,
                embed_perturb_mask=teacher_soft_mask[start:end] if teacher_soft_mask is not None else None,
                embed_perturb_vectors=teacher_soft_vectors[start:end] if teacher_soft_vectors is not None else None,
                **chunk_multimodal_kw,
            )
            per_sample_peer.append(
                self._per_token_hellinger_peer(student_all_logps.detach(), teacher_all_logps)
            )
            del student_all_logps, teacher_all_logps

        peer_h = torch.cat(per_sample_peer, dim=0)
        return (
            peer_h.float() * loss_completion_mask.float()
        ).sum(dim=-1) / loss_completion_mask.sum(dim=-1).clamp(min=1.0)

    @torch.no_grad()
    def _compute_mean_token_policy_entropy(
        self,
        model,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        forward_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Per-sample mean-token entropy H(p_theta) on a full generation batch."""
        loss_completion_mask = completion_mask
        if self.num_loss_tokens_to_skip > 0:
            batch_size, seq_len = completion_mask.shape
            token_positions = torch.arange(seq_len, device=completion_mask.device).unsqueeze(0).expand(batch_size, -1)
            skip_mask = (token_positions >= self.num_loss_tokens_to_skip).int()
            loss_completion_mask = completion_mask * skip_mask

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        multimodal_kw = dict(
            pixel_values=forward_kwargs.get("pixel_values"),
            image_grid_thw=forward_kwargs.get("image_grid_thw"),
            num_images=forward_kwargs.get("num_images"),
            pixel_attention_mask=forward_kwargs.get("pixel_attention_mask"),
            image_sizes=forward_kwargs.get("image_sizes"),
            token_type_ids=forward_kwargs.get("token_type_ids"),
        )

        per_sample_entropy: list[torch.Tensor] = []
        for start in range(input_ids.size(0)):
            end = start + 1
            chunk_multimodal_kw = self._slice_multimodal_forward_kwargs(multimodal_kw, start, end)
            _, _, token_entropies = self._get_per_token_logps_and_entropies(
                model,
                input_ids[start:end],
                attention_mask[start:end],
                logits_to_keep,
                batch_size=1,
                compute_entropy=True,
                compute_all_logps=False,
                return_energy=False,
                **chunk_multimodal_kw,
            )
            per_sample_entropy.append(token_entropies)
            del token_entropies

        policy_entropy = torch.cat(per_sample_entropy, dim=0)
        return (
            policy_entropy.float() * loss_completion_mask.float()
        ).sum(dim=-1) / loss_completion_mask.sum(dim=-1).clamp(min=1.0)

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = split_pixel_values_by_grid(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func

    def _generate_single_turn(self, prompts: list[str], images: Optional[list]):
        device = self.accelerator.device

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the sky?"}]}]
        kwargs = {}
        if images is not None:
            kwargs = {"images": images}
            for prompt, image_list in zip(prompts, images):
                if isinstance(prompt, list):  # i.e., when using conversational data
                    prepare_multimodal_messages(prompt, num_images=len(image_list))

        prompts_text = [
            maybe_apply_chat_template({"prompt": prompt}, self.processing_class)["prompt"] for prompt in prompts
        ]

        if images is not None:
            prompt_inputs = self.processing_class(text=prompts_text, padding=True, return_tensors="pt", **kwargs)
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # Generate completions using either vLLM or regular generation
        # Note: When generate_from_teacher=True, vLLM is initialized with teacher weights
        if self.use_vllm:
            if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
                # wake up colocated vLLM instances if needed
                torch.cuda.empty_cache()  # required to avoid OOM in some cases
                self.llm.wake_up()

            # First, update the vLLM weights if needed
            # When generate_from_teacher=True and sync_ref_model=False, teacher is static so no sync needed
            # (vLLM already loaded teacher weights at initialization)
            should_sync = self.state.global_step != self._last_loaded_step
            if self.generate_from_teacher and not self.args.sync_ref_model:
                should_sync = False  # Teacher is static, no need to sync
            if should_sync:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if images is not None:
                    all_images = gather_object(images)

                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]

                    if images is not None:
                        ordered_set_of_images = all_images[:: self.num_generations]
                    else:
                        ordered_set_of_images = None

                    with profiling_context(self, "vLLM.generate"):
                        output = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            images=ordered_set_of_images,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            truncate_prompt_tokens=self.max_prompt_length,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                        payload = (output["prompt_ids"], output["completion_ids"], output["logprobs"])
                else:
                    payload = None

                # Broadcast the completions from the main process to all processes, ensuring each process receives its corresponding slice.
                obj_list = [payload]
                broadcast_object_list(obj_list, from_process=0)
                all_prompt_ids, all_completion_ids, all_logprobs = obj_list[0]

                # At this point, we only get 1 copy of each prompt, so we need to repeat them num_generations times
                all_prompt_ids = [ids for ids in all_prompt_ids for _ in range(self.num_generations)]

                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                prompt_ids = all_prompt_ids[process_slice]
                completion_ids = all_completion_ids[process_slice]
                logprobs = all_logprobs[process_slice]

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "truncate_prompt_tokens": self.max_prompt_length,
                    "logprobs": 0,  # only return the logprob of the generated token
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]

                    if images is not None:
                        gathered_images = [None for _ in range(self.vllm_tensor_parallel_size)]
                        torch.distributed.all_gather_object(gathered_images, images, group=self.tp_group)
                        all_images = [img for sublist in gathered_images for img in sublist]
                    else:
                        all_images = None
                else:
                    all_prompts_text = prompts_text
                    all_images = images

                if images is not None and all_images:
                    vllm_inputs = []
                    for prompt, image_list in zip(all_prompts_text, all_images):
                        vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image_list}})

                else:
                    vllm_inputs = all_prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)

                all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
                all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
                all_logprobs = [
                    [next(iter(lp.values())).logprob for lp in output.logprobs]
                    for outputs in all_outputs
                    for output in outputs.outputs
                ]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    prompt_ids = all_prompt_ids[tp_slice]
                    completion_ids = all_completion_ids[tp_slice]
                    logprobs = all_logprobs[tp_slice]
                else:
                    prompt_ids = all_prompt_ids
                    completion_ids = all_completion_ids
                    logprobs = all_logprobs

                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)

        elif self.use_transformers_paged:
            # Re-process inputs for paged generation if needed
            # Note: images are already validated and preprocessed above
            paged_prompt_inputs = self.processing_class(text=prompts_text, **kwargs)
            previous_attn = self.model_wrapped.config._attn_implementation

            if is_flash_attn_2_available():
                self.model_wrapped.config._attn_implementation = "paged_attention"
            else:
                self.model_wrapped.config._attn_implementation = "sdpa_paged"
            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                with torch.inference_mode():
                    all_outputs = unwrapped_model.generate_batch(
                        paged_prompt_inputs.input_ids, generation_config=self.generation_config, progress_bar=False
                    )
                    unwrapped_model.train()  # restore training mode, as generate_batch forces eval mode
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            prompt_ids = paged_prompt_inputs.input_ids
            # Restore the original attention implementation, training mode
            self.model_wrapped.config._attn_implementation = previous_attn
            logprobs = None  # not used in this case

        else:
            # Regular generation path
            generate_inputs = self.processing_class(
                text=prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                max_length=self.max_prompt_length,
                truncation=True,
                add_special_tokens=False,
                **kwargs,
            )
            generate_inputs = super()._prepare_inputs(generate_inputs)

            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs, generation_config=self.generation_config, disable_compile=True
                )
            # Compute prompt length and extract completion ids
            prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool())]
            completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool())]
            logprobs = None  # not used in this case

        return prompt_ids, completion_ids, logprobs, forward_kwargs

    def _generate(self, prompts: list[str], images: Optional[list]):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompt_ids, completion_ids, logprobs, forward_kwargs = self._generate_single_turn(prompts, images)

        # Get completion length per sequence, used for logging
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()  # = num_items_in_batch, required for the DAPO loss

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        return prompt_ids, completion_ids, total_completion_tokens, logprobs, forward_kwargs

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        teacher_prompts = [x["teacher_prompt"] for x in inputs]
        soft_teacher_row_ids = None
        if self.soft_teacher_row_directions is not None:
            if "soft_teacher_row_id" not in inputs[0]:
                raise KeyError(
                    "soft_teacher_row_directions_pt requires train examples to include soft_teacher_row_id"
                )
            soft_teacher_row_ids = torch.tensor(
                [int(x["soft_teacher_row_id"]) for x in inputs],
                device=device,
                dtype=torch.long,
            )

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # Decide whether to generate from teacher (with context) or student (without context)
        generation_prompts = teacher_prompts if self.generate_from_teacher else prompts

        (
            _generation_prompt_ids_list,  # Discard - we'll compute student/teacher prompt IDs separately
            completion_ids_list,
            num_items_in_batch,
            sampling_per_token_logps_list,
            forward_kwargs,
        ) = self._generate(generation_prompts, images)

        # Process student prompts (always used for student training, regardless of generation source)
        prompts_text = [
            maybe_apply_chat_template({"prompt": prompt}, self.processing_class)["prompt"] for prompt in prompts
        ]
        if self.use_vllm:
            self.processing_class.truncation_side = "left"
        student_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        student_inputs = super()._prepare_inputs(student_inputs)
        student_prompt_ids, student_prompt_mask = student_inputs["input_ids"], student_inputs["attention_mask"]
        prompt_ids_list = [p[m].tolist() for p, m in zip(student_prompt_ids, student_prompt_mask.bool())]

        # Process teacher prompts (always used for teacher, regardless of generation source)
        teacher_prompts_text = [
            maybe_apply_chat_template({"prompt": prompt}, self.processing_class)["prompt"] for prompt in teacher_prompts
        ]
        teacher_inputs = self.processing_class(
            text=teacher_prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        teacher_inputs = super()._prepare_inputs(teacher_inputs)
        if self.use_vllm:
            self.processing_class.truncation_side = "right"
        teacher_prompt_ids, teacher_prompt_mask = teacher_inputs["input_ids"], teacher_inputs["attention_mask"]
        teacher_prompt_ids_list = [p[m].tolist() for p, m in zip(teacher_prompt_ids, teacher_prompt_mask.bool())]

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        teacher_prompt_ids = [torch.tensor(ids, device=device) for ids in teacher_prompt_ids_list]
        teacher_prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in teacher_prompt_ids]
        teacher_prompt_ids = pad(teacher_prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        teacher_prompt_mask = pad(teacher_prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        teacher_prompt_completion_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)  # (B, P+C)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)  # (B, P+C)
        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            # Skip when generate_from_teacher=True since importance sampling is not used in that case.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if not self.generate_from_teacher and (
                self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction)):
                old_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    compute_all_logps=False,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            # Skip when generate_from_teacher=True since vLLM has teacher weights (no mismatch to correct)
            if self.use_vllm and self.vllm_importance_sampling_correction and not self.generate_from_teacher:
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )
            else:
                importance_sampling_ratio = None

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        compute_all_logps=False,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            compute_all_logps=False,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )   
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
        
        # Not really necessary, but keeping for now
        rewards = torch.zeros_like(completion_ids, dtype=torch.float32)
        advantages = rewards
        
        # Keep a copy for logging (data is already local to each process, no slicing needed)
        all_process_advantages = advantages.clone()

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        self._logs["rewards"]["main"].extend(gather_object(rewards.mean(dim=-1).tolist()))
        self._logs["advantages"].extend(gather_object(all_process_advantages.mean(dim=-1).tolist()))
        reward_to_log = rewards.clone()
        reward_to_log = reward_to_log[completion_mask.bool()]
        mean_reward = torch.mean(reward_to_log) if reward_to_log.numel() > 0 else torch.tensor(0.0, device=device)
        self._metrics[mode]["rewards"].append(self.accelerator.gather(mean_reward).mean().item())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if importance_sampling_ratio is not None:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[completion_mask.bool()]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        online_sample_tau_k = None
        online_sample_tau_k_bar = None
        online_sample_tau_k0 = None
        online_sample_tau_k0_bar = None
        if self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_PEER_CACHE_MODES:
            # Per-sample mean-token peer signal on the full generation batch (e.g. 32 rows).
            # K_i = KL(q_o || p_theta) or H_i = H(p_theta); bar = batch mean (cached).
            # Sample modes also cache per-sample signal; token modes recompute per-token at loss.
            if self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_ENTROPY_MODES:
                generation_sample_signal = self._compute_mean_token_policy_entropy(
                    self.model,
                    prompt_ids,
                    prompt_mask,
                    completion_ids,
                    completion_mask,
                    forward_kwargs,
                )
            elif self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_VAR_MODES:
                generation_sample_signal = self._compute_mean_token_log_ratio_variance(
                    self.model,
                    prompt_ids,
                    prompt_mask,
                    completion_ids,
                    completion_mask,
                    teacher_prompt_ids,
                    teacher_prompt_mask,
                    forward_kwargs,
                    soft_teacher_row_ids=soft_teacher_row_ids,
                )
            elif self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_CHI2_MODES:
                generation_sample_signal = self._compute_mean_token_pearson_chi_square(
                    self.model,
                    prompt_ids,
                    prompt_mask,
                    completion_ids,
                    completion_mask,
                    teacher_prompt_ids,
                    teacher_prompt_mask,
                    forward_kwargs,
                    soft_teacher_row_ids=soft_teacher_row_ids,
                )
            elif self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_HELLINGER_MODES:
                generation_sample_signal = self._compute_mean_token_hellinger_peer(
                    self.model,
                    prompt_ids,
                    prompt_mask,
                    completion_ids,
                    completion_mask,
                    teacher_prompt_ids,
                    teacher_prompt_mask,
                    forward_kwargs,
                    soft_teacher_row_ids=soft_teacher_row_ids,
                )
            else:
                generation_sample_signal = self._compute_mean_token_kl_qo_to_policy(
                    self.model,
                    prompt_ids,
                    prompt_mask,
                    completion_ids,
                    completion_mask,
                    teacher_prompt_ids,
                    teacher_prompt_mask,
                    forward_kwargs,
                    soft_teacher_row_ids=soft_teacher_row_ids,
                )
            gathered_signal = self.accelerator.gather(generation_sample_signal.float())
            signal_bar_value = self._smooth_peer_signal_bar(gathered_signal.mean())
            online_sample_tau_k_bar = torch.full_like(
                generation_sample_signal, signal_bar_value
            )
            if self.online_sample_tau_mode in {
                "budget_forward_kl_relative",
                "budget_forward_kl_relative_exp",
                "budget_forward_kl_relative_exp_range",
                "budget_log_ratio_var_relative",
                "budget_log_ratio_var_relative_exp",
                "budget_log_ratio_var_relative_exp_range",
                "budget_chi2_relative",
                "budget_chi2_relative_exp",
                "budget_chi2_relative_exp_range",
                "budget_hellinger_relative",
                "budget_hellinger_relative_exp",
                "budget_hellinger_relative_exp_range",
                "softmax_router_relative",
                "budget_forward_kl_relative_entropy",
                "softmax_router_relative_entropy",
            }:
                online_sample_tau_k = generation_sample_signal
            if self.k0_kl_map is not None:
                if soft_teacher_row_ids is None:
                    if "soft_teacher_row_id" not in inputs[0]:
                        raise KeyError(
                            "online_sample_tau_k0_path requires train examples to include soft_teacher_row_id"
                        )
                    row_ids = [int(x["soft_teacher_row_id"]) for x in inputs]
                else:
                    row_ids = soft_teacher_row_ids.detach().cpu().tolist()
                k0_values: list[float] = []
                for row_id in row_ids:
                    if int(row_id) not in self.k0_kl_map:
                        raise KeyError(f"K0 map missing dataset index {row_id}")
                    k0_values.append(self.k0_kl_map[int(row_id)])
                online_sample_tau_k0 = torch.tensor(k0_values, device=device, dtype=torch.float32)
                assert self.k0_kl_global_mean is not None
                online_sample_tau_k0_bar = torch.full_like(
                    online_sample_tau_k0,
                    float(self.k0_kl_global_mean),
                )

        batch_kl_filter_keep = None
        batch_kl_filter_token_keep = None
        if self.batch_kl_filter_keep_frac is not None:
            loss_completion_mask_gen = self._loss_completion_mask(completion_mask)
            if self.batch_kl_filter_level == "token":
                per_token_kl = self._compute_per_token_kl_qo_to_policy(
                    self.model,
                    prompt_ids,
                    prompt_mask,
                    completion_ids,
                    completion_mask,
                    teacher_prompt_ids,
                    teacher_prompt_mask,
                    forward_kwargs,
                    soft_teacher_row_ids=soft_teacher_row_ids,
                )
                token_keep_mask = self._batch_kl_filter_token_keep_mask(
                    per_token_kl,
                    loss_completion_mask_gen,
                    self.batch_kl_filter_keep_frac,
                )
                batch_kl_filter_token_keep = self._batch_kl_filter_weights(
                    token_keep_mask, self.batch_kl_filter_drop_weight
                )
                valid = loss_completion_mask_gen.bool()
                keep_token_count = float((token_keep_mask & valid).sum().item())
                drop_token_count = float((~token_keep_mask & valid).sum().item())
                kept_kl = per_token_kl[token_keep_mask & valid]
                threshold = float(kept_kl.max().item()) if kept_kl.numel() > 0 else float("nan")
                self._metrics[mode]["batch_kl_filter/keep_token_count"].append(keep_token_count)
                self._metrics[mode]["batch_kl_filter/drop_token_count"].append(drop_token_count)
                self._metrics[mode]["batch_kl_filter/token_threshold"].append(threshold)
                self._metrics[mode]["batch_kl_filter/mean_weight"].append(
                    float(batch_kl_filter_token_keep[valid].mean().item()) if valid.any() else float("nan")
                )
            else:
                generation_kl = self._compute_mean_token_kl_qo_to_policy(
                    self.model,
                    prompt_ids,
                    prompt_mask,
                    completion_ids,
                    completion_mask,
                    teacher_prompt_ids,
                    teacher_prompt_mask,
                    forward_kwargs,
                    soft_teacher_row_ids=soft_teacher_row_ids,
                )
                keep_mask = self._batch_kl_filter_keep_mask(
                    generation_kl, self.batch_kl_filter_keep_frac
                )
                batch_kl_filter_keep = self._batch_kl_filter_weights(
                    keep_mask, self.batch_kl_filter_drop_weight
                )
                kept_kl = generation_kl[keep_mask]
                threshold = float(kept_kl.max().item()) if keep_mask.any() else float("nan")
                self._metrics[mode]["batch_kl_filter/keep_count"].append(float(keep_mask.sum()))
                self._metrics[mode]["batch_kl_filter/drop_count"].append(float((~keep_mask).sum()))
                self._metrics[mode]["batch_kl_filter/threshold"].append(threshold)
                self._metrics[mode]["batch_kl_filter/mean_weight"].append(
                    float(batch_kl_filter_keep.mean().item())
                )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "teacher_prompt_ids": teacher_prompt_ids,
            "teacher_prompt_mask": teacher_prompt_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if online_sample_tau_k is not None:
            output["online_sample_tau_k"] = online_sample_tau_k
        if online_sample_tau_k_bar is not None:
            output["online_sample_tau_k_bar"] = online_sample_tau_k_bar
        if online_sample_tau_k0 is not None:
            output["online_sample_tau_k0"] = online_sample_tau_k0
        if online_sample_tau_k0_bar is not None:
            output["online_sample_tau_k0_bar"] = online_sample_tau_k0_bar
        if batch_kl_filter_keep is not None:
            output["batch_kl_filter_keep"] = batch_kl_filter_keep
        if batch_kl_filter_token_keep is not None:
            output["batch_kl_filter_token_keep"] = batch_kl_filter_token_keep
        if soft_teacher_row_ids is not None:
            output["soft_teacher_row_ids"] = soft_teacher_row_ids
        for optional_key in ("sample_tau", "sample_avg64", "sample_uncertainty"):
            if optional_key in inputs[0]:
                output[optional_key] = torch.tensor(
                    [float(example[optional_key]) for example in inputs],
                    device=device,
                    dtype=torch.float32,
                )
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if importance_sampling_ratio is not None:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        return output


    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The DistilTrainer does not support returning outputs")
        return self._compute_loss(model, inputs)

    def _smooth_peer_signal_bar(self, batch_bar: Union[torch.Tensor, float]) -> float:
        """
        EMA-smooth the batch peer mean K_bar (or H_bar) across generation batches.

        K_bar_ema <- decay * K_bar_ema + (1 - decay) * K_bar_batch
        """
        batch_value = float(batch_bar.item() if isinstance(batch_bar, torch.Tensor) else batch_bar)
        decay = self.online_sample_tau_k_bar_ema_decay
        if decay <= 0.0:
            return batch_value
        if self._online_sample_tau_k_bar_ema is None:
            self._online_sample_tau_k_bar_ema = batch_value
        else:
            self._online_sample_tau_k_bar_ema = (
                decay * self._online_sample_tau_k_bar_ema + (1.0 - decay) * batch_value
            )
        return self._online_sample_tau_k_bar_ema

    @staticmethod
    def _batch_kl_filter_keep_mask(per_sample_kl: torch.Tensor, keep_frac: float) -> torch.Tensor:
        """Keep the lowest-keep_frac fraction of samples by per-sample KL(q_o || p)."""
        n = int(per_sample_kl.numel())
        if keep_frac >= 1.0 or n <= 1:
            return torch.ones(n, dtype=torch.bool, device=per_sample_kl.device)
        k_keep = max(1, int(round(n * keep_frac)))
        order = per_sample_kl.float().argsort(stable=True)
        keep_mask = torch.zeros(n, dtype=torch.bool, device=per_sample_kl.device)
        keep_mask[order[:k_keep]] = True
        return keep_mask

    @staticmethod
    def _batch_kl_filter_weights(keep_mask: torch.Tensor, drop_weight: float) -> torch.Tensor:
        """Map boolean keep mask to per-unit loss weights (1.0 kept, drop_weight filtered)."""
        if drop_weight <= 0.0:
            return keep_mask.float()
        return torch.where(
            keep_mask,
            torch.ones_like(keep_mask, dtype=torch.float32),
            torch.full_like(keep_mask, drop_weight, dtype=torch.float32),
        )

    def _resolve_peer_signal_bar(
        self,
        per_sample_signal: torch.Tensor,
        cached_k_bar: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the peer batch mean used for relative/router tau (optionally EMA-smoothed)."""
        if cached_k_bar is not None:
            return cached_k_bar.float().mean()
        gathered_signal = self.accelerator.gather(per_sample_signal.detach().float())
        smoothed_bar = self._smooth_peer_signal_bar(gathered_signal.mean())
        return torch.tensor(smoothed_bar, device=per_sample_signal.device, dtype=torch.float32)

    def _global_masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean over all valid positions in the local batch, gathered across processes."""
        weighted_sum = (values.float() * mask.float()).sum()
        token_count = mask.float().sum().clamp(min=1.0)
        stats = torch.stack([weighted_sum, token_count])
        gathered_stats = self.accelerator.gather(stats)
        if gathered_stats.dim() > 1:
            global_sum = gathered_stats[:, 0].sum()
            global_count = gathered_stats[:, 1].sum().clamp(min=1.0)
        else:
            global_sum = gathered_stats[0]
            global_count = gathered_stats[1].clamp(min=1.0)
        return global_sum / global_count

    @staticmethod
    def _peer_relative_softmax_router_tau(
        k: torch.Tensor,
        k_bar: torch.Tensor,
        alpha: float,
        gamma: float = 1.0,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> torch.Tensor:
        """tau = clip( (K/(gamma*Kbar))^alpha / (1 + (K/(gamma*Kbar))^alpha), clip_min, clip_max )."""
        scaled_k_bar = (k_bar.float() * gamma).clamp(min=1e-8)
        ratio = (k.float() / scaled_k_bar).clamp(min=1e-8)
        powered = ratio.pow(alpha)
        tau = powered / (1.0 + powered)
        return tau.clamp(min=clip_min, max=clip_max)

    @staticmethod
    def _fixed_c_budget_tau(
        k: torch.Tensor,
        budget_c: float,
        mode: str,
        tau_max: float,
        clip_min: float,
        clip_max: float,
    ) -> torch.Tensor:
        """
        Fixed-C online tau from per-sample peer signal (K_i or V_i).

        budget_forward_kl:
          tau_i = clip(max(0, 1 - sqrt(C / K_i)), clip_min, clip_max)

        budget_log_ratio_var* uses V_i=Var_p(log(q_o/p)); fixed-C modes use D_i≈V_i/2 in sqrt(C/D).
        budget_chi2* uses chi2_i=E_p[(q_o/p-1)^2]; fixed-C modes use chi2_i/2 in sqrt(2C/chi2).
        budget_hellinger* uses standard H^2: G≈(1/2)*peer_H*(1-tau)^2, so peer_H/2 in sqrt(C/·).
        Relative modes use sqrt(alpha * peer_bar / peer_i); (1/2) cancels in the ratio.
        """
        signal = k.float().clamp(min=1e-8)
        # Step-1 local G = (1/2)*peer*(1-tau)^2 for var / chi2 / standard Hellinger.
        if mode in {
            "budget_log_ratio_var",
            "budget_log_ratio_var_exp",
            "budget_chi2",
            "budget_chi2_exp",
            "budget_hellinger",
            "budget_hellinger_exp",
        }:
            signal = signal / 2.0
        sqrt_ratio = torch.sqrt(budget_c / signal)
        if mode in {
            "budget_forward_kl",
            "budget_log_ratio_var",
            "budget_chi2",
            "budget_hellinger",
        }:
            raw_tau = (1.0 - sqrt_ratio).clamp(min=0.0)
        elif mode in {
            "budget_forward_kl_exp",
            "budget_log_ratio_var_exp",
            "budget_chi2_exp",
            "budget_hellinger_exp",
        }:
            raw_tau = tau_max * torch.exp(-sqrt_ratio)
        else:
            raise ValueError(f"unsupported fixed-C budget mode: {mode!r}")
        return raw_tau.clamp(min=clip_min, max=clip_max)

    @staticmethod
    def _peer_relative_budget_tau(
        k: torch.Tensor,
        k_bar: torch.Tensor,
        alpha: float,
        mode: str,
        tau_max: float,
        clip_min: float,
        clip_max: float,
        k0: Optional[torch.Tensor] = None,
        k0_bar_for_ratio: Optional[float] = None,
        k0_gamma: float = 1.0,
    ) -> torch.Tensor:
        """
        Peer-normalized online tau from peer signal (K_i or V_i) and batch mean.

        budget_forward_kl_relative* / budget_log_ratio_var_relative*:
          tau_i = clip(max(0, 1 - sqrt(alpha * signal_bar / signal_i)), clip_min, clip_max)

        budget_forward_kl_relative_exp* / budget_log_ratio_var_relative_exp*:
          tau_i = clip(tau_max * exp(-sqrt(alpha * signal_bar / signal_i)), clip_min, clip_max)

        budget_forward_kl_relative_exp_range* / budget_log_ratio_var_relative_exp_range*:
          tau_i = tau_min + (tau_max - tau_min) * exp(-sqrt(alpha * signal_bar / signal_i))
          with tau_min=clip_min, tau_max=clip_max (no post-hoc clamp).

        With optional K0 dual-anchor, multiply the budget ratio by (Kbar_0/K_i,0)^gamma.
        """
        budget_ratio = alpha * k_bar.float() / k.float().clamp(min=1e-8)
        if k0 is not None:
            if k0_bar_for_ratio is None:
                raise ValueError("k0_bar_for_ratio is required when k0 is provided.")
            k0_prior_ratio = (float(k0_bar_for_ratio) / k0.float().clamp(min=1e-8)).pow(k0_gamma)
            budget_ratio = budget_ratio * k0_prior_ratio
        sqrt_ratio = torch.sqrt(budget_ratio)
        if mode in ONLINE_SAMPLE_TAU_RELATIVE_BUDGET_SQRT_MODES:
            raw_tau = (1.0 - sqrt_ratio).clamp(min=0.0)
        elif mode in ONLINE_SAMPLE_TAU_RELATIVE_BUDGET_EXP_MODES:
            raw_tau = tau_max * torch.exp(-sqrt_ratio)
        elif mode in ONLINE_SAMPLE_TAU_RELATIVE_BUDGET_EXP_RANGE_MODES:
            return clip_min + (clip_max - clip_min) * torch.exp(-sqrt_ratio)
        else:
            raise ValueError(f"unsupported peer-relative budget mode: {mode!r}")
        return raw_tau.clamp(min=clip_min, max=clip_max)

    @staticmethod
    def _build_proximal_teacher_log_probs(
        teacher_logps: torch.Tensor,
        current_logps: torch.Tensor,
        tau: torch.Tensor,
        selection: str,
    ) -> torch.Tensor:
        """
        Closed-form proximal teacher q_tau on the vocabulary simplex.

        forward_kl:
          log q_tau = (1 - tau) log q_o + tau log p - log Z   (geometric / logit mix)
        reverse_kl:
          q_tau(v) = (1 - tau) q_o(v) + tau p(v)            (arithmetic)
        hellinger:
          u = (1 - tau) sqrt(q_o) + tau sqrt(p); q_tau = u^2 / ||u||_2^2
        """
        if selection == "hellinger":
            teacher_probs = torch.exp(teacher_logps.float()).clamp(min=1e-12)
            current_probs = torch.exp(current_logps.float()).clamp(min=1e-12)
            u = (1.0 - tau) * teacher_probs.sqrt() + tau * current_probs.sqrt()
            qtau_probs = u.pow(2)
            qtau_probs = qtau_probs / qtau_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            return torch.log(qtau_probs.clamp(min=1e-12)).to(dtype=teacher_logps.dtype)
        if selection == "reverse_kl":
            teacher_probs = torch.exp(teacher_logps.float())
            current_probs = torch.exp(current_logps.float())
            qtau_probs = (1.0 - tau) * teacher_probs + tau * current_probs
            return torch.log(qtau_probs.clamp(min=1e-12)).to(dtype=teacher_logps.dtype)
        # forward_kl
        proximal_log_unnormalized = (1.0 - tau) * teacher_logps + tau * current_logps
        return proximal_log_unnormalized - torch.logsumexp(proximal_log_unnormalized, dim=-1, keepdim=True)

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        teacher_prompt_ids, teacher_prompt_mask = inputs["teacher_prompt_ids"], inputs["teacher_prompt_mask"]
        
        # Create a separate mask for loss computation that skips the first N tokens
        loss_completion_mask = self._loss_completion_mask(completion_mask)
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        teacher_input_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        multimodal_kw = dict(
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )
        use_distill_topk = self.distillation_topk is not None and self.distillation_topk > 0
        teacher_soft_mask = None
        teacher_soft_vectors = None
        teacher_policy_kl = None
        qo_to_student_kl = None
        policy_to_student_kl = None
        qtau_to_student_kl = None
        qtau_entropy = None
        teacher_entropy = None
        teacher_max_prob = None
        qtau_max_prob = None
        student_negative_energy = None
        teacher_negative_energy = None
        effective_tau = None
        effective_tau_clipped = None
        effective_tau_residual_ratio = None
        effective_tau_r2 = None
        effective_tau_direction_norm_sq = None
        original_teacher_policy_kl = None
        online_sample_tau_k = None
        online_sample_tau_k_bar = None
        online_sample_tau_k0 = None
        online_sample_tau_k0_bar = None
        token_tau_values = None
        sample_tau_values = inputs.get("sample_tau")
        if self.soft_teacher_w != 0.0 and (
            self.soft_teacher_v is not None or self.soft_teacher_row_directions is not None
        ):
            teacher_soft_mask = torch.cat(
                [
                    teacher_prompt_mask,
                    torch.zeros_like(completion_mask, dtype=teacher_prompt_mask.dtype),
                ],
                dim=1,
            )
        if self.soft_teacher_row_directions is not None and self.soft_teacher_w != 0.0:
            row_ids = inputs["soft_teacher_row_ids"].detach().cpu().long()
            if int(row_ids.max()) >= self.soft_teacher_row_directions.shape[0] or int(row_ids.min()) < 0:
                raise IndexError(
                    "soft_teacher_row_ids out of range for soft_teacher_row_directions: "
                    f"min={int(row_ids.min())} max={int(row_ids.max())} "
                    f"num_rows={self.soft_teacher_row_directions.shape[0]}"
                )
            teacher_soft_vectors = self.soft_teacher_row_directions[row_ids].to(device=teacher_prompt_mask.device)

        if use_distill_topk:
            if self.ref_model is None:
                raise ValueError("distillation_topk requires a teacher ref_model.")
            teacher_topk_logps, topk_indices, teacher_per_token_logps = self._get_teacher_topk_distillation_tensors(
                self.ref_model,
                teacher_input_ids,
                teacher_attention_mask,
                logits_to_keep,
                self.distillation_topk,
                embed_perturb_mask=teacher_soft_mask,
                embed_perturb_vectors=teacher_soft_vectors,
                **multimodal_kw,
            )
            per_token_logps, student_topk_logps, entropies = self._get_student_logps_entropy_distill_topk(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                topk_indices,
                **multimodal_kw,
            )
            teacher_distill = _prepare_distillation_log_probs(teacher_topk_logps, self.distillation_add_tail)
            student_distill = _prepare_distillation_log_probs(student_topk_logps, self.distillation_add_tail)
        else:
            per_token_logps, all_logps, entropies, student_negative_energy = self._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                compute_entropy=True,
                return_energy=True,
                **multimodal_kw,
            )
            current_policy_all_logps = all_logps.detach()
            with torch.no_grad():
                teacher_per_token_logps, teacher_all_logps, _, teacher_negative_energy = self._get_per_token_logps_and_entropies(
                    self.ref_model,
                    teacher_input_ids,
                    teacher_attention_mask,
                    logits_to_keep,
                    compute_entropy=True,
                    return_energy=True,
                    embed_perturb_mask=teacher_soft_mask,
                    embed_perturb_vectors=teacher_soft_vectors,
                    **multimodal_kw,
                )
                should_log_effective_tau = (
                    self.effective_tau_metrics_path is not None
                    and teacher_soft_mask is not None
                    and teacher_soft_vectors is None
                )
                if should_log_effective_tau:
                    _, original_teacher_all_logps, _, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        teacher_input_ids,
                        teacher_attention_mask,
                        logits_to_keep,
                        compute_entropy=False,
                        return_energy=True,
                        embed_perturb_mask=None,
                        embed_perturb_vectors=None,
                        **multimodal_kw,
                    )
                    original_to_policy = current_policy_all_logps.float() - original_teacher_all_logps.float()
                    global_shift = teacher_all_logps.float() - original_teacher_all_logps.float()
                    original_to_policy = original_to_policy - original_to_policy.mean(dim=-1, keepdim=True)
                    global_shift = global_shift - global_shift.mean(dim=-1, keepdim=True)
                    effective_tau_direction_norm_sq = (original_to_policy * original_to_policy).sum(dim=-1)
                    denominator = effective_tau_direction_norm_sq.clamp(min=1e-8)
                    effective_tau = (global_shift * original_to_policy).sum(dim=-1) / denominator
                    effective_tau_clipped = effective_tau.clamp(0.0, 1.0)
                    residual = global_shift - effective_tau.unsqueeze(-1) * original_to_policy
                    residual_norm_sq = (residual * residual).sum(dim=-1)
                    global_shift_norm_sq = (global_shift * global_shift).sum(dim=-1).clamp(min=1e-8)
                    effective_tau_residual_ratio = (
                        torch.sqrt(residual_norm_sq.clamp(min=0.0))
                        / torch.sqrt(global_shift_norm_sq)
                    )
                    effective_tau_r2 = 1.0 - residual_norm_sq / global_shift_norm_sq
                    original_teacher_policy_kl = (
                        torch.exp(original_teacher_all_logps.float())
                        * (original_teacher_all_logps.float() - current_policy_all_logps.float())
                    ).sum(dim=-1)
            teacher_distill = teacher_all_logps
            student_distill = all_logps
            teacher_policy_kl = (
                torch.exp(teacher_all_logps.float())
                * (teacher_all_logps.float() - current_policy_all_logps.float())
            ).sum(dim=-1)
            if self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_ALL_MODES:
                if self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_ENTROPY_MODES:
                    peer_signal = entropies.float()
                elif self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_VAR_MODES:
                    peer_signal = self._per_token_log_ratio_variance(
                        current_policy_all_logps.float(),
                        teacher_all_logps.float(),
                    )
                elif self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_CHI2_MODES:
                    peer_signal = self._per_token_pearson_chi_square(
                        current_policy_all_logps.float(),
                        teacher_all_logps.float(),
                    )
                elif self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_HELLINGER_MODES:
                    peer_signal = self._per_token_hellinger_peer(
                        current_policy_all_logps.float(),
                        teacher_all_logps.float(),
                    )
                else:
                    peer_signal = teacher_policy_kl.float()
                if self.online_sample_tau_mode in {
                    "budget_forward_kl_relative_token",
                    "budget_forward_kl_relative_exp_token",
                    "budget_forward_kl_relative_exp_range_token",
                    "budget_forward_kl_relative_entropy_token",
                    "budget_log_ratio_var_relative_token",
                    "budget_log_ratio_var_relative_exp_token",
                    "budget_log_ratio_var_relative_exp_range_token",
                    "budget_chi2_relative_token",
                    "budget_chi2_relative_exp_token",
                    "budget_chi2_relative_exp_range_token",
                    "budget_hellinger_relative_token",
                    "budget_hellinger_relative_exp_token",
                    "budget_hellinger_relative_exp_range_token",
                }:
                    online_sample_tau_k = peer_signal
                    cached_k_bar = inputs.get("online_sample_tau_k_bar")
                    if cached_k_bar is not None:
                        online_sample_tau_k_bar = cached_k_bar.float().mean()
                    else:
                        per_sample_k = (
                            online_sample_tau_k * loss_completion_mask.float()
                        ).sum(dim=-1) / loss_completion_mask.sum(dim=-1).clamp(min=1.0)
                        online_sample_tau_k_bar = self._resolve_peer_signal_bar(per_sample_k)
                    if self.online_sample_tau_mode == "budget_forward_kl_relative_entropy_token":
                        budget_ratio = (
                            self.online_sample_tau_budget_alpha
                            * online_sample_tau_k_bar
                            / online_sample_tau_k.clamp(min=1e-8)
                        )
                        raw_tau = 1.0 - torch.sqrt(budget_ratio)
                        token_tau_values = raw_tau.clamp(min=0.0).clamp(
                            min=self.online_sample_tau_clip_min,
                            max=self.online_sample_tau_clip_max,
                        ).to(dtype=torch.float32)
                    else:
                        token_tau_values = self._peer_relative_budget_tau(
                            online_sample_tau_k,
                            online_sample_tau_k_bar,
                            self.online_sample_tau_budget_alpha,
                            self.online_sample_tau_mode,
                            tau_max=self.online_sample_tau_clip_max,
                            clip_min=self.online_sample_tau_clip_min,
                            clip_max=self.online_sample_tau_clip_max,
                        ).to(dtype=torch.float32)
                elif self.online_sample_tau_mode in {
                    "softmax_router_relative_token",
                    "softmax_router_relative_entropy_token",
                }:
                    online_sample_tau_k = peer_signal
                    cached_k_bar = inputs.get("online_sample_tau_k_bar")
                    if cached_k_bar is not None:
                        online_sample_tau_k_bar = cached_k_bar.float().mean()
                    else:
                        per_sample_k = (
                            online_sample_tau_k * loss_completion_mask.float()
                        ).sum(dim=-1) / loss_completion_mask.sum(dim=-1).clamp(min=1.0)
                        online_sample_tau_k_bar = self._resolve_peer_signal_bar(per_sample_k)
                    token_tau_values = self._peer_relative_softmax_router_tau(
                        online_sample_tau_k,
                        online_sample_tau_k_bar,
                        self.online_sample_tau_budget_alpha,
                        gamma=self.online_sample_tau_router_gamma,
                        clip_min=self.online_sample_tau_clip_min,
                        clip_max=self.online_sample_tau_clip_max,
                    ).to(dtype=torch.float32)
                elif self.online_sample_tau_mode in {
                    "softmax_router_relative",
                    "softmax_router_relative_entropy",
                }:
                    cached_k = inputs.get("online_sample_tau_k")
                    cached_k_bar = inputs.get("online_sample_tau_k_bar")
                    if cached_k is not None:
                        online_sample_tau_k = cached_k.float()
                    else:
                        online_sample_tau_k = (
                            peer_signal * loss_completion_mask.float()
                        ).sum(dim=-1) / loss_completion_mask.sum(dim=-1).clamp(min=1.0)
                    if cached_k_bar is not None:
                        online_sample_tau_k_bar = cached_k_bar.float().mean()
                    else:
                        online_sample_tau_k_bar = self._resolve_peer_signal_bar(online_sample_tau_k)
                    sample_tau_values = self._peer_relative_softmax_router_tau(
                        online_sample_tau_k,
                        online_sample_tau_k_bar,
                        self.online_sample_tau_budget_alpha,
                        gamma=self.online_sample_tau_router_gamma,
                        clip_min=self.online_sample_tau_clip_min,
                        clip_max=self.online_sample_tau_clip_max,
                    ).to(dtype=torch.float32)
                else:
                    cached_k = inputs.get("online_sample_tau_k")
                    cached_k_bar = inputs.get("online_sample_tau_k_bar")
                    if cached_k is not None:
                        online_sample_tau_k = cached_k.float()
                    else:
                        online_sample_tau_k = (
                            peer_signal * loss_completion_mask.float()
                        ).sum(dim=-1) / loss_completion_mask.sum(dim=-1).clamp(min=1.0)
                    if self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_FIXED_C_MODES:
                        sample_tau_values = self._fixed_c_budget_tau(
                            online_sample_tau_k,
                            self.online_sample_tau_budget_c,
                            self.online_sample_tau_mode,
                            tau_max=self.online_sample_tau_clip_max,
                            clip_min=self.online_sample_tau_clip_min,
                            clip_max=self.online_sample_tau_clip_max,
                        ).to(dtype=torch.float32)
                    elif self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_RELATIVE_BUDGET_MODES:
                        if cached_k_bar is not None:
                            online_sample_tau_k_bar = cached_k_bar.float().mean()
                        else:
                            online_sample_tau_k_bar = self._resolve_peer_signal_bar(online_sample_tau_k)
                        cached_k0 = inputs.get("online_sample_tau_k0")
                        cached_k0_bar = inputs.get("online_sample_tau_k0_bar")
                        k0_for_ratio = None
                        k0_bar_for_ratio = None
                        if cached_k0 is not None:
                            online_sample_tau_k0 = cached_k0.float()
                            if self.k0_kl_global_mean is not None:
                                online_sample_tau_k0_bar = self.k0_kl_global_mean
                                k0_bar_for_ratio = self.k0_kl_global_mean
                            elif cached_k0_bar is not None:
                                online_sample_tau_k0_bar = cached_k0_bar.float().mean()
                                k0_bar_for_ratio = float(online_sample_tau_k0_bar.item())
                            else:
                                gathered_k0 = self.accelerator.gather(online_sample_tau_k0.detach().float())
                                online_sample_tau_k0_bar = gathered_k0.mean()
                                k0_bar_for_ratio = float(online_sample_tau_k0_bar.item())
                            k0_for_ratio = online_sample_tau_k0
                        sample_tau_values = self._peer_relative_budget_tau(
                            online_sample_tau_k,
                            online_sample_tau_k_bar,
                            self.online_sample_tau_budget_alpha,
                            self.online_sample_tau_mode,
                            tau_max=self.online_sample_tau_clip_max,
                            clip_min=self.online_sample_tau_clip_min,
                            clip_max=self.online_sample_tau_clip_max,
                            k0=k0_for_ratio,
                            k0_bar_for_ratio=k0_bar_for_ratio,
                            k0_gamma=self.online_sample_tau_k0_gamma,
                        ).to(dtype=torch.float32)
                    else:
                        if cached_k_bar is not None:
                            online_sample_tau_k_bar = cached_k_bar.float().mean()
                        else:
                            online_sample_tau_k_bar = self._resolve_peer_signal_bar(online_sample_tau_k)
                        budget_ratio = (
                            self.online_sample_tau_budget_alpha
                            * online_sample_tau_k_bar
                            / online_sample_tau_k.clamp(min=1e-8)
                        )
                        cached_k0 = inputs.get("online_sample_tau_k0")
                        cached_k0_bar = inputs.get("online_sample_tau_k0_bar")
                        if cached_k0 is not None:
                            online_sample_tau_k0 = cached_k0.float()
                            if self.k0_kl_global_mean is not None:
                                online_sample_tau_k0_bar = self.k0_kl_global_mean
                                k0_bar_for_ratio = self.k0_kl_global_mean
                            elif cached_k0_bar is not None:
                                online_sample_tau_k0_bar = cached_k0_bar.float().mean()
                                k0_bar_for_ratio = online_sample_tau_k0_bar
                            else:
                                gathered_k0 = self.accelerator.gather(online_sample_tau_k0.detach().float())
                                online_sample_tau_k0_bar = gathered_k0.mean()
                                k0_bar_for_ratio = online_sample_tau_k0_bar
                            k0_prior_ratio = (
                                k0_bar_for_ratio / online_sample_tau_k0.clamp(min=1e-8)
                            ).pow(self.online_sample_tau_k0_gamma)
                            budget_ratio = budget_ratio * k0_prior_ratio
                        raw_tau = 1.0 - torch.sqrt(budget_ratio)
                        raw_tau = raw_tau.clamp(min=0.0)
                        sample_tau_values = raw_tau.clamp(
                            min=self.online_sample_tau_clip_min,
                            max=self.online_sample_tau_clip_max,
                        ).to(dtype=torch.float32)
            teacher_probs = torch.exp(teacher_all_logps.float())
            teacher_entropy = -(teacher_probs * teacher_all_logps.float()).sum(dim=-1)
            teacher_max_prob = torch.exp(teacher_all_logps.float().max(dim=-1).values)
            use_proximal_teacher = (
                self.proximal_teacher_tau != 0.0
                or sample_tau_values is not None
                or self.uses_online_sample_tau
            )
            if use_proximal_teacher:
                if token_tau_values is not None:
                    tau = token_tau_values.to(device=teacher_all_logps.device, dtype=teacher_all_logps.dtype)
                    tau = tau.unsqueeze(-1)
                elif sample_tau_values is not None:
                    tau = sample_tau_values.to(device=teacher_all_logps.device, dtype=teacher_all_logps.dtype)
                    tau = tau.view(-1, 1, 1)
                else:
                    tau = torch.tensor(
                        self.proximal_teacher_tau,
                        device=teacher_all_logps.device,
                        dtype=teacher_all_logps.dtype,
                    )
                teacher_distill = self._build_proximal_teacher_log_probs(
                    teacher_all_logps,
                    current_policy_all_logps,
                    tau,
                    self._resolve_proximal_selection(),
                )
                teacher_per_token_logps = torch.gather(
                    teacher_distill, dim=-1, index=completion_ids.unsqueeze(-1)
                ).squeeze(-1)
                qtau_probs = torch.exp(teacher_distill.float())
                qtau_entropy = -(qtau_probs * teacher_distill.float()).sum(dim=-1)
                qtau_max_prob = torch.exp(teacher_distill.float().max(dim=-1).values)
                qtau_to_student_kl = (
                    qtau_probs * (teacher_distill.float() - all_logps.float())
                ).sum(dim=-1)

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, loss_completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Distillation loss D_f(teacher || student) or Squared Hellinger.
        # PyTorch F.kl_div swaps the usual math order vs the paper; keep that convention for KL.
        if self.distillation_divergence == "hellinger":
            # H^2(p, q) = 1 - sum_v sqrt(p(v) q(v)) = (1/2) ||sqrt(p)-sqrt(q)||^2
            # Symmetric; used as D_f between student p and teacher q (q_tau or q_o).
            student_probs = student_distill.float().exp().clamp(min=1e-12)
            teacher_probs = teacher_distill.float().exp().clamp(min=1e-12)
            per_token_loss = (
                1.0 - (student_probs.sqrt() * teacher_probs.sqrt()).sum(dim=-1)
            ).to(dtype=student_distill.dtype)
        elif self.alpha == 0:  # Forward KL
            kl_loss = kl_div(student_distill, teacher_distill, reduction="none", log_target=True)
            per_token_loss = kl_loss.sum(-1)
        elif self.alpha == 1:  # Reverse KL
            kl_loss = kl_div(teacher_distill, student_distill, reduction="none", log_target=True)
            per_token_loss = kl_loss.sum(-1)
        else:
            alpha_t = torch.tensor(
                self.alpha, dtype=student_distill.dtype, device=student_distill.device
            )
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_distill + torch.log(1 - alpha_t), teacher_distill + torch.log(alpha_t)]),
                dim=0,
            )
            kl_teacher = kl_div(mixture_log_probs, teacher_distill, reduction="none", log_target=True)
            kl_student = kl_div(mixture_log_probs, student_distill, reduction="none", log_target=True)
            kl_loss = alpha_t * kl_teacher + (1 - alpha_t) * kl_student
            per_token_loss = kl_loss.sum(-1)

        if self.use_vllm and self.vllm_importance_sampling_correction and not self.generate_from_teacher:
            ratio = inputs["importance_sampling_ratio"]
            importance_weights = (ratio * loss_completion_mask).sum(-1) / loss_completion_mask.sum(-1).clamp(min=1.0)
            importance_weights = importance_weights.unsqueeze(-1)
            per_token_loss = per_token_loss * importance_weights

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        per_sample_loss = (
            (per_token_loss * loss_completion_mask).sum(-1)
            / loss_completion_mask.sum(-1).clamp(min=1.0)
        )
        cached_batch_kl_filter_token_keep = inputs.get("batch_kl_filter_token_keep")
        cached_batch_kl_filter_keep = inputs.get("batch_kl_filter_keep")
        if cached_batch_kl_filter_token_keep is not None:
            token_keep_weight = cached_batch_kl_filter_token_keep.to(
                device=per_token_loss.device, dtype=per_token_loss.dtype
            )
            effective_mask = loss_completion_mask.float() * token_keep_weight
            per_sample_loss = (per_token_loss * effective_mask).sum(-1) / effective_mask.sum(-1).clamp(
                min=1.0
            )
            train_batch_loss = per_sample_loss.mean()
        elif cached_batch_kl_filter_keep is not None:
            keep_weight = cached_batch_kl_filter_keep.to(
                device=per_sample_loss.device, dtype=per_sample_loss.dtype
            ).view(-1)
            # Sum weighted losses; do not divide by keep_weight here. With bs=1 micro-batches,
            # (w*L)/w would cancel soft drop_weight; hard mask (w=0) still yields zero contribution.
            train_batch_loss = (per_sample_loss * keep_weight).sum()
        else:
            train_batch_loss = per_sample_loss.mean()
        loss = train_batch_loss / self.current_gradient_accumulation_steps

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        with torch.no_grad():
            kl_approx = (per_token_logps - teacher_per_token_logps) + torch.exp(teacher_per_token_logps - per_token_logps) - 1
            kl_approx_mean = (kl_approx * loss_completion_mask).sum() / loss_completion_mask.sum()
        self._metrics[mode]["kl_approx"].append(self.accelerator.gather(kl_approx_mean).nanmean().item())
        step = int(self.state.global_step)
        should_probe = (
            mode == "train"
            and self.probe_every_steps > 0
            and step % self.probe_every_steps == 0
            and self._last_probe_step != step
        )
        if should_probe:
            new_probe_loss = self._compute_new_probe_loss(model)
            old_probe_loss = self._compute_old_probe_loss(model)
            self._last_probe_step = step
            if new_probe_loss is not None:
                self._metrics[mode]["probe/l_new"].append(new_probe_loss)
            if old_probe_loss is not None:
                self._metrics[mode]["probe/l_old"].append(old_probe_loss)
        
        loss_completion_token_count = loss_completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * loss_completion_mask).sum() / loss_completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl_to_base_model"].append(self.accelerator.gather(mean_kl).nanmean().item())

        if token_tau_values is not None:
            tau_metric = masked_batch_mean(token_tau_values)
            self._metrics[mode]["proximal_teacher/tau"].append(
                self.accelerator.gather(tau_metric).nanmean().item()
            )
            token_tau_std = (
                (token_tau_values.float() - tau_metric.unsqueeze(-1)).pow(2) * loss_completion_mask.float()
            ).sum() / loss_completion_mask.sum().clamp(min=1.0)
            self._metrics[mode]["proximal_teacher/token_tau_std"].append(
                self.accelerator.gather(torch.sqrt(token_tau_std.clamp(min=0.0))).nanmean().item()
            )
        elif sample_tau_values is not None:
            tau_metric = sample_tau_values.float().mean()
            self._metrics[mode]["proximal_teacher/tau"].append(
                self.accelerator.gather(tau_metric).nanmean().item()
            )
        else:
            self._metrics[mode]["proximal_teacher/tau"].append(self.proximal_teacher_tau)
        if online_sample_tau_k_bar is not None:
            if self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_ENTROPY_MODES:
                peer_bar_metric = "proximal_teacher/online_sample_tau_h_bar"
            elif self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_VAR_MODES:
                peer_bar_metric = "proximal_teacher/online_sample_tau_v_bar"
            elif self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_CHI2_MODES:
                peer_bar_metric = "proximal_teacher/online_sample_tau_chi2_bar"
            elif self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_HELLINGER_MODES:
                peer_bar_metric = "proximal_teacher/online_sample_tau_peer_h_bar"
            else:
                peer_bar_metric = "proximal_teacher/online_sample_tau_k_bar"
            self._metrics[mode][peer_bar_metric].append(float(online_sample_tau_k_bar.item()))
        if online_sample_tau_k0_bar is not None:
            k0_bar_metric = (
                float(online_sample_tau_k0_bar)
                if isinstance(online_sample_tau_k0_bar, (int, float))
                else float(online_sample_tau_k0_bar.item())
            )
            self._metrics[mode]["proximal_teacher/online_sample_tau_k0_bar"].append(k0_bar_metric)
        if teacher_policy_kl is not None:
            mean_teacher_policy_kl = masked_batch_mean(teacher_policy_kl)
            self._metrics[mode]["proximal_teacher/kl_qo_to_current_policy"].append(
                self.accelerator.gather(mean_teacher_policy_kl).nanmean().item()
            )
        if qtau_to_student_kl is not None:
            self._metrics[mode]["proximal_teacher/kl_qtau_to_student"].append(
                self.accelerator.gather(masked_batch_mean(qtau_to_student_kl)).nanmean().item()
            )

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())
        if student_negative_energy is not None:
            self._metrics[mode]["energy/student_negative_energy"].append(
                self.accelerator.gather(masked_batch_mean(student_negative_energy)).nanmean().item()
            )
        if teacher_negative_energy is not None:
            self._metrics[mode]["energy/teacher_negative_energy"].append(
                self.accelerator.gather(masked_batch_mean(teacher_negative_energy)).nanmean().item()
            )
        if effective_tau is not None:
            self._metrics[mode]["effective_tau/mean"].append(
                self.accelerator.gather(masked_batch_mean(effective_tau)).nanmean().item()
            )
            self._metrics[mode]["effective_tau/clipped_mean"].append(
                self.accelerator.gather(masked_batch_mean(effective_tau_clipped)).nanmean().item()
            )
            self._metrics[mode]["effective_tau/residual_ratio"].append(
                self.accelerator.gather(masked_batch_mean(effective_tau_residual_ratio)).nanmean().item()
            )
            self._metrics[mode]["effective_tau/r2"].append(
                self.accelerator.gather(masked_batch_mean(effective_tau_r2)).nanmean().item()
            )
        if original_teacher_policy_kl is not None:
            self._metrics[mode]["effective_tau/kl_q_original_to_current_policy"].append(
                self.accelerator.gather(masked_batch_mean(original_teacher_policy_kl)).nanmean().item()
            )

        if mode == "train" and self.sample_metrics_path:
            def masked_sequence_mean(x: torch.Tensor) -> torch.Tensor:
                if x.shape[1] == 1:
                    return x.float().squeeze(1)
                return (x.float() * loss_completion_mask.float()).sum(dim=-1) / loss_completion_mask.sum(dim=-1).clamp(min=1.0)

            def masked_sequence_std(x: torch.Tensor) -> torch.Tensor:
                if x.shape[1] == 1:
                    return torch.zeros_like(x.float().squeeze(1))
                means = masked_sequence_mean(x).unsqueeze(1)
                centered = (x.float() - means) * loss_completion_mask.float()
                variance = (centered * centered).sum(dim=-1) / loss_completion_mask.sum(dim=-1).clamp(min=1.0)
                return torch.sqrt(variance.clamp(min=0.0))

            def masked_weighted_sequence_mean(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
                masked_weights = weights.float() * loss_completion_mask.float()
                numerator = (x.float() * masked_weights).sum(dim=-1)
                denominator = masked_weights.sum(dim=-1).clamp(min=1e-8)
                return numerator / denominator

            sample_ids = inputs.get("soft_teacher_row_ids")
            if sample_ids is None:
                sample_ids = torch.full((per_token_loss.shape[0],), -1, device=per_token_loss.device, dtype=torch.long)
            if token_tau_values is not None:
                sample_tau_for_log = (
                    (token_tau_values.float() * loss_completion_mask.float()).sum(dim=-1)
                    / loss_completion_mask.sum(dim=-1).clamp(min=1.0)
                )
            else:
                sample_tau_for_log = sample_tau_values
            if sample_tau_for_log is None:
                sample_tau_for_log = torch.full(
                    (per_token_loss.shape[0],),
                    float(self.proximal_teacher_tau),
                    device=per_token_loss.device,
                    dtype=torch.float32,
                )
            sample_avg64 = inputs.get("sample_avg64")
            sample_uncertainty = inputs.get("sample_uncertainty")
            if sample_avg64 is None:
                sample_avg64 = torch.full_like(sample_tau_for_log.float(), float("nan"))
            if sample_uncertainty is None:
                sample_uncertainty = torch.full_like(sample_tau_for_log.float(), float("nan"))

            student_max_prob = torch.exp(all_logps.float().max(dim=-1).values)
            sample_rank = torch.full_like(sample_ids.long(), int(self.accelerator.process_index))
            row_tensors = {
                "rank": sample_rank.long(),
                "sample_id": sample_ids.long(),
                "sample_tau": sample_tau_for_log.float(),
                "sample_avg64": sample_avg64.float(),
                "sample_uncertainty": sample_uncertainty.float(),
                "completion_tokens": loss_completion_mask.sum(dim=-1).long(),
                "train_loss": masked_sequence_mean(per_token_loss).float(),
                "student_entropy": masked_sequence_mean(entropies).float(),
                "student_max_prob": masked_sequence_mean(student_max_prob).float(),
            }
            if student_negative_energy is not None:
                row_tensors["student_negative_energy"] = masked_sequence_mean(student_negative_energy).float()
            if teacher_negative_energy is not None:
                row_tensors["teacher_negative_energy"] = masked_sequence_mean(teacher_negative_energy).float()
                if student_negative_energy is not None:
                    row_tensors["teacher_minus_student_negative_energy"] = (
                        masked_sequence_mean(teacher_negative_energy)
                        - masked_sequence_mean(student_negative_energy)
                    ).float()
            if teacher_policy_kl is not None:
                row_tensors["kl_qo_to_current_policy"] = masked_sequence_mean(teacher_policy_kl).float()
            if online_sample_tau_k is not None:
                if self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_ENTROPY_MODES:
                    peer_key = "online_sample_tau_h"
                    peer_bar_key = "online_sample_tau_h_bar"
                elif self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_VAR_MODES:
                    peer_key = "online_sample_tau_v"
                    peer_bar_key = "online_sample_tau_v_bar"
                elif self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_CHI2_MODES:
                    peer_key = "online_sample_tau_chi2"
                    peer_bar_key = "online_sample_tau_chi2_bar"
                elif self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_HELLINGER_MODES:
                    peer_key = "online_sample_tau_peer_h"
                    peer_bar_key = "online_sample_tau_peer_h_bar"
                else:
                    peer_key = "online_sample_tau_k"
                    peer_bar_key = "online_sample_tau_k_bar"
                if online_sample_tau_k.dim() == 1:
                    row_tensors[peer_key] = online_sample_tau_k.float()
                else:
                    row_tensors[peer_key] = masked_sequence_mean(online_sample_tau_k).float()
                    if token_tau_values is not None:
                        row_tensors["online_token_tau_std"] = masked_sequence_std(token_tau_values).float()
            if online_sample_tau_k_bar is not None:
                k_bar_scalar = float(online_sample_tau_k_bar.item())
                if self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_ENTROPY_MODES:
                    peer_bar_key = "online_sample_tau_h_bar"
                elif self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_VAR_MODES:
                    peer_bar_key = "online_sample_tau_v_bar"
                elif self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_CHI2_MODES:
                    peer_bar_key = "online_sample_tau_chi2_bar"
                elif self.online_sample_tau_mode in ONLINE_SAMPLE_TAU_HELLINGER_MODES:
                    peer_bar_key = "online_sample_tau_peer_h_bar"
                else:
                    peer_bar_key = "online_sample_tau_k_bar"
                if online_sample_tau_k.dim() == 1:
                    row_tensors[peer_bar_key] = torch.full_like(
                        online_sample_tau_k.float(),
                        k_bar_scalar,
                    )
                else:
                    row_tensors[peer_bar_key] = torch.full(
                        (online_sample_tau_k.shape[0],),
                        k_bar_scalar,
                        device=online_sample_tau_k.device,
                        dtype=torch.float32,
                    )
            if online_sample_tau_k0 is not None:
                row_tensors["online_sample_tau_k0"] = online_sample_tau_k0.float()
            if online_sample_tau_k0_bar is not None:
                k0_bar_log = (
                    float(online_sample_tau_k0_bar)
                    if isinstance(online_sample_tau_k0_bar, (int, float))
                    else float(online_sample_tau_k0_bar.item())
                )
                row_tensors["online_sample_tau_k0_bar"] = torch.full_like(
                    online_sample_tau_k0.float(),
                    k0_bar_log,
                )
            if qtau_to_student_kl is not None:
                row_tensors["kl_qtau_to_student"] = masked_sequence_mean(qtau_to_student_kl).float()
            if teacher_entropy is not None:
                row_tensors["teacher_entropy"] = masked_sequence_mean(teacher_entropy).float()
            if qtau_entropy is not None:
                row_tensors["qtau_entropy"] = masked_sequence_mean(qtau_entropy).float()
            if teacher_max_prob is not None:
                row_tensors["teacher_max_prob"] = masked_sequence_mean(teacher_max_prob).float()
            if qtau_max_prob is not None:
                row_tensors["qtau_max_prob"] = masked_sequence_mean(qtau_max_prob).float()
            if effective_tau is not None:
                row_tensors["effective_tau_mean"] = masked_sequence_mean(effective_tau).float()
                row_tensors["effective_tau_std"] = masked_sequence_std(effective_tau).float()
                row_tensors["effective_tau_clipped_mean"] = masked_sequence_mean(effective_tau_clipped).float()
                row_tensors["effective_tau_residual_ratio"] = masked_sequence_mean(effective_tau_residual_ratio).float()
                row_tensors["effective_tau_r2"] = masked_sequence_mean(effective_tau_r2).float()
                row_tensors["effective_tau_weighted_mean"] = masked_weighted_sequence_mean(
                    effective_tau,
                    effective_tau_direction_norm_sq,
                ).float()
                row_tensors["effective_tau_clipped_weighted_mean"] = masked_weighted_sequence_mean(
                    effective_tau_clipped,
                    effective_tau_direction_norm_sq,
                ).float()
            if original_teacher_policy_kl is not None:
                row_tensors["kl_q_original_to_current_policy"] = masked_sequence_mean(original_teacher_policy_kl).float()

            gathered_rows = {
                key: self.accelerator.gather(value.detach()).cpu().tolist()
                for key, value in row_tensors.items()
            }
            if self.accelerator.is_main_process:
                sample_metrics_dir = os.path.dirname(self.sample_metrics_path)
                if sample_metrics_dir:
                    os.makedirs(sample_metrics_dir, exist_ok=True)
                with open(self.sample_metrics_path, "a", encoding="utf-8") as handle:
                    batch_size = len(gathered_rows["sample_id"])
                    for row_idx in range(batch_size):
                        row = {
                            "step": int(self.state.global_step),
                            "micro_step": int(self._step),
                        }
                        for key, values in gathered_rows.items():
                            value = values[row_idx]
                            if key in {"rank", "sample_id", "completion_tokens"}:
                                row[key] = int(value)
                            else:
                                row[key] = float(value)
                        handle.write(json.dumps(row) + "\n")

        if mode == "train" and self.effective_tau_metrics_path and effective_tau is not None:
            sample_ids = inputs.get("soft_teacher_row_ids")
            if sample_ids is None:
                sample_ids = torch.full((effective_tau.shape[0],), -1, device=effective_tau.device, dtype=torch.long)
            local_rows: list[dict[str, Any]] = []
            token_ids_cpu = completion_ids.detach().cpu()
            mask_cpu = loss_completion_mask.detach().cpu().bool()
            tau_cpu = effective_tau.detach().float().cpu()
            tau_clipped_cpu = effective_tau_clipped.detach().float().cpu()
            residual_cpu = effective_tau_residual_ratio.detach().float().cpu()
            r2_cpu = effective_tau_r2.detach().float().cpu()
            direction_norm_sq_cpu = effective_tau_direction_norm_sq.detach().float().cpu()
            sample_ids_cpu = sample_ids.detach().cpu().long()
            for row_idx in range(effective_tau.shape[0]):
                valid = mask_cpu[row_idx]
                local_rows.append(
                    {
                        "step": int(self.state.global_step),
                        "micro_step": int(self._step),
                        "rank": int(self.accelerator.process_index),
                        "sample_id": int(sample_ids_cpu[row_idx]),
                        "token_ids": token_ids_cpu[row_idx][valid].tolist(),
                        "effective_tau": tau_cpu[row_idx][valid].tolist(),
                        "effective_tau_clipped": tau_clipped_cpu[row_idx][valid].tolist(),
                        "effective_tau_residual_ratio": residual_cpu[row_idx][valid].tolist(),
                        "effective_tau_r2": r2_cpu[row_idx][valid].tolist(),
                        "effective_tau_direction_norm_sq": direction_norm_sq_cpu[row_idx][valid].tolist(),
                    }
                )
            gathered_detail_rows = gather_object(local_rows)
            if self.accelerator.is_main_process:
                effective_tau_dir = os.path.dirname(self.effective_tau_metrics_path)
                if effective_tau_dir:
                    os.makedirs(effective_tau_dir, exist_ok=True)
                with open(self.effective_tau_metrics_path, "a", encoding="utf-8") as handle:
                    for item in gathered_detail_rows:
                        if isinstance(item, dict):
                            handle.write(json.dumps(item) + "\n")
                        else:
                            for row in item:
                                handle.write(json.dumps(row) + "\n")

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if mode == "train" and ("probe/l_new" in logs or "probe/l_old" in logs):
            os.makedirs(os.path.dirname(self.probe_metrics_path), exist_ok=True)
            row = {
                "step": int(self.state.global_step),
                "tau": logs.get("proximal_teacher/tau", float(self.proximal_teacher_tau)),
                "l_new": logs.get("probe/l_new"),
                "l_old": logs.get("probe/l_old"),
                "train_loss": logs.get("loss"),
                "kl_approx": logs.get("kl_approx"),
                "kl_qo_to_current_policy": logs.get("proximal_teacher/kl_qo_to_current_policy"),
                "kl_qtau_to_student": logs.get("proximal_teacher/kl_qtau_to_student"),
            }
            with open(self.probe_metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._logs["prompt"],
                    self._logs["completion"],
                    self._logs["rewards"],
                    self._logs["advantages"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._logs["prompt"]),
                    "prompt": self._logs["prompt"],
                    "completion": self._logs["completion"],
                    **self._logs["rewards"],
                    "advantage": self._logs["advantages"],
                }

                if self._logs["images"]:
                    table["images"] = []
                    for image_list in self._logs["images"]:
                        # Convert images to wandb Image objects for proper visualization
                        table["images"].append([wandb.Image(image) for image in image_list])

                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
