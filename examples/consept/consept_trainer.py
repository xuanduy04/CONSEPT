import inspect
import os
import textwrap
from collections import defaultdict, deque
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union, TYPE_CHECKING

import multiprocessing

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

from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
    prepare_multimodal_messages,
)
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_liger_kernel_available, is_vllm_available
from trl.models import prepare_deepspeed, prepare_fsdp, prepare_peft_model, unwrap_model_for_generation
from trl.models.utils import _ForwardRedirection
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
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

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, TrainOutput

from trl import GRPOTrainer
from .completion_length_scheduler import ConstantCompletionLengthScheduler
from .consept_config import CONSEPTConfig
from .sampler import DynamicRepeatSampler
from .collator import PromptSolutionCollator
from .log_samples import print_prompt_completion_solutions_sample
from .utils import save_dict_to_json, load_dict_from_json

if is_peft_available():
    from peft import PeftConfig, PeftModel

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb


if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized
    import optuna
    from trl.trainer.grpo_trainer import RewardFunc
    from .completion_length_scheduler import CompletionLengthScheduler


logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
COMPLETION_LENGTH_SCHEDULER_NAME = "completion_length_scheduler.pt"


class CONSEPTTrainer(GRPOTrainer):
    r"""
    CONtiual SEmantic PreTraining's Trainer.

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
        args ([`CONSEPTConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        completion_length_scheduler_cls: (type["CompletionLengthScheduler"], *optional*):
            The `completion_length_scheduler` class. If `None`, the `completion_length` will never change during training.
        completion_length_scheduler_kwargs: (dict[str, Any], *optional*):
            Additional keyword arguments for the `completion_length_scheduler` to use during its initialization.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"text"`. Any additional columns in the dataset is
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
        reward_processing_classes ([`~transformers.PreTrainedTokenizerBase`] or `list[PreTrainedTokenizerBase]`, *optional*):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using
            [`~transformers.AutoTokenizer.from_pretrained`]. For elements in `reward_funcs` that are custom reward
            functions (not [`~transformers.PreTrainedModel`]), the corresponding entries in `reward_processing_classes`
            are ignored.
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

    _name = "CONSEPT"

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union["RewardFunc", list["RewardFunc"]],
        args: Optional[CONSEPTConfig] = None,
        completion_length_scheduler_cls: Optional[type["CompletionLengthScheduler"]] = None,
        completion_length_scheduler_kwargs: Optional[dict[str, Any]] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = CONSEPTConfig(output_dir=f"{model_name}-CONSEPT")

        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        # Keep logs sized to the generation batch to record only outputs from the latest model update.
        self._logs["solution"] = deque(maxlen=args.generation_batch_size)

        # Dynamically changing the completion length during training
        self.initial_completion_length = self.args.initial_completion_length
        self._max_completion_length = self.max_completion_length
        self.prompt_length_remove_threshold = self.args.prompt_length_remove_threshold
        # vvvvvv This value is synced across workers to ensure the DataLoader only returns samples that satisfies our dynamically changing constraint.
        self.current_completion_length: "Synchronized" = multiprocessing.Value("i", self.initial_completion_length)
        # ^^^^^^
        self._set_max_completion_length()
        # NOTE! Because of backwards compatibility, please never use the variable `max_completion_length` directly

        if completion_length_scheduler_cls is None:
            # if no scheduler class is specified, the completion length will never change
            completion_length_scheduler_cls = ConstantCompletionLengthScheduler

        if completion_length_scheduler_kwargs is None:
            completion_length_scheduler_kwargs = {}

        self.completion_length_scheduler: "CompletionLengthScheduler" = completion_length_scheduler_cls(
            completion_length=self.current_completion_length,
            max_completion_length=self._max_completion_length,
            initial_completion_length=self.initial_completion_length,
            **completion_length_scheduler_kwargs,
        )

    # ================== SAVE completion_length_scheduler ================== #
    def _save_completion_length_scheduler(self, run_dir: str) -> None:
        """Saves the `completion_length_scheduler` at the same place as model checkpoints & training args"""
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        output_dir = os.path.join(run_dir, checkpoint_folder)
        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, COMPLETION_LENGTH_SCHEDULER_NAME))

    def _maybe_log_save_evaluate(
        self,
        tr_loss: torch.Tensor,
        grad_norm: torch.Tensor | float | None,
        model: nn.Module,
        trial: "optuna.Trial | dict[str, Any] | None",
        epoch: float,
        ignore_keys_for_eval: list[str] | None,
        start_time: float,
        learning_rate: float | None = None,
    ) -> None:
        if self.control.should_save and self.args.should_save and (not self.args.save_only_model):
            # according to super()._maybe_log_save_evaluate(...), this is the complete save condition
            run_dir = self._get_output_dir(trial=trial)
            self._save_completion_length_scheduler(run_dir=run_dir)

        super()._maybe_log_save_evaluate(
            tr_loss=tr_loss,
            grad_norm=grad_norm,
            model=model,
            trial=trial,
            epoch=epoch,
            ignore_keys_for_eval=ignore_keys_for_eval,
            start_time=start_time,
            learning_rate=learning_rate,
        )
        # NOTE: We update the completion length here,
        #       as this function called after sync_step's end.
        # Treat it like an... evaluation, ya know?
        self.completion_length_scheduler.step()

    # ================== LOAD completion_length_scheduler ================== #
    # These next 2 functions loads the completion_length_scheduler
    def _prepare_for_training(self, max_steps, train_dataloader, resume_from_checkpoint):
        output = super()._prepare_for_training(
            max_steps=max_steps, train_dataloader=train_dataloader, resume_from_checkpoint=resume_from_checkpoint
        )
        if resume_from_checkpoint is not None and resume_from_checkpoint is not False:
            self._load_completion_length_scheduler(resume_from_checkpoint)
        return output

    def _load_completion_length_scheduler(self, resume_from_checkpoint) -> None:
        """Loads the `completion_length_scheduler`"""
        self.completion_length_scheduler.load_state_dict(
            torch.load(os.path.join(resume_from_checkpoint, COMPLETION_LENGTH_SCHEDULER_NAME), weights_only=True)
        )
        self._set_max_completion_length()

    # ================== USE completion_length_scheduler ================== #
    def _set_max_completion_length(self):
        """Sync completion length, ensuring that new generations abide by
        the `current_completion_length`'s value"""
        value = self.current_completion_length.value

        self.max_completion_length = value
        if not self.use_vllm:
            self.generation_config.update(max_new_tokens=value)

    # This method overrides `GRPOTrainer._generate` to support dynamic completion length
    # Maintenance note: This method is a copy-paste of the original `GRPOTrainer._generate`
    # with only a single line modification.
    def _generate(self, prompts: list[str], images: Optional[list]):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        self._set_max_completion_length()  # <--- HERE is the modification
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

    # ================== Dynamic DataLoader ================== #
    # This method overrides `GRPOTrainer.get_train_dataloader` to support dynamic completion length
    # Maintenance note: This method is a copy-paste of the original `GRPOTrainer.get_train_dataloader`
    # with only one modification.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        # vvvvvv Here is the change
        data_collator = PromptSolutionCollator(
            data_collator, self.processing_class, self.current_completion_length, self.prompt_length_remove_threshold
        )
        # ^^^^^^

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,
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
        # Returns the GRPO Sampler, but "Dynamic" (i.e. only returns samples that satisfy a given function)
        if dataset is None:
            dataset = self.train_dataset

        def valid_item_fn(item: str) -> bool:
            return (
                len(self.processing_class.encode(item)) - self.current_completion_length.value
                >= self.prompt_length_remove_threshold
            )

        return DynamicRepeatSampler(
            data_source=dataset,
            valid_item_fn=valid_item_fn,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    # ================== Custom Logging ================== #
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Does exactly the same as `GRPOTrainer._generate_and_score_completions`
        output = super()._generate_and_score_completions(inputs=inputs)
        # but adds ground truths to logs
        self._logs["solution"].extend(gather_object([x["solution"] for x in inputs]))
        return output

    # This method overrides `GRPOTrainer.log` to support our logging (as we do not use chat template)
    # Maintenance note: This method is a copy-paste of the original `GRPOTrainer.log`
    # with only 3 modifications (add completion length to logs & changing the terminal/ console logging function).
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        # vvvvvv Here's a change
        if mode == "train":
            logs["training_completion_length"] = self.current_completion_length.value
        # ^^^^^^

        logs = {**logs, **metrics}
        # vvvvvv Here is another change. Though this one IS similar in function to GRPOTrainer,
        # calling BaseTrainer.log()
        super(GRPOTrainer, self).log(logs, start_time)
        # ^^^^^^
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                # vvvvvv Here is another change
                print_prompt_completion_solutions_sample(
                    self._logs["prompt"],
                    self._logs["completion"],
                    self._logs["solution"],
                    self._logs["rewards"],
                    self._logs["advantages"],
                    self.state.global_step,
                    self.processing_class.eos_token,
                    self.num_completions_to_print,
                )
                # ^^^^^^

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
