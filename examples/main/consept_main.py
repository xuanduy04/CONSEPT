import logging
from dataclasses import dataclass

import torch
from consept import CONSEPTConfig, CONSEPTTrainer
from consept.semantic_reward import get_semantic_reward
from datasets import load_dataset
from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
)


class _RightPaddingFilter(logging.Filter):
    def filter(self, record):
        return (
            "decoder-only architecture is being used, but right-padding was detected"
            not in record.getMessage().lower()
        )


logging.getLogger("transformers.generation.utils").addFilter(_RightPaddingFilter())


@dataclass
class DatasetConfig:
    dataset_path: str
    streaming: bool = True


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, CONSEPTConfig, ModelConfig, DatasetConfig))
    script_args, training_args, model_args, dataset_args = parser.parse_args_and_config()
    ################
    # Model & Processor
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )

    ################
    # Dataset
    ################
    train_dataset = load_dataset(
        "json", data_files=dataset_args.dataset_path, split="train", streaming=dataset_args.streaming
    )

    train_dataset = train_dataset.select_columns(["text"])

    ################
    # Training
    ################
    trainer = CONSEPTTrainer(
        model=model_args.model_name_or_path,
        # processing_class=AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side="left"),
        args=training_args,
        reward_funcs=get_semantic_reward(AutoTokenizer.from_pretrained(model_args.model_name_or_path).eos_token),
        train_dataset=train_dataset,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
