import logging

import torch
from consept import CONSEPTConfig, CONSEPTTrainer
from consept.semantic_reward import get_semantic_reward
from datasets import load_dataset
from transformers import AutoTokenizer

from trl import ModelConfig, ScriptArguments, TrlParser


class _RightPaddingLogFilter(logging.Filter):
    def filter(self, record):
        return (
            "decoder-only architecture is being used, but right-padding was detected"
            not in record.getMessage().lower()
        )


logging.getLogger("transformers.generation.utils").addFilter(_RightPaddingLogFilter())


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, CONSEPTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
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
        "json", data_files=script_args.dataset_name, split="train", streaming=script_args.dataset_streaming
    )

    train_dataset = train_dataset.select_columns(["text"])

    ################
    # Training
    ################
    print(f"Begin training CONSEPT for model {model_args.model_name_or_path}")
    trainer = CONSEPTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=get_semantic_reward(AutoTokenizer.from_pretrained(model_args.model_name_or_path).eos_token),
        train_dataset=train_dataset,
    )

    trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint if training_args.resume_from_checkpoint else None
    )
    trainer.save_model(training_args.output_dir)
