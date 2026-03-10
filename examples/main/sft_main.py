from dataclasses import dataclass

import torch
from datasets import load_dataset

from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser


@dataclass
class DatasetConfig:
    dataset_path: str
    streaming: bool = True


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, DatasetConfig))
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
    training_args.accelerator_config.dispatch_batches = False
    trainer = SFTTrainer(model=model_args.model_name_or_path, args=training_args, train_dataset=train_dataset)
    trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint if training_args.resume_from_checkpoint else None
    )
    trainer.save_model(training_args.output_dir)
