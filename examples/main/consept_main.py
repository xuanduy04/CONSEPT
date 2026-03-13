import torch
from consept import CONSEPTConfig, CONSEPTTrainer
from consept.completion_length_scheduler import LinearCompletionLengthScheduler
from consept.semantic_reward import get_semantic_reward
from datasets import load_dataset
from transformers import AutoTokenizer
from utils import validate_accelerator_config

from trl import ModelConfig, ScriptArguments, TrlParser


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
    processor = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side="left")

    ################
    # Dataset
    ################
    train_dataset = load_dataset(
        "json", data_files=script_args.dataset_name, split="train", streaming=script_args.dataset_streaming
    )

    train_dataset = train_dataset.select_columns(["text"])

    ################
    # Completion length scheduler
    ################
    completion_length_scheduler_cls = LinearCompletionLengthScheduler
    completion_length_scheduler_kwargs = {
        # currently setting this value for debugging purposes
        "warmup_steps": training_args.max_completion_length - training_args.initial_completion_length
    }

    ################
    # Training
    ################
    trainer = CONSEPTTrainer(
        model=model_args.model_name_or_path,
        processing_class=processor,
        completion_length_scheduler_cls=completion_length_scheduler_cls,
        completion_length_scheduler_kwargs=completion_length_scheduler_kwargs,
        args=training_args,
        reward_funcs=get_semantic_reward(processor.eos_token),
        train_dataset=train_dataset,
    )
    validate_accelerator_config(trainer.accelerator)
    trainer.accelerator.print(f"Begin training CONSEPT for model `{model_args.model_name_or_path}`")
    trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint if training_args.resume_from_checkpoint else None
    )
    trainer.save_model(training_args.output_dir)
