import torch
from datasets import load_dataset

from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser


def set_gradient_accumulation_steps(args, effective_batch_size: int = 256):
    assert torch.cuda.device_count() > 0, "No CUDA devices visible"
    number_of_devices = torch.cuda.device_count()
    assert effective_batch_size % (args.per_device_train_batch_size * number_of_devices) == 0, (
        f"Cannot evenly divide {effective_batch_size} by {args.per_device_train_batch_size=} and {number_of_devices=}"
    )
    args.gradient_accumulation_steps = effective_batch_size // (args.per_device_train_batch_size * number_of_devices)
    print(
        f"`gradient_accumulation_steps` was set to {args.gradient_accumulation_steps} inorder to make effective batch size equal {effective_batch_size}"
        f"\n\teffective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * number_of_devices"
        f"\n\t{effective_batch_size} = {args.per_device_train_batch_size} * {args.gradient_accumulation_steps} * {number_of_devices}"
    )
    return args


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
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

    # ======== IMPORTANT ======== #
    # Hard coding effective batch size.
    training_args = set_gradient_accumulation_steps(training_args)

    ################
    # Dataset
    ################
    if not script_args.dataset_streaming:
        print("`dataset_streaming` is False, loading dataset will take a while...")
    train_dataset = load_dataset(
        "json", data_files=script_args.dataset_name, split="train", streaming=script_args.dataset_streaming
    )

    train_dataset = train_dataset.select_columns(["text"])

    ################
    # Training
    ################
    print(f"Begin training SFT for model {model_args.model_name_or_path}")
    training_args.accelerator_config.dispatch_batches = False
    trainer = SFTTrainer(model=model_args.model_name_or_path, args=training_args, train_dataset=train_dataset)
    trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint if training_args.resume_from_checkpoint else None
    )
    trainer.save_model(training_args.output_dir)
