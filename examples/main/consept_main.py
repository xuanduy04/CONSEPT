"""
accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/gspo.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --output_dir gspo-Qwen3-0.6B \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj" \
    --log_completions \
    --per_device_train_batch_size 8 \
    --num_generations 8 \
    --importance_sampling_level sequence \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --beta 0.0 \
    --loss_type grpo \
    --gradient_accumulation_steps 2 \
    --steps_per_generation 8

"""


import torch
from consept import CONSEPTConfig, CONSEPTTrainer
from consept.semantic_reward import semantic_reward
from datasets import load_dataset

from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


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
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        training_args.model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.model_init_kwargs["quantization_config"] = quantization_config

    ################
    # Dataset
    ################
    # train_dataset = load_dataset("nvidia/Nemotron-CC-v2", name="HighQuality", split="train[:5%]")
    train_dataset = load_dataset(
        "nvidia/Nemotron-Pretraining-Dataset-sample", name="Nemotron-CC-High-Quality", split="train"
    )

    train_dataset = train_dataset.rename_column("text", "solution")

    ################
    # Training
    ################
    trainer = CONSEPTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=semantic_reward,
        train_dataset=train_dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
