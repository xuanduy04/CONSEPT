"""
accelerate launch \
    --config_file accelerate_configs/single_gpu.yaml \
    main/consept_main.py \
    --model_name_or_path Qwen/Qwen3-0.6B-Base \
    --output_dir consept-Qwen3-0.6B-Base \
    --dtype bfloat16 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --generation_batch_size 16 \
    --num_generations 4 \
    --gradient_checkpointing false \
    --loss_type dr_grpo \
    --entropy_coef 0.0 \
    --beta 0.0 \
    --learning_rate 2e-6 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --log_completions true \
    --num_completions_to_print 3 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 25 \
    --save_total_limit 5 \
    --num_train_epochs 1 \
    --report_to none \
    --seed 2212

"""

import logging

import torch
from consept import CONSEPTConfig, CONSEPTTrainer
from consept.semantic_reward import get_semantic_reward
from datasets import load_dataset
from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


class _RightPaddingFilter(logging.Filter):
    def filter(self, record):
        return (
            "decoder-only architecture is being used, but right-padding was detected"
            not in record.getMessage().lower()
        )


logging.getLogger("transformers.generation.utils").addFilter(_RightPaddingFilter())


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
        "HuggingFaceTB/cosmopedia", name="openstax", split="train"
    )

    train_dataset = train_datasettrain_dataset = train_dataset.remove_columns(
        [c for c in train_dataset.column_names if c != "text"]
    )

    ################
    # Training
    ################
    trainer = CONSEPTTrainer(
        model=model_args.model_name_or_path,
        # processing_class=AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side="left"),
        args=training_args,
        reward_funcs=get_semantic_reward(AutoTokenizer.from_pretrained(model_args.model_name_or_path).eos_token),
        train_dataset=train_dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
