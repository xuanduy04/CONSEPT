#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

(
    cd "$PARENT_DIR" || exit 1
    accelerate launch --config_file accelerate_configs/cpu_only.yaml \
        python scripts/consept_debug.py
        --model_name_or_path Qwen/Qwen3-0.6B \
        --output_dir gspo-Qwen3-0.6B \
        --learning_rate 1e-5 \
        --dtype bfloat16 \
        --max_prompt_length 2048 \
        --max_completion_length 1024 \
        --log_completions \
        --per_device_train_batch_size 1 \
        --num_generations 8 \
        --epsilon 3e-4 \
        --epsilon_high 4e-4 \
        --beta 0.0 \
        --loss_type grpo \
        --gradient_accumulation_steps 2
)
