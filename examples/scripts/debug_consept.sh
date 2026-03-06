#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

(
    cd "$PARENT_DIR" || exit 1
    PYTHONPATH="$PARENT_DIR" accelerate launch \
        --config_file accelerate_configs/cpu_only.yaml \
        main/consept_main.py \
        --model_name_or_path Qwen/Qwen3-0.6B \
        --output_dir consept-Qwen3-0.6B \
        --max_prompt_length 9999999 \
        --max_completion_length 1024 \
        --log_completions \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --num_generations 4 \
        --epsilon 3e-4 \
        --epsilon_high 4e-4 \
        --loss_type grpo \
        --use_cpu
)
