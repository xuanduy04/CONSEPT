#!/bin/bash

MODEL_PATH="Qwen/Qwen3-0.6B-Base"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_PATH="$PARENT_DIR/data_ready2train/*.jsonl"  # <--- DATA_PATH HERE


MODEL_NAME="$(basename "$MODEL_PATH")"
RUN_NAME="${MODEL_NAME}-$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="$PARENT_DIR/outputs/sft-$RUN_NAME"

(
    cd "$PARENT_DIR" || exit 1
    PYTHONPATH="$PARENT_DIR" accelerate launch \
        --config_file accelerate_configs/cpu_only.yaml \
        main/consept_main.py \
        --model_name_or_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --dataset_name "$DATA_PATH" \
        --dataset_streaming false \
        --dtype bfloat16 \
        --initial_completion_length 8 \
        --max_completion_length 1024 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --generation_batch_size 16 \
        --num_generations 4 \
        --gradient_checkpointing false \
        --loss_type dr_grpo \
        --beta 0.0 \
        --learning_rate 2e-6 \
        --max_step 100_000 \
        --warmup_ratio 0.01 \
        --max_grad_norm 1.0 \
        --log_completions true \
        --num_completions_to_print 1 \
        --logging_steps 10 \
        --save_strategy steps \
        --save_steps 100 \
        --save_total_limit 100 \
        --num_train_epochs 1 \
        --report_to none \
        --seed 2212 \
        --use_cpu
)
