#!/bin/bash

MODEL_PATH="Qwen/Qwen3-0.6B-Base"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_PATH="$PARENT_DIR/data_ready2train/*.jsonl"  # <--- DATA_PATH HERE


MODEL_NAME="$(basename "$MODEL_PATH")"
RUN_NAME="${MODEL_NAME}-$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="$PARENT_DIR/outputs/sft-$RUN_NAME"

(
    # export HF_HOME=""
    # export HF_DATASETS_CACHE=""
    # export HUGGINGFACE_HUB_CACHE=""
    export CUDA_VISIBLE_DEVICES="0"
    cd "$PARENT_DIR" || exit 1
    PYTHONPATH="$PARENT_DIR" accelerate launch \
        --config_file accelerate_configs/4gpus.yaml \
        main/sft_main.py \
        --model_name_or_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --dataset_name "$DATA_PATH" \
        --dataset_streaming true \
        --max_length 4096 \
        --per_device_train_batch_size 8 \
        --gradient_checkpointing false \
        --max_steps 2000 \
        --learning_rate 5e-6 \
        --warmup_ratio 0.01 \
        --max_grad_norm 1.0 \
        --logging_steps 20 \
        --save_strategy steps \
        --save_steps 500 \
        --dtype bfloat16 \
        --packing true \
        --ddp_find_unused_parameters false \
        --attn_implementation "flash_attention_2" \
        --report_to tensorboard \
        --seed 2212
)
