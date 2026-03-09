#!/bin/bash

MODEL_PATH="Qwen/Qwen3-0.6B-Base"
DATA_PATH=""


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"


MODEL_NAME="$(basename "$MODEL_PATH")"
RUN_NAME="${MODEL_NAME}-$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="$PARENT_DIR/sft-$RUN_NAME"

(
    cd "$PARENT_DIR" || exit 1
    PYTHONPATH="$PARENT_DIR" accelerate launch \
        --config_file accelerate_configs/multi_gpu.yaml \
        main/sft_baseline_main.py \
        --model_name_or_path "$MODEL_PATH" \
        --dataset_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --dtype bfloat16 \
        --seed 2212
)
