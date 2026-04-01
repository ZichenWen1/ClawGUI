#!/bin/bash

# ======================
# Experiment Configuration
# ======================
EXP_NAME="qwen3vl_transformers_2b_2dprompt"
BENCHMARK="screenspot-pro"

# ======================
# Path Configuration
# ======================
INPUT_FILE="output/${EXP_NAME}/${BENCHMARK}/predictions_judge.jsonl"
OUTPUT_FILE="output/${EXP_NAME}/${BENCHMARK}/metrics.json"

# ======================
# Run Metric Calculation
# ======================
python metric/screenspotpro_metric.py \
    --input_file ${INPUT_FILE} \
    --output_file ${OUTPUT_FILE} \
    --exp_name ${EXP_NAME} \
    --benchmark ${BENCHMARK}
