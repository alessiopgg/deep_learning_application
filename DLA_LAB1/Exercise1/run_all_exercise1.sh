#!/usr/bin/env bash

set -euo pipefail

EPOCHS="${1:-5}"

MODELS=(
  resnet18
  resnet50
)

STRATEGIES=(
  classifier
  last_block
  full
)

CLASSIFIERS=(
  linear
  mlp
)

LOG_DIR="Exercise1/outputs/logs"

mkdir -p "${LOG_DIR}"

run_experiment() {
  local experiment_name="$1"
  shift

  local log_file="${LOG_DIR}/${experiment_name}.log"

  echo
  echo "============================================================"
  echo "Starting experiment: ${experiment_name}"
  echo "Command: $*"
  echo "Log file: ${log_file}"
  echo "============================================================"

  "$@" 2>&1 | tee "${log_file}"

  echo
  echo "Completed experiment: ${experiment_name}"
}

echo "============================================================"
echo "Exercise 1 complete experiment campaign"
echo "Epochs for each fine-tuning run: ${EPOCHS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "============================================================"

# ------------------------------------------------------------
# Exercise 1.2
# 2 models × 3 classical classifiers = 6 experiments
# ------------------------------------------------------------

run_experiment \
  "exercise_1_2_all_models_all_classifiers" \
  python Exercise1/main.py baseline \
    --models all \
    --classifiers all \
    --wandb

# ------------------------------------------------------------
# Exercise 1.3
# 2 models × 3 strategies × 2 classifiers = 12 experiments
# ------------------------------------------------------------

for model in "${MODELS[@]}"; do
  for strategy in "${STRATEGIES[@]}"; do
    for classifier in "${CLASSIFIERS[@]}"; do

      experiment_name="exercise_1_3_${model}_${strategy}_${classifier}"

      run_experiment \
        "${experiment_name}" \
        python Exercise1/main.py finetune \
          --model "${model}" \
          --strategy "${strategy}" \
          --classifier "${classifier}" \
          --epochs "${EPOCHS}" \
          --wandb

    done
  done
done

echo
echo "============================================================"
echo "All Exercise 1 experiments completed successfully."
echo "Logs available in: ${LOG_DIR}"
echo "============================================================"