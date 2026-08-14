#!/usr/bin/env bash

set -o pipefail

mkdir -p Exercise1/logs

LR="0.001"
GAMMA="0.99"
HIDDEN_DIM="64"
EPISODES="2000"
EVAL_EVERY="25"
EVAL_EPISODES="20"

SEEDS=(42 123 456 789 1000)

for seed in "${SEEDS[@]}"
do
    run_name="reinforce_extended2000_lr${LR}_gamma${GAMMA}_h${HIDDEN_DIM}_seed${seed}"

    echo
    echo "============================================================"
    echo "Starting ${run_name}"
    echo "============================================================"

    python -m Exercise1.main \
        --lr "$LR" \
        --seed "$seed" \
        --episodes "$EPISODES" \
        --gamma "$GAMMA" \
        --hidden-dim "$HIDDEN_DIM" \
        --eval-every "$EVAL_EVERY" \
        --eval-episodes "$EVAL_EPISODES" \
        --run-name "$run_name" \
        2>&1 | tee "Exercise1/logs/${run_name}.log"

    status=${PIPESTATUS[0]}

    if [ "$status" -ne 0 ]; then
        echo
        echo "ERROR: ${run_name} failed with exit code ${status}."
        exit "$status"
    fi

    echo
    echo "Completed ${run_name}"
done

echo
echo "============================================================"
echo "All extended runs completed successfully."
echo "============================================================"
