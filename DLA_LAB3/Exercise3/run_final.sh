#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SEEDS=(42 123 456)

echo "===== CartPole-v1 ====="
for seed in "${SEEDS[@]}"; do
  rm -rf "Exercise3/runs/cartpole_seed${seed}"
  python -m Exercise3.main --seed "${seed}"
done

echo
echo "===== LunarLander-v3 ====="
for seed in "${SEEDS[@]}"; do
  rm -rf "Exercise3/runs/lunarlander_seed${seed}"
  python -m Exercise3.lunarlander_main --seed "${seed}"
done

echo
echo "===== Final evaluation ====="
python -m Exercise3.evaluate_results

echo
echo "===== Final plots ====="
python -m Exercise3.plot_results
