#!/usr/bin/env bash
# Track A: authors' own env + agent on our 28-ticker data (3 seeds x {A2C, PPO}, 50k steps each). Resumable.
cd "$(dirname "$0")/.."
for algo in a2c ppo; do
  for seed in 0 1 2; do
    [ -f results/trackA/trackA_${algo}_s${seed}_metrics.json ] && continue
    python3 -m src.track_a --seed $seed --algo $algo --timesteps 50000 --out results/trackA > logs/trackA_${algo}_s${seed}.log 2>&1
  done
done
echo DONE > logs/trackA_done.flag
