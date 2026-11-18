#!/usr/bin/env bash
# Real-time Interactive GPU Pipeline Runner
# Usage: ./train_gpu.sh [train|eval|alpha|all]

set -e
PROJECT_ROOT="/DATA/yogendra_2511ai38/projects/fintech/MARS-RL-Constrained-Regime-Aware-Reinforcement-Learning-for-Margin-FINTECH"
ENV_PYTHON="/DATA/yogendra_2511ai38/mars_gpu_env/bin/python3"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODE=${1:-"help"}

echo "=================================================================="
echo "⚡ MARS-RL Safe Shared-GPU Training & Validation Terminal Suite"
echo "=================================================================="

case "$MODE" in
  train)
    ROLE=${2:-"v2_b"}
    STEPS=${3:-50000}
    SEED=${4:-0}
    CHUNK=${5:-10000}
    echo "🎯 Running GPU RL Training: Role=$ROLE, Steps=$STEPS, Seed=$SEED"
    $ENV_PYTHON -m src.train --role "$ROLE" --seed "$SEED" --steps "$STEPS" --chunk "$CHUNK" --device auto --max-mem-fraction 0.25 --out models/gpu_runs
    ;;
  alpha)
    echo "🔬 Running Novel MARS Alpha Multi-Horizon Grid & Evaluation on GPU/Data splits..."
    $ENV_PYTHON run_novel_mars_alpha.py
    ;;
  eval_alpha)
    echo "📈 Running Best Novel MARS Alpha Evaluation on Train/Val/Test splits..."
    $ENV_PYTHON eval_best_novel_model.py
    ;;
  eval)
    echo "⚖️ Running Evaluation on ValTest across common-costs..."
    $ENV_PYTHON -m src.evaluate --models models/stage2 --seeds 0 1 2 --device auto --max-mem-fraction 0.20
    ;;
  test)
    echo "🧪 Running Test Suite..."
    /DATA/yogendra_2511ai38/mars_gpu_env/bin/pytest tests/ -v
    ;;
  all)
    echo "🚀 Running Complete Suite (Alpha Grid -> Eval -> Safe RL Training Step)..."
    $ENV_PYTHON run_novel_mars_alpha.py
    $ENV_PYTHON eval_best_novel_model.py
    $ENV_PYTHON -m src.train --role v2_b --seed 0 --steps 10000 --chunk 2500 --device auto --max-mem-fraction 0.25 --out models/gpu_runs
    ;;
  *)
    echo "Usage: ./train_gpu.sh [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  ./train_gpu.sh train [role] [steps] [seed]   - Train RL agent on GPU with live progress"
    echo "  ./train_gpu.sh alpha                         - Run Novel MARS Alpha parameter grid on Val"
    echo "  ./train_gpu.sh eval_alpha                    - Evaluate Best Novel Alpha on Train/Val/Test"
    echo "  ./train_gpu.sh eval                          - Run evaluation on validation/test splits"
    echo "  ./train_gpu.sh test                          - Run pytest unit tests"
    echo "  ./train_gpu.sh all                           - Run end-to-end full check"
    echo "=================================================================="
    ;;
esac
