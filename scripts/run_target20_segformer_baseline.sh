#!/usr/bin/env bash
set -euo pipefail

cd "$PROJECT_ROOT"

mkdir -p logs
mkdir -p results/target20_segformer_baseline

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

LOG_FILE="logs/target20_segformer_baseline_$(date +%Y%m%d_%H%M%S).log"

echo "[Run] Logging to ${LOG_FILE}"
echo "[Run] Start time: $(date)"

nohup python src/run_target20_segformer_baseline.py \
  --seeds 42,123,456 \
  --shot 20 \
  --source_epochs 30 \
  --target_epochs 120 \
  --batch_size 8 \
  --target_batch_size 4 \
  --num_workers 4 \
  --source_val_every 5 \
  --val_every 10 \
  --source_val_eval_limit 200 \
  --val_eval_limit 200 \
  --final_val_eval_limit 500 \
  --test_eval_limit 0 \
  --use_tta \
  --grid_mode full \
  --out_dir results/target20_segformer_baseline \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > logs/target20_segformer_baseline.pid
echo "[Run] PID=${PID}"
echo "[Run] tail -f ${LOG_FILE}"
