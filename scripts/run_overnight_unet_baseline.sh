#!/usr/bin/env bash
set -euo pipefail

cd "$PROJECT_ROOT"
mkdir -p logs

# 先确保脚本已经复制到 src/：
# cp "$PROJECT_ROOT"/run_overnight_unet_baseline.py src/run_overnight_unet_baseline.py

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

nohup python src/run_overnight_unet_baseline.py \
  --shot 20 \
  --seeds 42,123,456 \
  --source_epochs 12 \
  --target_epochs 120 \
  --batch_size 8 \
  --target_batch_size 4 \
  --num_workers 4 \
  --use_tta \
  --val_eval_limit 200 \
  --final_val_eval_limit 500 \
  --test_eval_limit 0 \
  --out_dir results/overnight_unet20_baseline \
  > logs/overnight_unet20_baseline.log 2>&1 &

echo $! > logs/overnight_unet20_baseline.pid
echo "Started PID=$(cat logs/overnight_unet20_baseline.pid)"
echo "Log: logs/overnight_unet20_baseline.log"
echo "Watch: tail -f logs/overnight_unet20_baseline.log"
