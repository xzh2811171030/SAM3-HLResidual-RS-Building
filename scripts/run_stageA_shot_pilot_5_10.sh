#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# conda activate (取消注释以启用):
# source /path/to/conda/bin/activate your_env 2>/dev/null || true

cd "$PROJECT_ROOT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

RUN_TAG="stageA_shotpilot_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/${RUN_TAG}"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "Stage A: 5/10-shot pilot for lora_light vs hl_residual_old"
echo "RUN_TAG=${RUN_TAG}"
echo "LOG_DIR=${LOG_DIR}"
echo "============================================================"

SHOTS=(5 10)
SEEDS=(42 123 456)

for SHOT in "${SHOTS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "============================================================"
    echo "[1/3] Train lora_light | shot=${SHOT} seed=${SEED}"
    echo "============================================================"

    LORA_OUT="results/${RUN_TAG}/train_lora_s${SHOT}_seed${SEED}"

    python -u src/run_hl_lite_pilot.py \
      --variants lora_light \
      --seed "${SEED}" \
      --shot "${SHOT}" \
      --epochs 20 \
      --batch_size 2 \
      --val_every 5 \
      --val_eval_limit 200 \
      --pilot_eval_limit 500 \
      --out_dir "${LORA_OUT}" \
      2>&1 | tee "${LOG_DIR}/train_lora_s${SHOT}_seed${SEED}.log"

    echo ""
    echo "============================================================"
    echo "[2/3] Train hl_residual_old | shot=${SHOT} seed=${SEED}"
    echo "============================================================"

    RES_OUT="results/${RUN_TAG}/train_hlres_s${SHOT}_seed${SEED}"
    BASE_LORA_CKPT="${LORA_OUT}/weights/e6v2_pilot_lora_light_best.pth"

    if [[ ! -f "${BASE_LORA_CKPT}" ]]; then
      echo "[ERROR] Missing base lora checkpoint: ${BASE_LORA_CKPT}"
      exit 1
    fi

    python -u src/run_hl_residual_refine_pilot.py \
      --seed "${SEED}" \
      --shot "${SHOT}" \
      --epochs 20 \
      --lr 1e-3 \
      --batch_size 2 \
      --val_every 5 \
      --val_eval_limit 200 \
      --pilot_eval_limit 500 \
      --base_lora_ckpt "${BASE_LORA_CKPT}" \
      --out_dir "${RES_OUT}" \
      2>&1 | tee "${LOG_DIR}/train_hlres_s${SHOT}_seed${SEED}.log"

    echo ""
    echo "============================================================"
    echo "[3/3] Eval TTA + validation-calibrated postprocess | shot=${SHOT} seed=${SEED}"
    echo "============================================================"

    RES_CKPT="${RES_OUT}/weights/hl_residual_best.pth"
    EVAL_OUT="results/${RUN_TAG}/eval_tta_post_s${SHOT}_seed${SEED}"

    if [[ ! -f "${RES_CKPT}" ]]; then
      echo "[ERROR] Missing residual checkpoint: ${RES_CKPT}"
      exit 1
    fi

    python -u src/run_calibrated_fulltest.py \
      --models lora_light,hl_residual_old \
      --use_tta \
      --val_eval_limit 500 \
      --test_manifest_name target_pilot_test_500.txt \
      --test_eval_limit 0 \
      --lora_light_ckpt "${BASE_LORA_CKPT}" \
      --hl_residual_old_ckpt "${RES_CKPT}" \
      --out_dir "${EVAL_OUT}" \
      2>&1 | tee "${LOG_DIR}/eval_tta_post_s${SHOT}_seed${SEED}.log"

    echo "[DONE] shot=${SHOT} seed=${SEED}"
  done
done

echo ""
echo "============================================================"
echo "Stage A finished."
echo "Results under: results/${RUN_TAG}"
echo "Logs under: ${LOG_DIR}"
echo "============================================================"
