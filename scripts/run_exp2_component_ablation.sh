#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ===================== EDIT THESE PATHS =====================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST_DIR="$PROJECT_ROOT/data/splits/e0_manifest"
SAM3_CKPT="$PROJECT_ROOT/weights/sam3.pt"

# Prefer the held-out non-pilot manifest. If your filename is different, change this line.
EVAL_MANIFEST="$MANIFEST_DIR/target_final_test_excluding_pilot500.txt"
if [[ ! -f "$EVAL_MANIFEST" ]]; then
  echo "[WARN] $EVAL_MANIFEST not found. Falling back to pilot500."
  EVAL_MANIFEST="$MANIFEST_DIR/target_pilot_test_500.txt"
fi

# Replace these with your trained Prompt-free LoRA checkpoints for each support seed.
declare -A LORA_CKPT
LORA_CKPT[42]="$PROJECT_ROOT/results/e6v2_pilot/weights/e6v2_pilot_lora_light_best.pth"
LORA_CKPT[123]="$PROJECT_ROOT/results/e6v3_hl_lite_seed123/weights/e6v2_pilot_lora_light_best.pth"
LORA_CKPT[456]="$PROJECT_ROOT/results/e6v3_hl_lite_seed456/weights/e6v2_pilot_lora_light_best.pth"
# ============================================================

OUT_ROOT="$PROJECT_ROOT/results/exp2_component_ablation"
mkdir -p "$OUT_ROOT/logs"

for seed in 42 123 456; do
  if [[ ! -f "${LORA_CKPT[$seed]}" ]]; then
    echo "[ERROR] LoRA checkpoint for seed=$seed not found: ${LORA_CKPT[$seed]}"
    echo "Please edit LORA_CKPT[$seed] in this script."
    exit 1
  fi
done

for variant in rgb_only sam_only; do
  for seed in 42 123 456; do
    OUT_DIR="$OUT_ROOT/${variant}_seed${seed}"
    LOG="$OUT_ROOT/logs/${variant}_seed${seed}.log"
    echo "============================================================"
    echo "[RUN] variant=$variant seed=$seed"
    echo "[OUT] $OUT_DIR"
    echo "[LOG] $LOG"
    echo "============================================================"

    python -u src/run_hl_residual_component_ablation.py \
      --project_root "$PROJECT_ROOT" \
      --manifest_dir "$MANIFEST_DIR" \
      --sam3_checkpoint "$SAM3_CKPT" \
      --base_lora_ckpt "${LORA_CKPT[$seed]}" \
      --variant "$variant" \
      --seed "$seed" \
      --shot 20 \
      --epochs 20 \
      --lr 5e-4 \
      --batch_size 2 \
      --num_workers 2 \
      --val_every 5 \
      --val_eval_limit 200 \
      --pilot_eval_limit 0 \
      --eval_manifest "$EVAL_MANIFEST" \
      --grad_accum_steps 2 \
      --train_erasing_prob 0.25 \
      --train_erasing_max_area 0.08 \
      --select_by_calibrated_val \
      --residual_l2_weight 0.01 \
      --use_tta_eval \
      --out_dir "$OUT_DIR" \
      2>&1 | tee "$LOG"
  done
done

python - <<'PY'
from pathlib import Path
import json, csv
import numpy as np

root = Path(__file__).resolve().parents[2] / "results" / "exp2_component_ablation"
rows = []
for variant in ["rgb_only", "sam_only"]:
    for seed in [42, 123, 456]:
        p = root / f"{variant}_seed{seed}" / f"hl_residual_{variant}_summary.json"
        if not p.exists():
            print(f"[WARN] missing {p}")
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        m = obj["test_calibrated"]
        rows.append({
            "variant": variant,
            "seed": seed,
            "mIoU": m["mIoU"] * 100,
            "F1": m["F1"] * 100,
            "BIoU": m["Boundary_IoU"] * 100,
            "threshold": obj["val_sweep_best"]["threshold"],
            "min_area": obj["val_sweep_best"]["min_area"],
            "json": str(p),
        })

summary_dir = root / "summary"
summary_dir.mkdir(parents=True, exist_ok=True)

with (summary_dir / "component_ablation_per_seed.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["variant", "seed", "mIoU", "F1", "BIoU", "threshold", "min_area", "json"])
    writer.writeheader()
    writer.writerows(rows)

summary = []
for variant in ["rgb_only", "sam_only"]:
    subset = [r for r in rows if r["variant"] == variant]
    if not subset:
        continue
    rec = {"variant": variant, "n_seeds": len(subset)}
    for metric in ["mIoU", "F1", "BIoU"]:
        vals = np.array([r[metric] for r in subset], dtype=float)
        rec[f"{metric}_mean"] = float(vals.mean())
        rec[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    summary.append(rec)

with (summary_dir / "component_ablation_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["variant", "n_seeds", "mIoU_mean", "mIoU_std", "F1_mean", "F1_std", "BIoU_mean", "BIoU_std"])
    writer.writeheader()
    writer.writerows(summary)

print("[OK] wrote", summary_dir / "component_ablation_per_seed.csv")
print("[OK] wrote", summary_dir / "component_ablation_summary.csv")
for r in summary:
    print(r)
PY

echo "[DONE] exp2 component ablation finished."
