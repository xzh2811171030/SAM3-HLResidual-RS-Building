# -*- coding: utf-8 -*-
"""
run_e8_regionwise_eval.py
=============================================================================
E8: Region-wise / city-wise generalization analysis

目的：
  对 full test 8402 进行 per-image 评估，然后按 region prefix 聚合。
  用于证明 HL-Residual 在不同地理区域上的稳定性与边界质量优势。

依赖：
  src/run_e7a_prompt_drift.py

默认模型：
  1. Prompt-free LoRA
  2. HL-Residual

默认数据：
  val  : target_val.txt，用于 threshold / postprocess calibration
  test : target_final_test_8402.txt，用于 region-wise full-test evaluation

推荐运行：
  cd <project_root>

  nohup bash -lc '
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  python -u src/run_e8_regionwise_eval.py \
    --models promptfree_lora,hl_residual \
    --seeds 42,123,456 \
    --use_tta \
    --val_eval_limit 500 \
    --test_manifest_name target_final_test_8402.txt \
    --test_eval_limit 0 \
    --batch_size 4 \
    --num_workers 2 \
    --out_dir results/e8_regionwise_full8402_tta
  ' > logs/e8_regionwise_full8402_tta.nohup.log 2>&1 &

  tail -f logs/e8_regionwise_full8402_tta.nohup.log
=============================================================================
"""

import argparse
import csv
import gc
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_e7a_prompt_drift import (
    DEVICE,
    PROJECT_ROOT_DEFAULT,
    ManifestDataset,
    collate_fn,
    load_promptfree_model,
    grid_search_postprocess,
    postprocess_mask,
    compute_iou,
    compute_f1,
    compute_biou,
)


# =============================================================================
# Region parsing
# =============================================================================

def infer_region_prefix(name: str) -> str:
    """
    从文件名推断 region prefix。

    注意：
      这里不强行声称一定是 city，而叫 region_prefix 更稳。
      示例：
        vienna_train_251 -> vienna
        chongqing_xxx    -> chongqing
        global_train_xxx -> global
    """
    stem = Path(name).stem.lower().replace("-", "_")
    parts = [p for p in stem.split("_") if p]

    if not parts:
        return "unknown"

    known = [
        "asia", "chongqing", "christchurch", "hangzhou", "hubei",
        "tianjin", "tyrol", "vienna", "wuxi", "potsdam",
        "dunedin", "kitsap", "khartoum", "global"
    ]

    for p in parts:
        if p in known:
            return p

    # 常见格式 city_train_001 / city_test_001
    if len(parts) >= 2 and parts[1] in {"train", "test", "val"}:
        return parts[0]

    return parts[0]


# =============================================================================
# DataLoader
# =============================================================================

def make_loader(ds, batch_size: int, num_workers: int):
    kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


# =============================================================================
# Prediction
# =============================================================================

@torch.no_grad()
def predict_batch_probs(extractor, model, imgs: torch.Tensor, use_tta: bool) -> np.ndarray:
    """
    返回 shape [B, H, W] 的概率图。
    """
    if not use_tta:
        feat = extractor.extract_batch(imgs)
        logits = model(feat.to(DEVICE), imgs.to(DEVICE))
        probs = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]
        return probs

    probs_list = []

    # original
    feat = extractor.extract_batch(imgs)
    logits = model(feat.to(DEVICE), imgs.to(DEVICE))
    probs_list.append(torch.sigmoid(logits.float()).cpu())

    # horizontal flip
    imgs_h = torch.flip(imgs, dims=[3])
    feat_h = extractor.extract_batch(imgs_h)
    logits_h = model(feat_h.to(DEVICE), imgs_h.to(DEVICE))
    ph = torch.sigmoid(logits_h.float()).cpu()
    ph = torch.flip(ph, dims=[3])
    probs_list.append(ph)

    # vertical flip
    imgs_v = torch.flip(imgs, dims=[2])
    feat_v = extractor.extract_batch(imgs_v)
    logits_v = model(feat_v.to(DEVICE), imgs_v.to(DEVICE))
    pv = torch.sigmoid(logits_v.float()).cpu()
    pv = torch.flip(pv, dims=[2])
    probs_list.append(pv)

    probs = torch.mean(torch.stack(probs_list, dim=0), dim=0).numpy()[:, 0]
    return probs


@torch.no_grad()
def collect_probs_for_calibration(extractor, model, loader, use_tta: bool):
    probs_all = []
    gts_all = []

    for batch in tqdm(loader, desc="  val calibration predict", ncols=100):
        imgs = batch["image"]
        masks = batch["mask"].numpy()[:, 0]

        probs = predict_batch_probs(extractor, model, imgs, use_tta=use_tta)

        probs_all.append(probs)
        gts_all.append(masks)

    return np.concatenate(probs_all, axis=0), np.concatenate(gts_all, axis=0)


@torch.no_grad()
def evaluate_per_image(
    extractor,
    model,
    loader,
    cfg: Dict,
    model_label: str,
    seed: int,
    use_tta: bool,
):
    rows = []

    for batch in tqdm(loader, desc=f"  test per-image {model_label} seed={seed}", ncols=100):
        imgs = batch["image"]
        masks = batch["mask"].numpy()[:, 0]
        names = batch["name"]
        paths = batch["path"] if "path" in batch else [""] * len(names)

        probs = predict_batch_probs(extractor, model, imgs, use_tta=use_tta)

        for i in range(len(names)):
            pred = postprocess_mask(
                probs[i],
                threshold=cfg["threshold"],
                min_area=cfg["min_area"],
                closing_kernel=cfg["closing_kernel"],
                fill_hole=cfg["fill_holes"],
            )

            gt = masks[i]

            miou = compute_iou(pred, gt)
            f1 = compute_f1(pred, gt)
            biou = compute_biou(pred, gt, d=5)

            region = infer_region_prefix(names[i])

            rows.append({
                "image_name": names[i],
                "image_path": paths[i],
                "region_prefix": region,
                "seed": seed,
                "model": model_label,
                "mIoU": miou,
                "F1": f1,
                "Boundary_IoU": biou,
            })

    return rows


# =============================================================================
# Aggregation
# =============================================================================

def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    if len(arr) == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def aggregate_region_seed(rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for r in rows:
        key = (r["model"], r["seed"], r["region_prefix"])
        grouped[key].append(r)

    out = []
    for (model, seed, region), items in sorted(grouped.items()):
        miou, miou_std = mean_std([x["mIoU"] for x in items])
        f1, f1_std = mean_std([x["F1"] for x in items])
        biou, biou_std = mean_std([x["Boundary_IoU"] for x in items])

        out.append({
            "model": model,
            "seed": seed,
            "region_prefix": region,
            "N": len(items),
            "mIoU": miou,
            "mIoU_std": miou_std,
            "F1": f1,
            "F1_std": f1_std,
            "Boundary_IoU": biou,
            "Boundary_IoU_std": biou_std,
        })

    return out


def aggregate_region_mean(region_seed_rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for r in region_seed_rows:
        key = (r["model"], r["region_prefix"])
        grouped[key].append(r)

    out = []
    for (model, region), items in sorted(grouped.items()):
        miou, miou_std = mean_std([x["mIoU"] for x in items])
        f1, f1_std = mean_std([x["F1"] for x in items])
        biou, biou_std = mean_std([x["Boundary_IoU"] for x in items])

        out.append({
            "model": model,
            "region_prefix": region,
            "num_seeds": len(items),
            "N_mean": float(np.mean([x["N"] for x in items])),
            "mIoU": miou,
            "mIoU_seed_std": miou_std,
            "F1": f1,
            "F1_seed_std": f1_std,
            "Boundary_IoU": biou,
            "Boundary_IoU_seed_std": biou_std,
        })

    return out


def aggregate_city_balanced(region_seed_rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for r in region_seed_rows:
        key = (r["model"], r["seed"])
        grouped[key].append(r)

    out = []
    for (model, seed), items in sorted(grouped.items()):
        region_mious = [x["mIoU"] for x in items]
        region_f1s = [x["F1"] for x in items]
        region_bious = [x["Boundary_IoU"] for x in items]

        out.append({
            "model": model,
            "seed": seed,
            "num_regions": len(items),
            "region_balanced_mIoU": float(np.mean(region_mious)),
            "region_balanced_F1": float(np.mean(region_f1s)),
            "region_balanced_Boundary_IoU": float(np.mean(region_bious)),
            "worst_region_mIoU": float(np.min(region_mious)),
            "worst_region_Boundary_IoU": float(np.min(region_bious)),
            "region_mIoU_std": float(np.std(region_mious, ddof=1)) if len(region_mious) > 1 else 0.0,
            "region_Boundary_IoU_std": float(np.std(region_bious, ddof=1)) if len(region_bious) > 1 else 0.0,
        })

    return out


def aggregate_city_balanced_mean(city_rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for r in city_rows:
        grouped[r["model"]].append(r)

    out = []
    for model, items in sorted(grouped.items()):
        for key in [
            "region_balanced_mIoU",
            "region_balanced_F1",
            "region_balanced_Boundary_IoU",
            "worst_region_mIoU",
            "worst_region_Boundary_IoU",
            "region_mIoU_std",
            "region_Boundary_IoU_std",
        ]:
            pass

        row = {
            "model": model,
            "num_seeds": len(items),
        }

        for key in [
            "region_balanced_mIoU",
            "region_balanced_F1",
            "region_balanced_Boundary_IoU",
            "worst_region_mIoU",
            "worst_region_Boundary_IoU",
            "region_mIoU_std",
            "region_Boundary_IoU_std",
        ]:
            mean, std = mean_std([x[key] for x in items])
            row[key] = mean
            row[f"{key}_seed_std"] = std

        out.append(row)

    return out


# =============================================================================
# Save utilities
# =============================================================================

def write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_region_bars(region_mean_rows: List[Dict], metric: str, out_path: Path):
    """
    metric: mIoU / Boundary_IoU
    """
    regions = sorted({r["region_prefix"] for r in region_mean_rows})
    models = sorted({r["model"] for r in region_mean_rows})

    data = {
        (r["model"], r["region_prefix"]): r[metric] * 100
        for r in region_mean_rows
    }

    x = np.arange(len(regions))
    width = 0.38 if len(models) <= 2 else 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(max(10, len(regions) * 0.9), 5.5))

    for idx, model in enumerate(models):
        offset = (idx - (len(models) - 1) / 2) * width
        ys = [data.get((model, reg), 0.0) for reg in regions]
        ax.bar(x + offset, ys, width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=35, ha="right")
    ax.set_ylabel(f"{metric} (%)")
    ax.set_title(f"Region-wise {metric}")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser("E8 region-wise full-test evaluation")

    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT_DEFAULT))
    parser.add_argument("--manifest_dir", type=str, default=None)

    parser.add_argument("--val_manifest_name", type=str, default="target_val.txt")
    parser.add_argument("--test_manifest_name", type=str, default="target_final_test_8402.txt")
    parser.add_argument("--val_eval_limit", type=int, default=500)
    parser.add_argument("--test_eval_limit", type=int, default=0)

    parser.add_argument("--sam3_checkpoint", type=str, default=None)
    parser.add_argument("--models", type=str, default="promptfree_lora,hl_residual")
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--use_tta", action="store_true")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--out_dir", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(args.project_root)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else project_root / "data/splits/e0_manifest"
    sam3_ckpt = Path(args.sam3_checkpoint) if args.sam3_checkpoint else project_root / "weights/sam3.pt"

    out_dir = Path(args.out_dir) if args.out_dir else project_root / "results/e8_regionwise_full8402_tta"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    val_limit = args.val_eval_limit
    if val_limit is not None and val_limit <= 0:
        val_limit = None

    test_limit = args.test_eval_limit
    if test_limit is not None and test_limit <= 0:
        test_limit = None

    model_label_map = {
        "promptfree_lora": "Prompt-free LoRA",
        "hl_residual": "HL-Residual (Ours)",
    }

    print("=" * 90)
    print("E8 Region-wise Evaluation")
    print("=" * 90)
    print(f"project_root : {project_root}")
    print(f"val_manifest : {args.val_manifest_name}")
    print(f"test_manifest: {args.test_manifest_name}")
    print(f"models       : {models}")
    print(f"seeds        : {seeds}")
    print(f"use_tta      : {args.use_tta}")
    print(f"batch_size   : {args.batch_size}")
    print(f"num_workers  : {args.num_workers}")
    print(f"device       : {DEVICE}")
    print(f"out_dir      : {out_dir}")
    print("=" * 90)

    val_ds = ManifestDataset(manifest_dir / args.val_manifest_name, limit=val_limit)
    test_ds = ManifestDataset(manifest_dir / args.test_manifest_name, limit=test_limit)

    val_loader = make_loader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = make_loader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    all_image_rows = []
    calibration_records = []

    for model_key in models:
        model_label = model_label_map.get(model_key, model_key)

        for seed in seeds:
            print("\n" + "=" * 90)
            print(f"[E8] model={model_label} seed={seed}")
            print("=" * 90)

            extractor, model, ckpt_path = load_promptfree_model(
                model_key,
                seed,
                project_root,
                sam3_ckpt,
            )

            print("[Calibration] predict target_val")
            val_probs, val_gts = collect_probs_for_calibration(
                extractor,
                model,
                val_loader,
                use_tta=args.use_tta,
            )

            print("[Calibration] grid search postprocess")
            search = grid_search_postprocess(val_probs, val_gts)
            best_cfg = search["best"]

            calibration_records.append({
                "model": model_label,
                "seed": seed,
                "checkpoint": str(ckpt_path),
                "val_best_cfg": best_cfg,
            })

            print(f"[Calibration] best_cfg={best_cfg}")

            print("[Test] full per-image evaluation")
            rows = evaluate_per_image(
                extractor=extractor,
                model=model,
                loader=test_loader,
                cfg=best_cfg,
                model_label=model_label,
                seed=seed,
                use_tta=args.use_tta,
            )

            all_image_rows.extend(rows)

            del extractor, model
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            # 每跑完一个 seed 保存一次，防止中断丢失
            write_csv(out_dir / "e8_per_image_metrics_partial.csv", all_image_rows)

    region_seed_rows = aggregate_region_seed(all_image_rows)
    region_mean_rows = aggregate_region_mean(region_seed_rows)
    city_balanced_rows = aggregate_city_balanced(region_seed_rows)
    city_balanced_mean_rows = aggregate_city_balanced_mean(city_balanced_rows)

    write_csv(out_dir / "e8_per_image_metrics.csv", all_image_rows)
    write_csv(out_dir / "e8_region_seed_metrics.csv", region_seed_rows)
    write_csv(out_dir / "e8_region_mean_metrics.csv", region_mean_rows)
    write_csv(out_dir / "e8_city_balanced_by_seed.csv", city_balanced_rows)
    write_csv(out_dir / "e8_city_balanced_summary.csv", city_balanced_mean_rows)

    summary = {
        "experiment": "E8_regionwise_generalization",
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "val_manifest": args.val_manifest_name,
            "test_manifest": args.test_manifest_name,
            "models": models,
            "seeds": seeds,
            "use_tta": args.use_tta,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "note": "region_prefix is inferred from filename prefixes; it is reported as region-wise rather than strict city-wise when source naming is ambiguous.",
        },
        "calibration_records": calibration_records,
        "city_balanced_summary": city_balanced_mean_rows,
    }

    json_path = out_dir / "e8_regionwise_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_region_bars(region_mean_rows, "mIoU", out_dir / "fig_region_miou.png")
    plot_region_bars(region_mean_rows, "Boundary_IoU", out_dir / "fig_region_biou.png")

    print("\n" + "=" * 90)
    print("E8 finished")
    print("=" * 90)
    print(f"Per-image CSV       : {out_dir / 'e8_per_image_metrics.csv'}")
    print(f"Region seed CSV     : {out_dir / 'e8_region_seed_metrics.csv'}")
    print(f"Region mean CSV     : {out_dir / 'e8_region_mean_metrics.csv'}")
    print(f"City-balanced CSV   : {out_dir / 'e8_city_balanced_summary.csv'}")
    print(f"Summary JSON        : {json_path}")
    print(f"Figure mIoU         : {out_dir / 'fig_region_miou.png'}")
    print(f"Figure Boundary IoU : {out_dir / 'fig_region_biou.png'}")

    print("\nCity-balanced summary:")
    for r in city_balanced_mean_rows:
        print(
            f"{r['model']:<22} "
            f"region-balanced mIoU={r['region_balanced_mIoU']*100:.2f} "
            f"BIoU={r['region_balanced_Boundary_IoU']*100:.2f} "
            f"worst-region mIoU={r['worst_region_mIoU']*100:.2f} "
            f"region-std={r['region_mIoU_std']*100:.2f}"
        )


if __name__ == "__main__":
    main()