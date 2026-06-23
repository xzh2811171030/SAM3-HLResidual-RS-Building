# -*- coding: utf-8 -*-
"""
run_qualitative_visualization.py

生成论文定性对比图：
RGB / GT / Prompt-free LoRA / HL-Residual / Error map

依赖：
  src/run_e7a_prompt_drift.py

推荐先跑 pilot500：
  python -u src/run_qualitative_visualization.py \
    --test_manifest_name target_pilot_test_500.txt \
    --test_eval_limit 500 \
    --seed 42 \
    --use_tta \
    --num_examples 12 \
    --out_dir results/qualitative_pilot500_seed42
"""

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

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


def read_rgb_from_path(path: str, size=512):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img


@torch.no_grad()
def predict_batch_probs(extractor, model, imgs: torch.Tensor, use_tta: bool):
    if not use_tta:
        feat = extractor.extract_batch(imgs)
        logits = model(feat.to(DEVICE), imgs.to(DEVICE))
        return torch.sigmoid(logits.float()).cpu().numpy()[:, 0]

    probs_list = []

    feat = extractor.extract_batch(imgs)
    logits = model(feat.to(DEVICE), imgs.to(DEVICE))
    probs_list.append(torch.sigmoid(logits.float()).cpu())

    imgs_h = torch.flip(imgs, dims=[3])
    feat_h = extractor.extract_batch(imgs_h)
    logits_h = model(feat_h.to(DEVICE), imgs_h.to(DEVICE))
    ph = torch.sigmoid(logits_h.float()).cpu()
    ph = torch.flip(ph, dims=[3])
    probs_list.append(ph)

    imgs_v = torch.flip(imgs, dims=[2])
    feat_v = extractor.extract_batch(imgs_v)
    logits_v = model(feat_v.to(DEVICE), imgs_v.to(DEVICE))
    pv = torch.sigmoid(logits_v.float()).cpu()
    pv = torch.flip(pv, dims=[2])
    probs_list.append(pv)

    return torch.mean(torch.stack(probs_list, dim=0), dim=0).numpy()[:, 0]


@torch.no_grad()
def calibrate_model(extractor, model, val_loader, use_tta: bool):
    probs_all, gts_all = [], []

    for batch in tqdm(val_loader, desc="  calibrate val", ncols=100):
        imgs = batch["image"]
        masks = batch["mask"].numpy()[:, 0]
        probs = predict_batch_probs(extractor, model, imgs, use_tta=use_tta)
        probs_all.append(probs)
        gts_all.append(masks)

    probs_all = np.concatenate(probs_all, axis=0)
    gts_all = np.concatenate(gts_all, axis=0)

    search = grid_search_postprocess(probs_all, gts_all)
    return search["best"]


def make_error_overlay(rgb, pred, gt):
    """
    输出 RGB overlay:
      TP: green
      FP: red
      FN: blue
    """
    rgb = rgb.astype(np.float32) / 255.0
    overlay = rgb.copy()

    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)

    tp = np.logical_and(pred_b, gt_b)
    fp = np.logical_and(pred_b, ~gt_b)
    fn = np.logical_and(~pred_b, gt_b)

    color = np.zeros_like(overlay)
    color[tp] = np.array([0.0, 1.0, 0.0])
    color[fp] = np.array([1.0, 0.0, 0.0])
    color[fn] = np.array([0.0, 0.2, 1.0])

    mask = np.logical_or(np.logical_or(tp, fp), fn)
    overlay[mask] = 0.45 * overlay[mask] + 0.55 * color[mask]

    return (overlay * 255).clip(0, 255).astype(np.uint8)


def save_panel(out_path: Path, rgb, gt, pred_lora, pred_hl, name, metrics):
    error_lora = make_error_overlay(rgb, pred_lora, gt)
    error_hl = make_error_overlay(rgb, pred_hl, gt)

    fig, axes = plt.subplots(1, 6, figsize=(18, 3.4))

    axes[0].imshow(rgb)
    axes[0].set_title("RGB")

    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT")

    axes[2].imshow(pred_lora, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"LoRA\nIoU {metrics['lora_mIoU']*100:.1f}")

    axes[3].imshow(pred_hl, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"HL-Residual\nIoU {metrics['hl_mIoU']*100:.1f}")

    axes[4].imshow(error_lora)
    axes[4].set_title("LoRA error\nG=TP R=FP B=FN")

    axes[5].imshow(error_hl)
    axes[5].set_title("HL error\nG=TP R=FP B=FN")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{name} | ΔIoU={metrics['delta_mIoU']*100:.2f}, "
        f"ΔBIoU={metrics['delta_Boundary_IoU']*100:.2f}",
        fontsize=11,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


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


def parse_args():
    parser = argparse.ArgumentParser("Qualitative visualization")

    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT_DEFAULT))
    parser.add_argument("--manifest_dir", type=str, default=None)
    parser.add_argument("--val_manifest_name", type=str, default="target_val.txt")
    parser.add_argument("--test_manifest_name", type=str, default="target_pilot_test_500.txt")
    parser.add_argument("--val_eval_limit", type=int, default=500)
    parser.add_argument("--test_eval_limit", type=int, default=500)
    parser.add_argument("--sam3_checkpoint", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_examples", type=int, default=12)
    parser.add_argument("--out_dir", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(args.project_root)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else project_root / "data/splits/e0_manifest"
    sam3_ckpt = Path(args.sam3_checkpoint) if args.sam3_checkpoint else project_root / "weights/sam3.pt"
    out_dir = Path(args.out_dir) if args.out_dir else project_root / "results/qualitative_visualization"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_limit = None if args.val_eval_limit <= 0 else args.val_eval_limit
    test_limit = None if args.test_eval_limit <= 0 else args.test_eval_limit

    print("=" * 90)
    print("Qualitative visualization")
    print("=" * 90)
    print(f"seed         : {args.seed}")
    print(f"use_tta      : {args.use_tta}")
    print(f"test_manifest: {args.test_manifest_name}")
    print(f"out_dir      : {out_dir}")
    print("=" * 90)

    val_ds = ManifestDataset(manifest_dir / args.val_manifest_name, limit=val_limit)
    test_ds = ManifestDataset(manifest_dir / args.test_manifest_name, limit=test_limit)

    val_loader = make_loader(val_ds, args.batch_size, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, args.num_workers)

    print("[Load] Prompt-free LoRA")
    lora_ext, lora_model, lora_ckpt = load_promptfree_model(
        "promptfree_lora", args.seed, project_root, sam3_ckpt
    )

    print("[Load] HL-Residual")
    hl_ext, hl_model, hl_ckpt = load_promptfree_model(
        "hl_residual", args.seed, project_root, sam3_ckpt
    )

    print("[Calibrate] LoRA")
    lora_cfg = calibrate_model(lora_ext, lora_model, val_loader, use_tta=args.use_tta)
    print(f"  LoRA cfg: {lora_cfg}")

    print("[Calibrate] HL-Residual")
    hl_cfg = calibrate_model(hl_ext, hl_model, val_loader, use_tta=args.use_tta)
    print(f"  HL cfg: {hl_cfg}")

    rows = []
    cache = {}

    for batch in tqdm(test_loader, desc="  evaluate candidates", ncols=100):
        imgs = batch["image"]
        masks = batch["mask"].numpy()[:, 0]
        names = batch["name"]
        paths = batch["path"]

        probs_lora = predict_batch_probs(lora_ext, lora_model, imgs, use_tta=args.use_tta)
        probs_hl = predict_batch_probs(hl_ext, hl_model, imgs, use_tta=args.use_tta)

        for i, name in enumerate(names):
            pred_lora = postprocess_mask(
                probs_lora[i],
                threshold=lora_cfg["threshold"],
                min_area=lora_cfg["min_area"],
                closing_kernel=lora_cfg["closing_kernel"],
                fill_hole=lora_cfg["fill_holes"],
            )

            pred_hl = postprocess_mask(
                probs_hl[i],
                threshold=hl_cfg["threshold"],
                min_area=hl_cfg["min_area"],
                closing_kernel=hl_cfg["closing_kernel"],
                fill_hole=hl_cfg["fill_holes"],
            )

            gt = masks[i]

            row = {
                "image_name": name,
                "image_path": paths[i],
                "lora_mIoU": compute_iou(pred_lora, gt),
                "hl_mIoU": compute_iou(pred_hl, gt),
                "lora_F1": 0.0,
                "hl_F1": 0.0,
                "lora_Boundary_IoU": compute_biou(pred_lora, gt, d=5),
                "hl_Boundary_IoU": compute_biou(pred_hl, gt, d=5),
            }
            row["delta_mIoU"] = row["hl_mIoU"] - row["lora_mIoU"]
            row["delta_Boundary_IoU"] = row["hl_Boundary_IoU"] - row["lora_Boundary_IoU"]

            rows.append(row)

            cache[name] = {
                "gt": gt,
                "pred_lora": pred_lora,
                "pred_hl": pred_hl,
                "path": paths[i],
                "metrics": row,
            }

    write_csv(out_dir / "qualitative_candidate_metrics.csv", rows)

    # 选择样例
    sorted_gain_iou = sorted(rows, key=lambda x: x["delta_mIoU"], reverse=True)
    sorted_gain_biou = sorted(rows, key=lambda x: x["delta_Boundary_IoU"], reverse=True)
    sorted_failure = sorted(rows, key=lambda x: x["delta_mIoU"])

    selected = []
    used = set()

    def add_samples(items, tag, n):
        for r in items:
            if len([x for x in selected if x["tag"] == tag]) >= n:
                break
            if r["image_name"] in used:
                continue
            used.add(r["image_name"])
            selected.append({**r, "tag": tag})

    n_each = max(2, args.num_examples // 4)
    add_samples(sorted_gain_iou, "best_delta_iou", n_each)
    add_samples(sorted_gain_biou, "best_delta_biou", n_each)
    add_samples(sorted_failure, "failure_cases", n_each)

    # 补足
    for r in rows:
        if len(selected) >= args.num_examples:
            break
        if r["image_name"] not in used:
            used.add(r["image_name"])
            selected.append({**r, "tag": "additional"})

    write_csv(out_dir / "qualitative_selected_examples.csv", selected)

    panel_dir = out_dir / "panels"
    for idx, r in enumerate(selected):
        item = cache[r["image_name"]]
        rgb = read_rgb_from_path(item["path"])
        out_path = panel_dir / f"{idx:02d}_{r['tag']}_{r['image_name']}.png"
        save_panel(
            out_path=out_path,
            rgb=rgb,
            gt=item["gt"],
            pred_lora=item["pred_lora"],
            pred_hl=item["pred_hl"],
            name=r["image_name"],
            metrics=item["metrics"],
        )

    summary = {
        "seed": args.seed,
        "use_tta": args.use_tta,
        "test_manifest": args.test_manifest_name,
        "lora_ckpt": str(lora_ckpt),
        "hl_ckpt": str(hl_ckpt),
        "lora_cfg": lora_cfg,
        "hl_cfg": hl_cfg,
        "num_candidates": len(rows),
        "num_selected": len(selected),
    }

    (out_dir / "qualitative_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Panels: {panel_dir}")
    print(f"Selected CSV: {out_dir / 'qualitative_selected_examples.csv'}")


if __name__ == "__main__":
    main()