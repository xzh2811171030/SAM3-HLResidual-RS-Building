# -*- coding: utf-8 -*-
"""
run_qualitative_visualization_v2.py

Publication-grade qualitative visualization for the HL-Residual manuscript.

Outputs:
  1) qualitative_candidate_metrics.csv          all evaluated pilot/test candidates
  2) qualitative_selected_examples.csv          balanced candidate pool for manual screening
  3) panels_candidate/*.png                     clean individual panels for manual selection
  4) figures/fig3_qualitative_main.png/pdf      main-paper qualitative montage
  5) figures/figS1_qualitative_additional.png/pdf supplementary montage

Main design principles:
  - one clean header row; no overlapping suptitle and subplot titles;
  - full journal-style labels: Ground Truth, Prompt-free LoRA, HL-Residual;
  - error overlay: TP=green, FP=red, FN=blue;
  - balanced sample selection, not only largest numerical gains;
  - avoid tiny-building-only examples dominating the main qualitative figure.

Recommended command:
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u src/run_qualitative_visualization_v2.py \
    --test_manifest_name target_pilot_test_500.txt \
    --test_eval_limit 500 \
    --seed 42 \
    --use_tta \
    --batch_size 4 \
    --num_workers 2 \
    --candidate_panel_count 80 \
    --main_examples 6 \
    --supp_examples 12 \
    --min_main_fg_ratio 0.01 \
    --out_dir results/qualitative_pilot500_seed42_v2
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
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


# -----------------------------------------------------------------------------
# Data loading and model inference
# -----------------------------------------------------------------------------

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


def read_rgb_from_path(path: str, size: int = 512) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img


@torch.no_grad()
def predict_batch_probs(extractor, model, imgs: torch.Tensor, use_tta: bool) -> np.ndarray:
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


# -----------------------------------------------------------------------------
# Metrics and selection helpers
# -----------------------------------------------------------------------------

def binary_stats(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = np.logical_and(pred_b, gt_b).sum()
    fp = np.logical_and(pred_b, ~gt_b).sum()
    fn = np.logical_and(~pred_b, gt_b).sum()
    tn = np.logical_and(~pred_b, ~gt_b).sum()
    total = pred_b.size
    return {
        "tp_ratio": tp / total,
        "fp_ratio": fp / total,
        "fn_ratio": fn / total,
        "tn_ratio": tn / total,
    }


def region_prefix(name: str) -> str:
    if not name:
        return "unknown"
    return str(name).split("_")[0].lower()


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


def add_ranked_samples(
    selected: List[Dict],
    used: set,
    rows: Sequence[Dict],
    tag: str,
    n: int,
    sort_key: str,
    reverse: bool = True,
    min_fg: float = 0.0,
    max_fg: float = 1.0,
    require_positive: Optional[str] = None,
):
    if n <= 0:
        return
    items = [r for r in rows if min_fg <= float(r.get("gt_fg_ratio", 0.0)) <= max_fg]
    if require_positive is not None:
        items = [r for r in items if float(r.get(require_positive, 0.0)) > 0]
    items = sorted(items, key=lambda r: float(r.get(sort_key, -1e9)), reverse=reverse)
    cnt = 0
    for r in items:
        if cnt >= n:
            break
        name = r["image_name"]
        if name in used:
            continue
        used.add(name)
        selected.append({**r, "tag": tag})
        cnt += 1


def select_balanced_examples(
    rows: List[Dict],
    candidate_count: int,
    main_count: int,
    min_main_fg: float,
    max_main_fg: float,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Returns:
      selected_pool: many candidate panels for manual screening;
      main_examples: 6/8 balanced examples for main paper;
      supp_examples: remaining good/failure cases for supplementary.
    """
    selected: List[Dict] = []
    used = set()

    # Candidate pool: balanced evidence categories.
    q = max(4, candidate_count // 8)
    add_ranked_samples(selected, used, rows, "boundary_gain", q, "delta_Boundary_IoU", True, 0.002, 0.80)
    add_ranked_samples(selected, used, rows, "region_iou_gain", q, "delta_mIoU", True, 0.002, 0.80)
    add_ranked_samples(selected, used, rows, "fp_reduction", q, "delta_fp_ratio", True, 0.002, 0.80, "delta_fp_ratio")
    add_ranked_samples(selected, used, rows, "fn_reduction", q, "delta_fn_ratio", True, 0.002, 0.80, "delta_fn_ratio")
    add_ranked_samples(selected, used, rows, "failure_or_regression", max(3, q // 2), "delta_mIoU", False, 0.002, 0.80)
    add_ranked_samples(selected, used, rows, "tiny_building_case", max(3, q // 2), "delta_Boundary_IoU", True, 0.0, 0.01)

    # Region-diverse examples: pick one or two strong boundary-gain examples per prefix.
    by_region: Dict[str, List[Dict]] = {}
    for r in rows:
        by_region.setdefault(str(r.get("region", "unknown")), []).append(r)
    for reg, items in sorted(by_region.items()):
        items = sorted(items, key=lambda r: float(r.get("delta_Boundary_IoU", 0.0)), reverse=True)
        for r in items[:2]:
            if len(selected) >= candidate_count:
                break
            if r["image_name"] not in used:
                used.add(r["image_name"])
                selected.append({**r, "tag": f"region_{reg}"})

    # Fill remaining with best boundary gains.
    add_ranked_samples(selected, used, rows, "additional", candidate_count - len(selected), "delta_Boundary_IoU", True, 0.0, 1.0)
    selected_pool = selected[:candidate_count]

    # Main examples: avoid extremely tiny masks; prioritize diverse categories and regions.
    main: List[Dict] = []
    main_used = set()
    preferred_tags = ["boundary_gain", "region_iou_gain", "fp_reduction", "fn_reduction"]
    for tag in preferred_tags:
        items = [r for r in selected_pool if r["tag"] == tag and min_main_fg <= float(r["gt_fg_ratio"]) <= max_main_fg]
        items = sorted(items, key=lambda r: (float(r["delta_Boundary_IoU"]), float(r["delta_mIoU"])), reverse=True)
        for r in items:
            if len(main) >= main_count:
                break
            if r["image_name"] not in main_used:
                main_used.add(r["image_name"])
                main.append(r)
                break

    # Add region-diverse candidates to main.
    for r in selected_pool:
        if len(main) >= main_count:
            break
        if r["image_name"] in main_used:
            continue
        if not (min_main_fg <= float(r["gt_fg_ratio"]) <= max_main_fg):
            continue
        # avoid all samples coming from same region
        main_regions = {x.get("region") for x in main}
        if r.get("region") not in main_regions or len(main_regions) >= 4:
            main_used.add(r["image_name"])
            main.append(r)

    # If still insufficient, fill by boundary gain with the same area filter.
    for r in selected_pool:
        if len(main) >= main_count:
            break
        if r["image_name"] not in main_used and min_main_fg <= float(r["gt_fg_ratio"]) <= max_main_fg:
            main_used.add(r["image_name"])
            main.append(r)

    supp = [r for r in selected_pool if r["image_name"] not in main_used]
    return selected_pool, main, supp


# -----------------------------------------------------------------------------
# Publication-grade rendering utilities
# -----------------------------------------------------------------------------

def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = []
    if bold:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def mask_to_rgb(mask: np.ndarray, fg=(255, 255, 255), bg=(0, 0, 0)) -> np.ndarray:
    mask_b = mask.astype(bool)
    out = np.zeros((mask_b.shape[0], mask_b.shape[1], 3), dtype=np.uint8)
    out[~mask_b] = np.array(bg, dtype=np.uint8)
    out[mask_b] = np.array(fg, dtype=np.uint8)
    return out


def make_error_overlay(rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray, alpha: float = 0.58) -> np.ndarray:
    """
    Overlay RGB error map:
      TP = green, FP = red, FN = blue.
    TN is unchanged RGB.
    """
    base = rgb.astype(np.float32) / 255.0
    overlay = base.copy()
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = np.logical_and(pred_b, gt_b)
    fp = np.logical_and(pred_b, ~gt_b)
    fn = np.logical_and(~pred_b, gt_b)

    color = np.zeros_like(overlay)
    color[tp] = np.array([0.0, 0.95, 0.0])   # green
    color[fp] = np.array([1.0, 0.0, 0.0])    # red
    color[fn] = np.array([0.0, 0.25, 1.0])   # blue

    mask = tp | fp | fn
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color[mask]
    return (overlay * 255).clip(0, 255).astype(np.uint8)


def resize_tile(img: np.ndarray, tile: int, is_mask: bool = False) -> Image.Image:
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    resized = cv2.resize(img, (tile, tile), interpolation=interp)
    return Image.fromarray(resized.astype(np.uint8)).convert("RGB")


def draw_centered_text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font, fill=(20, 20, 20)):
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((x - w / 2, y - h / 2), text, font=font, fill=fill)


def make_row_arrays(cache_item: Dict) -> List[np.ndarray]:
    rgb = cache_item["rgb"]
    gt = cache_item["gt"].astype(np.uint8)
    pred_lora = cache_item["pred_lora"].astype(np.uint8)
    pred_hl = cache_item["pred_hl"].astype(np.uint8)
    err_lora = make_error_overlay(rgb, pred_lora, gt)
    err_hl = make_error_overlay(rgb, pred_hl, gt)
    return [rgb, mask_to_rgb(gt), mask_to_rgb(pred_lora), mask_to_rgb(pred_hl), err_lora, err_hl]


def save_single_panel(
    out_path: Path,
    cache_item: Dict,
    row: Dict,
    tile: int = 320,
    show_metrics: bool = True,
):
    """Save one clean 1x6 panel. No title overlap."""
    cols = ["RGB", "Ground Truth", "Prompt-free LoRA", "HL-Residual", "LoRA error", "HL-Residual error"]
    arrays = make_row_arrays(cache_item)
    images = [resize_tile(a, tile, is_mask=(i in [1, 2, 3])) for i, a in enumerate(arrays)]

    gap = 16
    header_h = 58
    footer_h = 42 if show_metrics else 18
    W = 6 * tile + 5 * gap
    H = header_h + tile + footer_h
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    font_h = get_font(22, bold=False)
    font_f = get_font(17, bold=False)

    for j, (label, img) in enumerate(zip(cols, images)):
        x = j * (tile + gap)
        draw_centered_text(draw, (x + tile // 2, 28), label, font_h)
        canvas.paste(img, (x, header_h))

    # error legend and metrics below the image, not above, so it cannot overlap headers.
    if show_metrics:
        text = (
            f"{row['image_name']} | IoU: LoRA {row['lora_mIoU']*100:.1f}, "
            f"HL {row['hl_mIoU']*100:.1f} | "
            f"Delta IoU {row['delta_mIoU']*100:+.2f}, Delta BIoU {row['delta_Boundary_IoU']*100:+.2f} | "
            "error: green=TP, red=FP, blue=FN"
        )
        draw_centered_text(draw, (W // 2, header_h + tile + 22), text, font_f, fill=(35, 35, 35))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(600, 600))


def save_montage(
    out_png: Path,
    out_pdf: Optional[Path],
    selected_rows: Sequence[Dict],
    cache: Dict[str, Dict],
    tile: int = 230,
    row_label_w: int = 205,
    dpi: int = 600,
):
    """Save journal-style multi-row montage with one header row."""
    if not selected_rows:
        return

    cols = ["RGB", "Ground Truth", "Prompt-free LoRA", "HL-Residual", "LoRA error", "HL-Residual error"]
    gap = 10
    row_gap = 14
    header_h = 58
    footer_h = 42
    n_rows = len(selected_rows)
    W = row_label_w + 6 * tile + 5 * gap
    H = header_h + n_rows * tile + (n_rows - 1) * row_gap + footer_h
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    font_header = get_font(18, bold=False)
    font_label = get_font(16, bold=False)
    font_small = get_font(13, bold=False)

    # Header
    for j, label in enumerate(cols):
        x = row_label_w + j * (tile + gap)
        draw_centered_text(draw, (x + tile // 2, 30), label, font_header)

    # Rows
    y = header_h
    for row_idx, r in enumerate(selected_rows):
        name = r["image_name"]
        item = cache[name]
        arrays = make_row_arrays(item)
        tiles = [resize_tile(a, tile, is_mask=(i in [1, 2, 3])) for i, a in enumerate(arrays)]

        # row label with image id and compact improvements
        region = str(r.get("region", ""))
        line1 = name
        line2 = f"{region} | dIoU {r['delta_mIoU']*100:+.1f}"
        line3 = f"dBIoU {r['delta_Boundary_IoU']*100:+.1f}"
        draw.text((8, y + tile // 2 - 32), line1, font=font_label, fill=(30, 30, 30))
        draw.text((8, y + tile // 2 - 8), line2, font=font_small, fill=(75, 75, 75))
        draw.text((8, y + tile // 2 + 13), line3, font=font_small, fill=(75, 75, 75))

        for j, img in enumerate(tiles):
            x = row_label_w + j * (tile + gap)
            canvas.paste(img, (x, y))
        y += tile + row_gap

    legend = "Error overlay: green = true positive, red = false positive, blue = false negative."
    draw_centered_text(draw, (W // 2, H - 20), legend, font_small, fill=(50, 50, 50))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png, dpi=(dpi, dpi))
    if out_pdf is not None:
        canvas.save(out_pdf, "PDF", resolution=dpi)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser("Publication-grade qualitative visualization")

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
    parser.add_argument("--out_dir", type=str, default=None)

    # New visualization controls
    parser.add_argument("--candidate_panel_count", type=int, default=80,
                        help="Number of clean individual candidate panels to save for manual screening.")
    parser.add_argument("--main_examples", type=int, default=6,
                        help="Rows in the main-paper qualitative montage.")
    parser.add_argument("--supp_examples", type=int, default=12,
                        help="Rows in the supplementary montage.")
    parser.add_argument("--tile_size", type=int, default=230,
                        help="Tile size for montage rows. 220-260 is usually good for 6-column figures.")
    parser.add_argument("--panel_tile_size", type=int, default=320,
                        help="Tile size for individual candidate panels.")
    parser.add_argument("--min_main_fg_ratio", type=float, default=0.01,
                        help="Minimum GT foreground ratio for main montage; avoid tiny-building-only examples.")
    parser.add_argument("--max_main_fg_ratio", type=float, default=0.65,
                        help="Maximum GT foreground ratio for main montage; avoid nearly all-building patches.")
    parser.add_argument("--no_pdf", action="store_true",
                        help="Only save PNG montage, not PDF.")

    # Backward compatibility with old script; not used as main selector anymore.
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Deprecated. If given, used as candidate_panel_count when candidate_panel_count is not set.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_examples is not None and args.candidate_panel_count == 80:
        args.candidate_panel_count = max(args.num_examples, 24)

    project_root = Path(args.project_root)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else project_root / "data/splits/e0_manifest"
    sam3_ckpt = Path(args.sam3_checkpoint) if args.sam3_checkpoint else project_root / "weights/sam3.pt"
    out_dir = Path(args.out_dir) if args.out_dir else project_root / "results/qualitative_visualization_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_limit = None if args.val_eval_limit <= 0 else args.val_eval_limit
    test_limit = None if args.test_eval_limit <= 0 else args.test_eval_limit

    print("=" * 90)
    print("Publication-grade qualitative visualization")
    print("=" * 90)
    print(f"seed                  : {args.seed}")
    print(f"use_tta               : {args.use_tta}")
    print(f"test_manifest         : {args.test_manifest_name}")
    print(f"test_eval_limit       : {args.test_eval_limit}")
    print(f"candidate_panel_count : {args.candidate_panel_count}")
    print(f"main_examples         : {args.main_examples}")
    print(f"supp_examples         : {args.supp_examples}")
    print(f"out_dir               : {out_dir}")
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

    print("[Calibrate] Prompt-free LoRA")
    lora_cfg = calibrate_model(lora_ext, lora_model, val_loader, use_tta=args.use_tta)
    print(f"  LoRA cfg: {lora_cfg}")

    print("[Calibrate] HL-Residual")
    hl_cfg = calibrate_model(hl_ext, hl_model, val_loader, use_tta=args.use_tta)
    print(f"  HL cfg: {hl_cfg}")

    rows: List[Dict] = []
    cache: Dict[str, Dict] = {}

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
            ).astype(np.uint8)

            pred_hl = postprocess_mask(
                probs_hl[i],
                threshold=hl_cfg["threshold"],
                min_area=hl_cfg["min_area"],
                closing_kernel=hl_cfg["closing_kernel"],
                fill_hole=hl_cfg["fill_holes"],
            ).astype(np.uint8)

            gt = (masks[i] > 0.5).astype(np.uint8)
            stats_lora = binary_stats(pred_lora, gt)
            stats_hl = binary_stats(pred_hl, gt)

            row = {
                "image_name": name,
                "region": region_prefix(name),
                "image_path": paths[i],
                "gt_fg_ratio": float(gt.mean()),
                "lora_mIoU": compute_iou(pred_lora, gt),
                "hl_mIoU": compute_iou(pred_hl, gt),
                "lora_F1": compute_f1(pred_lora, gt),
                "hl_F1": compute_f1(pred_hl, gt),
                "lora_Boundary_IoU": compute_biou(pred_lora, gt, d=5),
                "hl_Boundary_IoU": compute_biou(pred_hl, gt, d=5),
                "lora_fp_ratio": stats_lora["fp_ratio"],
                "hl_fp_ratio": stats_hl["fp_ratio"],
                "lora_fn_ratio": stats_lora["fn_ratio"],
                "hl_fn_ratio": stats_hl["fn_ratio"],
            }
            row["delta_mIoU"] = row["hl_mIoU"] - row["lora_mIoU"]
            row["delta_F1"] = row["hl_F1"] - row["lora_F1"]
            row["delta_Boundary_IoU"] = row["hl_Boundary_IoU"] - row["lora_Boundary_IoU"]
            row["delta_fp_ratio"] = row["lora_fp_ratio"] - row["hl_fp_ratio"]
            row["delta_fn_ratio"] = row["lora_fn_ratio"] - row["hl_fn_ratio"]
            rows.append(row)

            # Store images only for candidate generation. RGB reading is cheap enough here.
            rgb = read_rgb_from_path(paths[i])
            cache[name] = {
                "rgb": rgb,
                "gt": gt,
                "pred_lora": pred_lora,
                "pred_hl": pred_hl,
                "path": paths[i],
                "metrics": row,
            }

    write_csv(out_dir / "qualitative_candidate_metrics.csv", rows)

    selected_pool, main_examples, supp_pool = select_balanced_examples(
        rows=rows,
        candidate_count=args.candidate_panel_count,
        main_count=args.main_examples,
        min_main_fg=args.min_main_fg_ratio,
        max_main_fg=args.max_main_fg_ratio,
    )
    supp_examples = supp_pool[:args.supp_examples]

    write_csv(out_dir / "qualitative_selected_examples.csv", selected_pool)
    write_csv(out_dir / "qualitative_main_examples.csv", main_examples)
    write_csv(out_dir / "qualitative_supp_examples.csv", supp_examples)

    # Save clean individual candidate panels for manual choice.
    panel_dir = out_dir / "panels_candidate"
    for idx, r in enumerate(selected_pool):
        item = cache[r["image_name"]]
        out_path = panel_dir / f"{idx:03d}_{r['tag']}_{r['image_name']}.png"
        save_single_panel(
            out_path=out_path,
            cache_item=item,
            row=r,
            tile=args.panel_tile_size,
            show_metrics=True,
        )

    # Main and supplementary montages.
    fig_dir = out_dir / "figures"
    save_montage(
        out_png=fig_dir / "fig3_qualitative_main.png",
        out_pdf=None if args.no_pdf else fig_dir / "fig3_qualitative_main.pdf",
        selected_rows=main_examples,
        cache=cache,
        tile=args.tile_size,
        dpi=600,
    )
    save_montage(
        out_png=fig_dir / "figS1_qualitative_additional.png",
        out_pdf=None if args.no_pdf else fig_dir / "figS1_qualitative_additional.pdf",
        selected_rows=supp_examples,
        cache=cache,
        tile=args.tile_size,
        dpi=600,
    )

    summary = {
        "seed": args.seed,
        "use_tta": args.use_tta,
        "test_manifest": args.test_manifest_name,
        "val_manifest": args.val_manifest_name,
        "lora_ckpt": str(lora_ckpt),
        "hl_ckpt": str(hl_ckpt),
        "lora_cfg": lora_cfg,
        "hl_cfg": hl_cfg,
        "num_candidates": len(rows),
        "num_selected_pool": len(selected_pool),
        "num_main_examples": len(main_examples),
        "num_supp_examples": len(supp_examples),
        "error_color_rule": "TP=green, FP=red, FN=blue",
    }
    (out_dir / "qualitative_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nDone.")
    print(f"All candidate metrics : {out_dir / 'qualitative_candidate_metrics.csv'}")
    print(f"Selected candidates   : {out_dir / 'qualitative_selected_examples.csv'}")
    print(f"Candidate panels      : {panel_dir}")
    print(f"Main montage          : {fig_dir / 'fig3_qualitative_main.png'}")
    print(f"Supplementary montage : {fig_dir / 'figS1_qualitative_additional.png'}")

    # Explicit cleanup for long AutoDL sessions.
    del lora_model, hl_model, lora_ext, hl_ext
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
