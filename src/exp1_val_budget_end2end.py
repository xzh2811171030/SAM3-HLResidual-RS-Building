#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end validation-budget sensitivity experiment.

This script does two things in one run:
1. Re-run inference for LoRA and HL-Residual checkpoints and save probability maps as .npz.
2. Sweep validation calibration budgets and evaluate on the held-out non-pilot test set.

Minimal required experiment:
    models: lora, hl
    seeds: 42, 123, 456
    budgets: 0, 50, 100, 500
    splits: target_val_500 and held-out non-pilot 7902

Important:
    You must adapt only the ADAPTER SECTION to your project.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError as exc:
    raise ImportError("Please install opencv-python: pip install opencv-python") from exc


SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_calibrated_fulltest import (
    DEVICE as CF_DEVICE,
    DEFAULT_PROJECT_ROOT,
    find_dual_label_for_image,
    load_model,
)

# 脚本内模型名 → calibrated_fulltest 模型名
MODEL_NAME_MAP = {"lora": "lora_light", "hl": "hl_residual_old"}
SAM3_CKPT = str(DEFAULT_PROJECT_ROOT / "weights" / "sam3.pt")


# ============================================================
# ADAPTER SECTION
# ============================================================

def build_dataset_items(manifest_path: str) -> List[Tuple[str, str, str]]:
    """
    Return a list of (image_path, mask_path, image_id).

    e0_manifest 文件每行只有一列 image_path，mask 通过 find_dual_label_for_image 自动查找。
    """
    items: List[Tuple[str, str, str]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().replace("\\", "/")
            if not line:
                continue

            image_path = line.split()[0]  # 只取第一列
            image_path = str(Path(image_path))

            mask_path_obj = find_dual_label_for_image(Path(image_path))
            if mask_path_obj is None:
                raise FileNotFoundError(f"Cannot find dual label for {image_path}")
            mask_path = str(mask_path_obj)

            image_id = Path(image_path).stem
            items.append((image_path, mask_path, image_id))

    if not items:
        raise RuntimeError(f"No items loaded from {manifest_path}")
    return items


def load_model_from_checkpoint(model_name: str, seed: int, ckpt_path: str, device: torch.device):
    """
    加载 Prompt-free LoRA 或 HL-Residual 模型。

    model_name: "lora" → lora_light, "hl" → hl_residual_old
    返回一个 ModelWrapper，将 SAM3 特征提取 + mask decoder 封装为统一前向接口。
    """
    mapped = MODEL_NAME_MAP.get(model_name, model_name)
    extractor, decoder_model = load_model(mapped, Path(SAM3_CKPT), Path(ckpt_path))

    # 将 extractor + decoder 封装为一个可调用对象，匹配 predict_probability 的调用方式
    class ModelWrapper:
        def __init__(self, extractor, model, device):
            self._extractor = extractor
            self._model = model
            self._device = device

        def eval(self):
            self._extractor.model.eval()
            self._model.eval()

        @torch.no_grad()
        def __call__(self, x: torch.Tensor):
            """x: (1, 3, 512, 512) float [0,1] → logits: (1, 1, 512, 512)"""
            feat = self._extractor.extract_batch(x.to(self._device))
            logits = self._model(feat.to(self._device), x.to(self._device))
            return logits

    return ModelWrapper(extractor, decoder_model, device)


def preprocess_image(image_path: str, device: torch.device) -> torch.Tensor:
    """
    图像预处理，与 SAM3 管线一致：
    - 读取 RGB
    - resize 到 512x512（SAM3 backbone 内部会再 resize 到 1008）
    - 归一化到 [0, 1]
    - 输出 shape: (1, 3, 512, 512)
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def load_mask(mask_path: str, out_hw: Tuple[int, int]) -> np.ndarray:
    """
    加载 GT mask。兼容 dual_channel_labels（H, W, 2）PNG 格式，取第一个通道。
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(mask_path)

    if mask.ndim == 3:
        mask = mask[..., 0]  # dual-channel label → 取第一个通道

    h, w = out_hw
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    mask = (mask > 0).astype(np.uint8)
    return mask


@torch.no_grad()
def predict_probability(model, image_tensor: torch.Tensor, use_tta: bool = True) -> torch.Tensor:
    """
    Return probability map, shape 1 x 1 x H x W, range [0, 1].

    You may need to modify the forward call according to your model.

    Common cases:
        logits = model(image_tensor)
        logits, _ = model(image_tensor)
        out = model(image_tensor)["logits"]

    Current default assumes:
        output = model(image_tensor)
        if tuple/list, use output[0]
        if dict, use output["logits"]
        then sigmoid.
    """
    def forward_once(x: torch.Tensor) -> torch.Tensor:
        out = model(x)

        if isinstance(out, dict):
            if "logits" in out:
                logits = out["logits"]
            elif "pred" in out:
                logits = out["pred"]
            else:
                raise KeyError(f"Model output dict has no logits/pred key: {out.keys()}")
        elif isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out

        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        return logits

    logits = forward_once(image_tensor)

    if use_tta:
        x_h = torch.flip(image_tensor, dims=[3])
        logit_h = torch.flip(forward_once(x_h), dims=[3])

        x_v = torch.flip(image_tensor, dims=[2])
        logit_v = torch.flip(forward_once(x_v), dims=[2])

        logits = (logits + logit_h + logit_v) / 3.0

    prob = torch.sigmoid(logits)
    return prob


# ============================================================
# Metric and post-processing utilities
# Usually no need to edit below.
# ============================================================

@dataclass(frozen=True)
class PostParams:
    threshold: float
    min_area: int
    closing_ks: int
    fill_holes: bool


@dataclass
class MetricResult:
    miou: float
    f1: float
    biou: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    out = np.zeros_like(mask, dtype=np.uint8)
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == idx] = 1
    return out


def fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    h, w = mask_u8.shape

    padded = np.pad(mask_u8, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    ff_mask = np.zeros((h + 4, w + 4), dtype=np.uint8)
    cv2.floodFill(padded, ff_mask, (0, 0), 1)

    flood = padded[1:-1, 1:-1]
    holes = (flood == 0).astype(np.uint8)
    return np.maximum(mask_u8, holes).astype(np.uint8)


def apply_postprocess(prob: np.ndarray, params: PostParams) -> np.ndarray:
    mask = (prob >= params.threshold).astype(np.uint8)

    if params.closing_ks and params.closing_ks > 1:
        k = int(params.closing_ks)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if params.fill_holes:
        mask = fill_binary_holes(mask)

    if params.min_area > 0:
        mask = remove_small_components(mask, params.min_area)

    return mask.astype(np.uint8)


def foreground_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def foreground_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0
    return float((2 * tp) / denom)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    return (mask - eroded).clip(0, 1).astype(np.uint8)


def boundary_iou(pred: np.ndarray, gt: np.ndarray, radius: int = 5) -> float:
    pred_b = mask_boundary(pred)
    gt_b = mask_boundary(gt)

    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0

    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    pred_band = cv2.dilate(pred_b, kernel, iterations=1) > 0
    gt_band = cv2.dilate(gt_b, kernel, iterations=1) > 0

    inter = np.logical_and(pred_band, gt_band).sum()
    union = np.logical_or(pred_band, gt_band).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def load_npz_item(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    if "prob" not in data or "gt" not in data:
        raise KeyError(f"{path} must contain keys: prob, gt")

    prob = data["prob"].astype(np.float32)
    gt = (data["gt"] > 0).astype(np.uint8)

    if prob.ndim == 3:
        prob = np.squeeze(prob)
    if gt.ndim == 3:
        gt = np.squeeze(gt)

    if prob.shape != gt.shape:
        raise ValueError(f"Shape mismatch: {path}, prob={prob.shape}, gt={gt.shape}")

    return prob, gt


def evaluate_npz_files(files: Sequence[Path], params: PostParams, boundary_radius: int) -> MetricResult:
    miou_list: List[float] = []
    f1_list: List[float] = []
    biou_list: List[float] = []

    for p in files:
        prob, gt = load_npz_item(p)
        pred = apply_postprocess(prob, params)

        miou_list.append(foreground_iou(pred, gt))
        f1_list.append(foreground_f1(pred, gt))
        biou_list.append(boundary_iou(pred, gt, radius=boundary_radius))

    return MetricResult(
        miou=float(np.mean(miou_list)),
        f1=float(np.mean(f1_list)),
        biou=float(np.mean(biou_list)),
    )


def parse_float_grid(spec: str) -> List[float]:
    spec = spec.strip()
    if ":" in spec:
        start, stop, step = map(float, spec.split(":"))
        vals = []
        x = start
        while x <= stop + 1e-9:
            vals.append(round(x, 6))
            x += step
        return vals
    return [float(x) for x in spec.split(",") if x.strip()]


def parse_int_list(spec: str) -> List[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def build_param_grid(
    thresholds: Sequence[float],
    min_areas: Sequence[int],
    closing_ks: Sequence[int],
    fill_holes_options: Sequence[bool],
) -> List[PostParams]:
    grid = []
    for th in thresholds:
        for area in min_areas:
            for ck in closing_ks:
                for fill in fill_holes_options:
                    grid.append(
                        PostParams(
                            threshold=float(th),
                            min_area=int(area),
                            closing_ks=int(ck),
                            fill_holes=bool(fill),
                        )
                    )
    return grid


def metric_value(result: MetricResult, select_metric: str) -> float:
    if select_metric == "miou":
        return result.miou
    if select_metric == "f1":
        return result.f1
    if select_metric == "biou":
        return result.biou
    if select_metric == "miou_biou_mean":
        return 0.5 * (result.miou + result.biou)
    raise ValueError(select_metric)


def choose_budget_subset(files: Sequence[Path], budget: int, calib_seed: int) -> List[Path]:
    files = list(files)
    if budget <= 0:
        return []
    if budget >= len(files):
        return files

    rng = random.Random(calib_seed)
    idxs = sorted(rng.sample(range(len(files)), budget))
    return [files[i] for i in idxs]


def calibrate_params(
    val_files: Sequence[Path],
    budget: int,
    grid: Sequence[PostParams],
    boundary_radius: int,
    select_metric: str,
    calib_seed: int,
) -> Tuple[PostParams, MetricResult]:
    if budget == 0:
        fixed = PostParams(threshold=0.5, min_area=0, closing_ks=0, fill_holes=False)
        return fixed, MetricResult(miou=float("nan"), f1=float("nan"), biou=float("nan"))

    subset = choose_budget_subset(val_files, budget, calib_seed)

    best_params = None
    best_result = None
    best_score = -1e18

    for params in grid:
        res = evaluate_npz_files(subset, params=params, boundary_radius=boundary_radius)
        score = metric_value(res, select_metric)
        if score > best_score:
            best_score = score
            best_params = params
            best_result = res

    assert best_params is not None and best_result is not None
    return best_params, best_result


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_probability_cache_for_split(
    model,
    items: List[Tuple[str, str, str]],
    out_split_dir: Path,
    device: torch.device,
    use_tta: bool,
    overwrite: bool,
) -> None:
    ensure_dir(out_split_dir)

    n = len(items)
    for i, (image_path, mask_path, image_id) in enumerate(items, 1):
        out_path = out_split_dir / f"{image_id}.npz"
        if out_path.exists() and not overwrite:
            if i % 200 == 0 or i == n:
                print(f"[CACHE] skip existing {i}/{n}: {out_split_dir}", flush=True)
            continue

        image_tensor = preprocess_image(image_path, device=device)

        prob_t = predict_probability(model, image_tensor, use_tta=use_tta)
        prob_t = prob_t.detach().float().cpu()

        prob = prob_t.squeeze().numpy().astype(np.float16)
        gt = load_mask(mask_path, out_hw=prob.shape).astype(np.uint8)

        np.savez_compressed(out_path, prob=prob, gt=gt)

        if i % 100 == 0 or i == n:
            print(f"[CACHE] saved {i}/{n}: {out_split_dir}", flush=True)


def make_cache(
    args,
    model_name: str,
    seed: int,
    ckpt_path: str,
    device: torch.device,
) -> None:
    print(f"\n[LOAD] model={model_name}, seed={seed}, ckpt={ckpt_path}", flush=True)
    model = load_model_from_checkpoint(model_name, seed, ckpt_path, device=device)
    model.eval()

    val_items = build_dataset_items(args.val_manifest)
    test_items = build_dataset_items(args.test_manifest)

    cache_root = Path(args.cache_root)
    base_dir = cache_root / model_name / f"seed{seed}"

    print(f"[CACHE] model={model_name}, seed={seed}, val items={len(val_items)}", flush=True)
    save_probability_cache_for_split(
        model=model,
        items=val_items,
        out_split_dir=base_dir / "val",
        device=device,
        use_tta=args.use_tta,
        overwrite=args.overwrite_cache,
    )

    print(f"[CACHE] model={model_name}, seed={seed}, test items={len(test_items)}", flush=True)
    save_probability_cache_for_split(
        model=model,
        items=test_items,
        out_split_dir=base_dir / "test",
        device=device,
        use_tta=args.use_tta,
        overwrite=args.overwrite_cache,
    )

    del model
    torch.cuda.empty_cache()


def percent(x: float) -> float:
    return 100.0 * x


def run_budget_sweep(args) -> None:
    thresholds = parse_float_grid(args.thresholds)
    min_areas = parse_int_list(args.min_areas)
    closing_ks = parse_int_list(args.closing_ks)
    fill_holes_options = [bool(int(x)) for x in args.fill_holes.split(",") if x.strip()]
    grid = build_param_grid(thresholds, min_areas, closing_ks, fill_holes_options)

    print(f"\n[SWEEP] grid size={len(grid)}", flush=True)
    print(f"[SWEEP] budgets={args.budgets}", flush=True)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    per_seed_rows: List[Dict] = []
    best_params_records: List[Dict] = []

    for model_name in args.models:
        for seed in args.seeds:
            val_dir = Path(args.cache_root) / model_name / f"seed{seed}" / "val"
            test_dir = Path(args.cache_root) / model_name / f"seed{seed}" / "test"

            val_files = sorted(val_dir.glob("*.npz"))
            test_files = sorted(test_dir.glob("*.npz"))

            if not val_files or not test_files:
                raise FileNotFoundError(
                    f"Missing cache for model={model_name}, seed={seed}: "
                    f"val={len(val_files)}, test={len(test_files)}"
                )

            print(
                f"[SWEEP] model={model_name}, seed={seed}, "
                f"val={len(val_files)}, test={len(test_files)}",
                flush=True,
            )

            for budget in args.budgets:
                best_params, val_result = calibrate_params(
                    val_files=val_files,
                    budget=budget,
                    grid=grid,
                    boundary_radius=args.boundary_radius,
                    select_metric=args.select_metric,
                    calib_seed=args.calib_seed,
                )

                test_result = evaluate_npz_files(
                    test_files,
                    params=best_params,
                    boundary_radius=args.boundary_radius,
                )

                row = {
                    "model": model_name,
                    "seed": seed,
                    "budget": budget,
                    "threshold": best_params.threshold,
                    "min_area": best_params.min_area,
                    "closing_ks": best_params.closing_ks,
                    "fill_holes": int(best_params.fill_holes),
                    "val_miou": "NA" if math.isnan(val_result.miou) else f"{percent(val_result.miou):.4f}",
                    "val_f1": "NA" if math.isnan(val_result.f1) else f"{percent(val_result.f1):.4f}",
                    "val_biou": "NA" if math.isnan(val_result.biou) else f"{percent(val_result.biou):.4f}",
                    "test_miou": f"{percent(test_result.miou):.4f}",
                    "test_f1": f"{percent(test_result.f1):.4f}",
                    "test_biou": f"{percent(test_result.biou):.4f}",
                    "n_val_used": 0 if budget == 0 else min(budget, len(val_files)),
                    "n_test": len(test_files),
                    "select_metric": args.select_metric,
                }
                per_seed_rows.append(row)

                best_params_records.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "budget": budget,
                        "params": asdict(best_params),
                        "test_metrics_percent": {
                            "miou": percent(test_result.miou),
                            "f1": percent(test_result.f1),
                            "biou": percent(test_result.biou),
                        },
                    }
                )

                print(
                    f"[DONE] {model_name} seed={seed} budget={budget} "
                    f"test mIoU={percent(test_result.miou):.2f}, "
                    f"F1={percent(test_result.f1):.2f}, "
                    f"BIoU={percent(test_result.biou):.2f}, "
                    f"params={best_params}",
                    flush=True,
                )

    write_csv(out_dir / "val_budget_per_seed.csv", per_seed_rows)

    summary_rows = summarize(per_seed_rows)
    write_csv(out_dir / "val_budget_summary.csv", summary_rows)

    gain_rows = summarize_gains(per_seed_rows, args.baseline_model, args.target_model)
    write_csv(out_dir / "val_budget_gain_summary.csv", gain_rows)

    with (out_dir / "best_params.json").open("w", encoding="utf-8") as f:
        json.dump(best_params_records, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] wrote {out_dir / 'val_budget_per_seed.csv'}", flush=True)
    print(f"[OK] wrote {out_dir / 'val_budget_summary.csv'}", flush=True)
    print(f"[OK] wrote {out_dir / 'val_budget_gain_summary.csv'}", flush=True)
    print(f"[OK] wrote {out_dir / 'best_params.json'}", flush=True)


def summarize(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, int], List[Dict]] = {}
    for r in rows:
        grouped.setdefault((r["model"], int(r["budget"])), []).append(r)

    out = []
    for (model, budget), rs in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        for metric in ["test_miou", "test_f1", "test_biou"]:
            vals = np.array([float(r[metric]) for r in rs], dtype=np.float64)
            out.append(
                {
                    "model": model,
                    "budget": budget,
                    "metric": metric.replace("test_", ""),
                    "mean_percent": f"{float(np.mean(vals)):.4f}",
                    "std_percent": f"{float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0:.4f}",
                    "n_seeds": len(vals),
                }
            )
    return out


def summarize_gains(rows: List[Dict], baseline_model: str, target_model: str) -> List[Dict]:
    by_key = {}
    for r in rows:
        by_key[(int(r["budget"]), int(r["seed"]), r["model"])] = r

    budgets = sorted({int(r["budget"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})

    out = []
    for budget in budgets:
        gain_records = []
        for seed in seeds:
            b = by_key.get((budget, seed, baseline_model))
            t = by_key.get((budget, seed, target_model))
            if b is None or t is None:
                continue
            gain_records.append(
                {
                    "gain_miou": float(t["test_miou"]) - float(b["test_miou"]),
                    "gain_f1": float(t["test_f1"]) - float(b["test_f1"]),
                    "gain_biou": float(t["test_biou"]) - float(b["test_biou"]),
                }
            )

        if not gain_records:
            continue

        for metric in ["gain_miou", "gain_f1", "gain_biou"]:
            vals = np.array([g[metric] for g in gain_records], dtype=np.float64)
            out.append(
                {
                    "budget": budget,
                    "gain_metric": metric,
                    "mean_gain_pp": f"{float(np.mean(vals)):.4f}",
                    "std_gain_pp": f"{float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0:.4f}",
                    "n_seeds": len(vals),
                }
            )
    return out


def parse_ckpt_map(ckpt_args: Sequence[str]) -> Dict[Tuple[str, int], str]:
    """
    Input format:
        lora:42:/path/a.pth hl:42:/path/b.pth
    """
    out: Dict[Tuple[str, int], str] = {}
    for item in ckpt_args:
        parts = item.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Bad --ckpt item: {item}. Expected model:seed:path, e.g. lora:42:/path/ckpt.pth"
            )
        model, seed_str, path = parts
        out[(model, int(seed_str))] = path
    return out


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--stage", choices=["cache", "sweep", "all"], default="all")

    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument("--test_manifest", type=str, required=True)

    parser.add_argument("--cache_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--models", nargs="+", default=["lora", "hl"])
    parser.add_argument("--baseline_model", type=str, default="lora")
    parser.add_argument("--target_model", type=str, default="hl")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument(
        "--ckpt",
        nargs="+",
        required=False,
        default=[],
        help="Format: model:seed:/path/to/checkpoint.pth",
    )

    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")

    parser.add_argument("--budgets", nargs="+", type=int, default=[0, 50, 100, 500])
    parser.add_argument("--thresholds", type=str, default="0.30:0.80:0.05")
    parser.add_argument("--min_areas", type=str, default="0,32,64,128,256")
    parser.add_argument("--closing_ks", type=str, default="0,3,5")
    parser.add_argument("--fill_holes", type=str, default="0,1")
    parser.add_argument("--boundary_radius", type=int, default=5)
    parser.add_argument(
        "--select_metric",
        choices=["miou", "f1", "biou", "miou_biou_mean"],
        default="miou",
    )
    parser.add_argument("--calib_seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}", flush=True)

    if args.stage in ["cache", "all"]:
        ckpt_map = parse_ckpt_map(args.ckpt)
        for model_name in args.models:
            for seed in args.seeds:
                key = (model_name, seed)
                if key not in ckpt_map:
                    raise KeyError(
                        f"Missing checkpoint for model={model_name}, seed={seed}. "
                        f"Use --ckpt {model_name}:{seed}:/path/to/ckpt.pth"
                    )
                make_cache(args, model_name, seed, ckpt_map[key], device=device)

    if args.stage in ["sweep", "all"]:
        run_budget_sweep(args)


if __name__ == "__main__":
    main()