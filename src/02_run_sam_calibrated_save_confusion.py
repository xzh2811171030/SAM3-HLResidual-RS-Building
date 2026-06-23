# -*- coding: utf-8 -*-
"""
run_calibrated_fulltest.py
=============================================================================
目的：
  不再训练新模型，只对已有 checkpoint 做统一的后处理校准与最终测试评估。
  支持任意 test manifest（pilot 500 / final 8402）。

比较对象：
  1. lora_light
  2. hl_lite
  3. hl_residual_old
  4. hl_residual_v2

搜索：
  threshold
  min component area
  closing kernel
  hole filling
  optional TTA

输出：
  results/calibrated_fulltest/calibrated_fulltest_summary.json

推荐运行（pilot 500）：
  cd <project_root>

  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python src/run_calibrated_fulltest.py \
    --models lora_light,hl_lite,hl_residual_old,hl_residual_v2 \
    --use_tta \
    --val_eval_limit 500 \
    --test_manifest_name target_pilot_test_500.txt \
    --test_eval_limit 500

推荐运行（final 8402）：
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python src/run_calibrated_fulltest.py \
    --models lora_light,hl_lite,hl_residual_old,hl_residual_v2 \
    --use_tta \
    --val_eval_limit 500 \
    --test_manifest_name target_final_test_8402.txt
=============================================================================
"""

import argparse
import csv
import gc
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from tqdm import tqdm


SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SAM3_SRC = SRC / "models" / "sam3"
if str(SAM3_SRC) not in sys.path:
    sys.path.insert(0, str(SAM3_SRC))

from sam3.model_builder import build_sam3_image_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_SIZE = 512
SAM3_INPUT_SIZE = 1008
FEATURE_SIZE = (64, 64)
FEATURE_CHANNELS = 256

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

LABEL_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


# =============================================================================
# Dataset
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def read_manifest(path: Path, limit: Optional[int] = None) -> List[Path]:
    if not path.exists():
        raise FileNotFoundError(path)

    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().replace("\\", "/")
        if line:
            lines.append(Path(line))

    if limit is not None and limit > 0:
        lines = lines[:limit]

    return lines


def find_dual_label_for_image(image_path: Path) -> Optional[Path]:
    stem = image_path.stem
    s = str(image_path).replace("\\", "/")

    candidates = []

    if "/images/" in s:
        candidates.append(Path(s.replace("/images/", "/dual_channel_labels/")).parent)

    if "/test/image/" in s:
        candidates.append(Path(s.replace("/test/image/", "/dual_channel_labels/")).parent)

    candidates.append(image_path.parent.parent / "dual_channel_labels")
    candidates.append(image_path.parent.parent / "label")
    candidates.append(image_path.parent.parent / "labels")

    seen = set()
    uniq = []

    for d in candidates:
        key = str(d).replace("\\", "/").lower()
        if key not in seen:
            seen.add(key)
            uniq.append(d)

    for d in uniq:
        if not d.exists():
            continue
        for ext in LABEL_EXTS:
            p = d / f"{stem}{ext}"
            if p.exists():
                return p

    return None


class ManifestDataset(Dataset):
    def __init__(self, manifest_path: Path, limit: Optional[int] = None):
        self.paths = read_manifest(manifest_path, limit=limit)
        if not self.paths:
            raise RuntimeError(f"empty manifest: {manifest_path}")
        print(f"  Dataset {manifest_path.name}: {len(self.paths)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        lab_path = find_dual_label_for_image(img_path)
        if lab_path is None:
            raise FileNotFoundError(f"missing label for {img_path}")

        lab = cv2.imread(str(lab_path), cv2.IMREAD_UNCHANGED)
        if lab is None:
            raise FileNotFoundError(lab_path)

        if lab.ndim == 2:
            lab = cv2.resize(lab, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)
            mask = (lab > 0).astype(np.float32)
        else:
            lab = cv2.resize(lab, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)
            mask = (lab[:, :, 0] > 0).astype(np.float32)

        return {
            "image": img_t,
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
            "name": img_path.stem,
        }


def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "name": [b["name"] for b in batch],
    }


# =============================================================================
# Metrics
# =============================================================================

def compute_iou(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + 1e-7) / (union + 1e-7))


def compute_f1(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return float((2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7))


def boundary_map(mask, d=5):
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask.astype(bool)

    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    b = mask - eroded

    if d > 1:
        b = cv2.dilate(b, np.ones((d, d), np.uint8), iterations=1)

    return b.astype(bool)


def compute_biou(pred, gt, d=5):
    pb = boundary_map(pred, d=d)
    gb = boundary_map(gt, d=d)
    inter = np.logical_and(pb, gb).sum()
    union = np.logical_or(pb, gb).sum()
    return float((inter + 1e-7) / (union + 1e-7))


def eval_arrays(preds, gts):
    ious, f1s, bious = [], [], []
    for p, g in zip(preds, gts):
        ious.append(compute_iou(p, g))
        f1s.append(compute_f1(p, g))
        bious.append(compute_biou(p, g, d=5))
    return {
        "mIoU": float(np.mean(ious)),
        "F1": float(np.mean(f1s)),
        "Boundary_IoU": float(np.mean(bious)),
    }


# =============================================================================
# Postprocess
# =============================================================================

def fill_holes(binary: np.ndarray) -> np.ndarray:
    binary = (binary > 0).astype(np.uint8)
    h, w = binary.shape

    flood = binary.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(flood, mask, (0, 0), 1)
    flood_inv = 1 - flood
    filled = np.logical_or(binary, flood_inv).astype(np.uint8)

    return filled


def postprocess_mask(
    prob: np.ndarray,
    threshold: float,
    min_area: int,
    closing_kernel: int,
    fill_hole: bool,
) -> np.ndarray:
    pred = (prob > threshold).astype(np.uint8)

    if closing_kernel and closing_kernel > 0:
        k = np.ones((closing_kernel, closing_kernel), np.uint8)
        pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, k, iterations=1)

    if fill_hole:
        pred = fill_holes(pred)

    if min_area and min_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
        out = np.zeros_like(pred)

        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                out[labels == i] = 1
        pred = out

    return pred.astype(np.float32)


def grid_search_postprocess(probs, gts, grid_mode="full"):
    if grid_mode == "fast":
        thresholds = [0.50, 0.55, 0.60, 0.65]
        min_areas = [0, 64, 128]
        closing_kernels = [0, 3]
        fill_holes = [False, True]
    else:
        thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        min_areas = [0, 16, 32, 64, 128]
        closing_kernels = [0, 3, 5]
        fill_holes = [False, True]

    total = len(thresholds) * len(min_areas) * len(closing_kernels) * len(fill_holes)
    print(f"[GridSearch] mode={grid_mode}, configs={total}, N={len(probs)}", flush=True)

    best = None
    rows = []
    idx = 0

    for th in thresholds:
        for area in min_areas:
            for ck in closing_kernels:
                for fh in fill_holes:
                    idx += 1
                    if idx == 1 or idx % 10 == 0 or idx == total:
                        print(
                            f"[GridSearch] {idx}/{total}: "
                            f"th={th}, area={area}, closing={ck}, fill={fh}",
                            flush=True,
                        )

                    preds = [
                        postprocess_mask(p, th, area, ck, fh)
                        for p in probs
                    ]
                    preds = np.stack(preds)
                    m = eval_arrays(preds, gts)

                    row = {
                        "threshold": th,
                        "min_area": area,
                        "closing_kernel": ck,
                        "fill_holes": fh,
                        **m,
                    }
                    rows.append(row)

                    if best is None or row["mIoU"] > best["mIoU"]:
                        best = row

    print(f"[GridSearch] best={best}", flush=True)

    return {
        "best": best,
        "all": rows,
    }


def eval_with_config(probs, gts, cfg):
    preds = [
        postprocess_mask(
            p,
            threshold=cfg["threshold"],
            min_area=cfg["min_area"],
            closing_kernel=cfg["closing_kernel"],
            fill_hole=cfg["fill_holes"],
        )
        for p in probs
    ]
    preds = np.stack(preds)
    return eval_arrays(preds, gts)




# =============================================================================
# Saving calibrated outputs and coverage diagnostics
# =============================================================================

def canonical_stem(x) -> str:
    """Convert a path or image id into a conservative matching stem."""
    s = str(x).strip().replace("\\", "/")
    p = Path(s)
    stem = p.stem if p.suffix else p.name
    for suf in ["_mask", "_label", "_labels", "_pred", "_prediction"]:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    return stem


def read_manifest_stems_for_filter(path: Optional[Path]) -> set:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    stems = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().replace("\\", "/")
        if not line or line.startswith("#"):
            continue
        stems.add(canonical_stem(line))
    return stems


def default_method_alias(model_name: str) -> str:
    aliases = {
        "lora_light": "Prompt-free LoRA",
        "hl_lite": "HL-Lite",
        "hl_residual_old": "HL-Residual-old",
        "hl_residual_v2": "HL-Residual",
    }
    return aliases.get(model_name, model_name)


def confusion_counts(pred: np.ndarray, gt: np.ndarray) -> Dict[str, int]:
    pred_b = (pred > 0).astype(bool)
    gt_b = (gt > 0).astype(bool)
    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    fn = int(np.logical_and(~pred_b, gt_b).sum())
    pred_area = int(pred_b.sum())
    gt_area = int(gt_b.sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_area": pred_area,
        "gt_area": gt_area,
    }


def save_calibrated_outputs(
    probs: np.ndarray,
    gts: np.ndarray,
    names: List[str],
    cfg: Dict,
    model_name: str,
    args,
    out_dir: Path,
) -> Optional[Path]:
    """Save final calibrated masks and/or per-image TP/FP/FN CSV.

    The saved mask is exactly the postprocessed binary prediction generated from
    validation-selected calibration parameters. This function is called only for
    the test manifest, not for the validation set.
    """
    if not args.save_masks and not args.save_per_image_confusion:
        return None

    if len(names) != len(probs):
        raise RuntimeError(f"name/prob length mismatch: {len(names)} vs {len(probs)}")

    method_alias = default_method_alias(model_name)
    seed_tag = f"seed{args.run_seed}"

    mask_dir = None
    if args.save_masks:
        mask_root = Path(args.save_mask_root) if args.save_mask_root else out_dir / "saved_masks"
        mask_dir = mask_root / model_name / seed_tag
        mask_dir.mkdir(parents=True, exist_ok=True)

    csv_path = None
    writer = None
    csv_f = None
    if args.save_per_image_confusion:
        confusion_dir = Path(args.confusion_dir) if args.confusion_dir else out_dir / "per_image_confusion"
        confusion_dir.mkdir(parents=True, exist_ok=True)
        csv_path = confusion_dir / f"{model_name}_{seed_tag}_calibrated_confusion.csv"
        csv_f = csv_path.open("w", newline="", encoding="utf-8")
        fieldnames = [
            "image_id", "method", "model_name", "seed",
            "tp", "fp", "fn", "pred_area", "gt_area",
            "iou", "f1",
        ]
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()

    for prob, gt, name in tqdm(
        list(zip(probs, gts, names)),
        desc=f"  save outputs [{model_name}]",
        ncols=100,
    ):
        pred = postprocess_mask(
            prob,
            threshold=cfg["threshold"],
            min_area=cfg["min_area"],
            closing_kernel=cfg["closing_kernel"],
            fill_hole=cfg["fill_holes"],
        ).astype(np.uint8)

        gt_u8 = (gt > 0).astype(np.uint8)
        stem = canonical_stem(name)

        if mask_dir is not None:
            # Store as 0/255 PNG for visual inspection and downstream tools.
            cv2.imwrite(str(mask_dir / f"{stem}.png"), pred * 255)

        if writer is not None:
            c = confusion_counts(pred, gt_u8)
            union = c["tp"] + c["fp"] + c["fn"]
            iou = 1.0 if union == 0 else c["tp"] / union
            denom_f1 = 2 * c["tp"] + c["fp"] + c["fn"]
            f1 = 1.0 if denom_f1 == 0 else 2 * c["tp"] / denom_f1
            writer.writerow({
                "image_id": stem,
                "method": method_alias,
                "model_name": model_name,
                "seed": args.run_seed,
                **c,
                "iou": iou,
                "f1": f1,
            })

    if csv_f is not None:
        csv_f.close()

    return csv_path


def _summarize_confusion_rows(rows: List[Dict]) -> Dict:
    tp = float(sum(float(r["tp"]) for r in rows))
    fp = float(sum(float(r["fp"]) for r in rows))
    fn = float(sum(float(r["fn"]) for r in rows))
    pred_area = float(sum(float(r["pred_area"]) for r in rows))
    gt_area = float(sum(float(r["gt_area"]) for r in rows))

    precision = 1.0 if (tp + fp) == 0 and gt_area == 0 else (tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    recall = 1.0 if gt_area == 0 and pred_area == 0 else (tp / gt_area if gt_area > 0 else 0.0)
    f1_micro = 1.0 if (2 * tp + fp + fn) == 0 else (2 * tp / (2 * tp + fp + fn))

    ious = []
    f1s = []
    for r in rows:
        rtp = float(r["tp"])
        rfp = float(r["fp"])
        rfn = float(r["fn"])
        union = rtp + rfp + rfn
        ious.append(1.0 if union == 0 else rtp / union)
        denom = 2 * rtp + rfp + rfn
        f1s.append(1.0 if denom == 0 else 2 * rtp / denom)

    return {
        "n_images": len(rows),
        "mIoU_image_avg": 100.0 * float(np.mean(ious)) if ious else float("nan"),
        "F1_image_avg": 100.0 * float(np.mean(f1s)) if f1s else float("nan"),
        "Precision_micro": 100.0 * precision,
        "Recall_micro": 100.0 * recall,
        "F1_micro": 100.0 * f1_micro,
        "Pred_GT_area_ratio": pred_area / gt_area if gt_area > 0 else float("nan"),
        "FP_GT_ratio": fp / gt_area if gt_area > 0 else float("nan"),
        "FN_GT_ratio": fn / gt_area if gt_area > 0 else float("nan"),
        "TP_pixels": tp,
        "FP_pixels": fp,
        "FN_pixels": fn,
        "GT_pixels": gt_area,
        "Pred_pixels": pred_area,
    }


def _fmt_mean_std(vals: List[float], digits: int = 2) -> str:
    vals = [float(v) for v in vals if np.isfinite(float(v))]
    if not vals:
        return "N/A"
    if len(vals) == 1:
        return f"${vals[0]:.{digits}f}\\pm0.00$"
    arr = np.asarray(vals, dtype=float)
    return f"${arr.mean():.{digits}f}\\pm{arr.std(ddof=1):.{digits}f}$"


def summarize_coverage_from_csvs(
    csv_paths: List[Path],
    eval_manifest: Optional[Path],
    exclude_manifest: Optional[Path],
    out_dir: Path,
) -> Dict:
    """Summarize saved per-image confusion CSVs into held-out non-pilot coverage table."""
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_stems = read_manifest_stems_for_filter(eval_manifest) if eval_manifest else set()
    exclude_stems = read_manifest_stems_for_filter(exclude_manifest) if exclude_manifest else set()
    keep_stems = eval_stems - exclude_stems if eval_stems else set()

    all_rows = []
    for p in csv_paths:
        if p is None:
            continue
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"confusion csv not found: {p}")

        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"image_id", "method", "seed", "tp", "fp", "fn", "pred_area", "gt_area"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{p} missing required columns: {sorted(missing)}")

            before = 0
            after = 0
            for row in reader:
                before += 1
                stem = canonical_stem(row["image_id"])
                if keep_stems and stem not in keep_stems:
                    continue
                if not keep_stems and exclude_stems and stem in exclude_stems:
                    continue
                row["image_id"] = stem
                all_rows.append(row)
                after += 1

        print(f"[Coverage] {p.name}: kept {after}/{before} rows after manifest filtering")

    if not all_rows:
        raise RuntimeError("No rows available for coverage summarization.")

    # Save filtered per-image rows
    filtered_csv = out_dir / "per_image_confusion_filtered.csv"
    with filtered_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(all_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Per method/seed summary
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for r in all_rows:
        groups.setdefault((str(r["method"]), str(r["seed"])), []).append(r)

    seed_rows = []
    for (method, seed), rows in sorted(groups.items()):
        s = _summarize_confusion_rows(rows)
        s.update({"method": method, "seed": seed})
        seed_rows.append(s)

    per_seed_csv = out_dir / "per_seed_coverage_summary.csv"
    with per_seed_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "seed"] + [k for k in seed_rows[0].keys() if k not in {"method", "seed"}]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(seed_rows)

    # Method mean/std across seeds
    method_groups: Dict[str, List[Dict]] = {}
    for r in seed_rows:
        method_groups.setdefault(str(r["method"]), []).append(r)

    method_rows = []
    for method, rows in sorted(method_groups.items()):
        method_rows.append({
            "method": method,
            "Precision_micro": _fmt_mean_std([r["Precision_micro"] for r in rows], 2),
            "Recall_micro": _fmt_mean_std([r["Recall_micro"] for r in rows], 2),
            "F1_micro": _fmt_mean_std([r["F1_micro"] for r in rows], 2),
            "Pred_GT_area_ratio": _fmt_mean_std([r["Pred_GT_area_ratio"] for r in rows], 3),
            "FP_GT_ratio": _fmt_mean_std([r["FP_GT_ratio"] for r in rows], 3),
            "FN_GT_ratio": _fmt_mean_std([r["FN_GT_ratio"] for r in rows], 3),
            "mIoU_image_avg": _fmt_mean_std([r["mIoU_image_avg"] for r in rows], 2),
            "F1_image_avg": _fmt_mean_std([r["F1_image_avg"] for r in rows], 2),
        })

    method_csv = out_dir / "method_coverage_summary.csv"
    with method_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "method", "Precision_micro", "Recall_micro", "F1_micro",
            "Pred_GT_area_ratio", "FP_GT_ratio", "FN_GT_ratio",
            "mIoU_image_avg", "F1_image_avg",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(method_rows)

    tex_lines = [
        r"\begin{table*}[!t]",
        r"\caption{Coverage-oriented diagnostic on the held-out non-pilot WHU-Mix subset. Precision and recall are micro-averaged over pixels; mIoU and F1 follow the image-averaged foreground-building definitions used in the manuscript.}",
        r"\label{tab:supp_coverage_diagnostic}",
        r"\centering",
        r"\footnotesize",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{l c c c c c c c}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Prec.} & \textbf{Recall} & \textbf{Pred./GT area} & \textbf{FP/GT} & \textbf{FN/GT} & \textbf{mIoU} & \textbf{F1} \\",
        r"\midrule",
    ]
    for r in method_rows:
        tex_lines.append(
            f"{r['method']} & {r['Precision_micro']} & {r['Recall_micro']} & "
            f"{r['Pred_GT_area_ratio']} & {r['FP_GT_ratio']} & {r['FN_GT_ratio']} & "
            f"{r['mIoU_image_avg']} & {r['F1_image_avg']} \\\\"
        )
    tex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table*}",
    ]
    coverage_tex = out_dir / "coverage_table.tex"
    coverage_tex.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    print(f"[Coverage] Saved filtered rows: {filtered_csv}")
    print(f"[Coverage] Saved per-seed summary: {per_seed_csv}")
    print(f"[Coverage] Saved method summary: {method_csv}")
    print(f"[Coverage] Saved LaTeX table: {coverage_tex}")
    if keep_stems:
        print(f"[Coverage] Expected held-out non-pilot image count per method/seed: {len(keep_stems)}")

    return {
        "filtered_csv": str(filtered_csv),
        "per_seed_csv": str(per_seed_csv),
        "method_csv": str(method_csv),
        "coverage_tex": str(coverage_tex),
        "n_rows": len(all_rows),
        "n_keep_stems": len(keep_stems) if keep_stems else None,
    }


# =============================================================================
# SAM3 + LoRA
# =============================================================================

def patch_vit_mlp(vit_peft_model):
    import types as _types
    n = 0

    for module in vit_peft_model.modules():
        if type(module).__name__ != "Mlp":
            continue

        def _safe_forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop1(x)
            x = self.norm(x)
            x = self.fc2(x)
            x = self.drop2(x)
            return x

        module.forward = _types.MethodType(_safe_forward, module)
        n += 1

    print(f"  patched MLP: {n}")


def inject_lora(model):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["qkv"],
        bias="none",
    )

    trunk = model.backbone.vision_backbone.trunk
    model.backbone.vision_backbone.trunk = get_peft_model(trunk, cfg)
    patch_vit_mlp(model.backbone.vision_backbone.trunk)

    for name, p in model.named_parameters():
        p.requires_grad = ("lora" in name.lower())


class SAM3Extractor:
    def __init__(self, checkpoint_path: Path):
        self.model = build_sam3_image_model(checkpoint_path=str(checkpoint_path))
        self.model.to(DEVICE)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        inject_lora(self.model)

        self.transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(SAM3_INPUT_SIZE, SAM3_INPUT_SIZE)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _prep(self, img):
        img_u8 = (img * 255).clamp(0, 255).to(torch.uint8)
        return self.transform(img_u8).unsqueeze(0).to(DEVICE)

    @torch.no_grad()
    def extract_batch(self, imgs):
        feats = []

        for i in range(imgs.shape[0]):
            x = self._prep(imgs[i])
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                out = self.model.backbone.forward_image(x)

            feat = out["vision_features"].float()
            if feat.shape[-2:] != FEATURE_SIZE:
                feat = F.interpolate(feat, size=FEATURE_SIZE, mode="bilinear", align_corners=False)

            feats.append(feat.detach())

        return torch.cat(feats, dim=0)


class LightweightMaskDecoder(nn.Module):
    def __init__(self, feat_channels=FEATURE_CHANNELS):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feat_channels, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, feat, rgb=None):
        return self.decoder(feat)


class OCLHighResPath(nn.Module):
    def __init__(self, out_ch: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(64, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = self.stem(rgb)
        x = self.down2(x)
        x = self.down4(x)
        return x


class HLCrossFusionLite(nn.Module):
    def __init__(self, low_ch: int = 256, high_ch: int = 128):
        super().__init__()
        self.high_to_low = nn.Sequential(
            nn.Conv2d(high_ch, low_ch, kernel_size=1),
            nn.BatchNorm2d(low_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(low_ch, low_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(low_ch),
            nn.ReLU(inplace=True),
        )
        self.low_gate = nn.Sequential(
            nn.Conv2d(low_ch * 2, low_ch, kernel_size=1),
            nn.Sigmoid(),
        )
        self.low_to_high = nn.Sequential(
            nn.Conv2d(low_ch, high_ch, kernel_size=1),
            nn.BatchNorm2d(high_ch),
            nn.ReLU(inplace=True),
        )
        self.high_gate = nn.Sequential(
            nn.Conv2d(high_ch * 2, high_ch, kernel_size=1),
            nn.Sigmoid(),
        )
        self.alpha_low = nn.Parameter(torch.tensor(0.0))
        self.alpha_high = nn.Parameter(torch.tensor(0.0))

    def forward(self, low: torch.Tensor, high: torch.Tensor):
        high_down = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        high_low_msg = self.high_to_low(high_down)
        low_gate = self.low_gate(torch.cat([low, high_low_msg], dim=1))
        low_refined = low + torch.tanh(self.alpha_low) * low_gate * high_low_msg

        low_up = F.interpolate(low_refined, size=high.shape[-2:], mode="bilinear", align_corners=False)
        low_high_msg = self.low_to_high(low_up)
        high_gate = self.high_gate(torch.cat([high, low_high_msg], dim=1))
        high_refined = high + torch.tanh(self.alpha_high) * high_gate * low_high_msg

        return low_refined, high_refined


class HLLiteDecoder(nn.Module):
    def __init__(self, low_ch: int = 256, high_ch: int = 128):
        super().__init__()
        self.high_path = OCLHighResPath(out_ch=high_ch)
        self.hl_fusion = HLCrossFusionLite(low_ch=low_ch, high_ch=high_ch)
        self.low_up128 = nn.Sequential(
            nn.ConvTranspose2d(low_ch, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.fuse128 = nn.Sequential(
            nn.Conv2d(128 + high_ch, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up256 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up512 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, feat, rgb):
        high = self.high_path(rgb)
        low_refined, high_refined = self.hl_fusion(feat, high)
        low_128 = self.low_up128(low_refined)
        x = self.fuse128(torch.cat([low_128, high_refined], dim=1))
        x = self.up256(x)
        x = self.up512(x)
        return self.out(x)


class HLResidualBranch(nn.Module):
    def __init__(self, low_ch=256):
        super().__init__()

        self.high_path = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.low_to_128 = nn.Sequential(
            nn.ConvTranspose2d(low_ch, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up256 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.up512 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        self.res_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, feat, rgb):
        high = self.high_path(rgb)
        low = self.low_to_128(feat)
        x = self.fuse(torch.cat([low, high], dim=1))
        x = self.up256(x)
        x = self.up512(x)
        res = self.out(x)
        return torch.tanh(self.res_scale) * res


class HLResidualRefiner(nn.Module):
    def __init__(self, base_decoder):
        super().__init__()
        self.base_decoder = base_decoder
        self.residual = HLResidualBranch()

        for p in self.base_decoder.parameters():
            p.requires_grad = False

    def forward(self, feat, rgb):
        with torch.no_grad():
            base_logits = self.base_decoder(feat, rgb)
        residual = self.residual(feat, rgb)
        return base_logits + residual


def _load_lora(extractor, lora_params: dict, model_name: str) -> None:
    state = extractor.model.state_dict()
    loaded = 0
    for k, v in lora_params.items():
        if k in state:
            state[k].copy_(v.to(DEVICE))
            loaded += 1
    print(f"  {model_name}: loaded LoRA params {loaded}")


def load_model(model_name, sam3_checkpoint, ckpt_path):
    extractor = SAM3Extractor(sam3_checkpoint)

    if model_name == "hl_lite":
        decoder = HLLiteDecoder().to(DEVICE)
    else:
        decoder = LightweightMaskDecoder().to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # --- hl_residual_old 兼容: checkpoint 只有 residual，decoder 在 base_lora_ckpt 里 ---
    if model_name.startswith("hl_residual") and "model" not in ckpt and "student" not in ckpt:
        base_lora_path = ckpt.get("base_lora_ckpt", None)
        if base_lora_path and Path(base_lora_path).exists():
            base_ckpt = torch.load(base_lora_path, map_location=DEVICE, weights_only=False)
            if "model" in base_ckpt:
                decoder.load_state_dict(base_ckpt["model"], strict=True)
            if "lora_params" in base_ckpt:
                _load_lora(extractor, base_ckpt["lora_params"], model_name)
            del base_ckpt
        else:
            # fallback: 尝试从同目录的 lora_light best 读取
            base_dir = Path(ckpt_path).parent
            lora_light_path = base_dir / "e6v2_pilot_lora_light_best.pth"
            if not lora_light_path.exists():
                lora_light_path = base_dir / "lora_light_best.pth"
            if lora_light_path.exists():
                base_ckpt = torch.load(lora_light_path, map_location=DEVICE, weights_only=False)
                if "model" in base_ckpt:
                    decoder.load_state_dict(base_ckpt["model"], strict=True)
                if "lora_params" in base_ckpt:
                    _load_lora(extractor, base_ckpt["lora_params"], model_name)
                del base_ckpt
            else:
                raise FileNotFoundError(
                    f"hl_residual checkpoint 缺少 decoder 权重，且找不到 base_lora_ckpt: {base_lora_path}"
                )

        model = HLResidualRefiner(decoder).to(DEVICE)
        model.residual.load_state_dict(ckpt["residual"], strict=True)
        model.eval()
        extractor.model.eval()
        print(f"  {model_name}: loaded residual from old-format checkpoint")
        return extractor, model

    # --- 正常路径: model/student key ---
    if "model" in ckpt:
        decoder.load_state_dict(ckpt["model"], strict=True)
    elif "student" in ckpt:
        decoder.load_state_dict(ckpt["student"], strict=True)
    else:
        raise KeyError(f"checkpoint 缺少 model/student: {ckpt_path}")

    if "lora_params" in ckpt:
        _load_lora(extractor, ckpt["lora_params"], model_name)

    if model_name.startswith("hl_residual"):
        model = HLResidualRefiner(decoder).to(DEVICE)

        if "residual" in ckpt:
            model.residual.load_state_dict(ckpt["residual"], strict=True)
        else:
            raise KeyError(f"{model_name} checkpoint 缺少 residual: {ckpt_path}")

    else:
        model = decoder

    model.eval()
    extractor.model.eval()

    return extractor, model


@torch.no_grad()
def predict_probs(extractor, model, loader, use_tta=False):
    probs_all = []
    gts_all = []
    names_all = []

    for batch in tqdm(loader, desc="  predict", ncols=100):
        imgs = batch["image"]
        gts = batch["mask"].numpy()[:, 0]

        if not use_tta:
            feat = extractor.extract_batch(imgs)
            rgb = imgs.to(DEVICE)
            logits = model(feat.to(DEVICE), rgb)
            probs = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]

        else:
            probs_list = []

            # original
            feat = extractor.extract_batch(imgs)
            rgb = imgs.to(DEVICE)
            logits = model(feat.to(DEVICE), rgb)
            probs_list.append(torch.sigmoid(logits.float()).cpu())

            # horizontal flip
            imgs_h = torch.flip(imgs, dims=[3])
            feat_h = extractor.extract_batch(imgs_h)
            logits_h = model(feat_h.to(DEVICE), imgs_h.to(DEVICE))
            prob_h = torch.sigmoid(logits_h.float()).cpu()
            prob_h = torch.flip(prob_h, dims=[3])
            probs_list.append(prob_h)

            # vertical flip
            imgs_v = torch.flip(imgs, dims=[2])
            feat_v = extractor.extract_batch(imgs_v)
            logits_v = model(feat_v.to(DEVICE), imgs_v.to(DEVICE))
            prob_v = torch.sigmoid(logits_v.float()).cpu()
            prob_v = torch.flip(prob_v, dims=[2])
            probs_list.append(prob_v)

            probs = torch.mean(torch.stack(probs_list, dim=0), dim=0).numpy()[:, 0]

        probs_all.append(probs)
        gts_all.append(gts)
        names_all.extend(batch["name"])

    return np.concatenate(probs_all), np.concatenate(gts_all), names_all


# =============================================================================
# CLI
# =============================================================================

def make_loader(ds, batch_size, shuffle, num_workers, collate_fn):
    kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs)


def parse_args():
    parser = argparse.ArgumentParser("Calibrated inference full test")

    parser.add_argument("--project_root", type=str, default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--manifest_dir", type=str, default=None)
    parser.add_argument("--sam3_checkpoint", type=str, default=None)

    parser.add_argument("--models", type=str, default="lora_light,hl_residual_old,hl_residual_v2")
    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--val_eval_limit", type=int, default=500)
    parser.add_argument("--pilot_eval_limit", type=int, default=500)

    parser.add_argument(
        "--test_manifest_name",
        type=str,
        default="target_pilot_test_500.txt",
        help="用于最终评估的 manifest 文件名，例如 target_pilot_test_500.txt 或 target_final_test_8402.txt",
    )

    parser.add_argument(
        "--test_eval_limit",
        type=int,
        default=None,
        help="测试集限制数量。None 或 <=0 表示全量。",
    )

    parser.add_argument("--lora_light_ckpt", type=str, default=None)
    parser.add_argument("--hl_lite_ckpt", type=str, default=None)
    parser.add_argument("--hl_residual_old_ckpt", type=str, default=None)
    parser.add_argument("--hl_residual_v2_ckpt", type=str, default=None)

    parser.add_argument(
        "--grid_mode",
        type=str,
        default="full",
        choices=["fast", "full"],
        help="后处理网格搜索范围。pilot/debug 用 fast，正式结果用 full。",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader worker 数。若阶段结束卡顿严重，可设为 0。",
    )

    parser.add_argument(
        "--disable_empty_cache",
        action="store_true",
        help="禁用每个模型结束后的 torch.cuda.empty_cache()，可减少阶段间卡顿。",
    )

    parser.add_argument("--out_dir", type=str, default=None)

    parser.add_argument(
        "--run_seed",
        type=int,
        default=42,
        help="Seed tag written into per-image confusion CSV. Use the actual checkpoint seed.",
    )
    parser.add_argument(
        "--save_masks",
        action="store_true",
        help="Save final validation-calibrated, post-processed binary masks for the test manifest.",
    )
    parser.add_argument(
        "--save_mask_root",
        type=str,
        default=None,
        help="Root directory for saved masks. Default: out_dir/saved_masks.",
    )
    parser.add_argument(
        "--save_per_image_confusion",
        action="store_true",
        help="Save per-image TP/FP/FN/pred_area/gt_area CSV for the final calibrated test predictions.",
    )
    parser.add_argument(
        "--confusion_dir",
        type=str,
        default=None,
        help="Directory for per-image confusion CSV files. Default: out_dir/per_image_confusion.",
    )
    parser.add_argument(
        "--make_coverage_table",
        action="store_true",
        help="After inference, summarize generated confusion CSVs into a held-out non-pilot coverage table.",
    )
    parser.add_argument(
        "--eval_manifest_name",
        type=str,
        default="target_final_test_8402.txt",
        help="Evaluation-pool manifest used for coverage filtering. Default: target_final_test_8402.txt.",
    )
    parser.add_argument(
        "--exclude_manifest_name",
        type=str,
        default="target_pilot_test_500.txt",
        help="Manifest excluded from coverage summary. Default: target_pilot_test_500.txt.",
    )
    parser.add_argument(
        "--coverage_out_dir",
        type=str,
        default=None,
        help="Output directory for coverage diagnostic table. Default: out_dir/coverage_nonpilot.",
    )
    parser.add_argument(
        "--extra_confusion_csv",
        type=str,
        nargs="*",
        default=[],
        help="Optional existing per-image confusion CSVs to merge into the coverage table, e.g., UNetFormer-style.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.run_seed)

    if args.make_coverage_table and not args.save_per_image_confusion:
        print("[Info] --make_coverage_table requires per-image confusion. Enabling --save_per_image_confusion.")
        args.save_per_image_confusion = True

    project_root = Path(args.project_root)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else project_root / "data" / "splits" / "e0_manifest"
    sam3_checkpoint = Path(args.sam3_checkpoint) if args.sam3_checkpoint else project_root / "weights" / "sam3.pt"
    out_dir = Path(args.out_dir) if args.out_dir else project_root / "results" / "calibrated_fulltest"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_map = {
        "lora_light": Path(args.lora_light_ckpt) if args.lora_light_ckpt else project_root / "results/e6v2_pilot/weights/e6v2_pilot_lora_light_best.pth",
        "hl_lite": Path(args.hl_lite_ckpt) if args.hl_lite_ckpt else project_root / "results/e6v2_pilot/weights/e6v2_pilot_hl_lite_best.pth",
        "hl_residual_old": Path(args.hl_residual_old_ckpt) if args.hl_residual_old_ckpt else project_root / "results/hl_residual_pilot/weights/hl_residual_best.pth",
        "hl_residual_v2": Path(args.hl_residual_v2_ckpt) if args.hl_residual_v2_ckpt else project_root / "results/hl_residual_v2_erase015/weights/hl_residual_best.pth",
    }

    val_manifest = manifest_dir / "target_val.txt"
    test_manifest = manifest_dir / args.test_manifest_name

    val_ds = ManifestDataset(val_manifest, limit=args.val_eval_limit)

    test_limit = args.test_eval_limit
    if test_limit is not None and test_limit <= 0:
        test_limit = None

    test_ds = ManifestDataset(test_manifest, limit=test_limit)

    val_loader = make_loader(
        val_ds,
        batch_size=4,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    test_loader = make_loader(
        test_ds,
        batch_size=4,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    results = []
    generated_confusion_csvs = []

    for model_name in models:
        ckpt_path = ckpt_map[model_name]
        if not ckpt_path.exists():
            print(f"[Skip] {model_name}: checkpoint 不存在 {ckpt_path}")
            continue

        print("\n" + "=" * 90)
        print(f"Evaluate model: {model_name}")
        print(f"Checkpoint: {ckpt_path}")
        print("=" * 90)

        extractor, model = load_model(model_name, sam3_checkpoint, ckpt_path)

        print("[Val] predict")
        val_probs, val_gts, val_names = predict_probs(extractor, model, val_loader, use_tta=args.use_tta)
        val_search = grid_search_postprocess(val_probs, val_gts, grid_mode=args.grid_mode)
        best_cfg = val_search["best"]

        print(f"  best val cfg: {best_cfg}")

        print(f"[Test] predict: {args.test_manifest_name}")
        test_probs, test_gts, test_names = predict_probs(extractor, model, test_loader, use_tta=args.use_tta)

        fixed = eval_with_config(
            test_probs,
            test_gts,
            {
                "threshold": 0.5,
                "min_area": 0,
                "closing_kernel": 0,
                "fill_holes": False,
            },
        )

        calib = eval_with_config(test_probs, test_gts, best_cfg)

        saved_confusion_csv = save_calibrated_outputs(
            probs=test_probs,
            gts=test_gts,
            names=test_names,
            cfg=best_cfg,
            model_name=model_name,
            args=args,
            out_dir=out_dir,
        )
        if saved_confusion_csv is not None:
            generated_confusion_csvs.append(saved_confusion_csv)

        row = {
            "model": model_name,
            "checkpoint": str(ckpt_path),
            "use_tta": args.use_tta,
            "val_best_cfg": best_cfg,
            "saved_confusion_csv": str(saved_confusion_csv) if saved_confusion_csv is not None else None,
            "test_manifest": args.test_manifest_name,
            "test_fixed_05": fixed,
            "test_calibrated_postprocess": calib,
            "pilot_fixed_05": fixed,
            "pilot_calibrated_postprocess": calib,
        }
        results.append(row)

        print(
            f"{model_name:<18} "
            f"fixed mIoU={fixed['mIoU']*100:.2f} F1={fixed['F1']*100:.2f} BIoU={fixed['Boundary_IoU']*100:.2f} | "
            f"calib-post mIoU={calib['mIoU']*100:.2f} F1={calib['F1']*100:.2f} BIoU={calib['Boundary_IoU']*100:.2f}"
        )

        del extractor, model
        gc.collect()
        if DEVICE == "cuda" and not args.disable_empty_cache:
            torch.cuda.empty_cache()

    summary = {
        "config": vars(args),
        "results": results,
    }

    out_path = out_dir / "calibrated_fulltest_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    coverage_summary = None
    if args.make_coverage_table:
        eval_manifest = manifest_dir / args.eval_manifest_name if args.eval_manifest_name else None
        exclude_manifest = manifest_dir / args.exclude_manifest_name if args.exclude_manifest_name else None
        coverage_out_dir = Path(args.coverage_out_dir) if args.coverage_out_dir else out_dir / "coverage_nonpilot"
        csvs_for_coverage = list(generated_confusion_csvs) + [Path(p) for p in args.extra_confusion_csv]
        if not csvs_for_coverage:
            raise RuntimeError("No confusion CSVs available for coverage summary.")
        coverage_summary = summarize_coverage_from_csvs(
            csv_paths=csvs_for_coverage,
            eval_manifest=eval_manifest,
            exclude_manifest=exclude_manifest,
            out_dir=coverage_out_dir,
        )
        summary["coverage_summary"] = coverage_summary
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)
    for r in results:
        f = r["test_fixed_05"]
        c = r["test_calibrated_postprocess"]
        print(
            f"{r['model']:<18} "
            f"fixed={f['mIoU']*100:.2f}/{f['F1']*100:.2f}/{f['Boundary_IoU']*100:.2f} | "
            f"calib-post={c['mIoU']*100:.2f}/{c['F1']*100:.2f}/{c['Boundary_IoU']*100:.2f}"
        )

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
