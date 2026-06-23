# -*- coding: utf-8 -*-
"""
run_target20_segformer_baseline.py
=============================================================================
Target 20-shot SegFormer-style fine-tuning baseline for WHU -> WHU-Mix building
extraction.

Purpose
-------
This script provides a stronger non-SAM baseline than Compact-ResUNet:
  1) source pre-train a compact SegFormer-style MiT encoder on source WHU;
  2) fine-tune on target 20-shot support sets for seeds 42/123/456;
  3) calibrate threshold/post-processing on target_val only;
  4) evaluate on target_final_test_8402 with optional TTA.

It intentionally does NOT use SAM/SAM3 features or prompts.

Recommended usage
-----------------
cd <project_root>
python src/run_target20_segformer_baseline.py \
  --seeds 42,123,456 \
  --source_epochs 30 \
  --target_epochs 120 \
  --batch_size 8 \
  --target_batch_size 4 \
  --use_tta \
  --val_eval_limit 200 \
  --final_val_eval_limit 500 \
  --test_eval_limit 0 \
  --out_dir results/target20_segformer_baseline

Notes
-----
- This is a self-contained SegFormer-style implementation to avoid dependence on
  MMSegmentation or Hugging Face downloads.
- The architecture is a compact MiT-B0-like encoder plus SegFormer decoder head.
- If you already have a trained source checkpoint, rerunning the script will reuse it
  unless --force_source_pretrain is specified.
=============================================================================
"""

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_SIZE = 512
LABEL_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def resolve_manifest(manifest_dir: Path, names: List[str]) -> Path:
    for name in names:
        p = manifest_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find manifest. Tried:\n" + "\n".join(str(manifest_dir / n) for n in names)
    )


def read_manifest(path: Path, limit: Optional[int] = None) -> List[Path]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip().replace("\\", "/")
        if s:
            lines.append(Path(s))
    if limit is not None and limit > 0:
        lines = lines[:limit]
    if not lines:
        raise RuntimeError(f"Empty manifest: {path}")
    return lines


def find_dual_label_for_image(image_path: Path) -> Optional[Path]:
    stem = image_path.stem
    s = str(image_path).replace("\\", "/")

    candidates = []
    if "/images/" in s:
        candidates.append(Path(s.replace("/images/", "/dual_channel_labels/")).parent)
        candidates.append(Path(s.replace("/images/", "/labels/")).parent)
        candidates.append(Path(s.replace("/images/", "/label/")).parent)
    if "/test/image/" in s:
        candidates.append(Path(s.replace("/test/image/", "/dual_channel_labels/")).parent)
        candidates.append(Path(s.replace("/test/image/", "/label/")).parent)
    if "/image/" in s:
        candidates.append(Path(s.replace("/image/", "/label/")).parent)
        candidates.append(Path(s.replace("/image/", "/labels/")).parent)

    candidates.append(image_path.parent.parent / "dual_channel_labels")
    candidates.append(image_path.parent.parent / "label")
    candidates.append(image_path.parent.parent / "labels")
    candidates.append(image_path.parent)

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


def infer_region(name: str) -> str:
    n = name.lower()
    for k in ["dunedin", "khartoum", "kitsap", "potsdam", "wuxi"]:
        if k in n:
            return k.capitalize()
    # fallback: filename prefix before first underscore
    return name.split("_")[0] if "_" in name else "Unknown"


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class ManifestSegDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        limit: Optional[int] = None,
        train_aug: bool = False,
        target_size: int = TARGET_SIZE,
    ):
        self.paths = read_manifest(manifest_path, limit=limit)
        self.train_aug = train_aug
        self.target_size = target_size
        print(f"[Dataset] {manifest_path.name}: N={len(self.paths)} train_aug={train_aug}", flush=True)

    def __len__(self):
        return len(self.paths)

    def _augment(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # simple segmentation-safe augmentations
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[::-1, :, :])
            mask = np.ascontiguousarray(mask[::-1, :])

        if random.random() < 0.35:
            # brightness/contrast jitter in numpy, mild to avoid damaging few-shot masks
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-12, 12)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        return img, mask

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.paths[idx]
        if not img_path.exists():
            raise FileNotFoundError(img_path)

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        lab_path = find_dual_label_for_image(img_path)
        if lab_path is None:
            raise FileNotFoundError(f"Cannot find label for {img_path}")
        lab = cv2.imread(str(lab_path), cv2.IMREAD_UNCHANGED)
        if lab is None:
            raise FileNotFoundError(lab_path)

        if lab.ndim == 2:
            lab = cv2.resize(lab, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
            mask = (lab > 0).astype(np.float32)
        else:
            lab = cv2.resize(lab, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
            mask = (lab[:, :, 0] > 0).astype(np.float32)

        if self.train_aug:
            img, mask = self._augment(img, mask)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # ImageNet-like normalization; useful for stable transformer optimization even from scratch/source pretrain.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std

        return {
            "image": img_t,
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
            "name": img_path.stem,
            "path": str(img_path),
        }


def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "name": [b["name"] for b in batch],
        "path": [b["path"] for b in batch],
    }


# -----------------------------------------------------------------------------
# SegFormer-style model
# -----------------------------------------------------------------------------

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int, stride: int):
        super().__init__()
        padding = patch_size // 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.norm(x_flat)
        x = x_flat.transpose(1, 2).reshape(b, c, h, w)
        return x, h, w


class DWConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MixFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DWConv(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, h, w)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_img = x.transpose(1, 2).reshape(b, c, h, w)
            x_sr = self.sr(x_img).reshape(b, c, -1).transpose(1, 2)
            x_sr = self.norm(x_sr)
        else:
            x_sr = x

        kv = self.kv(x_sr).reshape(b, -1, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        out = self.drop(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, sr_ratio: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x_img.shape
        x = x_img.flatten(2).transpose(1, 2)
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x), h, w)
        x_img = x.transpose(1, 2).reshape(b, c, h, w)
        return x_img


class TinyMixVisionTransformer(nn.Module):
    """MiT-B0-like encoder, compact enough for overnight baselines."""

    def __init__(
        self,
        in_chans: int = 3,
        embed_dims: Tuple[int, int, int, int] = (32, 64, 160, 256),
        depths: Tuple[int, int, int, int] = (2, 2, 2, 2),
        num_heads: Tuple[int, int, int, int] = (1, 2, 5, 8),
        sr_ratios: Tuple[int, int, int, int] = (8, 4, 2, 1),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch1 = OverlapPatchEmbed(in_chans, embed_dims[0], patch_size=7, stride=4)
        self.block1 = nn.ModuleList([
            TransformerBlock(embed_dims[0], num_heads[0], mlp_ratio, sr_ratios[0], dropout)
            for _ in range(depths[0])
        ])

        self.patch2 = OverlapPatchEmbed(embed_dims[0], embed_dims[1], patch_size=3, stride=2)
        self.block2 = nn.ModuleList([
            TransformerBlock(embed_dims[1], num_heads[1], mlp_ratio, sr_ratios[1], dropout)
            for _ in range(depths[1])
        ])

        self.patch3 = OverlapPatchEmbed(embed_dims[1], embed_dims[2], patch_size=3, stride=2)
        self.block3 = nn.ModuleList([
            TransformerBlock(embed_dims[2], num_heads[2], mlp_ratio, sr_ratios[2], dropout)
            for _ in range(depths[2])
        ])

        self.patch4 = OverlapPatchEmbed(embed_dims[2], embed_dims[3], patch_size=3, stride=2)
        self.block4 = nn.ModuleList([
            TransformerBlock(embed_dims[3], num_heads[3], mlp_ratio, sr_ratios[3], dropout)
            for _ in range(depths[3])
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []

        x, _, _ = self.patch1(x)
        for blk in self.block1:
            x = blk(x)
        outs.append(x)

        x, _, _ = self.patch2(x)
        for blk in self.block2:
            x = blk(x)
        outs.append(x)

        x, _, _ = self.patch3(x)
        for blk in self.block3:
            x = blk(x)
        outs.append(x)

        x, _, _ = self.patch4(x)
        for blk in self.block4:
            x = blk(x)
        outs.append(x)

        return outs


class SegFormerHead(nn.Module):
    def __init__(self, in_channels=(32, 64, 160, 256), embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, embed_dim, kernel_size=1) for c in in_channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(self, feats: List[torch.Tensor], out_size: Tuple[int, int]) -> torch.Tensor:
        target_hw = feats[0].shape[-2:]
        xs = []
        for f, p in zip(feats, self.proj):
            y = p(f)
            if y.shape[-2:] != target_hw:
                y = F.interpolate(y, size=target_hw, mode="bilinear", align_corners=False)
            xs.append(y)
        x = torch.cat(xs, dim=1)
        x = self.fuse(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class CompactSegFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyMixVisionTransformer()
        self.decode_head = SegFormerHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        logits = self.decode_head(feats, out_size=x.shape[-2:])
        return logits


# -----------------------------------------------------------------------------
# Loss and metrics
# -----------------------------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        b = probs.shape[0]
        probs = probs.reshape(b, -1)
        target = target.reshape(b, -1)
        inter = (probs * target).sum(dim=1)
        union = probs.sum(dim=1) + target.sum(dim=1)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return (1 - dice).mean()


def segmentation_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, masks)
    dice = DiceLoss()(logits, masks)
    return bce + dice


def compute_iou(pred, gt) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + 1e-7) / (union + 1e-7))


def compute_f1(pred, gt) -> float:
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


def compute_biou(pred, gt, d=5) -> float:
    pb = boundary_map(pred, d=d)
    gb = boundary_map(gt, d=d)
    inter = np.logical_and(pb, gb).sum()
    union = np.logical_or(pb, gb).sum()
    return float((inter + 1e-7) / (union + 1e-7))


def fill_holes(binary: np.ndarray) -> np.ndarray:
    binary = (binary > 0).astype(np.uint8)
    h, w = binary.shape
    flood = binary.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 1)
    flood_inv = 1 - flood
    filled = np.logical_or(binary, flood_inv).astype(np.uint8)
    return filled


def postprocess_mask(prob, threshold, min_area, closing_kernel, fill_hole) -> np.ndarray:
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
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 1
        pred = out

    return pred.astype(np.float32)


def update_metric_sums(sums: Dict, pred: np.ndarray, gt: np.ndarray):
    sums["mIoU"].append(compute_iou(pred, gt))
    sums["F1"].append(compute_f1(pred, gt))
    sums["Boundary_IoU"].append(compute_biou(pred, gt, d=5))


def summarize_metric_lists(sums: Dict) -> Dict:
    return {k: float(np.mean(v)) if len(v) else 0.0 for k, v in sums.items()}


# -----------------------------------------------------------------------------
# Prediction/evaluation
# -----------------------------------------------------------------------------

@torch.no_grad()
def predict_batch_probs(model: nn.Module, imgs: torch.Tensor, use_tta: bool = False) -> torch.Tensor:
    model.eval()
    imgs = imgs.to(DEVICE, non_blocking=True)

    with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
        logits = model(imgs)
        probs = torch.sigmoid(logits)

        if use_tta:
            imgs_h = torch.flip(imgs, dims=[3])
            ph = torch.sigmoid(model(imgs_h))
            ph = torch.flip(ph, dims=[3])

            imgs_v = torch.flip(imgs, dims=[2])
            pv = torch.sigmoid(model(imgs_v))
            pv = torch.flip(pv, dims=[2])

            probs = (probs + ph + pv) / 3.0

    return probs.detach().cpu()


@torch.no_grad()
def collect_probs_for_grid(model, loader, use_tta: bool) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    probs, gts = [], []
    for batch in tqdm(loader, desc="[Collect val probs]", leave=False):
        p = predict_batch_probs(model, batch["image"], use_tta=use_tta).numpy()
        g = batch["mask"].numpy()
        for i in range(p.shape[0]):
            probs.append(p[i, 0].astype(np.float32))
            gts.append(g[i, 0].astype(np.float32))
    return probs, gts


def eval_probs_with_config(probs: List[np.ndarray], gts: List[np.ndarray], cfg: Dict) -> Dict:
    sums = {"mIoU": [], "F1": [], "Boundary_IoU": []}
    for p, g in zip(probs, gts):
        pred = postprocess_mask(
            p,
            threshold=cfg["threshold"],
            min_area=cfg["min_area"],
            closing_kernel=cfg["closing_kernel"],
            fill_hole=cfg["fill_holes"],
        )
        update_metric_sums(sums, pred, g)
    return summarize_metric_lists(sums)


def grid_search_postprocess(probs: List[np.ndarray], gts: List[np.ndarray], grid_mode: str = "full") -> Dict:
    if grid_mode == "fast":
        thresholds = [0.50, 0.55, 0.60, 0.65]
        min_areas = [0, 64, 128]
        closing_kernels = [0, 3]
        fill_holes = [False, True]
    else:
        thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        min_areas = [0, 16, 32, 64, 128]
        closing_kernels = [0, 3, 5]
        fill_holes = [False, True]

    best, rows = None, []
    total = len(thresholds) * len(min_areas) * len(closing_kernels) * len(fill_holes)
    print(f"[GridSearch] mode={grid_mode} configs={total} N={len(probs)}", flush=True)

    idx = 0
    for th in thresholds:
        for area in min_areas:
            for ck in closing_kernels:
                for fh in fill_holes:
                    idx += 1
                    cfg = {"threshold": th, "min_area": area, "closing_kernel": ck, "fill_holes": fh}
                    m = eval_probs_with_config(probs, gts, cfg)
                    row = {**cfg, **m}
                    rows.append(row)
                    if best is None or row["mIoU"] > best["mIoU"]:
                        best = row
                    if idx == 1 or idx % 20 == 0 or idx == total:
                        print(f"[GridSearch] {idx}/{total} best mIoU={best['mIoU']:.4f}", flush=True)

    return {"best": best, "all": rows}


@torch.no_grad()
def eval_loader_with_config(
    model: nn.Module,
    loader: DataLoader,
    cfg: Dict,
    use_tta: bool,
    save_per_image_csv: Optional[Path] = None,
) -> Dict:
    sums = {"mIoU": [], "F1": [], "Boundary_IoU": []}
    per_rows = []

    for batch in tqdm(loader, desc="[Eval]", leave=False):
        probs = predict_batch_probs(model, batch["image"], use_tta=use_tta).numpy()
        gts = batch["mask"].numpy()
        for i in range(probs.shape[0]):
            p = probs[i, 0].astype(np.float32)
            g = gts[i, 0].astype(np.float32)
            pred = postprocess_mask(
                p,
                threshold=cfg["threshold"],
                min_area=cfg["min_area"],
                closing_kernel=cfg["closing_kernel"],
                fill_hole=cfg["fill_holes"],
            )
            miou = compute_iou(pred, g)
            f1 = compute_f1(pred, g)
            biou = compute_biou(pred, g, d=5)
            sums["mIoU"].append(miou)
            sums["F1"].append(f1)
            sums["Boundary_IoU"].append(biou)
            if save_per_image_csv is not None:
                per_rows.append({
                    "image_id": batch["name"][i],
                    "region": infer_region(batch["name"][i]),
                    "mIoU": miou,
                    "F1": f1,
                    "Boundary_IoU": biou,
                })

    if save_per_image_csv is not None:
        ensure_dir(save_per_image_csv.parent)
        with save_per_image_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "region", "mIoU", "F1", "Boundary_IoU"])
            writer.writeheader()
            writer.writerows(per_rows)

    return summarize_metric_lists(sums)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def make_loader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )


def train_epoch(model, loader, optimizer, scaler, epoch: int) -> float:
    model.train()
    losses = []
    pbar = tqdm(loader, desc=f"[Train epoch {epoch}]", leave=False)
    for batch in pbar:
        imgs = batch["image"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            logits = model(imgs)
            loss = segmentation_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.detach().cpu()))
        pbar.set_postfix(loss=np.mean(losses))

    return float(np.mean(losses))


@torch.no_grad()
def eval_fixed_05(model, loader, use_tta=False) -> Dict:
    cfg = {"threshold": 0.5, "min_area": 0, "closing_kernel": 0, "fill_holes": False}
    return eval_loader_with_config(model, loader, cfg=cfg, use_tta=use_tta)


def save_ckpt(path: Path, model: nn.Module, extra: Dict):
    ensure_dir(path.parent)
    torch.save({
        "model": model.state_dict(),
        "extra": extra,
    }, path)


def load_ckpt(path: Path, model: nn.Module, strict=True):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=strict)
    return ckpt


def source_pretrain(args, source_train_manifest: Path, source_val_manifest: Path, out_dir: Path) -> Path:
    source_dir = out_dir / "source_pretrain"
    ensure_dir(source_dir)
    best_path = source_dir / "segformer_source_best.pth"
    last_path = source_dir / "segformer_source_last.pth"

    if best_path.exists() and not args.force_source_pretrain:
        print(f"[Source] Reuse existing checkpoint: {best_path}", flush=True)
        return best_path

    print("[Source] Start source pretraining", flush=True)
    train_ds = ManifestSegDataset(source_train_manifest, train_aug=True)
    val_ds = ManifestSegDataset(source_val_manifest, limit=args.source_val_eval_limit, train_aug=False)

    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, args.num_workers, shuffle=False)

    model = CompactSegFormer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.source_lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    best_miou = -1
    history = []

    for epoch in range(1, args.source_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scaler, epoch)
        row = {"epoch": epoch, "loss": loss, "val_mIoU_05": None, "val_F1_05": None, "val_BIoU_05": None}

        if epoch == 1 or epoch % args.source_val_every == 0 or epoch == args.source_epochs:
            m = eval_fixed_05(model, val_loader, use_tta=False)
            row.update({
                "val_mIoU_05": m["mIoU"],
                "val_F1_05": m["F1"],
                "val_BIoU_05": m["Boundary_IoU"],
            })
            print(f"[Source] epoch={epoch} loss={loss:.4f} val_mIoU@0.5={m['mIoU']:.4f}", flush=True)
            if m["mIoU"] > best_miou:
                best_miou = m["mIoU"]
                save_ckpt(best_path, model, {"epoch": epoch, "val": m, "args": vars(args)})
                print(f"[Source] Saved best: {best_path}", flush=True)

        history.append(row)
        save_ckpt(last_path, model, {"epoch": epoch, "args": vars(args)})

    with (source_dir / "source_pretrain_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return best_path


def target_finetune_and_eval(
    args,
    seed: int,
    source_ckpt: Path,
    target_support_manifest: Path,
    target_val_manifest: Path,
    target_test_manifest: Path,
    out_dir: Path,
) -> Dict:
    print(f"\n[Target] Seed {seed}", flush=True)
    set_seed(seed)

    seed_dir = out_dir / f"seed{seed}"
    ensure_dir(seed_dir)
    best_path = seed_dir / "segformer_target_best.pth"

    train_ds = ManifestSegDataset(target_support_manifest, train_aug=True)
    val_ds_ckpt = ManifestSegDataset(target_val_manifest, limit=args.val_eval_limit, train_aug=False)

    train_loader = make_loader(train_ds, args.target_batch_size, args.num_workers, shuffle=True)
    val_loader_ckpt = make_loader(val_ds_ckpt, args.batch_size, args.num_workers, shuffle=False)

    model = CompactSegFormer().to(DEVICE)
    load_ckpt(source_ckpt, model, strict=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.target_lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    best_miou = -1
    history = []

    for epoch in range(1, args.target_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scaler, epoch)
        row = {"epoch": epoch, "loss": loss, "val_calib_mIoU": None, "val_calib_BIoU": None}

        if epoch == 1 or epoch % args.val_every == 0 or epoch == args.target_epochs:
            probs, gts = collect_probs_for_grid(model, val_loader_ckpt, use_tta=args.use_tta_for_val_selection)
            sweep = grid_search_postprocess(probs, gts, grid_mode="fast")
            m = sweep["best"]
            row.update({
                "val_calib_mIoU": m["mIoU"],
                "val_calib_F1": m["F1"],
                "val_calib_BIoU": m["Boundary_IoU"],
                "threshold": m["threshold"],
                "min_area": m["min_area"],
                "closing_kernel": m["closing_kernel"],
                "fill_holes": m["fill_holes"],
            })
            print(
                f"[Target seed={seed}] epoch={epoch} loss={loss:.4f} "
                f"val_calib_mIoU={m['mIoU']:.4f} BIoU={m['Boundary_IoU']:.4f}",
                flush=True,
            )

            if m["mIoU"] > best_miou:
                best_miou = m["mIoU"]
                save_ckpt(best_path, model, {"epoch": epoch, "val_best": m, "args": vars(args)})
                print(f"[Target seed={seed}] Saved best: {best_path}", flush=True)

        history.append(row)

    with (seed_dir / "target_train_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Reload best checkpoint for final calibration/evaluation.
    load_ckpt(best_path, model, strict=True)

    final_val_ds = ManifestSegDataset(target_val_manifest, limit=args.final_val_eval_limit, train_aug=False)
    final_val_loader = make_loader(final_val_ds, args.batch_size, args.num_workers, shuffle=False)

    print(f"[Target seed={seed}] Final target_val calibration", flush=True)
    val_probs, val_gts = collect_probs_for_grid(model, final_val_loader, use_tta=args.use_tta)
    val_sweep = grid_search_postprocess(val_probs, val_gts, grid_mode=args.grid_mode)
    best_cfg = val_sweep["best"]

    # Evaluate target final test streaming, not storing full probability maps.
    test_ds = ManifestSegDataset(target_test_manifest, limit=args.test_eval_limit, train_aug=False)
    test_loader = make_loader(test_ds, args.batch_size, args.num_workers, shuffle=False)

    print(f"[Target seed={seed}] Full test evaluation with cfg={best_cfg}", flush=True)
    per_csv = seed_dir / "segformer_per_image_metrics.csv"
    test_metrics = eval_loader_with_config(
        model,
        test_loader,
        cfg=best_cfg,
        use_tta=args.use_tta,
        save_per_image_csv=per_csv,
    )

    seed_summary = {
        "seed": seed,
        "method": "Target 20-shot SegFormer-style",
        "source_checkpoint": str(source_ckpt),
        "target_checkpoint": str(best_path),
        "support_manifest": str(target_support_manifest),
        "val_manifest": str(target_val_manifest),
        "test_manifest": str(target_test_manifest),
        "use_tta": bool(args.use_tta),
        "val_best_cfg": best_cfg,
        "test_calibrated_postprocess": test_metrics,
        "timestamp": datetime.now().isoformat(),
    }

    with (seed_dir / "segformer_seed_summary.json").open("w", encoding="utf-8") as f:
        json.dump(seed_summary, f, indent=2)

    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return seed_summary


def aggregate_and_save(rows: List[Dict], out_dir: Path):
    csv_path = out_dir / "target20_segformer_summary.csv"
    json_path = out_dir / "target20_segformer_summary.json"
    latex_path = out_dir / "target20_segformer_latex_table.txt"

    flat_rows = []
    for r in rows:
        m = r["test_calibrated_postprocess"]
        cfg = r["val_best_cfg"]
        flat_rows.append({
            "seed": r["seed"],
            "mIoU": m["mIoU"],
            "F1": m["F1"],
            "Boundary_IoU": m["Boundary_IoU"],
            "threshold": cfg["threshold"],
            "min_area": cfg["min_area"],
            "closing_kernel": cfg["closing_kernel"],
            "fill_holes": cfg["fill_holes"],
        })

    if flat_rows:
        keys = list(flat_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(flat_rows)

    metrics = ["mIoU", "F1", "Boundary_IoU"]
    agg = {}
    for k in metrics:
        vals = [row[k] for row in flat_rows]
        agg[k] = {
            "mean": float(np.mean(vals)) if vals else 0.0,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }

    summary = {
        "rows": flat_rows,
        "aggregate": agg,
        "timestamp": datetime.now().isoformat(),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    latex = (
        "% Add this line to the baseline table if the result is useful:\n"
        "Target-adapted SegFormer & Target 20-shot, full test & "
        f"{agg['mIoU']['mean']*100:.2f} $\\pm$ {agg['mIoU']['std']*100:.2f} & "
        f"{agg['F1']['mean']*100:.2f} $\\pm$ {agg['F1']['std']*100:.2f} & "
        f"{agg['Boundary_IoU']['mean']*100:.2f} $\\pm$ {agg['Boundary_IoU']['std']*100:.2f} \\\\\n"
    )
    latex_path.write_text(latex, encoding="utf-8")

    print("\n[Summary]", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[Saved] {csv_path}", flush=True)
    print(f"[Saved] {json_path}", flush=True)
    print(f"[Saved] {latex_path}", flush=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=str(DEFAULT_PROJECT_ROOT))
    ap.add_argument("--manifest_dir", type=str, default=None)

    ap.add_argument("--seeds", type=str, default="42,123,456")
    ap.add_argument("--shot", type=int, default=20)

    ap.add_argument("--source_epochs", type=int, default=30)
    ap.add_argument("--target_epochs", type=int, default=120)
    ap.add_argument("--source_lr", type=float, default=3e-4)
    ap.add_argument("--target_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--target_batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--source_val_every", type=int, default=5)
    ap.add_argument("--val_every", type=int, default=10)
    ap.add_argument("--source_val_eval_limit", type=int, default=200)
    ap.add_argument("--val_eval_limit", type=int, default=200)
    ap.add_argument("--final_val_eval_limit", type=int, default=500)
    ap.add_argument("--test_eval_limit", type=int, default=0)

    ap.add_argument("--use_tta", action="store_true")
    ap.add_argument("--use_tta_for_val_selection", action="store_true",
                    help="Use TTA during checkpoint-selection val sweep. Slower; off by default.")
    ap.add_argument("--grid_mode", type=str, default="full", choices=["fast", "full"])

    ap.add_argument("--force_source_pretrain", action="store_true")
    ap.add_argument("--out_dir", type=str, default="results/target20_segformer_baseline")

    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    project_root = Path(args.project_root)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else project_root / "data" / "splits" / "e0_manifest"
    out_dir = project_root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    ensure_dir(out_dir)

    print(f"[Config] DEVICE={DEVICE}", flush=True)
    print(f"[Config] project_root={project_root}", flush=True)
    print(f"[Config] manifest_dir={manifest_dir}", flush=True)
    print(f"[Config] out_dir={out_dir}", flush=True)
    print(f"[Config] args={vars(args)}", flush=True)

    # Robust candidate names for source manifests.
    source_train_manifest = resolve_manifest(
        manifest_dir,
        ["source_train.txt", "source_train_1000.txt", "source_whu_train.txt", "source_whu_train_1000.txt"],
    )
    source_val_manifest = resolve_manifest(
        manifest_dir,
        ["source_val.txt", "source_val_200.txt", "source_whu_val.txt", "source_whu_val_200.txt"],
    )
    target_val_manifest = resolve_manifest(manifest_dir, ["target_val.txt", "target_val_500.txt"])
    target_test_manifest = resolve_manifest(
        manifest_dir,
        ["target_final_test_8402.txt", "target_final_test.txt", "target_test_8402.txt"],
    )

    source_ckpt = source_pretrain(args, source_train_manifest, source_val_manifest, out_dir)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    seed_summaries = []

    for seed in seeds:
        support_manifest = resolve_manifest(
            manifest_dir,
            [
                f"target_support_{args.shot}_seed{seed}.txt",
                f"target_support_{args.shot}_seed_{seed}.txt",
                f"target_{args.shot}shot_seed{seed}.txt",
            ],
        )
        s = target_finetune_and_eval(
            args=args,
            seed=seed,
            source_ckpt=source_ckpt,
            target_support_manifest=support_manifest,
            target_val_manifest=target_val_manifest,
            target_test_manifest=target_test_manifest,
            out_dir=out_dir,
        )
        seed_summaries.append(s)

    aggregate_and_save(seed_summaries, out_dir)


if __name__ == "__main__":
    main()
