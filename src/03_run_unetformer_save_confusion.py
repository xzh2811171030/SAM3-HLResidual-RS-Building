#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matched UNetFormer-style baseline for low-label WHU-to-WHU-Mix building extraction.

Protocol implemented by this script:
1) Source pre-train on WHU aerial source_train/source_val.
2) Target 20-shot fine-tune for seeds 42/123/456 using the fixed target support manifests.
3) Validation-only post-processing calibration on target_val.
4) Full WHU-Mix test evaluation with optional TTA.
5) Primary aggregate on target_final_test_8402 excluding target_pilot_test_500.
6) Optionally save per-image TP/FP/FN/area CSVs and generate a coverage diagnostic table.

The code is intentionally self-contained and does not depend on segmentation_models_pytorch.
It uses torchvision's ResNet encoder when available, plus a lightweight UNetFormer-style
Transformer bottleneck and decoder. This is intended as a strong matched non-SAM baseline,
not as a replacement for the official UNetFormer authors' implementation.
"""

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    from torchvision.models import resnet34, resnet50
    try:
        from torchvision.models import ResNet34_Weights, ResNet50_Weights
    except Exception:
        ResNet34_Weights = None
        ResNet50_Weights = None
except Exception as e:  # pragma: no cover
    resnet34 = None
    resnet50 = None
    ResNet34_Weights = None
    ResNet50_Weights = None
    _TORCHVISION_IMPORT_ERROR = e
else:
    _TORCHVISION_IMPORT_ERROR = None


# -----------------------------
# Reproducibility utilities
# -----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_path(p: str) -> str:
    return os.path.normpath(os.path.abspath(os.path.expanduser(p)))


# -----------------------------
# Manifest parsing and masks
# -----------------------------

def _candidate_mask_paths(image_path: str) -> List[str]:
    """Infer mask path for common WHU/processed_slim layouts."""
    p = Path(image_path)
    suffixes = [p.suffix]
    if p.suffix.lower() not in [".png", ".tif", ".tiff", ".jpg", ".jpeg"]:
        suffixes += [".png", ".tif", ".tiff"]
    else:
        suffixes += [".png", ".tif", ".tiff", ".jpg"]

    candidates = []
    s = str(p)
    replacements = [
        ("/images/", "/dual_channel_labels/"),
        ("\\images\\", "\\dual_channel_labels\\"),
        ("/images/", "/labels/"),
        ("\\images\\", "\\labels\\"),
        ("/image/", "/label/"),
        ("\\image\\", "\\label\\"),
        ("/JPEGImages/", "/SegmentationClass/"),
        ("\\JPEGImages\\", "\\SegmentationClass\\"),
        ("/imgs/", "/masks/"),
        ("\\imgs\\", "\\masks\\"),
    ]
    for old, new in replacements:
        if old in s:
            base = s.replace(old, new)
            base_no_suffix = str(Path(base).with_suffix(""))
            for suf in suffixes:
                candidates.append(base_no_suffix + suf)

    # sibling folders, useful if manifests point to flat processed folders
    for folder in ["dual_channel_labels", "labels", "masks", "mask", "gt", "GT"]:
        base_no_suffix = str(p.parent.parent / folder / p.stem)
        for suf in suffixes:
            candidates.append(base_no_suffix + suf)

    # same directory with common suffix conventions
    for ext in suffixes:
        candidates.append(str(p.with_name(p.stem + "_mask" + ext)))
        candidates.append(str(p.with_name(p.stem + "_label" + ext)))
        candidates.append(str(p.with_suffix(ext)))

    # de-duplicate in order
    out = []
    seen = set()
    for c in candidates:
        c = os.path.normpath(c)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def infer_mask_path(image_path: str) -> str:
    for c in _candidate_mask_paths(image_path):
        if os.path.exists(c) and normalize_path(c) != normalize_path(image_path):
            return c
    msg = [f"Could not infer mask path for image: {image_path}", "Tried candidates:"]
    msg.extend(_candidate_mask_paths(image_path)[:30])
    raise FileNotFoundError("\n".join(msg))


def parse_manifest_line(line: str) -> Optional[Tuple[str, Optional[str]]]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    # Support CSV, tab, comma, or whitespace. If two paths are present, use image,mask.
    if "," in line:
        parts = [x.strip() for x in line.split(",") if x.strip()]
    else:
        parts = [x.strip() for x in line.split() if x.strip()]
    if len(parts) == 1:
        return parts[0], None
    return parts[0], parts[1]


def read_manifest(manifest_path: Path, project_root: Optional[Path] = None) -> List[Tuple[str, str]]:
    items = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for raw in f:
            parsed = parse_manifest_line(raw)
            if parsed is None:
                continue
            img, mask = parsed
            if project_root is not None:
                if not os.path.isabs(img):
                    img = str(project_root / img)
                if mask is not None and not os.path.isabs(mask):
                    mask = str(project_root / mask)
            if mask is None:
                mask = infer_mask_path(img)
            if not os.path.exists(img):
                raise FileNotFoundError(f"Image path does not exist: {img}")
            if not os.path.exists(mask):
                raise FileNotFoundError(f"Mask path does not exist: {mask}")
            items.append((normalize_path(img), normalize_path(mask)))
    if not items:
        raise RuntimeError(f"Empty manifest: {manifest_path}")
    return items


def load_image_rgb(path: str, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return img


def choose_mask_channel(arr: np.ndarray) -> np.ndarray:
    """Auto-detect the correct channel for 3-channel dual_channel_labels.
    
    For dual_channel_labels the building mask may be in channel 0, 1, or 2.
    This function picks the channel whose >0 binarization yields a plausible
    building occupancy ratio (0.001–0.80).  If no channel is plausible it falls
    back to the one closest to 0.20.
    """
    scored = []
    for ch in range(arr.shape[-1]):
        bin_arr = (arr[..., ch] > 0).astype(np.uint8)
        fg = float(bin_arr.mean())
        plausible = 0.001 <= fg <= 0.80
        scored.append((plausible, fg, ch, bin_arr))

    plausible_items = [x for x in scored if x[0]]
    if plausible_items:
        # Prefer the channel with the largest plausible foreground ratio:
        # in dual_channel_labels the building mask is denser than the boundary channel.
        _, _, ch, bin_arr = max(plausible_items, key=lambda x: x[1])
    else:
        _, _, ch, bin_arr = min(scored, key=lambda x: abs(x[1] - 0.20))

    # Auto-invert if fg > 0.80 (background channel misidentified as foreground)
    fg = float(bin_arr.mean())
    if fg > 0.80:
        inv = 1 - bin_arr
        inv_fg = float(inv.mean())
        if 0.001 <= inv_fg <= 0.80:
            bin_arr = inv

    return bin_arr


def load_mask_binary(path: str, size: int) -> Image.Image:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        bin_arr = choose_mask_channel(arr)
    else:
        bin_arr = (arr > 0).astype(np.uint8)
    arr255 = (bin_arr.astype(np.uint8)) * 255
    mask = Image.fromarray(arr255)
    if mask.size != (size, size):
        mask = mask.resize((size, size), Image.NEAREST)
    return mask


class BuildingDataset(Dataset):
    def __init__(self, items: Sequence[Tuple[str, str]], image_size: int = 512,
                 augment: bool = False, repeat: int = 1) -> None:
        self.items = list(items)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.repeat = max(1, int(repeat))

    def __len__(self) -> int:
        return len(self.items) * self.repeat

    def _augment_pair(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            k = random.randint(0, 3)
            if k:
                img = img.rotate(90 * k, resample=Image.BILINEAR)
                mask = mask.rotate(90 * k, resample=Image.NEAREST)
        if random.random() < 0.35:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))
        if random.random() < 0.35:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.25))
        return img, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path = self.items[idx % len(self.items)]
        img = load_image_rgb(img_path, self.image_size)
        mask = load_mask_binary(mask_path, self.image_size)
        if self.augment:
            img, mask = self._augment_pair(img, mask)
        img_arr = np.asarray(img).astype(np.float32) / 255.0
        # ImageNet normalization is used for the non-SAM baseline.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_arr = (img_arr - mean) / std
        img_arr = np.transpose(img_arr, (2, 0, 1))
        mask_arr = (np.asarray(mask).astype(np.float32) > 127).astype(np.float32)[None, ...]
        return {
            "image": torch.from_numpy(img_arr),
            "mask": torch.from_numpy(mask_arr),
            "path": img_path,
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
        "path": [b["path"] for b in batch],
    }


# -----------------------------
# UNetFormer-style architecture
# -----------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, ch: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(8, ch // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class TransformerBottleneck(nn.Module):
    def __init__(self, in_ch: int, dim: int = 256, heads: int = 8, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, 1, bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.out = nn.Conv2d(dim, in_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.proj(x)
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)  # B, HW, C
        y = self.norm1(seq)
        y, _ = self.attn(y, y, y, need_weights=False)
        seq = seq + y
        seq = seq + self.ffn(self.norm2(seq))
        x = seq.transpose(1, 2).reshape(b, c, h, w)
        return residual + self.out(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)
        self.se = SEBlock(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.se(x)


class UNetFormerBaseline(nn.Module):
    def __init__(self, encoder: str = "resnet34", imagenet_pretrained: bool = False) -> None:
        super().__init__()
        if resnet34 is None:
            raise ImportError(f"torchvision.models could not be imported: {_TORCHVISION_IMPORT_ERROR}")
        if encoder == "resnet50":
            weights = None
            if imagenet_pretrained and ResNet50_Weights is not None:
                try:
                    weights = ResNet50_Weights.IMAGENET1K_V2
                except Exception:
                    weights = None
            try:
                base = resnet50(weights=weights)
            except Exception as e:
                print(f"[WARN] Failed to load ImageNet weights for ResNet50 ({e}); using random init.")
                base = resnet50(weights=None)
            enc_ch = [64, 256, 512, 1024, 2048]
        else:
            weights = None
            if imagenet_pretrained and ResNet34_Weights is not None:
                try:
                    weights = ResNet34_Weights.IMAGENET1K_V1
                except Exception:
                    weights = None
            try:
                base = resnet34(weights=weights)
            except Exception as e:
                print(f"[WARN] Failed to load ImageNet weights for ResNet34 ({e}); using random init.")
                base = resnet34(weights=None)
            enc_ch = [64, 64, 128, 256, 512]

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu)  # H/2
        self.pool = base.maxpool
        self.layer1 = base.layer1  # H/4
        self.layer2 = base.layer2  # H/8
        self.layer3 = base.layer3  # H/16
        self.layer4 = base.layer4  # H/32

        self.bottleneck = nn.Sequential(
            ConvBNReLU(enc_ch[4], 512, 1, 0),
            TransformerBottleneck(512, dim=256, heads=8),
        )
        self.lat3 = ConvBNReLU(enc_ch[3], 256, 1, 0)
        self.lat2 = ConvBNReLU(enc_ch[2], 128, 1, 0)
        self.lat1 = ConvBNReLU(enc_ch[1], 64, 1, 0)
        self.lat0 = ConvBNReLU(enc_ch[0], 64, 1, 0)

        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)
        self.dec0 = DecoderBlock(64, 64, 64)
        self.head = nn.Sequential(
            ConvBNReLU(64, 32),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)          # H/2
        x1 = self.layer1(self.pool(x0))  # H/4
        x2 = self.layer2(x1)       # H/8
        x3 = self.layer3(x2)       # H/16
        x4 = self.layer4(x3)       # H/32
        x = self.bottleneck(x4)
        x = self.dec3(x, self.lat3(x3))
        x = self.dec2(x, self.lat2(x2))
        x = self.dec1(x, self.lat1(x1))
        x = self.dec0(x, self.lat0(x0))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(x)


# -----------------------------
# Loss and metrics
# -----------------------------

class BCEDiceLoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = torch.sum(probs * targets, dims)
        denom = torch.sum(probs + targets, dims)
        dice = 1.0 - torch.mean((2.0 * inter + 1.0) / (denom + 1.0))
        return bce + self.dice_weight * dice


def confusion_from_binary(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    return tp, fp, fn


def iou_f1(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    tp, fp, fn = confusion_from_binary(pred, gt)
    union = tp + fp + fn
    if union == 0:
        iou = 1.0
    else:
        iou = tp / union
    denom = 2 * tp + fp + fn
    f1 = 1.0 if denom == 0 else (2 * tp / denom)
    return float(iou), float(f1)


def boundary_map(mask: np.ndarray, d: int = 5) -> np.ndarray:
    mask = mask.astype(np.uint8)
    if cv2 is None:
        # Simple fallback: use binary difference after max-pool erosion approximation in numpy.
        from scipy.ndimage import binary_erosion, binary_dilation  # type: ignore
        eroded = binary_erosion(mask, iterations=1)
        boundary = np.logical_xor(mask.astype(bool), eroded)
        return binary_dilation(boundary, iterations=d).astype(np.uint8)
    k_erode = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, k_erode, iterations=1)
    boundary = (mask - eroded).astype(np.uint8)
    k = np.ones((2 * d + 1, 2 * d + 1), np.uint8)
    return cv2.dilate(boundary, k, iterations=1).astype(np.uint8)


def boundary_iou(pred: np.ndarray, gt: np.ndarray, d: int = 5) -> float:
    bp = boundary_map(pred, d=d).astype(bool)
    bg = boundary_map(gt, d=d).astype(bool)
    inter = np.logical_and(bp, bg).sum()
    union = np.logical_or(bp, bg).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def aggregate_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {"mIoU": float("nan"), "F1": float("nan"), "BIoU": float("nan"), "n": 0}
    return {
        "mIoU": float(np.mean([r["iou"] for r in rows]) * 100.0),
        "F1": float(np.mean([r["f1"] for r in rows]) * 100.0),
        "BIoU": float(np.mean([r["biou"] for r in rows]) * 100.0),
        "n": int(len(rows)),
    }


# -----------------------------
# Post-processing and inference
# -----------------------------

@dataclass
class PostConfig:
    threshold: float = 0.5
    min_area: int = 0
    closing: int = 0
    fill_holes: bool = False


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask.astype(np.uint8)
    if cv2 is None:
        from scipy import ndimage  # type: ignore
        lab, n = ndimage.label(mask)
        out = np.zeros_like(mask, dtype=np.uint8)
        for i in range(1, n + 1):
            comp = lab == i
            if int(comp.sum()) >= min_area:
                out[comp] = 1
        return out
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == i] = 1
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    if cv2 is None:
        from scipy.ndimage import binary_fill_holes  # type: ignore
        return binary_fill_holes(mask).astype(np.uint8)
    h, w = mask.shape
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = (flood == 0).astype(np.uint8)
    return np.maximum(mask, holes).astype(np.uint8)


def apply_postprocess(prob: np.ndarray, cfg: PostConfig) -> np.ndarray:
    mask = (prob >= cfg.threshold).astype(np.uint8)
    if cfg.closing and cfg.closing > 0:
        if cv2 is not None:
            k = np.ones((cfg.closing, cfg.closing), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        else:
            from scipy.ndimage import binary_closing  # type: ignore
            mask = binary_closing(mask, structure=np.ones((cfg.closing, cfg.closing))).astype(np.uint8)
    mask = remove_small_components(mask, cfg.min_area)
    if cfg.fill_holes:
        mask = fill_holes(mask)
    return mask.astype(np.uint8)


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device, use_tta: bool = False,
                  amp: bool = True) -> List[Dict[str, object]]:
    model.eval()
    records: List[Dict[str, object]] = []
    autocast_enabled = amp and device.type == "cuda"
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)  # type: ignore
        masks = batch["mask"].cpu().numpy()[:, 0].astype(np.uint8)  # type: ignore
        paths = batch["path"]  # type: ignore
        with torch.amp.autocast("cuda", enabled=autocast_enabled):
            if not use_tta:
                logits = model(images)
                probs = torch.sigmoid(logits)
            else:
                probs_list = []
                logits = model(images)
                probs_list.append(torch.sigmoid(logits))
                logits_h = model(torch.flip(images, dims=[3]))
                probs_list.append(torch.flip(torch.sigmoid(logits_h), dims=[3]))
                logits_v = model(torch.flip(images, dims=[2]))
                probs_list.append(torch.flip(torch.sigmoid(logits_v), dims=[2]))
                probs = torch.stack(probs_list, dim=0).mean(dim=0)
        probs_np = probs.detach().cpu().numpy()[:, 0].astype(np.float32)
        for p, m, path in zip(probs_np, masks, paths):
            records.append({"prob": p, "mask": m, "path": str(path)})
    return records


def evaluate_records(records: Sequence[Dict[str, object]], cfg: PostConfig, boundary_d: int = 5) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, float]] = []
    for rec in records:
        prob = rec["prob"]  # type: ignore
        gt = rec["mask"]  # type: ignore
        pred = apply_postprocess(prob, cfg)
        tp, fp, fn = confusion_from_binary(pred, gt)
        pred_area = int(pred.astype(bool).sum())
        gt_area = int(gt.astype(bool).sum())
        iou, f1 = iou_f1(pred, gt)
        biou = boundary_iou(pred, gt, d=boundary_d)
        row = {
            "path": rec["path"],
            "image_id": Path(str(rec["path"])).stem,
            "iou": iou,
            "f1": f1,
            "biou": biou,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "pred_area": pred_area,
            "gt_area": gt_area,
        }
        rows.append(row)
        metric_rows.append({"iou": iou, "f1": f1, "biou": biou})
    return aggregate_metrics(metric_rows), rows



def calibrate(records: Sequence[Dict[str, object]], boundary_d: int,
              thresholds: Sequence[float], min_areas: Sequence[int], closings: Sequence[int],
              fill_holes_options: Sequence[bool]) -> Tuple[PostConfig, Dict[str, float]]:
    best_cfg = PostConfig()
    best_metrics = {"mIoU": -1.0, "F1": -1.0, "BIoU": -1.0, "n": 0}
    for th in thresholds:
        for ma in min_areas:
            for cl in closings:
                for fh in fill_holes_options:
                    cfg = PostConfig(float(th), int(ma), int(cl), bool(fh))
                    metrics, _ = evaluate_records(records, cfg, boundary_d=boundary_d)
                    # Match the manuscript: select by validation mIoU.
                    if metrics["mIoU"] > best_metrics["mIoU"]:
                        best_cfg = cfg
                        best_metrics = metrics
    return best_cfg, best_metrics


# -----------------------------
# Training / evaluation loops
# -----------------------------

def train_one_stage(model: nn.Module, train_items: Sequence[Tuple[str, str]], val_items: Sequence[Tuple[str, str]],
                    device: torch.device, output_ckpt: Path, image_size: int, epochs: int, batch_size: int,
                    eval_batch_size: int, lr: float, weight_decay: float, repeat: int, eval_interval: int,
                    amp: bool, num_workers: int, stage_name: str) -> Dict[str, float]:
    ensure_dir(output_ckpt.parent)
    train_ds = BuildingDataset(train_items, image_size=image_size, augment=True, repeat=repeat)
    val_ds = BuildingDataset(val_items, image_size=image_size, augment=False, repeat=1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=False, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True, collate_fn=collate_batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = BCEDiceLoss(dice_weight=1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))
    best = {"mIoU": -1.0, "F1": -1.0, "BIoU": -1.0, "epoch": -1}
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)  # type: ignore
            msk = batch["mask"].to(device, non_blocking=True)  # type: ignore
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(amp and device.type == "cuda")):
                logits = model(img)
                if logits.shape[-2:] != msk.shape[-2:]:
                    logits = F.interpolate(logits, size=msk.shape[-2:], mode="bilinear", align_corners=False)
                loss = loss_fn(logits, msk)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % eval_interval == 0 or epoch == epochs:
            val_records = predict_probs(model, val_loader, device, use_tta=False, amp=amp)
            cfg = PostConfig(threshold=0.5, min_area=0, closing=0, fill_holes=False)
            metrics, _ = evaluate_records(val_records, cfg, boundary_d=5)
            elapsed = (time.time() - start_time) / 60.0
            print(f"[{stage_name}] epoch {epoch:03d}/{epochs} loss={np.mean(losses):.4f} "
                  f"val_mIoU={metrics['mIoU']:.2f} val_F1={metrics['F1']:.2f} val_BIoU={metrics['BIoU']:.2f} "
                  f"elapsed={elapsed:.1f}m")
            if metrics["mIoU"] > best["mIoU"]:
                best = {**metrics, "epoch": epoch}
                torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": best}, output_ckpt)
                print(f"[{stage_name}] saved best checkpoint to {output_ckpt}")
    return best


def load_model_ckpt(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)


def write_per_image_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "iou", "f1", "biou"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "path": r["path"],
                "iou": f"{float(r['iou']):.8f}",
                "f1": f"{float(r['f1']):.8f}",
                "biou": f"{float(r['biou']):.8f}",
            })


def write_confusion_csv(path: Path, rows: Sequence[Dict[str, object]], method: str, seed: int) -> None:
    """Write per-image TP/FP/FN/area CSV without saving prediction masks."""
    ensure_dir(path.parent)
    fields = [
        "method", "seed", "image_id", "path",
        "tp", "fp", "fn", "pred_area", "gt_area",
        "iou", "f1", "biou",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "method": method,
                "seed": seed,
                "image_id": r.get("image_id", Path(str(r["path"])).stem),
                "path": r["path"],
                "tp": int(r["tp"]),
                "fp": int(r["fp"]),
                "fn": int(r["fn"]),
                "pred_area": int(r["pred_area"]),
                "gt_area": int(r["gt_area"]),
                "iou": f"{float(r['iou']):.8f}",
                "f1": f"{float(r['f1']):.8f}",
                "biou": f"{float(r['biou']):.8f}",
            })


def canonical_stem(x: str) -> str:
    """Convert image path or id to a conservative filename stem for manifest matching."""
    s = str(x).strip()
    p = Path(s)
    stem = p.stem if p.suffix else Path(s).name
    for suf in ["_mask", "_label", "_labels", "_pred", "_prediction"]:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    return stem


def read_manifest_stems_for_coverage(manifest_path: Optional[Path], project_root: Optional[Path] = None) -> set:
    if manifest_path is None:
        return set()
    stems = set()
    with manifest_path.open("r", encoding="utf-8") as f:
        for raw in f:
            parsed = parse_manifest_line(raw)
            if parsed is None:
                continue
            img, _ = parsed
            if project_root is not None and not os.path.isabs(img):
                img = str(project_root / img)
            stems.add(canonical_stem(img))
    return stems


def _safe_mean_std(vals: Sequence[float], digits: int = 2) -> str:
    arr = np.asarray(list(vals), dtype=np.float64)
    if arr.size == 0:
        return "N/A"
    if arr.size == 1:
        return f"{arr.mean():.{digits}f}"
    return f"{arr.mean():.{digits}f}$\\pm${arr.std(ddof=1):.{digits}f}"


def _coverage_one_group(rows: "np.ndarray") -> Dict[str, float]:
    # rows is a pandas DataFrame at runtime, but avoid importing pandas at module import.
    tp = float(rows["tp"].sum())
    fp = float(rows["fp"].sum())
    fn = float(rows["fn"].sum())
    pred = float(rows["pred_area"].sum())
    gt = float(rows["gt_area"].sum())

    precision = 1.0 if (tp + fp) == 0 and gt == 0 else (tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    recall = 1.0 if gt == 0 and pred == 0 else (tp / gt if gt > 0 else 0.0)
    f1_micro = 1.0 if (2 * tp + fp + fn) == 0 else (2 * tp / (2 * tp + fp + fn))

    denom_iou = rows["tp"] + rows["fp"] + rows["fn"]
    iou_img = np.where(denom_iou == 0, 1.0, rows["tp"] / denom_iou)
    denom_f1 = 2 * rows["tp"] + rows["fp"] + rows["fn"]
    f1_img = np.where(denom_f1 == 0, 1.0, 2 * rows["tp"] / denom_f1)

    return {
        "n_images": int(len(rows)),
        "Precision_micro": 100.0 * float(precision),
        "Recall_micro": 100.0 * float(recall),
        "F1_micro": 100.0 * float(f1_micro),
        "Pred_GT_area_ratio": float(pred / gt) if gt > 0 else float("nan"),
        "FP_GT_ratio": float(fp / gt) if gt > 0 else float("nan"),
        "FN_GT_ratio": float(fn / gt) if gt > 0 else float("nan"),
        "mIoU_image_avg": 100.0 * float(np.mean(iou_img)),
        "F1_image_avg": 100.0 * float(np.mean(f1_img)),
    }


def write_coverage_outputs_from_csvs(
    csv_paths: Sequence[Path],
    out_dir: Path,
    eval_manifest: Optional[Path] = None,
    exclude_manifest: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> None:
    """Combine per-image confusion CSVs and write held-out coverage diagnostics."""
    import pandas as pd

    ensure_dir(out_dir)
    eval_stems = read_manifest_stems_for_coverage(eval_manifest, project_root=project_root)
    exclude_stems = read_manifest_stems_for_coverage(exclude_manifest, project_root=project_root)
    keep_stems = eval_stems - exclude_stems if eval_stems else set()

    frames = []
    required = {"tp", "fp", "fn", "pred_area", "gt_area"}
    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")
        if "image_id" not in df.columns:
            if "path" in df.columns:
                df["image_id"] = df["path"].astype(str).map(canonical_stem)
            else:
                raise ValueError(f"{csv_path} must contain image_id or path")
        df["stem"] = df["image_id"].astype(str).map(canonical_stem)

        if "method" not in df.columns:
            df["method"] = csv_path.stem
        if "seed" not in df.columns:
            df["seed"] = 0

        before = len(df)
        if keep_stems:
            df = df[df["stem"].isin(keep_stems)].copy()
        elif exclude_stems:
            df = df[~df["stem"].isin(exclude_stems)].copy()
        after = len(df)
        print(f"[Coverage] {csv_path.name}: kept {after}/{before} rows")
        if after == 0:
            raise RuntimeError(f"No rows remain after filtering for {csv_path}")
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out_dir / "per_image_confusion_filtered.csv", index=False)

    seed_rows = []
    for (method, seed), g in all_df.groupby(["method", "seed"], dropna=False):
        row = _coverage_one_group(g)
        row.update({"method": str(method), "seed": seed})
        seed_rows.append(row)
    per_seed = pd.DataFrame(seed_rows).sort_values(["method", "seed"])
    per_seed.to_csv(out_dir / "per_seed_coverage_summary.csv", index=False)

    summary_rows = []
    for method, g in per_seed.groupby("method", dropna=False):
        row = {"method": str(method)}
        for c in ["Precision_micro", "Recall_micro", "mIoU_image_avg", "F1_image_avg"]:
            row[c] = _safe_mean_std(g[c].tolist(), digits=2)
        for c in ["Pred_GT_area_ratio", "FP_GT_ratio", "FN_GT_ratio"]:
            row[c] = _safe_mean_std(g[c].tolist(), digits=3)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("method")
    summary.to_csv(out_dir / "method_coverage_summary.csv", index=False)

    lines = [
        r"% Auto-generated coverage diagnostic table.",
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
    for _, r in summary.iterrows():
        lines.append(
            f"{r['method']} & {r['Precision_micro']} & {r['Recall_micro']} & "
            f"{r['Pred_GT_area_ratio']} & {r['FP_GT_ratio']} & {r['FN_GT_ratio']} & "
            f"{r['mIoU_image_avg']} & {r['F1_image_avg']} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table*}",
    ]
    (out_dir / "coverage_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Coverage] Saved coverage table: {out_dir / 'coverage_table.tex'}")



def filter_nonpilot(rows: Sequence[Dict[str, object]], pilot_items: Sequence[Tuple[str, str]]) -> List[Dict[str, object]]:
    pilot_abs = {normalize_path(x[0]) for x in pilot_items}
    pilot_base = {Path(x[0]).name for x in pilot_items}
    out = []
    for r in rows:
        p = normalize_path(str(r["path"]))
        if p in pilot_abs or Path(p).name in pilot_base:
            continue
        out.append(r)
    return out


def rows_to_metric_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, float]]:
    return [{"iou": float(r["iou"]), "f1": float(r["f1"]), "biou": float(r["biou"])} for r in rows]


def write_latex_table(path: Path, seed_results: Dict[str, Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    lines = []
    lines.append("% Auto-generated by run_unetformer_matched_baseline.py")
    lines.append("\\begin{table}[!t]")
    lines.append("\\caption{Matched UNetFormer-Style Non-SAM Baseline Under Source Pretraining and Target 20-Shot Fine-Tuning}")
    lines.append("\\label{tab:unetformer_matched}")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append("\\begin{tabular}{c c c c}")
    lines.append("\\toprule")
    lines.append("Seed & mIoU (\\%) & F1 (\\%) & BIoU (\\%) " + "\\\\")
    lines.append("\\midrule")
    vals = []
    for seed in sorted(seed_results, key=lambda x: int(x)):
        m = seed_results[seed]["nonpilot_metrics"]
        vals.append((m["mIoU"], m["F1"], m["BIoU"]))
        lines.append(f"{seed} & {m['mIoU']:.2f} & {m['F1']:.2f} & {m['BIoU']:.2f} " + "\\\\")
    if vals:
        arr = np.array(vals, dtype=np.float32)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=1) if len(vals) > 1 else np.zeros(3)
        lines.append("\\midrule")
        lines.append(f"Mean$\\pm$Std & {mean[0]:.2f}$\\pm${std[0]:.2f} & {mean[1]:.2f}$\\pm${std[1]:.2f} & {mean[2]:.2f}$\\pm${std[2]:.2f} " + "\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default="./data")
    parser.add_argument("--manifest_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/unetformer_matched")
    parser.add_argument("--source_train", type=str, default="source_train.txt")
    parser.add_argument("--source_val", type=str, default="source_val.txt")
    parser.add_argument("--target_val", type=str, default="target_val.txt")
    parser.add_argument("--target_test", type=str, default="target_final_test_8402.txt")
    parser.add_argument("--pilot_manifest", type=str, default="target_pilot_test_500.txt")
    parser.add_argument("--support_pattern", type=str, default="target_support_20_seed{seed}.txt")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--encoder", type=str, default="resnet34", choices=["resnet34", "resnet50"])
    parser.add_argument("--imagenet_pretrained", action="store_true")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--source_epochs", type=int, default=60)
    parser.add_argument("--target_epochs", type=int, default=120)
    parser.add_argument("--source_repeat", type=int, default=1)
    parser.add_argument("--target_repeat", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--source_lr", type=float, default=3e-4)
    parser.add_argument("--target_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--skip_source_pretrain", action="store_true")
    parser.add_argument("--source_ckpt", type=str, default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    parser.add_argument("--min_areas", type=int, nargs="+", default=[0, 16, 32, 64, 128])
    parser.add_argument("--closings", type=int, nargs="+", default=[0, 3, 5])
    parser.add_argument("--fill_holes", type=int, nargs="+", default=[0, 1], help="0/1 candidates")
    parser.add_argument("--save_confusion_csv", action="store_true",
                        help="Save per-image TP/FP/FN/pred_area/gt_area CSVs. No masks are saved.")
    parser.add_argument("--method_name", type=str, default="UNetFormer-style ResNet34",
                        help="Method name written into confusion CSVs and coverage table.")
    parser.add_argument("--make_coverage_table", action="store_true",
                        help="Generate a coverage diagnostic table from this baseline and optional extra CSVs.")
    parser.add_argument("--extra_confusion_csv", type=str, nargs="*", default=[],
                        help="Optional additional confusion CSVs, e.g., Prompt-free LoRA and HL-Residual CSVs.")
    parser.add_argument("--coverage_output_dir", type=str, default=None,
                        help="Output directory for coverage diagnostic table. Default: output_dir/coverage_nonpilot.")
    parser.add_argument("--skip_target_finetune_if_exists", action="store_true",
                        help="If the seed target checkpoint already exists, skip target 20-shot fine-tuning and only evaluate.")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    manifest_dir = Path(args.manifest_dir).expanduser().resolve() if args.manifest_dir else project_root / "data" / "splits" / "e0_manifest"
    output_dir = (project_root / args.output_dir).resolve() if not os.path.isabs(args.output_dir) else Path(args.output_dir).resolve()
    ensure_dir(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp
    print(f"[INFO] project_root={project_root}")
    print(f"[INFO] manifest_dir={manifest_dir}")
    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] device={device}, amp={amp}")

    source_train_items = read_manifest(manifest_dir / args.source_train, project_root=project_root)
    source_val_items = read_manifest(manifest_dir / args.source_val, project_root=project_root)
    target_val_items = read_manifest(manifest_dir / args.target_val, project_root=project_root)
    test_items = read_manifest(manifest_dir / args.target_test, project_root=project_root)
    pilot_items = read_manifest(manifest_dir / args.pilot_manifest, project_root=project_root)
    print(f"[INFO] source_train={len(source_train_items)}, source_val={len(source_val_items)}, "
          f"target_val={len(target_val_items)}, test={len(test_items)}, pilot={len(pilot_items)}")

    # 1) Source pretraining
    if args.source_ckpt:
        source_ckpt = Path(args.source_ckpt).expanduser().resolve()
    else:
        source_ckpt = output_dir / f"unetformer_{args.encoder}_source_best.pth"

    if args.skip_source_pretrain and source_ckpt.exists():
        print(f"[INFO] Skipping source pretrain; using {source_ckpt}")
        source_best = {"ckpt": str(source_ckpt)}
    elif source_ckpt.exists() and args.skip_source_pretrain:
        raise FileNotFoundError(f"--skip_source_pretrain was set but source_ckpt does not exist: {source_ckpt}")
    else:
        seed_everything(3407)
        model = UNetFormerBaseline(encoder=args.encoder, imagenet_pretrained=args.imagenet_pretrained).to(device)
        source_best_metrics = train_one_stage(
            model=model,
            train_items=source_train_items,
            val_items=source_val_items,
            device=device,
            output_ckpt=source_ckpt,
            image_size=args.image_size,
            epochs=args.source_epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            lr=args.source_lr,
            weight_decay=args.weight_decay,
            repeat=args.source_repeat,
            eval_interval=args.eval_interval,
            amp=amp,
            num_workers=args.num_workers,
            stage_name="source_pretrain",
        )
        source_best = {"ckpt": str(source_ckpt), "metrics": source_best_metrics}
        del model
        torch.cuda.empty_cache()

    results: Dict[str, object] = {
        "args": vars(args),
        "source_pretrain": source_best,
        "seeds": {},
    }

    # Validation records for calibration are generated per seed after model fine-tuning.
    val_loader = DataLoader(BuildingDataset(target_val_items, image_size=args.image_size, augment=False),
                            batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, collate_fn=collate_batch)
    test_loader = DataLoader(BuildingDataset(test_items, image_size=args.image_size, augment=False),
                             batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, collate_fn=collate_batch)

    for seed in args.seeds:
        seed_str = str(seed)
        seed_everything(seed)
        support_manifest = manifest_dir / args.support_pattern.format(seed=seed)
        support_items = read_manifest(support_manifest, project_root=project_root)
        seed_dir = output_dir / f"seed{seed}"
        ensure_dir(seed_dir)
        target_ckpt = seed_dir / f"unetformer_{args.encoder}_source_target20_seed{seed}_best.pth"

        print(f"[INFO][seed={seed}] support={support_manifest} ({len(support_items)} images)")
        model = UNetFormerBaseline(encoder=args.encoder, imagenet_pretrained=False).to(device)
        load_model_ckpt(model, source_ckpt, device)
        if args.skip_target_finetune_if_exists and target_ckpt.exists():
            print(f"[INFO][seed={seed}] Skipping target fine-tuning; using existing checkpoint: {target_ckpt}")
            target_best = {"skipped": True, "checkpoint": str(target_ckpt)}
        else:
            target_best = train_one_stage(
                model=model,
                train_items=support_items,
                val_items=target_val_items,
                device=device,
                output_ckpt=target_ckpt,
                image_size=args.image_size,
                epochs=args.target_epochs,
                batch_size=max(1, min(args.batch_size, len(support_items))),
                eval_batch_size=args.eval_batch_size,
                lr=args.target_lr,
                weight_decay=args.weight_decay,
                repeat=args.target_repeat,
                eval_interval=args.eval_interval,
                amp=amp,
                num_workers=args.num_workers,
                stage_name=f"target20_seed{seed}",
            )

        load_model_ckpt(model, target_ckpt, device)
        print(f"[INFO][seed={seed}] Predicting target_val for calibration...")
        val_records = predict_probs(model, val_loader, device, use_tta=args.use_tta, amp=amp)
        cfg, val_metrics = calibrate(
            val_records,
            boundary_d=5,
            thresholds=args.thresholds,
            min_areas=args.min_areas,
            closings=args.closings,
            fill_holes_options=[bool(x) for x in args.fill_holes],
        )
        print(f"[INFO][seed={seed}] best_cfg={asdict(cfg)}, val_metrics={val_metrics}")

        print(f"[INFO][seed={seed}] Evaluating official test pool, use_tta={args.use_tta}...")
        test_records = predict_probs(model, test_loader, device, use_tta=args.use_tta, amp=amp)
        full_metrics, full_rows = evaluate_records(test_records, cfg, boundary_d=5)
        nonpilot_rows = filter_nonpilot(full_rows, pilot_items)
        nonpilot_metrics = aggregate_metrics(rows_to_metric_rows(nonpilot_rows))
        write_per_image_csv(seed_dir / "per_image_full_test.csv", full_rows)
        write_per_image_csv(seed_dir / "per_image_nonpilot_test.csv", nonpilot_rows)

        full_confusion_csv = seed_dir / "per_image_full_confusion.csv"
        nonpilot_confusion_csv = seed_dir / "per_image_nonpilot_confusion.csv"
        if args.save_confusion_csv or args.make_coverage_table:
            write_confusion_csv(full_confusion_csv, full_rows, method=args.method_name, seed=seed)
            write_confusion_csv(nonpilot_confusion_csv, nonpilot_rows, method=args.method_name, seed=seed)
            print(f"[INFO][seed={seed}] confusion CSV saved: {nonpilot_confusion_csv}")

        print(f"[RESULT][seed={seed}] full={full_metrics}")
        print(f"[RESULT][seed={seed}] nonpilot={nonpilot_metrics}")

        results["seeds"][seed_str] = {
            "support_manifest": str(support_manifest),
            "checkpoint": str(target_ckpt),
            "target_best_val_fixed05": target_best,
            "calibration": asdict(cfg),
            "calibration_val_metrics": val_metrics,
            "full_metrics": full_metrics,
            "nonpilot_metrics": nonpilot_metrics,
            "per_image_full_csv": str(seed_dir / "per_image_full_test.csv"),
            "per_image_nonpilot_csv": str(seed_dir / "per_image_nonpilot_test.csv"),
            "per_image_full_confusion_csv": str(full_confusion_csv),
            "per_image_nonpilot_confusion_csv": str(nonpilot_confusion_csv),
        }
        with (output_dir / "unetformer_matched_summary.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        write_latex_table(output_dir / "unetformer_matched_table.tex", results["seeds"])  # type: ignore
        del model
        torch.cuda.empty_cache()

    if args.make_coverage_table:
        own_csvs = []
        for seed in args.seeds:
            p = output_dir / f"seed{seed}" / "per_image_nonpilot_confusion.csv"
            if p.exists():
                own_csvs.append(p)
        extra_csvs = [Path(x).expanduser().resolve() for x in args.extra_confusion_csv]
        coverage_out = Path(args.coverage_output_dir).expanduser().resolve() if args.coverage_output_dir else output_dir / "coverage_nonpilot"
        write_coverage_outputs_from_csvs(
            csv_paths=[*extra_csvs, *own_csvs],
            out_dir=coverage_out,
            eval_manifest=manifest_dir / args.target_test,
            exclude_manifest=manifest_dir / args.pilot_manifest,
            project_root=project_root,
        )
        print("[DONE] Coverage diagnostic written to:", coverage_out)

    print("[DONE] Summary written to:", output_dir / "unetformer_matched_summary.json")
    print("[DONE] LaTeX table written to:", output_dir / "unetformer_matched_table.tex")


if __name__ == "__main__":
    main()
