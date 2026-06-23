# -*- coding: utf-8 -*-
"""
run_e6v2_pilot.py
=============================================================================
E6-v2 Pilot 实验脚本

目的：
  在正式全量重跑 E5/E6/E7 前，用低成本 pilot 验证 E6-v2 是否值得继续。

默认实验：
  train: data/splits/e0_manifest/target_support_20_seed42.txt
  val:   data/splits/e0_manifest/target_val.txt
  test:  data/splits/e0_manifest/target_pilot_test_500.txt

默认变体：
  1. lora_light  = LoRA + LightweightDecoder
  2. msr         = LoRA + Multi-Scale Refinement Decoder
  3. msr_bga     = LoRA + MSR Decoder + Dual-Gated Boundary Adapter

输出：
  results/e6v2_pilot/
    weights/
    e6v2_pilot_summary.json
    per_variant_predictions_metrics.json

运行：
  cd <project_root>
  python src/run_e6v2_pilot.py --variants lora_light,msr,msr_bga --epochs 20

如果显存不够：
  python src/run_e6v2_pilot.py --batch_size 1 --val_eval_limit 150

=============================================================================
"""

import argparse
import gc
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from tqdm import tqdm


# =============================================================================
# 0. 路径与常量
# =============================================================================

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SAM3_SRC = SRC / "models" / "sam3"
if str(SAM3_SRC) not in sys.path:
    sys.path.insert(0, str(SAM3_SRC))

from sam3.model_builder import build_sam3_image_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_SIZE = 512
SAM3_INPUT_SIZE = 1008
FEATURE_SIZE = (64, 64)
FEATURE_CHANNELS = 256

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
LABEL_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# 1. 可复现设置
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 2. Manifest Dataset
# =============================================================================

def read_manifest(path: Path, limit: Optional[int] = None) -> List[Path]:
    if not path.exists():
        raise FileNotFoundError(f"manifest 不存在: {path}")

    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().replace("\\", "/")
        if line:
            lines.append(Path(line))

    if limit is not None and limit > 0:
        lines = lines[:limit]

    return lines


def find_dual_label_for_image(image_path: Path) -> Optional[Path]:
    """
    根据 image 路径寻找 dual_channel_label。

    支持：
      processed_slim/target_whu_mix/train_pool/images/xxx.tif
      processed_slim/target_whu_mix/val/images/xxx.tif
      whu_mix_full_test/test/image/xxx.tif
    """
    stem = image_path.stem
    s = str(image_path).replace("\\", "/")

    candidates = []

    if "/images/" in s:
        candidates.append(Path(s.replace("/images/", "/dual_channel_labels/")).parent)

    if "/test/image/" in s:
        candidates.append(Path(s.replace("/test/image/", "/dual_channel_labels/")).parent)

    # 兄弟目录兜底
    candidates.append(image_path.parent.parent / "dual_channel_labels")
    candidates.append(image_path.parent.parent / "label")
    candidates.append(image_path.parent.parent / "labels")

    seen = set()
    unique_dirs = []
    for d in candidates:
        key = str(d).lower().replace("\\", "/")
        if key not in seen:
            seen.add(key)
            unique_dirs.append(d)

    for d in unique_dirs:
        if not d.exists():
            continue
        for ext in LABEL_EXTS:
            p = d / f"{stem}{ext}"
            if p.exists():
                return p

    return None


def make_boundary_from_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = cv2.subtract(mask, eroded)
    return boundary


class ManifestSegDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        target_size: int = TARGET_SIZE,
        limit: Optional[int] = None,
    ):
        self.image_paths = read_manifest(manifest_path, limit=limit)
        self.target_size = target_size

        if len(self.image_paths) == 0:
            raise RuntimeError(f"manifest 为空: {manifest_path}")

        print(f"  Dataset: {manifest_path.name} | N={len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.image_paths[idx]
        if not img_path.exists():
            raise FileNotFoundError(f"image 不存在: {img_path}")

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取 image: {img_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        label_path = find_dual_label_for_image(img_path)
        if label_path is None:
            raise FileNotFoundError(f"找不到 dual label: {img_path}")

        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if label is None:
            raise FileNotFoundError(f"无法读取 label: {label_path}")

        if label.ndim == 2:
            mask = (label > 0).astype(np.uint8) * 255
            boundary = make_boundary_from_mask(mask)
        else:
            label = cv2.resize(label, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
            # OpenCV BGR: B=mask, G=boundary
            mask = label[:, :, 0]
            boundary = label[:, :, 1] if label.shape[2] > 1 else make_boundary_from_mask(mask)
            mask = (mask > 0).astype(np.float32)
            boundary = (boundary > 0).astype(np.float32)
            return {
                "image": img_tensor,
                "mask": torch.from_numpy(mask).unsqueeze(0).float(),
                "boundary": torch.from_numpy(boundary).unsqueeze(0).float(),
                "name": img_path.stem,
                "path": str(img_path),
            }

        mask = (mask > 0).astype(np.float32)
        boundary = (boundary > 0).astype(np.float32)

        return {
            "image": img_tensor,
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
            "boundary": torch.from_numpy(boundary).unsqueeze(0).float(),
            "name": img_path.stem,
            "path": str(img_path),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "boundary": torch.stack([b["boundary"] for b in batch]),
        "name": [b["name"] for b in batch],
        "path": [b["path"] for b in batch],
    }


# =============================================================================
# 3. 指标
# =============================================================================

def compute_iou_np(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + 1e-7) / (union + 1e-7))


def compute_f1_np(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return float((2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7))


def boundary_map(mask: np.ndarray, d: int = 5) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask.astype(bool)

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = mask - eroded

    if d > 1:
        k = np.ones((d, d), np.uint8)
        boundary = cv2.dilate(boundary, k, iterations=1)

    return boundary.astype(bool)


def compute_boundary_iou_np(pred: np.ndarray, gt: np.ndarray, d: int = 5) -> float:
    pb = boundary_map(pred, d=d)
    gb = boundary_map(gt, d=d)
    inter = np.logical_and(pb, gb).sum()
    union = np.logical_or(pb, gb).sum()
    return float((inter + 1e-7) / (union + 1e-7))


def evaluate_arrays(preds: np.ndarray, gts: np.ndarray) -> Dict[str, float]:
    ious, f1s, bious = [], [], []
    for p, g in zip(preds, gts):
        ious.append(compute_iou_np(p, g))
        f1s.append(compute_f1_np(p, g))
        bious.append(compute_boundary_iou_np(p, g, d=5))
    return {
        "mIoU": float(np.mean(ious)),
        "F1": float(np.mean(f1s)),
        "Boundary_IoU": float(np.mean(bious)),
    }


def remove_small_components(pred: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return pred.astype(np.float32)

    pred_u8 = (pred > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(pred_u8, connectivity=8)
    out = np.zeros_like(pred_u8)

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 1

    return out.astype(np.float32)


# =============================================================================
# 4. Dice Loss
# =============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        b = probs.shape[0]
        probs = probs.view(b, -1)
        target = target.view(b, -1)
        inter = (probs * target).sum(dim=1)
        union = probs.sum(dim=1) + target.sum(dim=1)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return (1 - dice).mean()


# =============================================================================
# 5. SAM3 + LoRA
# =============================================================================

def patch_vit_mlp_for_grad_compat(vit_peft_model) -> None:
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

    print(f"  MLP patch: {n} modules")


def inject_lora_to_vit(model, rank: int = 8, alpha: int = 16, dropout: float = 0.05) -> None:
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["qkv"],
        bias="none",
    )

    vit_trunk = model.backbone.vision_backbone.trunk
    model.backbone.vision_backbone.trunk = get_peft_model(vit_trunk, lora_config)
    patch_vit_mlp_for_grad_compat(model.backbone.vision_backbone.trunk)

    for name, p in model.named_parameters():
        p.requires_grad = ("lora" in name.lower())

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


class SAM3Extractor:
    def __init__(self, checkpoint_path: Path, device: str = DEVICE):
        self.device = device
        self.model = build_sam3_image_model(checkpoint_path=str(checkpoint_path))
        self.model.to(device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        inject_lora_to_vit(self.model, LORA_RANK, LORA_ALPHA, LORA_DROPOUT)

        self.transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(SAM3_INPUT_SIZE, SAM3_INPUT_SIZE)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _preprocess_one(self, image_tensor: torch.Tensor) -> torch.Tensor:
        img_uint8 = (image_tensor * 255.0).clamp(0, 255).to(torch.uint8)
        return self.transform(img_uint8).unsqueeze(0).to(self.device)

    def extract_with_grad(self, image_tensor: torch.Tensor) -> torch.Tensor:
        img_processed = self._preprocess_one(image_tensor)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            out = self.model.backbone.forward_image(img_processed)

        feat = out["vision_features"].float()
        if feat.shape[-2:] != FEATURE_SIZE:
            feat = F.interpolate(feat, size=FEATURE_SIZE, mode="bilinear", align_corners=False)
        return feat

    @torch.no_grad()
    def extract_eval_batch(self, image_batch: torch.Tensor) -> torch.Tensor:
        feats = []
        for i in range(image_batch.shape[0]):
            img_processed = self._preprocess_one(image_batch[i])
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                out = self.model.backbone.forward_image(img_processed)
            feat = out["vision_features"].float()
            if feat.shape[-2:] != FEATURE_SIZE:
                feat = F.interpolate(feat, size=FEATURE_SIZE, mode="bilinear", align_corners=False)
            feats.append(feat.detach())
        return torch.cat(feats, dim=0)


# =============================================================================
# 6. 模型变体
# =============================================================================

class LightweightMaskDecoder(nn.Module):
    def __init__(self, feat_channels: int = FEATURE_CHANNELS):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feat_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor, rgb: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.decoder(feat)


class ASPP(nn.Module):
    def __init__(self, in_ch: int = 256, out_ch: int = 256):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_ch, out_ch // 4, 1), nn.BatchNorm2d(out_ch // 4), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv2d(in_ch, out_ch // 4, 3, padding=2, dilation=2), nn.BatchNorm2d(out_ch // 4), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(nn.Conv2d(in_ch, out_ch // 4, 3, padding=4, dilation=4), nn.BatchNorm2d(out_ch // 4), nn.ReLU(inplace=True))
        self.b4 = nn.Sequential(nn.Conv2d(in_ch, out_ch // 4, 3, padding=8, dilation=8), nn.BatchNorm2d(out_ch // 4), nn.ReLU(inplace=True))
        self.proj = nn.Sequential(nn.Conv2d(out_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1))


class RGBPyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.s2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.s3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.s4 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

    def forward(self, rgb: torch.Tensor):
        f1 = self.s1(rgb)     # 512
        f2 = self.s2(f1)      # 256
        f3 = self.s3(f2)      # 128
        f4 = self.s4(f3)      # 64
        return f1, f2, f3, f4


class MSRDecoder(nn.Module):
    """
    Multi-Scale Refinement Decoder:
      SAM feature 64x64 + RGB shallow pyramid
      重点测试：decoder 是否是旧 E6/E5 的瓶颈。
    """
    def __init__(self, feat_ch: int = 256):
        super().__init__()
        self.rgb_pyr = RGBPyramid()
        self.aspp = ASPP(feat_ch, 256)

        self.fuse64 = nn.Sequential(nn.Conv2d(256 + 128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.up128 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.fuse128 = nn.Sequential(nn.Conv2d(128 + 128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        self.up256 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.fuse256 = nn.Sequential(nn.Conv2d(64 + 64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.up512 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.fuse512 = nn.Sequential(nn.Conv2d(32 + 32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, feat: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        f1, f2, f3, f4 = self.rgb_pyr(rgb)
        x = self.aspp(feat)
        x = self.fuse64(torch.cat([x, f4], dim=1))

        x = self.up128(x)
        x = self.fuse128(torch.cat([x, f3], dim=1))

        x = self.up256(x)
        x = self.fuse256(torch.cat([x, f2], dim=1))

        x = self.up512(x)
        x = self.fuse512(torch.cat([x, f1], dim=1))

        return self.out(x)


class DualGatedBoundaryAdapter(nn.Module):
    """
    比旧 alpha_gate 更积极的边界注入：
      channel gate + spatial gate + zero conv residual
    """
    def __init__(self, feat_ch: int = 256):
        super().__init__()
        self.edge = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feat_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
        )
        self.channel_gate = nn.Parameter(torch.zeros(feat_ch, 1, 1))
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(feat_ch, 1, 1),
            nn.Sigmoid(),
        )
        self.zero_conv = nn.Conv2d(feat_ch, feat_ch, 1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, feat: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        e = self.edge(rgb)
        sg = self.spatial_gate(e)
        residual = self.zero_conv(e * sg) * self.channel_gate
        return feat + residual


class MSRWithBGA(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = DualGatedBoundaryAdapter()
        self.decoder = MSRDecoder()

    def forward(self, feat: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        fused = self.adapter(feat, rgb)
        return self.decoder(fused, rgb)


def build_variant(variant: str) -> nn.Module:
    if variant == "lora_light":
        return LightweightMaskDecoder()
    if variant == "msr":
        return MSRDecoder()
    if variant == "msr_bga":
        return MSRWithBGA()
    raise ValueError(f"未知 variant: {variant}")


# =============================================================================
# 7. 训练和评估
# =============================================================================

def collect_trainable_params(model: nn.Module, extractor: SAM3Extractor) -> List[nn.Parameter]:
    params = list(model.parameters())
    for p in extractor.model.parameters():
        if p.requires_grad:
            params.append(p)
    return params


@torch.no_grad()
def predict_probs(
    model: nn.Module,
    extractor: SAM3Extractor,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_gts = [], []

    for batch in tqdm(loader, desc="  predict", ncols=100):
        imgs = batch["image"]
        masks = batch["mask"].numpy()[:, 0]

        feat = extractor.extract_eval_batch(imgs)
        rgb = imgs.to(DEVICE)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            logits = model(feat.to(DEVICE), rgb)

        probs = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]

        all_probs.append(probs)
        all_gts.append(masks)

    return np.concatenate(all_probs, axis=0), np.concatenate(all_gts, axis=0)


def evaluate_with_threshold_sweep(
    probs: np.ndarray,
    gts: np.ndarray,
    thresholds: List[float],
    min_areas: List[int],
) -> Dict:
    best = None
    all_rows = []

    for th in thresholds:
        for area in min_areas:
            preds = []
            for p in probs:
                pred = (p > th).astype(np.float32)
                pred = remove_small_components(pred, area)
                preds.append(pred)

            preds_arr = np.stack(preds, axis=0)
            m = evaluate_arrays(preds_arr, gts)
            row = {
                "threshold": th,
                "min_area": area,
                **m,
            }
            all_rows.append(row)

            score = m["mIoU"]
            if best is None or score > best["mIoU"]:
                best = row

    return {
        "best": best,
        "all": all_rows,
    }


def evaluate_fixed(
    probs: np.ndarray,
    gts: np.ndarray,
    threshold: float,
    min_area: int,
) -> Dict:
    preds = []
    for p in probs:
        pred = (p > threshold).astype(np.float32)
        pred = remove_small_components(pred, min_area)
        preds.append(pred)
    preds = np.stack(preds, axis=0)
    return evaluate_arrays(preds, gts)


def train_one_variant(
    variant: str,
    project_root: Path,
    checkpoint_path: Path,
    train_loader: DataLoader,
    val_loader_for_ckpt: DataLoader,
    val_loader_full: DataLoader,
    pilot_loader: DataLoader,
    epochs: int,
    lr: float,
    val_every: int,
    out_dir: Path,
) -> Dict:
    print("\n" + "=" * 90)
    print(f"Training variant: {variant}")
    print("=" * 90)

    extractor = SAM3Extractor(checkpoint_path, DEVICE)
    model = build_variant(variant).to(DEVICE)

    optimizer = torch.optim.AdamW(collect_trainable_params(model, extractor), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None

    best_val = -1.0
    best_path = out_dir / "weights" / f"e6v2_pilot_{variant}_best.pth"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"  {variant} epoch {epoch}/{epochs}", ncols=100)

        for batch in pbar:
            imgs = batch["image"]
            masks = batch["mask"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            # 少样本 pilot，为节省显存，逐张提 SAM 特征
            batch_logits = []
            for i in range(imgs.shape[0]):
                feat = extractor.extract_with_grad(imgs[i])
                rgb = imgs[i].unsqueeze(0).to(DEVICE)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                    logit = model(feat.to(DEVICE), rgb)

                batch_logits.append(logit)

            logits = torch.cat(batch_logits, dim=0)

            loss = bce(logits.float(), masks.float()) + dice(logits.float(), masks.float())

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss /= max(1, len(train_loader))

        do_val = (epoch % val_every == 0) or (epoch == epochs)

        val_metric = None
        if do_val:
            val_probs, val_gts = predict_probs(model, extractor, val_loader_for_ckpt)
            val_fixed = evaluate_fixed(val_probs, val_gts, threshold=0.5, min_area=0)
            val_metric = val_fixed["mIoU"]

            print(f"  epoch={epoch} loss={epoch_loss:.4f} val_mIoU@0.5={val_metric:.4f}")

            if val_metric > best_val:
                best_val = val_metric
                torch.save({
                    "variant": variant,
                    "model": model.state_dict(),
                    "lora_params": {
                        k: v.detach().cpu()
                        for k, v in extractor.model.state_dict().items()
                        if "lora" in k.lower()
                    },
                    "epoch": epoch,
                    "best_val_mIoU_05": best_val,
                }, best_path)
                print(f"  ★ saved best: {best_path}")

        history.append({
            "epoch": epoch,
            "loss": epoch_loss,
            "val_mIoU_05": val_metric,
        })

        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # reload best
    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)

    model_state = extractor.model.state_dict()
    for k, v in ckpt["lora_params"].items():
        if k in model_state:
            model_state[k].copy_(v.to(DEVICE))

    # full val threshold sweep
    print("\n[Val full] threshold sweep ...")
    val_probs, val_gts = predict_probs(model, extractor, val_loader_full)
    sweep = evaluate_with_threshold_sweep(
        val_probs,
        val_gts,
        thresholds=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
        min_areas=[0, 16, 32, 64, 128],
    )
    best_th = float(sweep["best"]["threshold"])
    best_area = int(sweep["best"]["min_area"])

    print(f"  best threshold={best_th}, min_area={best_area}, val_mIoU={sweep['best']['mIoU']:.4f}")

    # pilot eval
    print("\n[Pilot500] evaluation ...")
    pilot_probs, pilot_gts = predict_probs(model, extractor, pilot_loader)
    pilot_05 = evaluate_fixed(pilot_probs, pilot_gts, threshold=0.5, min_area=0)
    pilot_calib = evaluate_fixed(pilot_probs, pilot_gts, threshold=best_th, min_area=best_area)

    result = {
        "variant": variant,
        "best_weight": str(best_path),
        "history": history,
        "val_sweep_best": sweep["best"],
        "pilot_fixed_05": pilot_05,
        "pilot_calibrated": pilot_calib,
        "timestamp": datetime.now().isoformat(),
    }

    result_path = out_dir / f"result_{variant}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    del model, extractor
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return result


# =============================================================================
# 8. CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser("E6-v2 pilot runner")

    parser.add_argument("--project_root", type=str, default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--manifest_dir", type=str, default=None)
    parser.add_argument("--sam3_checkpoint", type=str, default=None)

    parser.add_argument("--variants", type=str, default="lora_light,msr,msr_bga")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shot", type=int, default=20)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--val_eval_limit", type=int, default=200)
    parser.add_argument("--pilot_eval_limit", type=int, default=500)

    parser.add_argument("--out_dir", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    project_root = Path(args.project_root)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else project_root / "data" / "splits" / "e0_manifest"

    checkpoint_path = Path(args.sam3_checkpoint) if args.sam3_checkpoint else project_root / "weights" / "sam3.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM3 checkpoint 不存在: {checkpoint_path}")

    out_dir = Path(args.out_dir) if args.out_dir else project_root / "results" / "e6v2_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = manifest_dir / f"target_support_{args.shot}_seed{args.seed}.txt"
    val_manifest = manifest_dir / "target_val.txt"
    pilot_manifest = manifest_dir / "target_pilot_test_500.txt"

    print("=" * 90)
    print("E6-v2 Pilot")
    print("=" * 90)
    print(f"project_root : {project_root}")
    print(f"manifest_dir : {manifest_dir}")
    print(f"checkpoint   : {checkpoint_path}")
    print(f"train        : {train_manifest.name}")
    print(f"val          : {val_manifest.name}")
    print(f"pilot        : {pilot_manifest.name}")
    print(f"variants     : {args.variants}")
    print(f"device       : {DEVICE}")
    print("=" * 90)

    train_ds = ManifestSegDataset(train_manifest, limit=None)
    val_ckpt_ds = ManifestSegDataset(val_manifest, limit=args.val_eval_limit)
    val_full_ds = ManifestSegDataset(val_manifest, limit=None)
    pilot_ds = ManifestSegDataset(pilot_manifest, limit=args.pilot_eval_limit)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_ckpt_loader = DataLoader(
        val_ckpt_ds,
        batch_size=4,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_full_loader = DataLoader(
        val_full_ds,
        batch_size=4,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    pilot_loader = DataLoader(
        pilot_ds,
        batch_size=4,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    all_results = []

    for v in variants:
        r = train_one_variant(
            variant=v,
            project_root=project_root,
            checkpoint_path=checkpoint_path,
            train_loader=train_loader,
            val_loader_for_ckpt=val_ckpt_loader,
            val_loader_full=val_full_loader,
            pilot_loader=pilot_loader,
            epochs=args.epochs,
            lr=args.lr,
            val_every=args.val_every,
            out_dir=out_dir,
        )
        all_results.append(r)

    summary = {
        "config": vars(args),
        "results": all_results,
    }

    summary_path = out_dir / "e6v2_pilot_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("Pilot summary")
    print("=" * 90)

    for r in all_results:
        v = r["variant"]
        fixed = r["pilot_fixed_05"]
        calib = r["pilot_calibrated"]
        print(
            f"{v:<12} "
            f"fixed mIoU={fixed['mIoU']*100:.2f} F1={fixed['F1']*100:.2f} BIoU={fixed['Boundary_IoU']*100:.2f} | "
            f"calib mIoU={calib['mIoU']*100:.2f} F1={calib['F1']*100:.2f} BIoU={calib['Boundary_IoU']*100:.2f}"
        )

    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()