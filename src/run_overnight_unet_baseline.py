# -*- coding: utf-8 -*-
"""
run_overnight_unet_baseline.py
=============================================================================
Purpose
  Overnight supplementary experiment for the RS manuscript:
  train a conventional target-adapted Compact-ResUNet baseline under the same
  WHU-Mix 20-shot protocol, validation calibration, TTA, and final-test setting.

Why this experiment is worth running
  The current manuscript mainly compares HL-Residual against a SAM-family LoRA
  prompt-free baseline. A reviewer may ask whether a conventional segmentation
  model, when also adapted to the 20-shot target support set, has been compared.
  This script fills that gap with a reproducible, non-SAM baseline.

Default behavior
  1) Optionally source-pretrain Compact-ResUNet on source_train manifest if found.
  2) Fine-tune/evaluate target support seeds 42/123/456.
  3) Select checkpoints using target_val only.
  4) Calibrate threshold/post-processing on target_val only.
  5) Evaluate target_final_test_8402 with optional TTA.
  6) Save JSON/CSV/LaTeX summaries for direct insertion into the paper.

Typical command
  cd <project_root>
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python src/run_overnight_unet_baseline.py \
    --shot 20 \
    --seeds 42,123,456 \
    --source_epochs 12 \
    --target_epochs 120 \
    --batch_size 8 \
    --target_batch_size 4 \
    --use_tta \
    --val_eval_limit 200 \
    --final_val_eval_limit 500 \
    --test_eval_limit 0 \
    --out_dir results/overnight_unet20_baseline
=============================================================================
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_SIZE = 512
LABEL_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_manifest(path: Path, limit: Optional[int] = None) -> List[Path]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    lines: List[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().replace("\\", "/")
        if line:
            lines.append(Path(line))
    if limit is not None and limit > 0:
        lines = lines[:limit]
    if not lines:
        raise RuntimeError(f"empty manifest: {path}")
    return lines


def pick_manifest(manifest_dir: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        p = manifest_dir / name
        if p.exists():
            return p
    return None


def find_dual_label_for_image(image_path: Path) -> Optional[Path]:
    """Robust label resolver compatible with the user's processed_slim structure."""
    stem = image_path.stem
    s = str(image_path).replace("\\", "/")

    candidate_dirs: List[Path] = []
    replacement_pairs = [
        ("/images/", "/dual_channel_labels/"),
        ("/image/", "/dual_channel_labels/"),
        ("/imgs/", "/dual_channel_labels/"),
        ("/test/image/", "/dual_channel_labels/"),
        ("/train/image/", "/dual_channel_labels/"),
        ("/val/image/", "/dual_channel_labels/"),
        ("/images/", "/label/"),
        ("/image/", "/label/"),
        ("/images/", "/labels/"),
        ("/image/", "/labels/"),
        ("/images/", "/mask/"),
        ("/image/", "/mask/"),
        ("/images/", "/masks/"),
        ("/image/", "/masks/"),
    ]
    for old, new in replacement_pairs:
        if old in s:
            candidate_dirs.append(Path(s.replace(old, new)).parent)

    parent = image_path.parent
    for dname in ["dual_channel_labels", "label", "labels", "mask", "masks", "gt", "annotation", "annotations"]:
        candidate_dirs.append(parent.parent / dname)
        candidate_dirs.append(parent / dname)

    seen = set()
    uniq: List[Path] = []
    for d in candidate_dirs:
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


# =============================================================================
# Dataset and augmentation
# =============================================================================

class ManifestSegDataset(Dataset):
    def __init__(self, manifest_path: Path, limit: Optional[int] = None, train_aug: bool = False):
        self.manifest_path = manifest_path
        self.paths = read_manifest(manifest_path, limit=limit)
        self.train_aug = train_aug
        print(f"[Dataset] {manifest_path.name}: N={len(self.paths)} train_aug={train_aug}", flush=True)

    def __len__(self):
        return len(self.paths)

    def _augment(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # geometric transforms shared by image and mask
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[::-1, :, :])
            mask = np.ascontiguousarray(mask[::-1, :])
        k = random.randint(0, 3)
        if k:
            img = np.ascontiguousarray(np.rot90(img, k))
            mask = np.ascontiguousarray(np.rot90(mask, k))

        # mild photometric augmentation for image only
        if random.random() < 0.8:
            alpha = random.uniform(0.80, 1.20)  # contrast
            beta = random.uniform(-0.08, 0.08)  # brightness, normalized scale
            img_f = img.astype(np.float32) / 255.0
            img_f = np.clip(img_f * alpha + beta, 0.0, 1.0)
            img = (img_f * 255.0 + 0.5).astype(np.uint8)

        if random.random() < 0.25:
            # small cutout, image only; avoids changing labels under sparse supervision
            h, w = mask.shape
            area = h * w
            target = random.uniform(0.01, 0.06) * area
            ratio = random.uniform(0.5, 2.0)
            eh = int(round(math.sqrt(target * ratio)))
            ew = int(round(math.sqrt(target / ratio)))
            if 2 <= eh < h and 2 <= ew < w:
                y = random.randint(0, h - eh)
                x = random.randint(0, w - ew)
                img[y:y + eh, x:x + ew, :] = 127
        return img, mask

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"cannot read image: {img_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)

        lab_path = find_dual_label_for_image(img_path)
        if lab_path is None:
            raise FileNotFoundError(f"missing label for image: {img_path}")
        lab = cv2.imread(str(lab_path), cv2.IMREAD_UNCHANGED)
        if lab is None:
            raise FileNotFoundError(f"cannot read label: {lab_path}")
        if lab.ndim == 2:
            mask = cv2.resize(lab, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.float32)
        else:
            lab = cv2.resize(lab, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)
            mask = (lab[:, :, 0] > 0).astype(np.float32)

        if self.train_aug:
            img, mask = self._augment(img, mask)

        img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).float()
        return {"image": img_t, "mask": mask_t, "name": img_path.stem, "path": str(img_path)}


def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "name": [b["name"] for b in batch],
        "path": [b["path"] for b in batch],
    }


# =============================================================================
# Model: compact ResUNet baseline
# =============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class CompactResUNet(nn.Module):
    def __init__(self, base_ch: int = 32):
        super().__init__()
        c = base_ch
        self.e1 = ResidualBlock(3, c)
        self.e2 = ResidualBlock(c, c * 2)
        self.e3 = ResidualBlock(c * 2, c * 4)
        self.e4 = ResidualBlock(c * 4, c * 8)
        self.bottleneck = ResidualBlock(c * 8, c * 16)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.d4 = ResidualBlock(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.d3 = ResidualBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.d2 = ResidualBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.d1 = ResidualBlock(c * 2, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        x = self.up4(b)
        x = self.d4(torch.cat([x, e4], dim=1))
        x = self.up3(x)
        x = self.d3(torch.cat([x, e3], dim=1))
        x = self.up2(x)
        x = self.d2(torch.cat([x, e2], dim=1))
        x = self.up1(x)
        x = self.d1(torch.cat([x, e1], dim=1))
        return self.out(x)


# =============================================================================
# Loss and metrics
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


def boundary_map(mask: np.ndarray, d: int = 5) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask.astype(bool)
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    b = mask - eroded
    if d > 1:
        b = cv2.dilate(b, np.ones((d, d), np.uint8), iterations=1)
    return b.astype(bool)


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + 1e-7) / (union + 1e-7))


def compute_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return float((2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7))


def compute_biou(pred: np.ndarray, gt: np.ndarray, d: int = 5) -> float:
    pb = boundary_map(pred, d=d)
    gb = boundary_map(gt, d=d)
    inter = np.logical_and(pb, gb).sum()
    union = np.logical_or(pb, gb).sum()
    return float((inter + 1e-7) / (union + 1e-7))


def eval_arrays(preds: np.ndarray, gts: np.ndarray) -> Dict[str, float]:
    ious, f1s, bious = [], [], []
    for p, g in zip(preds, gts):
        ious.append(compute_iou(p, g))
        f1s.append(compute_f1(p, g))
        bious.append(compute_biou(p, g, d=5))
    return {"mIoU": float(np.mean(ious)), "F1": float(np.mean(f1s)), "Boundary_IoU": float(np.mean(bious))}


# =============================================================================
# Post-processing and evaluation
# =============================================================================

def fill_holes(binary: np.ndarray) -> np.ndarray:
    binary = (binary > 0).astype(np.uint8)
    h, w = binary.shape
    flood = binary.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 1)
    flood_inv = 1 - flood
    return np.logical_or(binary, flood_inv).astype(np.uint8)


def postprocess_mask(prob: np.ndarray, threshold: float, min_area: int, closing_kernel: int, fill_hole: bool) -> np.ndarray:
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


def grid_search_postprocess(probs: np.ndarray, gts: np.ndarray, grid_mode: str = "full") -> Dict:
    if grid_mode == "fast":
        thresholds = [0.45, 0.50, 0.55, 0.60, 0.65]
        min_areas = [0, 64, 128]
        closing_kernels = [0, 3]
        fill_holes_options = [False, True]
    else:
        thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        min_areas = [0, 16, 32, 64, 128]
        closing_kernels = [0, 3, 5]
        fill_holes_options = [False, True]

    best = None
    rows = []
    for th in thresholds:
        for ma in min_areas:
            for ck in closing_kernels:
                for fh in fill_holes_options:
                    preds = np.stack([postprocess_mask(p, th, ma, ck, fh) for p in probs])
                    m = eval_arrays(preds, gts)
                    row = {"threshold": th, "min_area": ma, "closing_kernel": ck, "fill_holes": fh, **m}
                    rows.append(row)
                    if best is None or row["mIoU"] > best["mIoU"]:
                        best = row
    return {"best": best, "all": rows}


def eval_with_config(probs: np.ndarray, gts: np.ndarray, cfg: Dict) -> Dict[str, float]:
    preds = np.stack([
        postprocess_mask(p, cfg["threshold"], cfg["min_area"], cfg["closing_kernel"], cfg["fill_holes"])
        for p in probs
    ])
    return eval_arrays(preds, gts)


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, use_tta: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    probs_all, gts_all, names_all = [], [], []
    for batch in tqdm(loader, desc="predict", ncols=100):
        imgs = batch["image"].to(DEVICE, non_blocking=True)
        gts = batch["mask"].numpy()[:, 0]
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            if use_tta:
                imgs_h = torch.flip(imgs, dims=[3])
                p_h = torch.sigmoid(model(imgs_h))
                p_h = torch.flip(p_h, dims=[3])
                imgs_v = torch.flip(imgs, dims=[2])
                p_v = torch.sigmoid(model(imgs_v))
                p_v = torch.flip(p_v, dims=[2])
                probs = (probs + p_h + p_v) / 3.0
        probs_all.append(probs.detach().cpu().float().numpy()[:, 0])
        gts_all.append(gts)
        names_all.extend(batch["name"])
    return np.concatenate(probs_all, axis=0), np.concatenate(gts_all, axis=0), names_all


def make_loader(manifest: Path, batch_size: int, num_workers: int, train_aug: bool, limit: Optional[int] = None, shuffle: bool = False) -> DataLoader:
    ds = ManifestSegDataset(manifest, limit=limit, train_aug=train_aug)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )


# =============================================================================
# Training
# =============================================================================

def train_one_epoch(model, loader, optimizer, scaler, bce_loss, dice_loss, epoch: int, max_grad_norm: float = 1.0) -> float:
    model.train()
    losses = []
    pbar = tqdm(loader, desc=f"train epoch {epoch}", ncols=100)
    for batch in pbar:
        imgs = batch["image"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
            logits = model(imgs)
            loss = bce_loss(logits, masks) + dice_loss(logits, masks)
        scaler.scale(loss).backward()
        if max_grad_norm and max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
        pbar.set_postfix(loss=np.mean(losses[-20:]))
    return float(np.mean(losses))


def quick_val_score(model, val_loader, use_tta: bool, grid_mode: str) -> Dict:
    probs, gts, _ = predict_probs(model, val_loader, use_tta=use_tta)
    return grid_search_postprocess(probs, gts, grid_mode=grid_mode)["best"]


def save_ckpt(path: Path, model: nn.Module, meta: Dict):
    ensure_dir(path.parent)
    torch.save({"model": model.state_dict(), "meta": meta}, path)


def load_ckpt(path: Path, model: nn.Module):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    return ckpt.get("meta", {})


def source_pretrain(args, source_train: Path, source_val: Optional[Path], out_dir: Path) -> Optional[Path]:
    if args.source_epochs <= 0:
        print("[Source] source_epochs <= 0, skip source pretraining.")
        return None

    print("\n========== Stage A: source pretraining ==========")
    set_seed(args.source_seed)
    model = CompactResUNet(base_ch=args.base_ch).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.source_lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    train_loader = make_loader(source_train, args.batch_size, args.num_workers, train_aug=True, shuffle=True)
    val_loader = None
    if source_val is not None:
        val_loader = make_loader(source_val, args.batch_size, args.num_workers, train_aug=False, limit=args.val_eval_limit)

    best_metric = -1.0
    best_path = out_dir / "weights" / "compact_resunet_source_pretrained_best.pth"
    last_path = out_dir / "weights" / "compact_resunet_source_pretrained_last.pth"
    history = []

    for ep in range(1, args.source_epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scaler, bce_loss, dice_loss, ep)
        row = {"epoch": ep, "loss": loss}
        if val_loader is not None and (ep == args.source_epochs or ep % max(1, args.source_val_every) == 0):
            best_cfg = quick_val_score(model, val_loader, use_tta=False, grid_mode="fast")
            row.update({f"val_{k}": v for k, v in best_cfg.items() if k in ["mIoU", "F1", "Boundary_IoU"]})
            metric = best_cfg["mIoU"]
            if metric > best_metric:
                best_metric = metric
                save_ckpt(best_path, model, {"stage": "source_pretrain", "epoch": ep, "val_best_cfg": best_cfg})
                print(f"[Source] new best epoch={ep} mIoU={metric:.4f}", flush=True)
        history.append(row)

    save_ckpt(last_path, model, {"stage": "source_pretrain", "epoch": args.source_epochs})
    hist_path = out_dir / "source_pretrain_history.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    if best_path.exists():
        return best_path
    return last_path


def train_target_seed(args, seed: int, support_manifest: Path, val_manifest: Path, test_manifest: Path, init_ckpt: Optional[Path], out_dir: Path) -> Dict:
    print(f"\n========== Stage B/C: target fine-tune seed={seed} ==========")
    set_seed(seed)
    seed_dir = ensure_dir(out_dir / f"seed{seed}")
    model = CompactResUNet(base_ch=args.base_ch).to(DEVICE)
    init_meta = {}
    if init_ckpt is not None and init_ckpt.exists():
        init_meta = load_ckpt(init_ckpt, model)
        print(f"[Seed {seed}] initialized from {init_ckpt}")
    else:
        print(f"[Seed {seed}] no source checkpoint; train from scratch on target support.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.target_lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    train_loader = make_loader(support_manifest, args.target_batch_size, args.num_workers, train_aug=True, shuffle=True)
    val_select_loader = make_loader(val_manifest, args.batch_size, args.num_workers, train_aug=False, limit=args.val_eval_limit)

    best_metric = -1.0
    best_path = seed_dir / "compact_resunet_target_best.pth"
    history = []

    for ep in range(1, args.target_epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scaler, bce_loss, dice_loss, ep)
        row = {"epoch": ep, "loss": loss}
        if ep == args.target_epochs or ep % max(1, args.target_val_every) == 0:
            best_cfg = quick_val_score(model, val_select_loader, use_tta=args.val_tta_for_selection, grid_mode="fast")
            row.update({f"val_{k}": v for k, v in best_cfg.items() if k in ["mIoU", "F1", "Boundary_IoU"]})
            row["val_threshold"] = best_cfg["threshold"]
            metric = best_cfg["mIoU"]
            if metric > best_metric:
                best_metric = metric
                save_ckpt(best_path, model, {
                    "stage": "target_finetune",
                    "seed": seed,
                    "epoch": ep,
                    "support_manifest": str(support_manifest),
                    "init_ckpt": str(init_ckpt) if init_ckpt else None,
                    "val_select_cfg": best_cfg,
                    "init_meta": init_meta,
                })
                print(f"[Seed {seed}] new best epoch={ep} val_mIoU={metric:.4f}", flush=True)
        history.append(row)

    (seed_dir / "target_train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # final calibration on full target_val (or requested final_val_eval_limit)
    load_ckpt(best_path, model)
    final_val_loader = make_loader(val_manifest, args.batch_size, args.num_workers, train_aug=False, limit=args.final_val_eval_limit)
    val_probs, val_gts, _ = predict_probs(model, final_val_loader, use_tta=args.use_tta)
    val_sweep = grid_search_postprocess(val_probs, val_gts, grid_mode="full")
    val_best_cfg = val_sweep["best"]

    # final test evaluation
    test_limit = None if args.test_eval_limit is None or args.test_eval_limit <= 0 else args.test_eval_limit
    test_loader = make_loader(test_manifest, args.batch_size, args.num_workers, train_aug=False, limit=test_limit)
    test_probs, test_gts, test_names = predict_probs(model, test_loader, use_tta=args.use_tta)

    fixed_cfg = {"threshold": 0.50, "min_area": 0, "closing_kernel": 0, "fill_holes": False}
    test_fixed_05 = eval_with_config(test_probs, test_gts, fixed_cfg)
    test_calibrated = eval_with_config(test_probs, test_gts, val_best_cfg)

    # per-image calibrated metrics for optional significance checks
    per_image_path = seed_dir / "compact_resunet_per_image_metrics.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "seed", "method", "mIoU", "F1", "Boundary_IoU"])
        writer.writeheader()
        for name, p, g in zip(test_names, test_probs, test_gts):
            pred = postprocess_mask(p, val_best_cfg["threshold"], val_best_cfg["min_area"], val_best_cfg["closing_kernel"], val_best_cfg["fill_holes"])
            writer.writerow({
                "image_name": name,
                "seed": seed,
                "method": "Compact-ResUNet target-adapted",
                "mIoU": compute_iou(pred, g),
                "F1": compute_f1(pred, g),
                "Boundary_IoU": compute_biou(pred, g, d=5),
            })

    result = {
        "seed": seed,
        "method": "Compact-ResUNet target-adapted",
        "support_manifest": str(support_manifest),
        "init_ckpt": str(init_ckpt) if init_ckpt else None,
        "use_tta": bool(args.use_tta),
        "val_best_cfg": val_best_cfg,
        "test_fixed_05": test_fixed_05,
        "test_calibrated_postprocess": test_calibrated,
        "best_ckpt": str(best_path),
        "per_image_csv": str(per_image_path),
    }
    (seed_dir / "compact_resunet_seed_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[Seed {seed}] FINAL calibrated: {test_calibrated}", flush=True)
    return result


# =============================================================================
# Summary writers
# =============================================================================

def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def write_summaries(results: List[Dict], out_dir: Path):
    summary_json = out_dir / "overnight_compact_resunet_summary.json"
    summary_json.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")

    csv_path = out_dir / "overnight_compact_resunet_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "seed", "method", "use_tta",
            "val_threshold", "val_min_area", "val_closing_kernel", "val_fill_holes",
            "mIoU", "F1", "Boundary_IoU",
            "fixed_mIoU", "fixed_F1", "fixed_Boundary_IoU",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            cfg = r["val_best_cfg"]
            cal = r["test_calibrated_postprocess"]
            fix = r["test_fixed_05"]
            writer.writerow({
                "seed": r["seed"],
                "method": r["method"],
                "use_tta": r["use_tta"],
                "val_threshold": cfg["threshold"],
                "val_min_area": cfg["min_area"],
                "val_closing_kernel": cfg["closing_kernel"],
                "val_fill_holes": cfg["fill_holes"],
                "mIoU": cal["mIoU"] * 100,
                "F1": cal["F1"] * 100,
                "Boundary_IoU": cal["Boundary_IoU"] * 100,
                "fixed_mIoU": fix["mIoU"] * 100,
                "fixed_F1": fix["F1"] * 100,
                "fixed_Boundary_IoU": fix["Boundary_IoU"] * 100,
            })

    miou_m, miou_s = mean_std([r["test_calibrated_postprocess"]["mIoU"] * 100 for r in results])
    f1_m, f1_s = mean_std([r["test_calibrated_postprocess"]["F1"] * 100 for r in results])
    biou_m, biou_s = mean_std([r["test_calibrated_postprocess"]["Boundary_IoU"] * 100 for r in results])

    tex_path = out_dir / "overnight_compact_resunet_latex_table.txt"
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Target-adapted conventional baseline under the same 20-shot WHU-Mix protocol. The Compact-ResUNet baseline is source-pretrained when the source manifest is available and then fine-tuned on each 20-shot target support set. Threshold and post-processing parameters are selected only on target validation images and then frozen for final-test evaluation.}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & mIoU (\%) & F1 (\%) & BIoU (\%) \\")
    lines.append(r"\midrule")
    lines.append(f"Compact-ResUNet target-adapted & {miou_m:.2f}$\\pm${miou_s:.2f} & {f1_m:.2f}$\\pm${f1_s:.2f} & {biou_m:.2f}$\\pm${biou_s:.2f} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    tex_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n========== Summary ==========")
    print(f"CSV:   {csv_path}")
    print(f"JSON:  {summary_json}")
    print(f"LaTeX: {tex_path}")
    print(f"Mean calibrated mIoU/F1/BIoU = {miou_m:.2f}±{miou_s:.2f} / {f1_m:.2f}±{f1_s:.2f} / {biou_m:.2f}±{biou_s:.2f}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--manifest_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="results/overnight_unet20_baseline")

    parser.add_argument("--shot", type=int, default=20)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--test_manifest_name", type=str, default="target_final_test_8402.txt")
    parser.add_argument("--test_eval_limit", type=int, default=0, help="0 means full test")
    parser.add_argument("--val_eval_limit", type=int, default=200, help="fast checkpoint selection subset")
    parser.add_argument("--final_val_eval_limit", type=int, default=500, help="final calibration set size; 0 means all")

    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--source_epochs", type=int, default=12)
    parser.add_argument("--target_epochs", type=int, default=120)
    parser.add_argument("--source_val_every", type=int, default=3)
    parser.add_argument("--target_val_every", type=int, default=10)
    parser.add_argument("--source_lr", type=float, default=3e-4)
    parser.add_argument("--target_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--target_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--source_seed", type=int, default=2026)

    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--val_tta_for_selection", action="store_true", help="slower; usually keep false")
    parser.add_argument("--skip_source_pretrain", action="store_true")
    parser.add_argument("--source_ckpt", type=str, default=None, help="optional existing CompactResUNet source checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    project_root = Path(args.project_root)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else project_root / "data" / "splits" / "e0_manifest"
    out_dir = ensure_dir(project_root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    ensure_dir(out_dir / "weights")

    print("========== Overnight Compact-ResUNet target-adapted baseline ==========")
    print(f"DEVICE={DEVICE}")
    print(f"project_root={project_root}")
    print(f"manifest_dir={manifest_dir}")
    print(f"out_dir={out_dir}")
    print(f"args={vars(args)}")

    val_manifest = manifest_dir / "target_val.txt"
    test_manifest = manifest_dir / args.test_manifest_name
    if not val_manifest.exists():
        raise FileNotFoundError(f"target_val manifest not found: {val_manifest}")
    if not test_manifest.exists():
        raise FileNotFoundError(f"test manifest not found: {test_manifest}")

    source_train = pick_manifest(manifest_dir, [
        "source_train.txt", "source_whu_train.txt", "whu_source_train.txt",
        "source_train_1000.txt", "source_whu_train_1000.txt",
    ])
    source_val = pick_manifest(manifest_dir, [
        "source_val.txt", "source_whu_val.txt", "whu_source_val.txt",
        "source_val_200.txt", "source_whu_val_200.txt",
    ])

    # Source pretraining is optional but useful if manifests exist.
    init_ckpt: Optional[Path] = None
    if args.source_ckpt:
        init_ckpt = Path(args.source_ckpt)
        if not init_ckpt.is_absolute():
            init_ckpt = project_root / init_ckpt
        if not init_ckpt.exists():
            raise FileNotFoundError(f"source_ckpt not found: {init_ckpt}")
    elif not args.skip_source_pretrain and source_train is not None:
        init_ckpt = source_pretrain(args, source_train, source_val, out_dir)
    else:
        print("[Source] No source pretraining will be used.")
        if source_train is None:
            print("[Source] Source train manifest not found. This is acceptable; target-only baseline will still run.")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    results = []
    for seed in seeds:
        support = manifest_dir / f"target_support_{args.shot}_seed{seed}.txt"
        if not support.exists():
            print(f"[WARN] missing support manifest for seed {seed}: {support}; skip", flush=True)
            continue
        try:
            r = train_target_seed(args, seed, support, val_manifest, test_manifest, init_ckpt, out_dir)
            results.append(r)
        except Exception as e:
            err_path = out_dir / f"error_seed{seed}.txt"
            err_path.write_text(repr(e), encoding="utf-8")
            print(f"[ERROR] seed={seed}: {e}", flush=True)
            raise

    if not results:
        raise RuntimeError("No seed finished successfully.")
    write_summaries(results, out_dir)
    elapsed_h = (time.time() - t0) / 3600.0
    print(f"Done. elapsed={elapsed_h:.2f} hours")


if __name__ == "__main__":
    main()
