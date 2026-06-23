# -*- coding: utf-8 -*-
"""
run_e7a_prompt_drift.py
=============================================================================
E7a: Prompt Drift / Box Jitter Robustness

目的：
  评估 box-prompt 方法在 GT bbox 被高斯扰动时的性能退化；
  同时验证 prompt-free LoRA / HL-Residual 不依赖 bbox，因此对 box prompt drift 免疫。

默认评估：
  test: target_pilot_test_500.txt
  seeds: 42,123,456
  sigmas: 0,2,5,10,20
  models:
    1. Box-prompt LoRA
    2. Prompt-free LoRA-light
    3. HL-Residual (Ours)

运行示例：
  cd <project_root>

  nohup bash -lc '
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  python -u src/run_e7a_prompt_drift.py \
    --use_tta \
    --test_manifest_name target_pilot_test_500.txt \
    --test_eval_limit 0 \
    --sigmas 0,2,5,10,20 \
    --seeds 42,123,456 \
    --out_dir results/e7a_prompt_drift_pilot500
  ' > logs/e7a_prompt_drift_pilot500.nohup.log 2>&1 &

  tail -f logs/e7a_prompt_drift_pilot500.nohup.log
=============================================================================
"""

import argparse
import gc
import json
import random
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
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
from sam3.model.sam3_image_processor import Sam3Processor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

TARGET_SIZE = 512
SAM3_INPUT_SIZE = 1008
FEATURE_SIZE = (64, 64)
FEATURE_CHANNELS = 256
BOUNDARY_D = 5

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
SAM3_CONFIDENCE_THRESHOLD = 0.20

LABEL_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


# =============================================================================
# 基础工具
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def read_manifest(path: Path, limit: Optional[int] = None) -> List[Path]:
    if not path.exists():
        raise FileNotFoundError(f"manifest 不存在: {path}")

    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().replace("\\", "/")
        if line:
            out.append(Path(line))

    if limit is not None and limit > 0:
        out = out[:limit]

    return out


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
        k = str(d).replace("\\", "/").lower()
        if k not in seen:
            seen.add(k)
            uniq.append(d)

    for d in uniq:
        if not d.exists():
            continue
        for ext in LABEL_EXTS:
            p = d / f"{stem}{ext}"
            if p.exists():
                return p

    return None


def load_bboxes(path: Path) -> Dict[str, List[List[int]]]:
    if not path.exists():
        print(f"[Warning] bbox json 不存在: {path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def lookup_bboxes(bboxes_all: Dict[str, List[List[int]]], stem: str) -> List[List[int]]:
    candidates = [
        stem,
        f"{stem}.tif",
        f"{stem}.tiff",
        f"{stem}.png",
        f"{stem}.jpg",
    ]
    for k in candidates:
        if k in bboxes_all:
            return bboxes_all[k]
    return []


# =============================================================================
# Dataset
# =============================================================================

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

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(img_path)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

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
            "path": str(img_path),
        }


def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "name": [b["name"] for b in batch],
        "path": [b["path"] for b in batch],
    }


def make_loader(ds, batch_size=4, num_workers=2):
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
        bious.append(compute_biou(p, g, d=BOUNDARY_D))
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
    return np.logical_or(binary, flood_inv).astype(np.uint8)


def postprocess_mask(prob, threshold, min_area, closing_kernel, fill_hole):
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


def eval_with_cfg(probs, gts, cfg):
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
    return eval_arrays(np.stack(preds), gts)


def grid_search_postprocess(probs, gts):
    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    min_areas = [0, 16, 32, 64, 128]
    closing_kernels = [0, 3, 5]
    fill_holes = [False, True]

    total = len(thresholds) * len(min_areas) * len(closing_kernels) * len(fill_holes)
    print(f"[GridSearch] configs={total}, N={len(probs)}", flush=True)

    best = None
    rows = []
    idx = 0

    for th in thresholds:
        for area in min_areas:
            for ck in closing_kernels:
                for fh in fill_holes:
                    idx += 1
                    if idx == 1 or idx % 20 == 0 or idx == total:
                        print(f"[GridSearch] {idx}/{total}", flush=True)

                    cfg = {
                        "threshold": th,
                        "min_area": area,
                        "closing_kernel": ck,
                        "fill_holes": fh,
                    }
                    m = eval_with_cfg(probs, gts, cfg)
                    row = {**cfg, **m}
                    rows.append(row)

                    if best is None or row["mIoU"] > best["mIoU"]:
                        best = row

    print(f"[GridSearch] best={best}", flush=True)
    return {"best": best, "all": rows}


# =============================================================================
# SAM3 LoRA / prompt-free models
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
    if n:
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

    def load_lora_params(self, lora_params: Dict):
        state = self.model.state_dict()
        loaded = 0
        for k, v in lora_params.items():
            if k in state:
                state[k].copy_(v.to(DEVICE))
                loaded += 1
        print(f"  loaded LoRA params: {loaded}")

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


def default_lora_ckpt(project_root: Path, seed: int) -> Path:
    if seed == 42:
        return project_root / "results/e6v2_pilot/weights/e6v2_pilot_lora_light_best.pth"
    return project_root / f"results/e6v3_hl_lite_seed{seed}/weights/e6v2_pilot_lora_light_best.pth"


def default_hlres_ckpt(project_root: Path, seed: int) -> Path:
    if seed == 42:
        return project_root / "results/hl_residual_pilot/weights/hl_residual_best.pth"
    return project_root / f"results/hl_residual_pilot_seed{seed}/weights/hl_residual_best.pth"


def load_promptfree_model(model_name: str, seed: int, project_root: Path, sam3_ckpt: Path):
    if model_name == "promptfree_lora":
        ckpt_path = default_lora_ckpt(project_root, seed)
    elif model_name == "hl_residual":
        ckpt_path = default_hlres_ckpt(project_root, seed)
    else:
        raise ValueError(model_name)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    extractor = SAM3Extractor(sam3_ckpt)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # --- hl_residual 旧格式兼容: 只有 residual，decoder+lora 在 base_lora_ckpt ---
    if model_name == "hl_residual" and "model" not in ckpt and "student" not in ckpt:
        decoder = LightweightMaskDecoder().to(DEVICE)

        base_lora_path = ckpt.get("base_lora_ckpt", None)
        if base_lora_path:
            base_lora_path = Path(base_lora_path)
        else:
            # fallback: 同种子的 lora_light ckpt
            base_lora_path = default_lora_ckpt(project_root, seed)

        if not base_lora_path.exists():
            raise FileNotFoundError(f"hl_residual 的 base_lora_ckpt 不存在: {base_lora_path}")

        base_ckpt = torch.load(base_lora_path, map_location=DEVICE, weights_only=False)
        if "model" in base_ckpt:
            decoder.load_state_dict(base_ckpt["model"], strict=True)
        if "lora_params" in base_ckpt:
            extractor.load_lora_params(base_ckpt["lora_params"])
        del base_ckpt

        model = HLResidualRefiner(decoder).to(DEVICE)
        model.residual.load_state_dict(ckpt["residual"], strict=True)
        model.eval()
        extractor.model.eval()

        print(f"  hl_residual: loaded from old-format checkpoint (seed={seed})")
        return extractor, model, ckpt_path

    # --- 正常路径: 有 model/student key ---
    decoder = LightweightMaskDecoder().to(DEVICE)

    if "model" in ckpt:
        decoder.load_state_dict(ckpt["model"], strict=True)
    elif "student" in ckpt:
        decoder.load_state_dict(ckpt["student"], strict=True)
    else:
        raise KeyError(f"{ckpt_path} 缺少 model/student")

    if "lora_params" in ckpt:
        extractor.load_lora_params(ckpt["lora_params"])

    if model_name == "hl_residual":
        model = HLResidualRefiner(decoder).to(DEVICE)
        if "residual" not in ckpt:
            raise KeyError(f"{ckpt_path} 缺少 residual")
        model.residual.load_state_dict(ckpt["residual"], strict=True)
    else:
        model = decoder

    model.eval()
    extractor.model.eval()

    return extractor, model, ckpt_path


@torch.no_grad()
def predict_promptfree_probs(extractor, model, loader, use_tta: bool):
    probs_all = []
    gts_all = []

    for batch in tqdm(loader, desc="  predict prompt-free", ncols=100):
        imgs = batch["image"]
        gts = batch["mask"].numpy()[:, 0]

        if not use_tta:
            feat = extractor.extract_batch(imgs)
            rgb = imgs.to(DEVICE)
            logits = model(feat.to(DEVICE), rgb)
            probs = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]
        else:
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

            probs = torch.mean(torch.stack(probs_list, dim=0), dim=0).numpy()[:, 0]

        probs_all.append(probs)
        gts_all.append(gts)

    return np.concatenate(probs_all), np.concatenate(gts_all)


# =============================================================================
# Box-prompt LoRA
# =============================================================================

def load_box_prompt_lora(project_root: Path, sam3_ckpt: Path, box_lora_dir: Path, seed: int):
    weight_path = box_lora_dir / f"sam3_lora_tgt_20shot_seed{seed}.pth"
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Box-prompt LoRA 权重不存在: {weight_path}\n"
            f"如果你没有旧 E5 box-prompt LoRA 权重，需要先跑旧 E5。"
        )

    model = build_sam3_image_model(checkpoint_path=str(sam3_ckpt))
    model.to(DEVICE)
    model.eval()
    inject_lora(model)

    ckpt = torch.load(weight_path, map_location=DEVICE, weights_only=False)
    if "lora_params" in ckpt:
        state = model.state_dict()
        loaded = 0
        for k, v in ckpt["lora_params"].items():
            if k in state:
                state[k].copy_(v.to(DEVICE))
                loaded += 1
        print(f"  loaded box-prompt LoRA params: {loaded}")

    processor = Sam3Processor(model, confidence_threshold=SAM3_CONFIDENCE_THRESHOLD)
    return model, processor, weight_path


def jitter_box_to_prompt(box, sigma, rng):
    x1, y1, x2, y2 = box[:4]

    x1 = x1 + rng.normal(0, sigma)
    y1 = y1 + rng.normal(0, sigma)
    x2 = x2 + rng.normal(0, sigma)
    y2 = y2 + rng.normal(0, sigma)

    cx = ((x1 + x2) / 2.0) / TARGET_SIZE
    cy = ((y1 + y2) / 2.0) / TARGET_SIZE
    w = abs(x2 - x1) / TARGET_SIZE
    h = abs(y2 - y1) / TARGET_SIZE

    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.01, min(1.0, w))
    h = max(0.01, min(1.0, h))

    return [cx, cy, w, h]


@torch.no_grad()
def evaluate_box_prompt_lora(
    dataset: ManifestDataset,
    bboxes_all: Dict[str, List[List[int]]],
    project_root: Path,
    sam3_ckpt: Path,
    box_lora_dir: Path,
    seed: int,
    sigma: float,
):
    _, processor, weight_path = load_box_prompt_lora(project_root, sam3_ckpt, box_lora_dir, seed)

    preds = []
    gts = []
    rng = np.random.RandomState(seed + int(sigma * 100))

    for idx in tqdm(range(len(dataset)), desc=f"  box-prompt seed={seed} sigma={sigma}", ncols=100):
        item = dataset[idx]

        img_tensor = item["image"]
        mask = item["mask"].squeeze(0).numpy()
        stem = item["name"]

        img_uint8 = (img_tensor * 255).clamp(0, 255).to(torch.uint8)
        img_np = img_uint8.permute(1, 2, 0).cpu().numpy()
        pil = Image.fromarray(img_np)

        bboxes = lookup_bboxes(bboxes_all, stem)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            state = processor.set_image(pil)

            if bboxes:
                for box in bboxes:
                    if len(box) < 4:
                        continue
                    prompt = jitter_box_to_prompt(box, sigma, rng)
                    state = processor.add_geometric_prompt(prompt, True, state)
            else:
                # 没有 bbox 通常说明 GT 无建筑，此时输出空 mask；
                # 不用 text fallback，避免混入另一种 prompt 范式。
                pass

        masks_out = state.get("masks", None)
        if masks_out is None or masks_out.numel() == 0:
            pred = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
        else:
            masks_np = masks_out.detach().cpu().numpy() if isinstance(masks_out, torch.Tensor) else np.array(masks_out)
            if masks_np.ndim == 2:
                masks_np = masks_np[None, ...]
            pred = np.any(masks_np > 0.5, axis=0).astype(np.float32)
            pred = np.squeeze(pred)
            if pred.ndim != 2:
                pred = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
            pred = cv2.resize(pred, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)

        preds.append(pred)
        gts.append(mask)

    del processor
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return eval_arrays(np.stack(preds), np.stack(gts)), str(weight_path)


# =============================================================================
# Plot / Save
# =============================================================================

def aggregate_seed_metrics(seed_metrics: List[Dict[str, float]]):
    out = {}
    for k in ["mIoU", "F1", "Boundary_IoU"]:
        vals = [m[k] for m in seed_metrics]
        out[k] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


def plot_curves(results, sigmas, out_path: Path):
    model_order = ["Box-prompt LoRA", "Prompt-free LoRA", "HL-Residual (Ours)"]
    colors = {
        "Box-prompt LoRA": "#2E7D32",
        "Prompt-free LoRA": "#1976D2",
        "HL-Residual (Ours)": "#C2185B",
    }
    markers = {
        "Box-prompt LoRA": "^",
        "Prompt-free LoRA": "o",
        "HL-Residual (Ours)": "*",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for model_name in model_order:
        ys_miou = []
        ys_biou = []
        xs = []

        for s in sigmas:
            if str(s) in results[model_name]:
                xs.append(s)
                ys_miou.append(results[model_name][str(s)]["mIoU"] * 100)
                ys_biou.append(results[model_name][str(s)]["Boundary_IoU"] * 100)

        if not xs:
            continue

        axes[0].plot(xs, ys_miou, marker=markers[model_name], color=colors[model_name],
                     linewidth=2, markersize=8, label=model_name)
        axes[1].plot(xs, ys_biou, marker=markers[model_name], color=colors[model_name],
                     linewidth=2, markersize=8, label=model_name)

    axes[0].set_title("Prompt Drift Robustness: mIoU")
    axes[0].set_xlabel("Box jitter σ (pixels)")
    axes[0].set_ylabel("mIoU (%)")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].set_title("Prompt Drift Robustness: Boundary IoU")
    axes[1].set_xlabel("Box jitter σ (pixels)")
    axes[1].set_ylabel("Boundary IoU (%)")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser("E7a prompt drift robustness")

    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT_DEFAULT))
    parser.add_argument("--manifest_dir", type=str, default=None)
    parser.add_argument("--test_manifest_name", type=str, default="target_pilot_test_500.txt")
    parser.add_argument("--test_eval_limit", type=int, default=500)
    parser.add_argument("--val_manifest_name", type=str, default="target_val.txt")
    parser.add_argument("--val_eval_limit", type=int, default=500)

    parser.add_argument("--sam3_checkpoint", type=str, default=None)
    parser.add_argument("--bbox_json", type=str, default=None)
    parser.add_argument("--box_lora_dir", type=str, default=None)

    parser.add_argument("--sigmas", type=str, default="0,2,5,10,20")
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(args.project_root)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else project_root / "data/splits/e0_manifest"
    sam3_ckpt = Path(args.sam3_checkpoint) if args.sam3_checkpoint else project_root / "weights/sam3.pt"
    bbox_json = Path(args.bbox_json) if args.bbox_json else project_root / "data/raw/whu_mix_full_test/bbox.json"
    box_lora_dir = Path(args.box_lora_dir) if args.box_lora_dir else project_root / "weights"
    out_dir = Path(args.out_dir) if args.out_dir else project_root / "results/e7a_prompt_drift"
    out_dir.mkdir(parents=True, exist_ok=True)

    sigmas = [int(x.strip()) for x in args.sigmas.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    test_limit = args.test_eval_limit
    if test_limit is not None and test_limit <= 0:
        test_limit = None

    val_limit = args.val_eval_limit
    if val_limit is not None and val_limit <= 0:
        val_limit = None

    print("=" * 90)
    print("E7a Prompt Drift Robustness")
    print("=" * 90)
    print(f"project_root : {project_root}")
    print(f"test_manifest: {args.test_manifest_name}")
    print(f"val_manifest : {args.val_manifest_name}")
    print(f"sigmas       : {sigmas}")
    print(f"seeds        : {seeds}")
    print(f"use_tta      : {args.use_tta}")
    print(f"device       : {DEVICE}")
    print("=" * 90)

    test_ds = ManifestDataset(manifest_dir / args.test_manifest_name, limit=test_limit)
    val_ds = ManifestDataset(manifest_dir / args.val_manifest_name, limit=val_limit)
    test_loader = make_loader(test_ds, batch_size=4, num_workers=args.num_workers)
    val_loader = make_loader(val_ds, batch_size=4, num_workers=args.num_workers)

    bboxes_all = load_bboxes(bbox_json)

    results = {
        "Box-prompt LoRA": {},
        "Prompt-free LoRA": {},
        "HL-Residual (Ours)": {},
    }
    raw = {
        "Box-prompt LoRA": defaultdict(list),
        "Prompt-free LoRA": defaultdict(list),
        "HL-Residual (Ours)": defaultdict(list),
    }

    # 1) Box-prompt LoRA: 每个 sigma 都重新评估
    for sigma in sigmas:
        seed_metrics = []
        for seed in seeds:
            try:
                m, ckpt_used = evaluate_box_prompt_lora(
                    dataset=test_ds,
                    bboxes_all=bboxes_all,
                    project_root=project_root,
                    sam3_ckpt=sam3_ckpt,
                    box_lora_dir=box_lora_dir,
                    seed=seed,
                    sigma=sigma,
                )
                m["checkpoint"] = ckpt_used
                seed_metrics.append(m)
            except Exception as e:
                print(f"[Box-prompt skip] seed={seed}, sigma={sigma}, error={e}")

        if seed_metrics:
            results["Box-prompt LoRA"][str(sigma)] = aggregate_seed_metrics(seed_metrics)
            raw["Box-prompt LoRA"][str(sigma)] = seed_metrics

    # 2) Prompt-free 模型：每 seed 只预测一次，然后复制到所有 sigma
    for model_key, model_label in [
        ("promptfree_lora", "Prompt-free LoRA"),
        ("hl_residual", "HL-Residual (Ours)"),
    ]:
        clean_seed_metrics = []

        for seed in seeds:
            try:
                print(f"\n[Prompt-free] {model_label}, seed={seed}: calibrating on val")
                extractor, model, ckpt_path = load_promptfree_model(model_key, seed, project_root, sam3_ckpt)

                val_probs, val_gts = predict_promptfree_probs(extractor, model, val_loader, use_tta=args.use_tta)
                search = grid_search_postprocess(val_probs, val_gts)
                best_cfg = search["best"]

                print(f"[Prompt-free] {model_label}, seed={seed}: eval on test")
                test_probs, test_gts = predict_promptfree_probs(extractor, model, test_loader, use_tta=args.use_tta)
                m = eval_with_cfg(test_probs, test_gts, best_cfg)
                m["checkpoint"] = str(ckpt_path)
                m["val_best_cfg"] = best_cfg

                clean_seed_metrics.append(m)

                del extractor, model
                gc.collect()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"[Prompt-free skip] {model_label}, seed={seed}, error={e}")
                import traceback
                traceback.print_exc()

        if clean_seed_metrics:
            agg = aggregate_seed_metrics(clean_seed_metrics)
            for sigma in sigmas:
                results[model_label][str(sigma)] = agg
                raw[model_label][str(sigma)] = clean_seed_metrics

    summary = {
        "experiment": "E7a_prompt_drift_box_jitter",
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "test_manifest": args.test_manifest_name,
            "val_manifest": args.val_manifest_name,
            "sigmas": sigmas,
            "seeds": seeds,
            "use_tta": args.use_tta,
            "bbox_json": str(bbox_json),
            "note": "Prompt-free models do not consume bbox prompts; their metrics are repeated across sigma levels.",
        },
        "results": results,
        "raw_per_seed": raw,
    }

    json_path = out_dir / "e7a_prompt_drift_metrics.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fig_path = out_dir / "e7a_prompt_drift_curve.png"
    plot_curves(results, sigmas, fig_path)

    print("\n" + "=" * 90)
    print("E7a finished")
    print("=" * 90)
    print(f"JSON: {json_path}")
    print(f"FIG : {fig_path}")

    for model_name, sigma_dict in results.items():
        print(f"\n{model_name}")
        for s in sigmas:
            if str(s) in sigma_dict:
                m = sigma_dict[str(s)]
                print(
                    f"  sigma={s:<3} "
                    f"mIoU={m['mIoU']*100:.2f} "
                    f"F1={m['F1']*100:.2f} "
                    f"BIoU={m['Boundary_IoU']*100:.2f}"
                )


if __name__ == "__main__":
    main()