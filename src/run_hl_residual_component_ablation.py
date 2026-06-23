# -*- coding: utf-8 -*-
"""
run_hl_residual_refine_pilot.py
=============================================================================
E6-v3R: HL-Residual Refinement Pilot

目的：
  当前 HL-Lite calibrated 有小幅提升，但 fixed mIoU 大幅下降。
  因此改成残差式高分辨率修正：
      final_logits = frozen_lora_light_logits + residual_scale * hl_residual_logits

核心思想：
  1. 加载已经训练好的 lora_light checkpoint；
  2. 冻结 lora_light decoder 和 LoRA encoder；
  3. 只训练一个轻量 HL residual branch；
  4. 用 target_val 做 threshold calibration；
  5. 在 pilot500 上评估。

输入：
  results/e6v2_pilot/weights/e6v2_pilot_lora_light_best.pth
  data/splits/e0_manifest/target_support_20_seed42.txt
  data/splits/e0_manifest/target_val.txt
  data/splits/e0_manifest/target_pilot_test_500.txt

运行：
  cd <project_root>

  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python src/run_hl_residual_refine_pilot.py \
    --epochs 20 \
    --batch_size 2 \
    --val_every 5 \
    --val_eval_limit 200 \
    --pilot_eval_limit 500
=============================================================================
"""

import argparse
import gc
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple

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

IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
LABEL_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


# =============================================================================
# 1. Seed
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 2. Dataset
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


def make_boundary(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return (mask.astype(np.uint8) - eroded).astype(np.float32)


def random_erasing_image_only(
    img_t: torch.Tensor,
    p: float = 0.25,
    scale=(0.02, 0.08),
    ratio=(0.3, 3.3),
    fill: float = 0.5,
) -> torch.Tensor:
    """
    Segmentation-safe random erasing:
      只擦除 image，不改 mask。
    注意:
      p 不要太高，scale 不要太大，否则 20-shot 会被增强噪声淹没。
    """
    if p <= 0 or random.random() > p:
        return img_t

    c, h, w = img_t.shape
    area = h * w

    for _ in range(10):
        target_area = random.uniform(scale[0], scale[1]) * area
        aspect = random.uniform(ratio[0], ratio[1])

        erase_h = int(round((target_area * aspect) ** 0.5))
        erase_w = int(round((target_area / aspect) ** 0.5))

        if erase_h < h and erase_w < w:
            y = random.randint(0, h - erase_h)
            x = random.randint(0, w - erase_w)
            img_t[:, y:y + erase_h, x:x + erase_w] = fill
            return img_t

    return img_t


class ManifestSegDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        limit: Optional[int] = None,
        train_aug: bool = False,
        erase_prob: float = 0.0,
        erase_max_area: float = 0.08,
    ):
        self.paths = read_manifest(manifest_path, limit=limit)
        self.train_aug = train_aug
        self.erase_prob = erase_prob
        self.erase_max_area = erase_max_area

        if not self.paths:
            raise RuntimeError(f"manifest 为空: {manifest_path}")

        print(
            f"  Dataset {manifest_path.name}: {len(self.paths)} | "
            f"train_aug={train_aug} erase_prob={erase_prob}"
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        if not img_path.exists():
            raise FileNotFoundError(img_path)

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(img_path)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        if self.train_aug and self.erase_prob > 0:
            img_t = random_erasing_image_only(
                img_t,
                p=self.erase_prob,
                scale=(0.02, self.erase_max_area),
                fill=0.5,
            )

        label_path = find_dual_label_for_image(img_path)
        if label_path is None:
            raise FileNotFoundError(f"找不到 label: {img_path}")

        lab = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if lab is None:
            raise FileNotFoundError(label_path)

        if lab.ndim == 2:
            mask = cv2.resize(lab, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.float32)
            boundary = make_boundary(mask)
        else:
            lab = cv2.resize(lab, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)
            mask = (lab[:, :, 0] > 0).astype(np.float32)
            boundary = (lab[:, :, 1] > 0).astype(np.float32) if lab.shape[2] > 1 else make_boundary(mask)

        return {
            "image": img_t,
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
            "boundary": torch.from_numpy(boundary).unsqueeze(0).float(),
            "name": img_path.stem,
        }


def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "boundary": torch.stack([b["boundary"] for b in batch]),
        "name": [b["name"] for b in batch],
    }


# =============================================================================
# 3. Metrics
# =============================================================================

def compute_iou_np(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + 1e-7) / (union + 1e-7))


def compute_f1_np(pred, gt):
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

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    b = mask - eroded

    if d > 1:
        b = cv2.dilate(b, np.ones((d, d), np.uint8), iterations=1)

    return b.astype(bool)


def compute_biou_np(pred, gt, d=5):
    pb = boundary_map(pred, d=d)
    gb = boundary_map(gt, d=d)
    inter = np.logical_and(pb, gb).sum()
    union = np.logical_or(pb, gb).sum()
    return float((inter + 1e-7) / (union + 1e-7))


def evaluate_arrays(preds, gts):
    ious, f1s, bious = [], [], []
    for p, g in zip(preds, gts):
        ious.append(compute_iou_np(p, g))
        f1s.append(compute_f1_np(p, g))
        bious.append(compute_biou_np(p, g, d=5))
    return {
        "mIoU": float(np.mean(ious)),
        "F1": float(np.mean(f1s)),
        "Boundary_IoU": float(np.mean(bious)),
    }


def remove_small_components(pred, min_area):
    if min_area <= 0:
        return pred.astype(np.float32)

    pred = (pred > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
    out = np.zeros_like(pred)

    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1

    return out.astype(np.float32)


def evaluate_fixed(probs, gts, threshold, min_area):
    preds = []
    for p in probs:
        pred = (p > threshold).astype(np.float32)
        pred = remove_small_components(pred, min_area)
        preds.append(pred)
    return evaluate_arrays(np.stack(preds), gts)


def threshold_sweep(probs, gts):
    best = None
    rows = []

    for th in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        for min_area in [0, 16, 32, 64, 128]:
            row = {
                "threshold": th,
                "min_area": min_area,
                **evaluate_fixed(probs, gts, th, min_area),
            }
            rows.append(row)
            if best is None or row["mIoU"] > best["mIoU"]:
                best = row

    return {"best": best, "all": rows}


@torch.no_grad()
def quick_val_select_score(model, extractor, loader, use_tta: bool = False):
    """
    用 val subset 做 threshold sweep，按 calibrated mIoU 选择 best checkpoint。
    这是比 val_mIoU@0.5 更合理的 checkpoint 选择方式。
    """
    probs, gts = predict_probs(model, extractor, loader, use_tta=use_tta)
    sweep = threshold_sweep(probs, gts)
    return sweep["best"]


# =============================================================================
# 4. Loss
# =============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        b = probs.shape[0]
        probs = probs.view(b, -1)
        target = target.view(b, -1)
        inter = (probs * target).sum(dim=1)
        union = probs.sum(dim=1) + target.sum(dim=1)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return (1 - dice).mean()


# =============================================================================
# 5. SAM3 Extractor + LoRA
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

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA trainable {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")


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

    def extract_one_grad(self, img):
        x = self._prep(img)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            out = self.model.backbone.forward_image(x)
        feat = out["vision_features"].float()
        if feat.shape[-2:] != FEATURE_SIZE:
            feat = F.interpolate(feat, size=FEATURE_SIZE, mode="bilinear", align_corners=False)
        return feat

    @torch.no_grad()
    def extract_batch_eval(self, imgs):
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


# =============================================================================
# 6. Model
# =============================================================================

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
    """
    Residual branch with ablation switches.

    variant:
      - full:     use both low-resolution SAM3 features and high-resolution RGB details.
      - rgb_only: use only the RGB high-resolution path; the SAM low path is zeroed after projection.
      - sam_only: use only the SAM low-resolution path; the RGB high path is zeroed after projection.

    This keeps the parameterization and downstream layers unchanged across ablations,
    so the comparison isolates information source rather than changing tensor shapes.
    """
    def __init__(self, low_ch=256, variant: str = "full"):
        super().__init__()
        if variant not in {"full", "rgb_only", "sam_only"}:
            raise ValueError(f"Unknown residual variant: {variant}")
        self.variant = variant

        self.high_path = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # 256
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.low_to_128 = nn.Sequential(
            nn.ConvTranspose2d(low_ch, 128, 4, 2, 1),    # 128
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

        # Initialize residual close to zero, avoiding early damage to the LoRA baseline.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        self.res_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, feat, rgb):
        high = self.high_path(rgb)
        low = self.low_to_128(feat)

        if self.variant == "rgb_only":
            low = torch.zeros_like(low)
        elif self.variant == "sam_only":
            high = torch.zeros_like(high)

        x = self.fuse(torch.cat([low, high], dim=1))
        x = self.up256(x)
        x = self.up512(x)
        res = self.out(x)
        return torch.tanh(self.res_scale) * res


class HLResidualRefiner(nn.Module):
    """
    frozen base decoder + trainable residual branch
    """
    def __init__(self, base_decoder: LightweightMaskDecoder, variant: str = "full"):
        super().__init__()
        self.variant = variant
        self.base_decoder = base_decoder
        self.residual = HLResidualBranch(variant=variant)

        for p in self.base_decoder.parameters():
            p.requires_grad = False

    def forward(self, feat, rgb):
        with torch.no_grad():
            base_logits = self.base_decoder(feat, rgb)
        residual_logits = self.residual(feat, rgb)
        return base_logits + residual_logits


# =============================================================================
# 7. Loading base checkpoint
# =============================================================================

def load_lora_light_checkpoint(
    checkpoint_path: Path,
    base_ckpt_path: Path,
) -> Tuple[SAM3Extractor, LightweightMaskDecoder]:
    extractor = SAM3Extractor(checkpoint_path)
    base_decoder = LightweightMaskDecoder().to(DEVICE)

    if not base_ckpt_path.exists():
        raise FileNotFoundError(f"lora_light checkpoint 不存在: {base_ckpt_path}")

    ckpt = torch.load(base_ckpt_path, map_location=DEVICE, weights_only=False)

    # 兼容两种 key: "model"（新格式）与 "decoder"（20-shot LoRA checkpoint 格式）
    if "model" in ckpt:
        decoder_state = ckpt["model"]
    elif "decoder" in ckpt:
        decoder_state = ckpt["decoder"]
    else:
        raise KeyError(f"checkpoint 缺少 model/decoder: {base_ckpt_path} (keys: {list(ckpt.keys())[:10]})")

    base_decoder.load_state_dict(decoder_state, strict=True)

    if "lora_params" in ckpt:
        state = extractor.model.state_dict()
        loaded = 0
        for k, v in ckpt["lora_params"].items():
            if k in state:
                state[k].copy_(v.to(DEVICE))
                loaded += 1
        print(f"  loaded LoRA params: {loaded}")

    extractor.model.eval()
    for p in extractor.model.parameters():
        p.requires_grad = False

    base_decoder.eval()
    for p in base_decoder.parameters():
        p.requires_grad = False

    return extractor, base_decoder


# =============================================================================
# 8. Train / Eval
# =============================================================================

def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def forward_train(model, extractor, imgs):
    logits = []
    for i in range(imgs.shape[0]):
        with torch.no_grad():
            feat = extractor.extract_batch_eval(imgs[i:i+1])
        rgb = imgs[i:i+1].to(DEVICE)
        logit = model(feat.to(DEVICE), rgb)
        logits.append(logit)
    return torch.cat(logits, dim=0)


@torch.no_grad()
def predict_probs(model, extractor, loader, use_tta: bool = False):
    model.eval()
    probs_all, gts_all = [], []

    def _logits_for(imgs_batch):
        feat = extractor.extract_batch_eval(imgs_batch)
        rgb = imgs_batch.to(DEVICE)
        return model(feat.to(DEVICE), rgb)

    for batch in tqdm(loader, desc="  predict", ncols=100):
        imgs = batch["image"]
        masks = batch["mask"].numpy()[:, 0]

        logits = _logits_for(imgs)

        if use_tta:
            imgs_h = torch.flip(imgs, dims=[3])
            logits_h = torch.flip(_logits_for(imgs_h), dims=[3])

            imgs_v = torch.flip(imgs, dims=[2])
            logits_v = torch.flip(_logits_for(imgs_v), dims=[2])

            logits = (logits + logits_h + logits_v) / 3.0

        probs = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]

        probs_all.append(probs)
        gts_all.append(masks)

    return np.concatenate(probs_all), np.concatenate(gts_all)


def train_refiner(
    project_root: Path,
    sam3_checkpoint: Path,
    base_lora_ckpt: Path,
    train_loader,
    val_ckpt_loader,
    val_full_loader,
    pilot_loader,
    epochs: int,
    lr: float,
    val_every: int,
    out_dir: Path,
    grad_accum_steps: int = 1,
    select_by_calibrated_val: bool = True,
    residual_l2_weight: float = 0.01,
    variant: str = "full",
    use_tta_eval: bool = False,
):
    extractor, base_decoder = load_lora_light_checkpoint(sam3_checkpoint, base_lora_ckpt)

    model = HLResidualRefiner(base_decoder, variant=variant).to(DEVICE)

    optimizer = torch.optim.AdamW(trainable_params(model), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    best_val = -1.0
    best_path = out_dir / "weights" / f"hl_residual_{variant}_best.pth"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    history = []

    print("\n" + "=" * 90)
    print(f"Train HL-Residual Refiner | variant={variant}")
    print("=" * 90)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(train_loader, desc=f"  epoch {epoch}/{epochs}", ncols=100), start=1):
            imgs = batch["image"]
            masks = batch["mask"].to(DEVICE)

            logits = forward_train(model, extractor, imgs)

            loss_main = bce(logits.float(), masks.float()) + dice(logits.float(), masks.float())

            # 残差分支轻微 L2，防止为了过拟合 20-shot 产生大幅不稳定修正
            l2_reg = torch.tensor(0.0, device=DEVICE)
            for p in model.residual.parameters():
                l2_reg = l2_reg + torch.sum(p.float() ** 2)

            loss = loss_main + residual_l2_weight * l2_reg / 1e6

            loss_to_backward = loss / max(1, grad_accum_steps)
            loss_to_backward.backward()

            if step % grad_accum_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params(model), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            losses.append(float(loss.item()))

        val_miou = None
        val_select_info = None

        if epoch % val_every == 0 or epoch == epochs:
            if select_by_calibrated_val:
                val_select_info = quick_val_select_score(model, extractor, val_ckpt_loader, use_tta=use_tta_eval)
                val_miou = val_select_info["mIoU"]
                print(
                    f"  epoch={epoch} loss={np.mean(losses):.4f} "
                    f"val_calib_mIoU={val_miou:.4f} "
                    f"th={val_select_info['threshold']} area={val_select_info['min_area']}"
                )
            else:
                val_probs, val_gts = predict_probs(model, extractor, val_ckpt_loader, use_tta=use_tta_eval)
                val_fixed = evaluate_fixed(val_probs, val_gts, threshold=0.5, min_area=0)
                val_miou = val_fixed["mIoU"]
                val_select_info = {
                    "threshold": 0.5,
                    "min_area": 0,
                    **val_fixed,
                }
                print(f"  epoch={epoch} loss={np.mean(losses):.4f} val_mIoU@0.5={val_miou:.4f}")

            if val_miou > best_val:
                best_val = val_miou
                torch.save({
                    "variant": variant,
                    "residual": model.residual.state_dict(),
                    "base_lora_ckpt": str(base_lora_ckpt),
                    "epoch": epoch,
                    "best_val_select_mIoU": best_val,
                    "val_select_info": val_select_info,
                    "grad_accum_steps": grad_accum_steps,
                    "select_by_calibrated_val": select_by_calibrated_val,
                    "residual_l2_weight": residual_l2_weight,
                }, best_path)
                print(f"  ★ saved: {best_path}")

        history.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_select_mIoU": val_miou,
            "val_select_info": val_select_info,
        })

        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.residual.load_state_dict(ckpt["residual"], strict=True)

    print("\n[Val full] threshold sweep")
    val_probs, val_gts = predict_probs(model, extractor, val_full_loader, use_tta=use_tta_eval)
    sweep = threshold_sweep(val_probs, val_gts)

    th = float(sweep["best"]["threshold"])
    area = int(sweep["best"]["min_area"])

    print(f"  best threshold={th}, min_area={area}, val_mIoU={sweep['best']['mIoU']:.4f}")

    print("\n[Pilot500] evaluation")
    pilot_probs, pilot_gts = predict_probs(model, extractor, pilot_loader, use_tta=use_tta_eval)
    pilot_05 = evaluate_fixed(pilot_probs, pilot_gts, 0.5, 0)
    pilot_calib = evaluate_fixed(pilot_probs, pilot_gts, th, area)

    result = {
        "variant": variant,
        "base_lora_ckpt": str(base_lora_ckpt),
        "best_weight": str(best_path),
        "history": history,
        "val_sweep_best": sweep["best"],
        "test_fixed_05": pilot_05,
        "test_calibrated": pilot_calib,
        "pilot_fixed_05": pilot_05,
        "pilot_calibrated": pilot_calib,
        "use_tta_eval": use_tta_eval,
        "timestamp": datetime.now().isoformat(),
    }

    result_path = out_dir / f"hl_residual_{variant}_summary.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print(f"HL-Residual summary | variant={variant}")
    print("=" * 90)
    print(
        f"fixed mIoU={pilot_05['mIoU']*100:.2f} "
        f"F1={pilot_05['F1']*100:.2f} "
        f"BIoU={pilot_05['Boundary_IoU']*100:.2f}"
    )
    print(
        f"calib mIoU={pilot_calib['mIoU']*100:.2f} "
        f"F1={pilot_calib['F1']*100:.2f} "
        f"BIoU={pilot_calib['Boundary_IoU']*100:.2f}"
    )
    print(f"Saved: {result_path}")

    return result


# =============================================================================
# 9. CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser("HL-Residual refinement pilot")

    parser.add_argument("--project_root", type=str, default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--manifest_dir", type=str, default=None)
    parser.add_argument("--sam3_checkpoint", type=str, default=None)
    parser.add_argument("--base_lora_ckpt", type=str, default=None)
    parser.add_argument(
        "--variant",
        type=str,
        default="full",
        choices=["full", "rgb_only", "sam_only"],
        help="Residual ablation variant: full / rgb_only / sam_only.",
    )
    parser.add_argument(
        "--eval_manifest",
        type=str,
        default=None,
        help="Evaluation manifest. If not set, use target_final_test_excluding_pilot500.txt when available, otherwise target_pilot_test_500.txt.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shot", type=int, default=20)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--val_eval_limit", type=int, default=200)
    parser.add_argument("--pilot_eval_limit", type=int, default=None)
    parser.add_argument(
        "--use_tta_eval",
        action="store_true",
        help="Use original + horizontal flip + vertical flip during validation sweep and evaluation.",
    )

    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=2,
        help="梯度累积步数。20-shot 推荐 2 或 4。",
    )

    parser.add_argument(
        "--train_erasing_prob",
        type=float,
        default=0.25,
        help="训练图像随机擦除概率。建议 0.0/0.15/0.25 做 pilot。",
    )

    parser.add_argument(
        "--train_erasing_max_area",
        type=float,
        default=0.08,
        help="随机擦除最大面积比例。建议不要超过 0.08。",
    )

    parser.add_argument(
        "--select_by_calibrated_val",
        action="store_true",
        help="按 val calibrated mIoU 保存最佳 checkpoint。",
    )

    parser.add_argument(
        "--residual_l2_weight",
        type=float,
        default=0.01,
        help="残差分支 L2 正则权重。",
    )

    parser.add_argument("--out_dir", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    project_root = Path(args.project_root)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else project_root / "data" / "splits" / "e0_manifest"

    sam3_checkpoint = Path(args.sam3_checkpoint) if args.sam3_checkpoint else project_root / "weights" / "sam3.pt"

    base_lora_ckpt = (
        Path(args.base_lora_ckpt)
        if args.base_lora_ckpt
        else project_root / "results" / "e6v2_pilot" / "weights" / "e6v2_pilot_lora_light_best.pth"
    )

    out_dir = Path(args.out_dir) if args.out_dir else project_root / "results" / "hl_residual_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = manifest_dir / f"target_support_{args.shot}_seed{args.seed}.txt"
    val_manifest = manifest_dir / "target_val.txt"
    if args.eval_manifest:
        pilot_manifest = Path(args.eval_manifest)
    else:
        heldout = manifest_dir / "target_final_test_excluding_pilot500.txt"
        pilot_manifest = heldout if heldout.exists() else manifest_dir / "target_pilot_test_500.txt"

    print("=" * 90)
    print("HL-Residual Refinement Pilot")
    print("=" * 90)
    print(f"project_root   : {project_root}")
    print(f"manifest_dir   : {manifest_dir}")
    print(f"sam3_checkpoint: {sam3_checkpoint}")
    print(f"base_lora_ckpt : {base_lora_ckpt}")
    print(f"variant        : {args.variant}")
    print(f"train_manifest : {train_manifest.name}")
    print(f"val_manifest   : {val_manifest.name}")
    print(f"eval_manifest  : {pilot_manifest.name}")
    print(f"use_tta_eval   : {args.use_tta_eval}")
    print(f"device         : {DEVICE}")
    print("=" * 90)

    train_ds = ManifestSegDataset(
        train_manifest,
        train_aug=True,
        erase_prob=args.train_erasing_prob,
        erase_max_area=args.train_erasing_max_area,
    )
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

    train_refiner(
        project_root=project_root,
        sam3_checkpoint=sam3_checkpoint,
        base_lora_ckpt=base_lora_ckpt,
        train_loader=train_loader,
        val_ckpt_loader=val_ckpt_loader,
        val_full_loader=val_full_loader,
        pilot_loader=pilot_loader,
        epochs=args.epochs,
        lr=args.lr,
        val_every=args.val_every,
        out_dir=out_dir,
        grad_accum_steps=args.grad_accum_steps,
        select_by_calibrated_val=args.select_by_calibrated_val,
        residual_l2_weight=args.residual_l2_weight,
        variant=args.variant,
        use_tta_eval=args.use_tta_eval,
    )


if __name__ == "__main__":
    main()