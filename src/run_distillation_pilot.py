# -*- coding: utf-8 -*-
"""
run_distillation_pilot.py
=============================================================================
Text-prompt distillation pilot

目的：
  当 MSR / BGA 没有提升时，使用 SAM3 text prompt "building" 作为 teacher，
  在 target distill pool 上生成 pseudo masks，
  然后训练 prompt-free student。

默认比较：
  1. sup_light      : 只用 20-shot GT 监督
  2. distill_light  : 20-shot GT + SAM3 text pseudo masks

输入：
  data/splits/e0_manifest/target_support_20_seed42.txt
  data/splits/e0_manifest/target_val.txt
  data/splits/e0_manifest/target_pilot_test_500.txt
  data/splits/e0_manifest/target_distill_pool_1000.txt

输出：
  results/distill_pilot/
    pseudo_masks/
    weights/
    distill_pilot_summary.json

运行：
  cd <project_root>

  python src/run_distillation_pilot.py \
    --variants sup_light,distill_light \
    --epochs 20 \
    --pseudo_limit 1000
=============================================================================
"""

import argparse
import gc
import json
import random
import sys
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
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
# 2. Manifest path resolver
# =============================================================================

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


def resolve_by_basename(paths: List[Path], fallback_dir: Path) -> List[Path]:
    resolved = []
    missing = []

    for p in paths:
        if p.exists():
            resolved.append(p)
            continue

        cand = fallback_dir / p.name
        if cand.exists():
            resolved.append(cand)
        else:
            stem = p.stem
            found = None
            for ext in IMAGE_EXTS:
                c = fallback_dir / f"{stem}{ext}"
                if c.exists():
                    found = c
                    break
            if found is not None:
                resolved.append(found)
            else:
                missing.append(str(p))

    if missing:
        raise FileNotFoundError(
            f"有 {len(missing)} 个 manifest 图像无法解析，示例: {missing[:5]}"
        )

    return resolved


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


# =============================================================================
# 3. Dataset
# =============================================================================

def make_boundary(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return (mask.astype(np.uint8) - eroded).astype(np.float32)


class SupervisedManifestDataset(Dataset):
    def __init__(self, manifest_path: Path, limit: Optional[int] = None):
        self.paths = read_manifest(manifest_path, limit=limit)
        if not self.paths:
            raise RuntimeError(f"空 supervised manifest: {manifest_path}")
        print(f"  SupervisedDataset {manifest_path.name}: {len(self.paths)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        if not img_path.exists():
            raise FileNotFoundError(img_path)

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        label_path = find_dual_label_for_image(img_path)
        if label_path is None:
            raise FileNotFoundError(f"missing label for {img_path}")

        lab = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if lab is None:
            raise FileNotFoundError(label_path)

        if lab.ndim == 2:
            mask = (cv2.resize(lab, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
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


class DistillPseudoDataset(Dataset):
    def __init__(self, pseudo_index_path: Path):
        if not pseudo_index_path.exists():
            raise FileNotFoundError(pseudo_index_path)

        data = json.loads(pseudo_index_path.read_text(encoding="utf-8"))
        self.items = [x for x in data["items"] if x.get("valid", False)]

        if not self.items:
            raise RuntimeError("没有有效 pseudo masks")

        print(f"  DistillPseudoDataset valid pseudo: {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = Path(item["image"])
        pseudo_path = Path(item["pseudo_mask"])

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        pseudo = cv2.imread(str(pseudo_path), cv2.IMREAD_GRAYSCALE)
        if pseudo is None:
            raise FileNotFoundError(pseudo_path)

        pseudo = cv2.resize(pseudo, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)
        pseudo = (pseudo > 0).astype(np.float32)

        return {
            "image": img_t,
            "pseudo": torch.from_numpy(pseudo).unsqueeze(0).float(),
            "name": img_path.stem,
        }


def collate_supervised(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "boundary": torch.stack([b["boundary"] for b in batch]),
        "name": [b["name"] for b in batch],
    }


def collate_pseudo(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "pseudo": torch.stack([b["pseudo"] for b in batch]),
        "name": [b["name"] for b in batch],
    }


# =============================================================================
# 4. Metrics
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
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
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


# =============================================================================
# 5. Pseudo generation
# =============================================================================

def _tensor_to_numpy(x):
    """安全地将 tensor / array 转为 numpy float。"""
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.array(x)


def extract_masks_from_state(state: dict) -> Tuple[Optional[np.ndarray], Dict]:
    """
    从 Sam3Processor state 中尽可能稳健地提取 masks。

    返回：
      masks_np: shape 尽量为 [N, H, W]，找不到返回 None
      debug: state keys / shape / value range
    """
    debug = {
        "state_keys": list(state.keys()) if isinstance(state, dict) else [],
        "used_key": None,
        "raw_shape": None,
        "raw_min": None,
        "raw_max": None,
    }

    if not isinstance(state, dict):
        return None, debug

    candidate_keys = [
        "masks",
        "pred_masks",
        "mask",
        "mask_logits",
        "logits",
    ]

    obj = None
    used_key = None

    for k in candidate_keys:
        if k in state and state[k] is not None:
            obj = state[k]
            used_key = k
            break

    if obj is None:
        return None, debug

    arr = _tensor_to_numpy(obj)

    debug["used_key"] = used_key
    debug["raw_shape"] = list(arr.shape)
    if arr.size > 0:
        debug["raw_min"] = float(np.nanmin(arr))
        debug["raw_max"] = float(np.nanmax(arr))

    if arr.size == 0:
        return None, debug

    # squeeze 掉 batch/channel 的单维
    arr = np.squeeze(arr)

    # 可能形状：
    # [H, W]
    # [N, H, W]
    # [N, 1, H, W]
    # [1, N, H, W]
    if arr.ndim == 2:
        arr = arr[None, ...]

    elif arr.ndim == 3:
        # OK: [N,H,W] 或 [C,H,W]
        pass

    elif arr.ndim == 4:
        # 尝试 squeeze channel 维
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim != 3:
            return None, debug
    else:
        return None, debug

    # 如果是 logits，转 sigmoid；如果已经是 0/1 或 0~1，则保持
    if arr.size > 0:
        amin, amax = float(np.nanmin(arr)), float(np.nanmax(arr))
        if amin < 0.0 or amax > 1.0:
            arr = 1.0 / (1.0 + np.exp(-arr))

    return arr, debug


@torch.no_grad()
def generate_pseudo_masks(
    project_root: Path,
    checkpoint_path: Path,
    distill_manifest: Path,
    out_dir: Path,
    prompt: str,
    pseudo_limit: Optional[int],
    min_area_ratio: float,
    max_area_ratio: float,
    confidence_threshold: float,
    force_regen: bool = False,
    debug_pseudo_n: int = 20,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    pseudo_index_path = out_dir / "pseudo_index.json"

    if pseudo_index_path.exists() and not force_regen:
        print(f"[Pseudo] 已存在: {pseudo_index_path}")
        print("         如需重新生成，请使用 --force_pseudo_regen 或删除 pseudo_masks 目录。")
        return pseudo_index_path

    if pseudo_index_path.exists() and force_regen:
        print(f"[Pseudo] force_regen=True，删除旧 pseudo 目录: {out_dir}")
        import shutil
        shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    distill_paths = read_manifest(distill_manifest, limit=pseudo_limit)

    fallback_dir = project_root / "data" / "raw" / "processed_slim" / "target_whu" / "distill_pool" / "images"
    distill_paths = resolve_by_basename(distill_paths, fallback_dir)

    print(f"[Pseudo] generating teacher masks: N={len(distill_paths)} prompt='{prompt}'")

    teacher = build_sam3_image_model(checkpoint_path=str(checkpoint_path))
    teacher.to(DEVICE)
    teacher.eval()

    processor = Sam3Processor(teacher, confidence_threshold=confidence_threshold)

    items = []

    for img_path in tqdm(distill_paths, desc="  SAM3 text teacher", ncols=100):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            items.append({"image": str(img_path), "valid": False, "reason": "read_fail"})
            continue

        # 【修复】统一预处理: 对齐 E7 的处理管线
        #   1. 16-bit → 8-bit 归一化 (WHU-Mix 原始 tif 可能为 16-bit)
        #   2. resize 到 512×512 (SAM3 text prompt 零样本在原分辨率下置信度更高)
        if img_bgr.dtype != np.uint8:
            img_bgr = (img_bgr.astype(np.float32) / img_bgr.max() * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
        pil = Image.fromarray(img_rgb, mode="RGB")

        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                state = processor.set_image(pil)
                state = processor.set_text_prompt(state=state, prompt=prompt)

            masks_np, debug_info = extract_masks_from_state(state)

            if masks_np is None:
                pred = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
                reason = "no_mask_in_state"
            else:
                pred = np.any(masks_np > 0.5, axis=0).astype(np.uint8)
                pred = np.squeeze(pred)

                if pred.ndim != 2:
                    pred = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
                    reason = f"bad_pred_shape_after_agg:{list(np.array(pred).shape)}"
                else:
                    reason = "ok"

                pred = cv2.resize(
                    pred,
                    (TARGET_SIZE, TARGET_SIZE),
                    interpolation=cv2.INTER_NEAREST,
                )

            area_ratio = float(pred.mean())
            valid = (area_ratio >= min_area_ratio) and (area_ratio <= max_area_ratio)

            pseudo_path = out_dir / f"{img_path.stem}.png"
            cv2.imwrite(str(pseudo_path), pred * 255)

            # 保存前 N 张 debug 可视化图
            if len(items) < debug_pseudo_n:
                debug_dir = out_dir / "debug_vis"
                debug_dir.mkdir(parents=True, exist_ok=True)

                vis_img = img_rgb.copy()
                vis_mask = (pred * 255).astype(np.uint8)

                cv2.imwrite(
                    str(debug_dir / f"{img_path.stem}_image.png"),
                    cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(debug_dir / f"{img_path.stem}_pseudo.png"),
                    vis_mask,
                )

            debug_record = {
                "image": str(img_path),
                "pseudo_mask": str(pseudo_path),
                "valid": bool(valid),
                "area_ratio": area_ratio,
                "reason": reason,
                "state_debug": debug_info,
            }

            items.append(debug_record)

        except Exception as e:
            items.append({
                "image": str(img_path),
                "valid": False,
                "reason": repr(e),
            })

    area_values = [
        float(x.get("area_ratio", 0.0))
        for x in items
        if "area_ratio" in x
    ]

    reason_counts = {}
    for x in items:
        r = x.get("reason", "unknown")
        reason_counts[r] = reason_counts.get(r, 0) + 1

    if area_values:
        area_arr = np.array(area_values, dtype=float)
        area_stats = {
            "min": float(area_arr.min()),
            "mean": float(area_arr.mean()),
            "max": float(area_arr.max()),
            "num_zero": int((area_arr <= 0).sum()),
            "num_below_min": int((area_arr < min_area_ratio).sum()),
            "num_above_max": int((area_arr > max_area_ratio).sum()),
        }
    else:
        area_stats = {}

    index = {
        "prompt": prompt,
        "confidence_threshold": confidence_threshold,
        "min_area_ratio": min_area_ratio,
        "max_area_ratio": max_area_ratio,
        "items": items,
        "valid_count": sum(1 for x in items if x.get("valid", False)),
        "total_count": len(items),
        "area_stats": area_stats,
        "reason_counts": reason_counts,
    }

    pseudo_index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    del teacher, processor
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"[Pseudo] saved: {pseudo_index_path}")
    print(f"[Pseudo] valid: {index['valid_count']} / {index['total_count']}")

    return pseudo_index_path


# =============================================================================
# 6. Student model
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


class SAM3StudentExtractor:
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


def student_params(model, extractor):
    params = list(model.parameters())
    for p in extractor.model.parameters():
        if p.requires_grad:
            params.append(p)
    return params


def forward_batch(model, extractor, imgs):
    logits = []
    for i in range(imgs.shape[0]):
        feat = extractor.extract_one_grad(imgs[i])
        rgb = imgs[i].unsqueeze(0).to(DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            logit = model(feat.to(DEVICE), rgb)
        logits.append(logit)
    return torch.cat(logits, dim=0)


@torch.no_grad()
def predict_probs(model, extractor, loader):
    model.eval()
    probs_all, gts_all = [], []

    for batch in tqdm(loader, desc="  predict", ncols=100):
        imgs = batch["image"]
        masks = batch["mask"].numpy()[:, 0]

        feat = extractor.extract_batch_eval(imgs)
        rgb = imgs.to(DEVICE)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            logits = model(feat.to(DEVICE), rgb)

        probs = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]

        probs_all.append(probs)
        gts_all.append(masks)

    return np.concatenate(probs_all), np.concatenate(gts_all)


def evaluate_threshold_sweep(probs, gts):
    best = None
    rows = []

    for th in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        for min_area in [0, 16, 32, 64, 128]:
            preds = []
            for p in probs:
                pred = (p > th).astype(np.float32)
                pred = remove_small_components(pred, min_area)
                preds.append(pred)
            preds = np.stack(preds)
            m = eval_arrays(preds, gts)
            row = {
                "threshold": th,
                "min_area": min_area,
                **m,
            }
            rows.append(row)
            if best is None or row["mIoU"] > best["mIoU"]:
                best = row

    return {"best": best, "all": rows}


def evaluate_fixed(probs, gts, th, min_area):
    preds = []
    for p in probs:
        pred = (p > th).astype(np.float32)
        pred = remove_small_components(pred, min_area)
        preds.append(pred)
    preds = np.stack(preds)
    return eval_arrays(preds, gts)


# =============================================================================
# 7. Training
# =============================================================================

def train_variant(
    variant,
    checkpoint_path,
    train_loader,
    pseudo_loader,
    val_ckpt_loader,
    val_full_loader,
    pilot_loader,
    epochs,
    lr,
    lambda_pseudo,
    out_dir,
):
    print("\n" + "=" * 90)
    print(f"Train variant: {variant}")
    print("=" * 90)

    use_pseudo = variant == "distill_light"

    extractor = SAM3StudentExtractor(checkpoint_path)
    model = LightweightMaskDecoder().to(DEVICE)

    optimizer = torch.optim.AdamW(student_params(model, extractor), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    best_val = -1.0
    weight_path = out_dir / "weights" / f"{variant}_best.pth"
    weight_path.parent.mkdir(parents=True, exist_ok=True)

    history = []

    for epoch in range(1, epochs + 1):
        model.train()

        pseudo_iter = cycle(pseudo_loader) if (use_pseudo and pseudo_loader is not None) else None

        losses = []

        for sup_batch in tqdm(train_loader, desc=f"  {variant} epoch {epoch}/{epochs}", ncols=100):
            optimizer.zero_grad(set_to_none=True)

            imgs = sup_batch["image"]
            masks = sup_batch["mask"].to(DEVICE)

            logits = forward_batch(model, extractor, imgs)

            loss_sup = bce(logits.float(), masks.float()) + dice(logits.float(), masks.float())
            loss = loss_sup

            loss_pseudo_val = None

            if use_pseudo:
                pseudo_batch = next(pseudo_iter)
                p_imgs = pseudo_batch["image"]
                p_masks = pseudo_batch["pseudo"].to(DEVICE)

                p_logits = forward_batch(model, extractor, p_imgs)
                loss_pseudo = bce(p_logits.float(), p_masks.float()) + dice(p_logits.float(), p_masks.float())
                loss = loss + lambda_pseudo * loss_pseudo
                loss_pseudo_val = float(loss_pseudo.item())

            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        # checkpoint val @0.5
        if epoch % 5 == 0 or epoch == epochs:
            val_probs, val_gts = predict_probs(model, extractor, val_ckpt_loader)
            val_m = evaluate_fixed(val_probs, val_gts, 0.5, 0)
            val_score = val_m["mIoU"]

            print(f"  epoch={epoch} loss={np.mean(losses):.4f} val_mIoU@0.5={val_score:.4f}")

            if val_score > best_val:
                best_val = val_score
                torch.save({
                    "variant": variant,
                    "student": model.state_dict(),
                    "lora_params": {
                        k: v.detach().cpu()
                        for k, v in extractor.model.state_dict().items()
                        if "lora" in k.lower()
                    },
                    "epoch": epoch,
                    "best_val_mIoU_05": best_val,
                }, weight_path)
                print(f"  ★ saved: {weight_path}")

        history.append({"epoch": epoch, "loss": float(np.mean(losses))})

        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # reload best
    ckpt = torch.load(weight_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["student"], strict=True)

    state = extractor.model.state_dict()
    for k, v in ckpt["lora_params"].items():
        if k in state:
            state[k].copy_(v.to(DEVICE))

    # val sweep
    print("[Val] threshold sweep")
    val_probs, val_gts = predict_probs(model, extractor, val_full_loader)
    sweep = evaluate_threshold_sweep(val_probs, val_gts)

    th = float(sweep["best"]["threshold"])
    area = int(sweep["best"]["min_area"])

    print(f"  best threshold={th}, min_area={area}, val_mIoU={sweep['best']['mIoU']:.4f}")

    # pilot eval
    print("[Pilot500] eval")
    pilot_probs, pilot_gts = predict_probs(model, extractor, pilot_loader)
    pilot_05 = evaluate_fixed(pilot_probs, pilot_gts, 0.5, 0)
    pilot_calib = evaluate_fixed(pilot_probs, pilot_gts, th, area)

    result = {
        "variant": variant,
        "use_pseudo": use_pseudo,
        "lambda_pseudo": lambda_pseudo if use_pseudo else 0.0,
        "best_weight": str(weight_path),
        "history": history,
        "val_sweep_best": sweep["best"],
        "pilot_fixed_05": pilot_05,
        "pilot_calibrated": pilot_calib,
        "timestamp": datetime.now().isoformat(),
    }

    (out_dir / f"result_{variant}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    del model, extractor
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return result


# =============================================================================
# 8. CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser("Text prompt distillation pilot")

    parser.add_argument("--project_root", type=str, default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--manifest_dir", type=str, default=None)
    parser.add_argument("--sam3_checkpoint", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shot", type=int, default=20)
    parser.add_argument("--variants", type=str, default="sup_light,distill_light")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--pseudo_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lambda_pseudo", type=float, default=0.30)

    parser.add_argument("--pseudo_limit", type=int, default=1000)
    parser.add_argument("--pseudo_prompt", type=str, default="building")
    parser.add_argument("--teacher_confidence_threshold", type=float, default=0.20)
    parser.add_argument("--pseudo_min_area_ratio", type=float, default=0.0005)
    parser.add_argument("--pseudo_max_area_ratio", type=float, default=0.60)

    parser.add_argument("--val_eval_limit", type=int, default=200)
    parser.add_argument("--pilot_eval_limit", type=int, default=500)
    parser.add_argument(
        "--force_pseudo_regen",
        action="store_true",
        help="强制删除并重新生成 pseudo masks，避免复用旧的 0-valid pseudo_index.json。",
    )
    parser.add_argument(
        "--debug_pseudo_n",
        type=int,
        default=20,
        help="保存前 N 张 pseudo 生成调试图和 state 信息。",
    )
    parser.add_argument(
        "--allow_invalid_pseudo_for_debug",
        action="store_true",
        help="仅调试用：即使 valid=0 也不立刻崩溃，而是只训练 sup_light。",
    )

    parser.add_argument("--out_dir", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    project_root = Path(args.project_root)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else project_root / "data" / "splits" / "e0_manifest"
    checkpoint_path = Path(args.sam3_checkpoint) if args.sam3_checkpoint else project_root / "weights" / "sam3.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    out_dir = Path(args.out_dir) if args.out_dir else project_root / "results" / "distill_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = manifest_dir / f"target_support_{args.shot}_seed{args.seed}.txt"
    val_manifest = manifest_dir / "target_val.txt"
    pilot_manifest = manifest_dir / "target_pilot_test_500.txt"
    distill_manifest = manifest_dir / f"target_distill_pool_{args.pseudo_limit}.txt"

    print("=" * 90)
    print("Text-prompt distillation pilot")
    print("=" * 90)
    print(f"project_root     : {project_root}")
    print(f"manifest_dir     : {manifest_dir}")
    print(f"checkpoint       : {checkpoint_path}")
    print(f"train_manifest   : {train_manifest.name}")
    print(f"val_manifest     : {val_manifest.name}")
    print(f"pilot_manifest   : {pilot_manifest.name}")
    print(f"distill_manifest : {distill_manifest.name}")
    print(f"variants         : {args.variants}")
    print(f"device           : {DEVICE}")
    print("=" * 90)

    pseudo_dir = out_dir / "pseudo_masks" / f"{args.pseudo_prompt}_n{args.pseudo_limit}"
    pseudo_index = generate_pseudo_masks(
        project_root=project_root,
        checkpoint_path=checkpoint_path,
        distill_manifest=distill_manifest,
        out_dir=pseudo_dir,
        prompt=args.pseudo_prompt,
        pseudo_limit=args.pseudo_limit,
        min_area_ratio=args.pseudo_min_area_ratio,
        max_area_ratio=args.pseudo_max_area_ratio,
        confidence_threshold=args.teacher_confidence_threshold,
        force_regen=args.force_pseudo_regen,
        debug_pseudo_n=args.debug_pseudo_n,
    )

    train_ds = SupervisedManifestDataset(train_manifest)
    val_ckpt_ds = SupervisedManifestDataset(val_manifest, limit=args.val_eval_limit)
    val_full_ds = SupervisedManifestDataset(val_manifest, limit=None)
    pilot_ds = SupervisedManifestDataset(pilot_manifest, limit=args.pilot_eval_limit)
    pseudo_ds = None
    try:
        pseudo_ds = DistillPseudoDataset(pseudo_index)
    except RuntimeError as e:
        print(f"[Pseudo Warning] {e}")
        if not args.allow_invalid_pseudo_for_debug:
            raise
        print("[Pseudo Warning] allow_invalid_pseudo_for_debug=True，只允许运行 sup_light。")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_supervised,
        pin_memory=True,
    )

    pseudo_loader = None
    if pseudo_ds is not None:
        pseudo_loader = DataLoader(
            pseudo_ds,
            batch_size=args.pseudo_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_pseudo,
            pin_memory=True,
        )

    val_ckpt_loader = DataLoader(
        val_ckpt_ds,
        batch_size=4,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_supervised,
        pin_memory=True,
    )

    val_full_loader = DataLoader(
        val_full_ds,
        batch_size=4,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_supervised,
        pin_memory=True,
    )

    pilot_loader = DataLoader(
        pilot_ds,
        batch_size=4,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_supervised,
        pin_memory=True,
    )

    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    if pseudo_ds is None:
        variants = [v for v in variants if v == "sup_light"]
        if not variants:
            raise RuntimeError("pseudo_ds=None 且 variants 中没有 sup_light，无法继续。")
    results = []

    for v in variants:
        if v not in {"sup_light", "distill_light"}:
            raise ValueError(f"未知 variant: {v}")

        r = train_variant(
            variant=v,
            checkpoint_path=checkpoint_path,
            train_loader=train_loader,
            pseudo_loader=pseudo_loader,
            val_ckpt_loader=val_ckpt_loader,
            val_full_loader=val_full_loader,
            pilot_loader=pilot_loader,
            epochs=args.epochs,
            lr=args.lr,
            lambda_pseudo=args.lambda_pseudo,
            out_dir=out_dir,
        )
        results.append(r)

    summary = {
        "config": vars(args),
        "results": results,
        "pseudo_index": str(pseudo_index),
    }

    summary_path = out_dir / "distill_pilot_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("Distillation pilot summary")
    print("=" * 90)

    for r in results:
        fixed = r["pilot_fixed_05"]
        calib = r["pilot_calibrated"]
        print(
            f"{r['variant']:<14} "
            f"fixed mIoU={fixed['mIoU']*100:.2f} F1={fixed['F1']*100:.2f} BIoU={fixed['Boundary_IoU']*100:.2f} | "
            f"calib mIoU={calib['mIoU']*100:.2f} F1={calib['F1']*100:.2f} BIoU={calib['Boundary_IoU']*100:.2f}"
        )

    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()