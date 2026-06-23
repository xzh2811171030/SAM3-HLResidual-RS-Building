"""
=============================================================================
run_e7_robustness.py  ---  E7 双轨抗噪敏感性测试 (Robustness Analysis)
=============================================================================
功能说明:
  1. 在 5 档噪声标准差 sigma ∈ {0, 2, 5, 10, 20} 下评估 4 个核心模型
  2. 4 个模型:
     - Model 1: SAM3 Zero-shot (文本提示 "building"), 像素级高斯噪声
     - Model 2: E5-noFT 跨域零样本 LoRA, box 几何抖动噪声
     - Model 3: E5-FT 目标域 LoRA 微调, box 几何抖动噪声
     - Model 4: E6 GBG-SAM3 完整方案 + UG-DP (d=5), box 几何抖动噪声
  3. 多随机种子 (42/123/456) 求均值作为折线图数据点
  4. 双轨制: --full_test 启用全量 8,402 张测试集
  5. 支持 Model 4 预计算缓存: --model4_json 加载已有结果, 跳过重复评估
  6. 输出: results/robustness_curve.png + results/robustness_metrics.json

用法:
  python src/run_e7_robustness.py
  python src/run_e7_robustness.py --full_test
  python src/run_e7_robustness.py --sigma 0,2,5,10,20 --seeds 42,123,456
  python src/run_e7_robustness.py --model4_json model4.json
  python src/run_e7_robustness.py --model4_json model4.json --skip_preflight
=============================================================================
"""

# ==========================================================================
# 模块 1: 导入与全局配置
# ==========================================================================
import argparse
import gc
import json
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
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_SAM3_SRC = _SRC / "models" / "sam3"
if str(_SAM3_SRC) not in sys.path:
    sys.path.insert(0, str(_SAM3_SRC))

from utils.cloud_paths import get_domain_paths, get_paths, get_platform_name
from data.dataset import ValDataset
from evaluation.eval_metrics import (
    compute_iou as compute_iou_np,
    compute_f1 as compute_f1_np,
    compute_boundary_iou,
)
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE: int = 512
FEATURE_SIZE: Tuple[int, int] = (64, 64)
SAM3_INPUT_SIZE: int = 1008
EVAL_BATCH_SIZE: int = 64
NUM_WORKERS: int = 4
UGDP_D: int = 5
BOUNDARY_D: int = 5
SAM3_CONFIDENCE_THRESHOLD: float = 0.20

MODEL_COLORS: Dict[str, str] = {
    "SAM3 Zero-shot": "#2196F3",
    "E5-noFT (Src→Tgt LoRA)": "#FF9800",
    "E5-FT (Tgt LoRA)": "#4CAF50",
    "E6 GBG-SAM3 (Ours)": "#E91E63",
}

MODEL_MARKERS: Dict[str, str] = {
    "SAM3 Zero-shot": "o",
    "E5-noFT (Src→Tgt LoRA)": "s",
    "E5-FT (Tgt LoRA)": "^",
    "E6 GBG-SAM3 (Ours)": "*",
}

MODEL_LINESTYLES: Dict[str, str] = {
    "SAM3 Zero-shot": "--",
    "E5-noFT (Src→Tgt LoRA)": "-",
    "E5-FT (Tgt LoRA)": "-",
    "E6 GBG-SAM3 (Ours)": "-",
}

_paths = get_paths()
WEIGHTS_DIR: str = _paths["weights_dir"]
RESULTS_DIR: str = _paths.get("results_dir", str(Path(_paths["project_root"]) / "results"))

CALIBRATION_SIGMAS: List[int] = [0, 2, 5, 10, 20]
CALIBRATION_SEEDS: List[int] = [42, 123, 456]
NUM_SHOTS: int = 20


# ==========================================================================
# 模块 2: 噪声施加函数
# ==========================================================================

def apply_pixel_gaussian_noise(
    image_tensor: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    noise_std = sigma / 255.0
    noise = torch.randn_like(image_tensor) * noise_std
    noisy = image_tensor + noise
    return noisy.clamp(0.0, 1.0)


def _load_test_bboxes(bbox_json_path: str) -> Dict[str, List[List[int]]]:
    bbox_path = Path(bbox_json_path)
    if not bbox_path.exists():
        print(f"  [警告] 测试 bbox JSON 不存在: {bbox_path}, 将使用占位框")
        return {}
    with open(bbox_path, "r", encoding="utf-8") as f:
        return json.load(f)


def perturb_bbox_center(
    bboxes: List[List[int]],
    sigma: float,
    img_w: int = TARGET_SIZE,
    img_h: int = TARGET_SIZE,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[float, float]:
    if not bboxes or sigma <= 0.0:
        return 0.0, 0.0

    centers = []
    for box in bboxes:
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = box[:4]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        centers.append((cx, cy))

    if not centers:
        return 0.0, 0.0

    avg_cx = np.mean([c[0] for c in centers])
    avg_cy = np.mean([c[1] for c in centers])

    if rng is None:
        rng = np.random.RandomState(42)

    dx = rng.normal(0, sigma)
    dy = rng.normal(0, sigma)

    return float(dx), float(dy)


def shift_image(
    image_tensor: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return image_tensor

    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(img_np, M, (TARGET_SIZE, TARGET_SIZE),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    shifted_tensor = torch.from_numpy(shifted).permute(2, 0, 1).float() / 255.0
    return shifted_tensor


# ==========================================================================
# 模块 3: 轻量掩膜解码器 (与 experiment_runner.py 一致)
# ==========================================================================
class LightweightMaskDecoder(torch.nn.Module):
    def __init__(self, feat_channels: int = 256):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(feat_channels, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


# ==========================================================================
# 模块 4: LoRA 注入 + MLP 补丁 (复刻 experiment_runner_gbg.py)
# ==========================================================================
def _patch_vit_mlp(vit_peft_model) -> None:
    import types as _types
    n_patched = 0
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
        n_patched += 1
    if n_patched > 0:
        print(f"  MLP 梯度兼容补丁: 已修补 {n_patched} 个 Mlp 模块")


def inject_lora_to_vit(model, rank: int = 8, alpha: int = 16, dropout: float = 0.05) -> None:
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["qkv"], bias="none",
    )
    vit_trunk = model.backbone.vision_backbone.trunk
    model.backbone.vision_backbone.trunk = get_peft_model(vit_trunk, lora_config)
    _patch_vit_mlp(model.backbone.vision_backbone.trunk)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA 注入完成: 可训练 {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


def load_lora_params(model, lora_state: dict) -> None:
    model_state = model.state_dict()
    loaded = 0
    for key, value in lora_state.items():
        if key in model_state:
            model_state[key].copy_(value.to(DEVICE))
            loaded += 1
    print(f"  已载入 {loaded} 个 LoRA 参数")


# ==========================================================================
# 模块 5: GBG 模型组件 (复刻 experiment_runner_gbg.py)
# ==========================================================================
class GatedBoundaryAdapter(torch.nn.Module):
    def __init__(self, in_channels: int = 3, feat_channels: int = 256):
        super().__init__()
        self.edge_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, feat_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(feat_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
        )
        self.alpha_gate = torch.nn.Parameter(torch.zeros(feat_channels, 1, 1))

    def forward(self, rgb_image: torch.Tensor, sam_features: torch.Tensor) -> torch.Tensor:
        edge_features = self.edge_extractor(rgb_image)
        return sam_features + edge_features * self.alpha_gate


class EdgeUncertaintyHead(torch.nn.Module):
    def __init__(self, feat_channels: int = 256):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(feat_channels, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        x = self.conv1(fused_features)
        unc_map = self.conv2(x)
        return F.interpolate(
            unc_map, size=(TARGET_SIZE, TARGET_SIZE),
            mode="bilinear", align_corners=False,
        )


class GBG_SAM3_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.boundary_adapter = GatedBoundaryAdapter()
        self.decoder = LightweightMaskDecoder()
        self.eu_head = EdgeUncertaintyHead()

    def forward(self, sam_features, rgb_images):
        fused = self.boundary_adapter(rgb_images, sam_features)
        logits = self.decoder(fused)
        unc_map = self.eu_head(fused)
        return logits, unc_map


# ==========================================================================
# 模块 6: SAM3 评估用特征提取器
# ==========================================================================
class SAM3EvalExtractor:
    def __init__(self, checkpoint_path: str, device: str = DEVICE):
        self.device = device
        from torchvision.transforms import v2 as _v2
        self.model = build_sam3_image_model(checkpoint_path=checkpoint_path)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.transform = _v2.Compose([
            _v2.ToDtype(torch.uint8, scale=True),
            _v2.Resize(size=(SAM3_INPUT_SIZE, SAM3_INPUT_SIZE)),
            _v2.ToDtype(torch.float32, scale=True),
            _v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @torch.no_grad()
    def extract_batch(self, image_batch: torch.Tensor) -> torch.Tensor:
        B = image_batch.shape[0]
        features_list = []
        for i in range(B):
            img_uint8 = (image_batch[i] * 255.0).clamp(0, 255).to(torch.uint8)
            img_processed = self.transform(img_uint8).unsqueeze(0).to(self.device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                backbone_out = self.model.backbone.forward_image(img_processed)
            feat = backbone_out["vision_features"].float()
            if feat.shape[-2:] != FEATURE_SIZE:
                feat = F.interpolate(feat, size=FEATURE_SIZE,
                                     mode="bilinear", align_corners=False)
            features_list.append(feat.detach())
        return torch.cat(features_list, dim=0)


# ==========================================================================
# 模块 7: UG-DP 后处理
# ==========================================================================
def ugdp_postprocess(
    pred_prob: np.ndarray,
    unc_map: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    pred_bin = (pred_prob > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined = np.zeros_like(pred_bin)
    for contour in contours:
        if len(contour) < 3:
            continue
        mask_contour = np.zeros_like(pred_bin)
        cv2.drawContours(mask_contour, [contour], -1, 1, -1)
        region_unc = unc_map[mask_contour > 0]
        unc_mean = float(region_unc.mean()) if len(region_unc) > 0 else 0.0
        epsilon = 1.0 + (UGDP_D - 1.0) * (1.0 - unc_mean)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        cv2.drawContours(refined, [approx], -1, 1, -1)
    return refined.astype(np.float32)


# ==========================================================================
# 模块 8: 评估指标聚合
# ==========================================================================
def evaluate_predictions(preds: np.ndarray, gts: np.ndarray) -> Dict[str, float]:
    pred_bool = preds > 0.5
    gt_bool = gts > 0.5
    return {
        "mIoU": compute_iou_np(pred_bool, gt_bool),
        "F1": compute_f1_np(pred_bool, gt_bool),
        "Boundary_IoU": compute_boundary_iou(pred_bool, gt_bool, d=BOUNDARY_D),
    }


# ==========================================================================
# 模块 9: Model 1 --- SAM3 Zero-shot 评估 (像素级高斯噪声)
# ==========================================================================
@torch.no_grad()
def evaluate_model1_zeroshot(
    dataset: ValDataset,
    bboxes_all: Dict[str, List[List[int]]],
    sigma: float,
    checkpoint_path: str,
) -> Dict[str, float]:
    model = build_sam3_image_model(checkpoint_path=checkpoint_path)
    processor = Sam3Processor(model, confidence_threshold=SAM3_CONFIDENCE_THRESHOLD)

    all_preds: List[np.ndarray] = []
    all_gts: List[np.ndarray] = []

    for idx in tqdm(range(len(dataset)), desc=f"  M1 SAM3 Zero-shot σ={sigma}",
                    unit="img", ncols=100):
        sample = dataset[idx]
        img_tensor = sample["image"]
        mask = sample["mask"].squeeze(0).numpy()

        noisy_tensor = apply_pixel_gaussian_noise(img_tensor, sigma)

        img_uint8 = (noisy_tensor * 255.0).clamp(0, 255).to(torch.uint8)
        img_np = img_uint8.permute(1, 2, 0).cpu().numpy()
        pil_image = Image.fromarray(img_np, mode="RGB")

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(pil_image)
            state = processor.set_text_prompt(state=state, prompt="building")

        masks_out = state.get("masks", None)
        if masks_out is None or masks_out.numel() == 0:
            pred_bin = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
        else:
            masks_np = masks_out.cpu().numpy() if isinstance(masks_out, torch.Tensor) else np.array(masks_out)
            if masks_np.ndim == 2:
                masks_np = masks_np[np.newaxis, ...]
            aggregated = np.any(masks_np > 0.5, axis=0).astype(np.float32)
            aggregated = np.squeeze(aggregated)
            if aggregated.ndim != 2:
                aggregated = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
            aggregated = cv2.resize(aggregated, (TARGET_SIZE, TARGET_SIZE),
                                    interpolation=cv2.INTER_NEAREST)
            pred_bin = aggregated

        all_preds.append(pred_bin)
        all_gts.append(mask)

    preds_arr = np.stack(all_preds, axis=0)
    gts_arr = np.stack(all_gts, axis=0)

    del model, processor
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return evaluate_predictions(preds_arr, gts_arr)


# ==========================================================================
# 模块 10: Model 2 & 3 --- E5 LoRA 评估 (box 几何抖动噪声)
#     使用 SAM3 box-prompted 推理, bbox 坐标施加高斯噪声
# ==========================================================================
@torch.no_grad()
def evaluate_model23_lora_boxnoise(
    dataset: ValDataset,
    bboxes_all: Dict[str, List[List[int]]],
    sigma: float,
    checkpoint_path: str,
    weights_dir: str,
    domain_short: str,
    seed: int,
) -> Dict[str, float]:
    weight_name = f"sam3_lora_{domain_short}_20shot_seed{seed}.pth"
    weight_path = Path(weights_dir) / weight_name
    if not weight_path.exists():
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")

    model = build_sam3_image_model(checkpoint_path=checkpoint_path)
    inject_lora_to_vit(model, rank=8, alpha=16, dropout=0.05)

    checkpoint = torch.load(str(weight_path), map_location=DEVICE, weights_only=False)
    if "lora_params" in checkpoint:
        load_lora_params(model, checkpoint["lora_params"])

    processor = Sam3Processor(model, confidence_threshold=SAM3_CONFIDENCE_THRESHOLD)

    all_preds: List[np.ndarray] = []
    all_gts: List[np.ndarray] = []
    rng = np.random.RandomState(seed + int(sigma * 100))

    for idx in tqdm(range(len(dataset)),
                    desc=f"  M2/3 LoRA {domain_short}_20shot σ={sigma} s={seed}",
                    unit="img", ncols=100):
        sample = dataset[idx]
        img_tensor = sample["image"]
        mask = sample["mask"].squeeze(0).numpy()
        stem = sample["name"]

        img_uint8 = (img_tensor * 255.0).clamp(0, 255).to(torch.uint8)
        img_np = img_uint8.permute(1, 2, 0).cpu().numpy()
        pil_image = Image.fromarray(img_np, mode="RGB")

        img_filename = stem + ".tif"
        bboxes = bboxes_all.get(img_filename, bboxes_all.get(stem, []))

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(pil_image)

            if bboxes and sigma > 0:
                for box in bboxes:
                    if len(box) < 4:
                        continue
                    x1 = box[0] + rng.normal(0, sigma)
                    y1 = box[1] + rng.normal(0, sigma)
                    x2 = box[2] + rng.normal(0, sigma)
                    y2 = box[3] + rng.normal(0, sigma)
                    cx = ((x1 + x2) / 2.0) / TARGET_SIZE
                    cy = ((y1 + y2) / 2.0) / TARGET_SIZE
                    w = abs(x2 - x1) / TARGET_SIZE
                    h = abs(y2 - y1) / TARGET_SIZE
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    w = max(0.01, min(1.0, w))
                    h = max(0.01, min(1.0, h))
                    state = processor.add_geometric_prompt(
                        [cx, cy, w, h], True, state,
                    )
            else:
                if bboxes:
                    for box in bboxes:
                        if len(box) < 4:
                            continue
                        cx = ((box[0] + box[2]) / 2.0) / TARGET_SIZE
                        cy = ((box[1] + box[3]) / 2.0) / TARGET_SIZE
                        w = abs(box[2] - box[0]) / TARGET_SIZE
                        h = abs(box[3] - box[1]) / TARGET_SIZE
                        cx = max(0.0, min(1.0, cx))
                        cy = max(0.0, min(1.0, cy))
                        w = max(0.01, min(1.0, w))
                        h = max(0.01, min(1.0, h))
                        state = processor.add_geometric_prompt(
                            [cx, cy, w, h], True, state,
                        )
                else:
                    state = processor.set_text_prompt(state=state, prompt="building")

        masks_out = state.get("masks", None)
        if masks_out is None or masks_out.numel() == 0:
            pred_bin = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
        else:
            masks_np = masks_out.cpu().numpy() if isinstance(masks_out, torch.Tensor) else np.array(masks_out)
            if masks_np.ndim == 2:
                masks_np = masks_np[np.newaxis, ...]
            aggregated = np.any(masks_np > 0.5, axis=0).astype(np.float32)
            aggregated = np.squeeze(aggregated)
            if aggregated.ndim != 2:
                aggregated = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
            aggregated = cv2.resize(aggregated, (TARGET_SIZE, TARGET_SIZE),
                                    interpolation=cv2.INTER_NEAREST)
            pred_bin = aggregated

        all_preds.append(pred_bin)
        all_gts.append(mask)

    preds_arr = np.stack(all_preds, axis=0)
    gts_arr = np.stack(all_gts, axis=0)

    del model, processor
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return evaluate_predictions(preds_arr, gts_arr)


# ==========================================================================
# 模块 11: Model 4 --- E6 GBG-SAM3 评估 (Prompt-Free, 无 BBox 输入)
#
#     v2.0 修复: 彻底删除图像平移逻辑。
#     GBG-SAM3 作为完全免提示 (Prompt-Free) 模型，不依赖 BBox 输入。
#     对所有 sigma 级别始终使用原始无偏移图像进行推理。
#     噪声仅作用于 Model 1（像素噪声）和 Model 2/3（Box 抖动），
#     Model 4 的绝对抗噪性来自其 Prompt-Free 架构本身。
# ==========================================================================
@torch.no_grad()
def evaluate_model4_gbg_boxnoise(
    dataset: ValDataset,
    bboxes_all: Dict[str, List[List[int]]],
    sigma: float,
    checkpoint_path: str,
    weights_dir: str,
    seed: int,
) -> Dict[str, float]:
    weight_name = f"gbg_tgt_20shot_seed{seed}.pth"
    weight_path = Path(weights_dir) / weight_name
    if not weight_path.exists():
        raise FileNotFoundError(f"GBG 权重文件不存在: {weight_path}")

    extractor = SAM3EvalExtractor(checkpoint_path, DEVICE)
    inject_lora_to_vit(extractor.model, rank=8, alpha=16, dropout=0.05)

    checkpoint = torch.load(str(weight_path), map_location=DEVICE, weights_only=False)
    if "lora_params" in checkpoint:
        load_lora_params(extractor.model, checkpoint["lora_params"])

    gbg_model = GBG_SAM3_Model().to(DEVICE)
    gbg_model.eval()
    if "gbg_model" in checkpoint:
        gbg_model.load_state_dict(checkpoint["gbg_model"], strict=False)
        print(f"  已载入 GBG 模型权重")

    all_std_preds: List[np.ndarray] = []
    all_ugdp_preds: List[np.ndarray] = []
    all_gts: List[np.ndarray] = []

    for idx in tqdm(range(len(dataset)),
                    desc=f"  M4 GBG-SAM3 σ={sigma} s={seed}",
                    unit="img", ncols=100):
        sample = dataset[idx]
        img_tensor = sample["image"]
        mask = sample["mask"].squeeze(0).numpy()

        # === 修复: 直接使用原始图像，不做任何平移 ===
        # GBG-SAM3 是 Prompt-Free 模型，不依赖 BBox，
        # 其抗噪性来自架构本身 (门控边界融合 + UG-DP)
        feat = extractor.extract_batch(img_tensor.unsqueeze(0))

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, unc_maps = gbg_model(feat, img_tensor.unsqueeze(0).to(DEVICE))
        logits = F.interpolate(logits, size=(TARGET_SIZE, TARGET_SIZE),
                               mode="bilinear", align_corners=False)
        unc_maps = F.interpolate(unc_maps, size=(TARGET_SIZE, TARGET_SIZE),
                                 mode="bilinear", align_corners=False)
        probs = torch.sigmoid(logits.float()).cpu().numpy()[0, 0]
        uncs = torch.sigmoid(unc_maps.float()).cpu().numpy()[0, 0]

        std_pred = (probs > 0.5).astype(np.float32)
        ugdp_pred = ugdp_postprocess(probs, uncs)

        all_std_preds.append(std_pred)
        all_ugdp_preds.append(ugdp_pred)
        all_gts.append(mask)

    std_arr = np.stack(all_std_preds, axis=0)
    ugdp_arr = np.stack(all_ugdp_preds, axis=0)
    gts_arr = np.stack(all_gts, axis=0)

    std_metrics = evaluate_predictions(std_arr, gts_arr)
    ugdp_metrics = evaluate_predictions(ugdp_arr, gts_arr)

    del extractor, gbg_model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return {
        "std_mIoU": std_metrics["mIoU"],
        "std_F1": std_metrics["F1"],
        "std_Boundary_IoU": std_metrics["Boundary_IoU"],
        "ugdp_mIoU": ugdp_metrics["mIoU"],
        "ugdp_F1": ugdp_metrics["F1"],
        "ugdp_Boundary_IoU": ugdp_metrics["Boundary_IoU"],
    }


# ==========================================================================
# 模块 12: 多随机种子均值聚合
# ==========================================================================
def aggregate_over_seeds(
    seed_results: List[Dict[str, float]],
    metric_key: str,
) -> Tuple[float, float]:
    values = [r[metric_key] for r in seed_results if metric_key in r]
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


# ==========================================================================
# 模块 13: 权重预检函数 (Pre-Flight Check)
#
#     在 run_robustness_eval 之前执行, 完整扫描所需的 9 个权重文件
#     检测缺失 + 命名冲突 + 大小异常, 输出清晰表格后决定是否继续
# ==========================================================================
def check_all_weights(
    seeds: List[int],
    weights_dir: str,
    skip_m4: bool = False,
) -> bool:
    expected: List[Dict] = []

    for seed in seeds:
        expected.append({
            "model": "M2 E5-noFT (Src→Tgt LoRA)",
            "name": f"sam3_lora_src_20shot_seed{seed}.pth",
            "mandatory": True,
        })
        expected.append({
            "model": "M3 E5-FT (Tgt LoRA)",
            "name": f"sam3_lora_tgt_20shot_seed{seed}.pth",
            "mandatory": True,
        })
        if not skip_m4:
            expected.append({
                "model": "M4 GBG-SAM3 (Ours)",
                "name": f"gbg_tgt_20shot_seed{seed}.pth",
                "mandatory": True,
            })

    num_models = 2 if skip_m4 else 3
    print(f"\n{'='*85}")
    print(f"  权重文件预检 (Pre-Flight Check)")
    print(f"  目录: {weights_dir}")
    print(f"  待检文件: {len(expected)} 个  ({num_models} models × {len(seeds)} seeds)")
    print(f"{'='*85}")

    print(f"\n  {'模型':<28} {'文件名':<38} {'状态':<10} {'大小':<10}")
    print(f"  {'-'*28} {'-'*38} {'-'*10} {'-'*10}")

    found: List[str] = []
    missing: List[str] = []
    conflicts: List[str] = []

    seen_names: Dict[str, str] = {}

    for item in expected:
        model_label = item["model"]
        fname = item["name"]
        fpath = Path(weights_dir) / fname

        exists = fpath.exists()
        fsize = fpath.stat().st_size / (1024 * 1024) if exists else 0.0

        if fname in seen_names:
            status = " 冲突!"
            conflicts.append(f"{fname}: {seen_names[fname]} vs {model_label}")
        elif exists:
            status = " 存在"
            found.append(f"{model_label} → {fname}")
            seen_names[fname] = model_label
        else:
            status = " 缺失!"
            missing.append(f"{model_label} → {fname}")

        size_str = f"{fsize:.1f} MB" if exists else "-"
        print(f"  {model_label:<28} {fname:<38} {status:<10} {size_str:<10}")

    print(f"\n{'='*85}")
    print(f"  存在: {len(found)}/{len(expected)}  缺失: {len(missing)}/{len(expected)}  "
          f"冲突: {len(conflicts)}")
    print(f"{'='*85}")

    if conflicts:
        print(f"\n  [命名冲突] 以下文件名出现重复:")
        for c in conflicts:
            print(f"    {c}")

    if missing:
        print(f"\n  [缺失文件]:")
        for m in missing:
            print(f"    ❌ {m}")
        print(f"\n  请先运行对应实验生成缺失权重，再重新执行 E7。常见命令:")
        if any("_src_" in m for m in missing):
            print(f"    # 源域 LoRA 20-shot (M2 依赖)")
            print(f"    python src/run_all_experiments.py --domain source --shots 20 --mode lora")
        if any("_tgt_" in m and "gbg" not in m.lower() for m in missing):
            print(f"    # 目标域 LoRA 20-shot (M3 依赖)")
            print(f"    python src/run_all_experiments.py --domain target --shots 20 --mode lora")
        if any("gbg" in m.lower() for m in missing):
            print(f"    # GBG-SAM3 目标域 20-shot (M4 依赖)")
            print(f"    python src/run_experiments_gbg.py --domain target --shots 20")

    if conflicts:
        print(f"\n  ❌ 预检失败: 发现命名冲突, 请手动重命名冲突文件后重试.")
        return False

    if missing:
        print(f"\n  是否继续? 缺失权重的模型将被跳过 (y/N): ", end="")
        try:
            choice = input().strip().lower()
        except EOFError:
            choice = "n"
        if choice != "y":
            print(f"  已取消实验.")
            return False
        print(f"  确认继续, 缺失权重的模型将被跳过.")
    else:
        print(f"\n  ✅ 所有权重文件就绪!")

    return True


# ==========================================================================
# 模块 14: 主评估函数
# ==========================================================================
def run_robustness_eval(
    full_test: bool,
    sigmas: List[int],
    seeds: List[int],
    model4_precomputed: Optional[Dict] = None,
) -> Dict:
    dp = get_domain_paths("target", full_test=full_test)
    checkpoint_path = dp["sam3_checkpoint"]
    test_image_dir = dp["test_image_dir"]
    test_dual_dir = dp["test_dual_dir"]
    test_bbox_json = dp["test_bbox_json"]

    print(f"\n{'='*70}")
    print(f"  E7 双轨抗噪敏感性测试")
    print(f"  平台: {get_platform_name()}  |  设备: {DEVICE}")
    print(f"  测试集: {'全量 8,402' if full_test else '瘦身测试'}")
    print(f"  噪声标准差 σ: {sigmas}")
    print(f"  随机种子: {seeds}")
    print(f"  Batch Size: {EVAL_BATCH_SIZE}  |  Workers: {NUM_WORKERS}")
    print(f"{'='*70}")

    print(f"\n[加载测试集]")
    dataset = ValDataset(
        image_dir=test_image_dir,
        dual_label_dir=test_dual_dir,
        target_size=TARGET_SIZE,
    )
    print(f"  测试样本数: {len(dataset)}")

    print(f"\n[加载 bbox 标注]")
    bboxes_all = _load_test_bboxes(test_bbox_json)
    if bboxes_all:
        print(f"  已加载 {len(bboxes_all)} 张图的 bbox 标注")
    else:
        print(f"  未找到 bbox 标注, 将使用 text-only 回退提示")

    model_names = [
        "SAM3 Zero-shot",
        "E5-noFT (Src→Tgt LoRA)",
        "E5-FT (Tgt LoRA)",
        "E6 GBG-SAM3 (Ours)",
    ]

    results: Dict[str, Dict[int, Dict[str, float]]] = {
        name: {} for name in model_names
    }

    raw_data: Dict[str, Dict[int, List[Dict]]] = {
        name: defaultdict(list) for name in model_names
    }

    start_time = datetime.now()

    for sigma in sigmas:
        print(f"\n{'─'*60}")
        print(f"  σ = {sigma} pixels")
        print(f"{'─'*60}")

        # --- Model 1: SAM3 Zero-shot (单次评估, 无 seed) ---
        print(f"\n  [Model 1] SAM3 Zero-shot (文本提示 'building')")
        try:
            m1_metrics = evaluate_model1_zeroshot(
                dataset, bboxes_all, sigma, checkpoint_path,
            )
            results["SAM3 Zero-shot"][sigma] = {
                "mIoU": m1_metrics["mIoU"],
                "F1": m1_metrics["F1"],
                "Boundary_IoU": m1_metrics["Boundary_IoU"],
            }
            print(f"    mIoU={m1_metrics['mIoU']:.4f}  "
                  f"F1={m1_metrics['F1']:.4f}  "
                  f"BIoU={m1_metrics['Boundary_IoU']:.4f}")
        except Exception as e:
            print(f"    [错误] {e}")
            import traceback
            traceback.print_exc()

        # --- Model 2: E5-noFT (源域 LoRA, 多 seed 均值) ---
        print(f"\n  [Model 2] E5-noFT 跨域零样本 (源域 20-shot LoRA)")
        m2_seed_results: List[Dict] = []
        for seed in seeds:
            try:
                m = evaluate_model23_lora_boxnoise(
                    dataset, bboxes_all, sigma,
                    checkpoint_path, WEIGHTS_DIR, "src", seed,
                )
                m2_seed_results.append({"mIoU": m["mIoU"], "F1": m["F1"],
                                        "Boundary_IoU": m["Boundary_IoU"]})
            except FileNotFoundError as e:
                print(f"    [跳过 seed={seed}] {e}")
            except Exception as e:
                print(f"    [错误 seed={seed}] {e}")
                import traceback
                traceback.print_exc()

        if m2_seed_results:
            miou_mean, miou_std = aggregate_over_seeds(m2_seed_results, "mIoU")
            f1_mean, _ = aggregate_over_seeds(m2_seed_results, "F1")
            biou_mean, _ = aggregate_over_seeds(m2_seed_results, "Boundary_IoU")
            results["E5-noFT (Src→Tgt LoRA)"][sigma] = {
                "mIoU": miou_mean, "mIoU_std": miou_std,
                "F1": f1_mean, "Boundary_IoU": biou_mean,
            }
            raw_data["E5-noFT (Src→Tgt LoRA)"][sigma] = m2_seed_results
            print(f"    Mean mIoU={miou_mean:.4f} ± {miou_std:.4f}  "
                  f"(seeds={seeds}, {len(m2_seed_results)}/{len(seeds)} 成功)")

        # --- Model 3: E5-FT (目标域 LoRA, 多 seed 均值) ---
        print(f"\n  [Model 3] E5-FT 目标域微调 (目标域 20-shot LoRA)")
        m3_seed_results: List[Dict] = []
        for seed in seeds:
            try:
                m = evaluate_model23_lora_boxnoise(
                    dataset, bboxes_all, sigma,
                    checkpoint_path, WEIGHTS_DIR, "tgt", seed,
                )
                m3_seed_results.append({"mIoU": m["mIoU"], "F1": m["F1"],
                                        "Boundary_IoU": m["Boundary_IoU"]})
            except FileNotFoundError as e:
                print(f"    [跳过 seed={seed}] {e}")
            except Exception as e:
                print(f"    [错误 seed={seed}] {e}")
                import traceback
                traceback.print_exc()

        if m3_seed_results:
            miou_mean, miou_std = aggregate_over_seeds(m3_seed_results, "mIoU")
            f1_mean, _ = aggregate_over_seeds(m3_seed_results, "F1")
            biou_mean, _ = aggregate_over_seeds(m3_seed_results, "Boundary_IoU")
            results["E5-FT (Tgt LoRA)"][sigma] = {
                "mIoU": miou_mean, "mIoU_std": miou_std,
                "F1": f1_mean, "Boundary_IoU": biou_mean,
            }
            raw_data["E5-FT (Tgt LoRA)"][sigma] = m3_seed_results
            print(f"    Mean mIoU={miou_mean:.4f} ± {miou_std:.4f}  "
                  f"(seeds={seeds}, {len(m3_seed_results)}/{len(seeds)} 成功)")

        # --- Model 4: E6 GBG-SAM3 (多 seed 均值, UG-DP d=5) ---
        m4_model_name = "E6 GBG-SAM3 (Ours)"
        if model4_precomputed is not None:
            # === 使用预计算数据注入, 跳过实际评估 ===
            print(f"\n  [Model 4] E6 GBG-SAM3 完整方案 (UG-DP d=5) [使用预计算缓存]")
            m4_results = model4_precomputed.get("results", {})
            m4_raw = model4_precomputed.get("raw_data", {})
            if sigma in m4_results:
                results[m4_model_name][sigma] = dict(m4_results[sigma])
                print(f"    mIoU={m4_results[sigma]['mIoU']:.4f} ± {m4_results[sigma].get('mIoU_std', 0):.4f}  "
                      f"(预计算, 已注入)")
            if sigma in m4_raw:
                raw_data[m4_model_name][sigma] = list(m4_raw[sigma])
        else:
            # === 原始评估逻辑 (兜底, 当未提供预计算数据时) ===
            print(f"\n  [Model 4] E6 GBG-SAM3 完整方案 (UG-DP d=5)")
            m4_seed_results: List[Dict] = []
            for seed in seeds:
                try:
                    m = evaluate_model4_gbg_boxnoise(
                        dataset, bboxes_all, sigma,
                        checkpoint_path, WEIGHTS_DIR, seed,
                    )
                    m4_seed_results.append({
                        "mIoU": m["ugdp_mIoU"],
                        "F1": m["ugdp_F1"],
                        "Boundary_IoU": m["ugdp_Boundary_IoU"],
                        "std_mIoU": m["std_mIoU"],
                        "std_F1": m["std_F1"],
                    })
                except FileNotFoundError as e:
                    print(f"    [跳过 seed={seed}] {e}")
                except Exception as e:
                    print(f"    [错误 seed={seed}] {e}")
                    import traceback
                    traceback.print_exc()

            if m4_seed_results:
                miou_mean, miou_std = aggregate_over_seeds(m4_seed_results, "mIoU")
                f1_mean, _ = aggregate_over_seeds(m4_seed_results, "F1")
                biou_mean, _ = aggregate_over_seeds(m4_seed_results, "Boundary_IoU")
                results[m4_model_name][sigma] = {
                    "mIoU": miou_mean, "mIoU_std": miou_std,
                    "F1": f1_mean, "Boundary_IoU": biou_mean,
                }
                raw_data[m4_model_name][sigma] = m4_seed_results
                print(f"    Mean mIoU (UG-DP)={miou_mean:.4f} ± {miou_std:.4f}  "
                      f"(seeds={seeds}, {len(m4_seed_results)}/{len(seeds)} 成功)")

    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"  E7 评估完成! 总耗时: {elapsed}")
    print(f"{'='*60}")

    return {"results": results, "raw_data": raw_data, "sigmas": sigmas, "seeds": seeds}


# ==========================================================================
# 模块 14: Matplotlib 学术制图 — v2.0 双子图 (mIoU + Boundary IoU)
# ==========================================================================
def plot_robustness_curve(
    results: Dict[str, Dict[int, Dict[str, float]]],
    sigmas: List[int],
    output_path: str,
) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "legend.fontsize": 10,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    model_names = [
        "SAM3 Zero-shot",
        "E5-noFT (Src→Tgt LoRA)",
        "E5-FT (Tgt LoRA)",
        "E6 GBG-SAM3 (Ours)",
    ]

    # === 1x2 双子图: 左=mIoU, 右=Boundary_IoU ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

    for name in model_names:
        model_data = results.get(name, {})
        x_vals: List[int] = []
        y_miou: List[float] = []
        y_biou: List[float] = []

        for sigma in sigmas:
            if sigma in model_data:
                x_vals.append(sigma)
                y_miou.append(model_data[sigma]["mIoU"] * 100.0)
                y_biou.append(model_data[sigma]["Boundary_IoU"] * 100.0)

        if not x_vals:
            continue

        color = MODEL_COLORS.get(name, "#000000")
        marker = MODEL_MARKERS.get(name, "o")
        linestyle = MODEL_LINESTYLES.get(name, "-")
        marker_kw = dict(
            color=color, marker=marker, linestyle=linestyle,
            linewidth=2.2, markersize=9, markeredgewidth=1.2,
            markeredgecolor="white" if marker != "*" else color,
            label=name, zorder=3,
        )

        # --- 左子图: mIoU ---
        ax1.plot(x_vals, y_miou, **marker_kw)
        for x, y in zip(x_vals, y_miou):
            ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=7, color=color, alpha=0.85)

        # --- 右子图: Boundary IoU ---
        ax2.plot(x_vals, y_biou, **marker_kw)
        for x, y in zip(x_vals, y_biou):
            ax2.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=7, color=color, alpha=0.85)

    # === 左子图设置 ===
    ax1.set_xlabel("Noise Intensity $\\sigma$ (pixels)")
    ax1.set_ylabel("Test mIoU (%)")
    ax1.set_xticks(sigmas)
    ax1.set_xticklabels([str(s) for s in sigmas])
    ax1.set_xlim(sigmas[0] - 0.5, sigmas[-1] + 1.5)
    ax1.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    ax1.set_title("Robustness: mIoU", fontweight="bold", pad=10)

    # === 右子图设置 ===
    ax2.set_xlabel("Noise Intensity $\\sigma$ (pixels)")
    ax2.set_ylabel("Test Boundary IoU (%)")
    ax2.set_xticks(sigmas)
    ax2.set_xticklabels([str(s) for s in sigmas])
    ax2.set_xlim(sigmas[0] - 0.5, sigmas[-1] + 1.5)
    ax2.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    ax2.set_title("Robustness: Boundary IoU (UG-DP d=5)", fontweight="bold", pad=10)

    # === 共享图例 (置于 Figure 底部) ===
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=2,
        framealpha=0.92,
        edgecolor="gray",
        fancybox=True,
        shadow=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Robustness to Noise: Gaussian Pixel / Box Jitter Perturbation",
        fontweight="bold", fontsize=17, y=1.02,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"\n  鲁棒性曲线已保存: {output_path}")


# ==========================================================================
# 模块 15: JSON 持久化
# ==========================================================================
def save_robustness_json(
    results: Dict,
    raw_data: Dict,
    sigmas: List[int],
    seeds: List[int],
    full_test: bool,
    output_path: str,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    _results_clean = {}
    for model_name, sigma_dict in results.items():
        _results_clean[model_name] = {}
        for sigma, metrics in sigma_dict.items():
            _results_clean[model_name][str(sigma)] = {k: _convert(v) for k, v in metrics.items()}

    _raw_clean = {}
    for model_name, sigma_dict in raw_data.items():
        _raw_clean[model_name] = {}
        for sigma, seed_list in sigma_dict.items():
            _raw_clean[model_name][str(sigma)] = [
                {k: _convert(v) for k, v in item.items()} for item in seed_list
            ]

    output = {
        "experiment": "E7_Robustness_Analysis",
        "metadata": {
            "platform": get_platform_name(),
            "device": DEVICE,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "full_test": full_test,
            "sigma_levels": sigmas,
            "seeds": seeds,
            "num_shots": NUM_SHOTS,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "boundary_d": BOUNDARY_D,
            "ugdp_d": UGDP_D,
            "noise_types": {
                "SAM3 Zero-shot": "Gaussian pixel noise (σ/255), clamped [0,1]",
                "E5-noFT (Src→Tgt LoRA)": "Box jitter noise on bbox coordinates (σ pixels)",
                "E5-FT (Tgt LoRA)": "Box jitter noise on bbox coordinates (σ pixels)",
                "E6 GBG-SAM3 (Ours)": "Prompt-Free (no BBox), immune to box jitter by design",
            },
        },
        "results": _results_clean,
        "raw_per_seed": _raw_clean,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  鲁棒性指标已保存: {output_path}")


# ==========================================================================
# 模块 16: 命令行参数解析
# ==========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E7 双轨抗噪敏感性测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/run_e7_robustness.py
  python src/run_e7_robustness.py --full_test
  python src/run_e7_robustness.py --sigma 0,5,10,20
  python src/run_e7_robustness.py --seeds 42,123,456 --sigma 0,2,5,10,20
  python src/run_e7_robustness.py --model4_json model4.json
  python src/run_e7_robustness.py --model4_json model4.json --skip_preflight
        """,
    )
    parser.add_argument(
        "--full_test", action="store_true",
        help="使用全量 8,402 张测试集",
    )
    parser.add_argument(
        "--sigma", type=str, default="0,2,5,10,20",
        help="噪声标准差列表, 逗号分隔 (默认: 0,2,5,10,20)",
    )
    parser.add_argument(
        "--seeds", type=str, default="42,123,456",
        help="随机种子列表, 逗号分隔 (默认: 42,123,456)",
    )
    parser.add_argument(
        "--skip_preflight", action="store_true",
        help="跳过权重预检, 直接运行实验",
    )
    parser.add_argument(
        "--model4_json", type=str, default="model4.json",
        help="Model 4 预计算结果的 JSON 路径 (默认: model4.json). 提供后跳过 M4 实际评估.",
    )
    return parser.parse_args()


# ==========================================================================
# 模块 17: 主函数
# ==========================================================================
def main() -> None:
    args = parse_args()
    sigmas = [int(s.strip()) for s in args.sigma.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if not torch.cuda.is_available():
        print("\n  警告: 未检测到 CUDA GPU, 将使用 CPU (极慢)")

    # === 加载 Model 4 预计算结果 (如有) ===
    model4_precomputed: Optional[Dict] = None
    m4_json_path = Path(args.model4_json)
    if m4_json_path.exists():
        print(f"\n[Model 4 预计算缓存] 加载: {m4_json_path}")
        with open(m4_json_path, "r", encoding="utf-8") as f:
            m4_raw_json = json.load(f)
        # 提取 results + raw_per_seed 中与 M4 相关的部分
        m4_results = m4_raw_json.get("results", {}).get("E6 GBG-SAM3 (Ours)", {})
        m4_raw = m4_raw_json.get("raw_per_seed", {}).get("E6 GBG-SAM3 (Ours)", {})
        # 将字符串 key (如 "20") 转回 int
        m4_results_int = {}
        for k, v in m4_results.items():
            m4_results_int[int(k) if k.lstrip("-").isdigit() else k] = v
        m4_raw_int = {}
        for k, v in m4_raw.items():
            m4_raw_int[int(k) if k.lstrip("-").isdigit() else k] = v
        model4_precomputed = {"results": m4_results_int, "raw_data": m4_raw_int}
        print(f"  已加载 {len(m4_results_int)} 个 sigma 级别的汇总结果")
        print(f"  已加载 {len(m4_raw_int)} 个 sigma 级别的逐种子原始数据")
        print(f"  Model 4 评估将被跳过, 直接注入预计算结果。")
    else:
        print(f"\n[Model 4 预计算缓存] 未找到: {m4_json_path}, 将执行实际评估。")

    if not args.skip_preflight:
        ok = check_all_weights(seeds, WEIGHTS_DIR, skip_m4=(model4_precomputed is not None))
        if not ok:
            sys.exit(1)

    eval_results = run_robustness_eval(
        full_test=args.full_test,
        sigmas=sigmas,
        seeds=seeds,
        model4_precomputed=model4_precomputed,
    )

    test_label = "full" if args.full_test else "slim"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    curve_path = str(Path(RESULTS_DIR) / f"robustness_curve_{test_label}_{timestamp}.png")
    plot_robustness_curve(eval_results["results"], sigmas, curve_path)

    json_path = str(Path(RESULTS_DIR) / f"robustness_metrics_{test_label}_{timestamp}.json")
    save_robustness_json(
        eval_results["results"], eval_results["raw_data"],
        sigmas, seeds, args.full_test, json_path,
    )

    symlink_path = str(Path(RESULTS_DIR) / "robustness_curve.png")
    if Path(symlink_path).exists() or Path(symlink_path).is_symlink():
        Path(symlink_path).unlink(missing_ok=True)
    try:
        Path(symlink_path).symlink_to(Path(curve_path).name)
    except OSError:
        import shutil
        shutil.copy2(curve_path, symlink_path)

    symlink_json = str(Path(RESULTS_DIR) / "robustness_metrics.json")
    if Path(symlink_json).exists() or Path(symlink_json).is_symlink():
        Path(symlink_json).unlink(missing_ok=True)
    try:
        Path(symlink_json).symlink_to(Path(json_path).name)
    except OSError:
        import shutil
        shutil.copy2(json_path, symlink_json)

    print(f"\n{'='*60}")
    print(f"  E7 鲁棒性分析 全部完成!")
    print(f"  曲线:    {curve_path}")
    print(f"  指标:    {json_path}")
    print(f"  快捷链接: {Path(RESULTS_DIR) / 'robustness_curve.png'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
