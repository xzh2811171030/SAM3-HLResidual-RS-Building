"""
=============================================================================
eval_metrics.py  ---  多基线统一评估与对比指标生成
=============================================================================
功能说明:
  1. 读取 val_demo 测试集 (影像 + 双通道 Label)
  2. 计算 IoU / F1-score / Boundary IoU (d=5) 三大指标
  3. 载入并评估 5 个模型:
     - UNet 全监督       (unet_best.pth)
     - SegFormer 全监督  (segformer_best.pth)
     - Vanilla SAM3 零样本 (Zero-shot)
     - SAM3 Decoder-Only (sam3_decoder_only.pth)
     - SAM3 LoRA         (sam3_lora.pth)
  4. 统计各模型可训练参数量 (M)
  5. 控制台输出期刊规范 Markdown 对比表格

用法:
  python src/evaluation/eval_metrics.py
=============================================================================
"""

# ==========================================================================
# 模块 1: 导入与全局配置
# ==========================================================================
import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_SAM3_SRC = _SRC / "models" / "sam3"
if str(_SAM3_SRC) not in sys.path:
    sys.path.insert(0, str(_SAM3_SRC))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from torchvision.transforms import v2

VAL_IMAGE_DIR: str = r"/path/to/project\data\raw\val_demo\image"
VAL_LABEL_DIR: str = r"/path/to/project\data\raw\val_demo\label"

CHECKPOINT_PATH: str = r"/path/to/project\weights\sam3.pt"
UNET_WEIGHT: str = r"/path/to/project\weights\unet_best.pth"
SEGFORMER_WEIGHT: str = r"/path/to/project\weights\segformer_best.pth"
DECODER_ONLY_WEIGHT: str = r"/path/to/project\weights\sam3_decoder_only.pth"
LORA_WEIGHT: str = r"/path/to/project\weights\sam3_lora.pth"

TARGET_SIZE: int = 512
FEATURE_SIZE: Tuple[int, int] = (64, 64)
SAM3_INPUT_SIZE: int = 1008
BOUNDARY_D: int = 5
SAM3_CONFIDENCE_THRESHOLD: float = 0.20

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_EXTS: tuple = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


# ==========================================================================
# 模块 2: 轻量掩膜解码器 (LightweightMaskDecoder)
#     与 train_peft_baselines.py 完全一致
#     输入 [B,256,64,64] 上采样 → [B,1,512,512]
# ==========================================================================
class LightweightMaskDecoder(nn.Module):
    def __init__(self, feat_channels: int = 256):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


# ==========================================================================
# 模块 3: 测试数据集加载
#     val_demo/image/*.tif + val_demo/label/*.tif
#     标签为单通道二值掩膜, 需动态计算边界 (d=3 结构元素)
# ==========================================================================
class ValDemoDataset(torch.utils.data.Dataset):
    image_dir: Path
    label_dir: Path
    file_paths: List[Path]

    def __init__(
        self,
        image_dir: str = VAL_IMAGE_DIR,
        label_dir: str = VAL_LABEL_DIR,
        target_size: int = TARGET_SIZE,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.target_size = target_size

        image_candidates = sorted([
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        ])

        self.file_paths = []
        for img_path in image_candidates:
            label_path = self.label_dir / img_path.name
            if label_path.exists():
                self.file_paths.append(img_path)

        if not self.file_paths:
            raise FileNotFoundError(
                f"未找到配对的影像与标签。\n"
                f"  影像目录: {self.image_dir}\n"
                f"  标签目录: {self.label_dir}"
            )

        print(f"ValDemoDataset: {len(self.file_paths)} 张配对样本已就绪")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        img_path = self.file_paths[index]
        stem = img_path.stem

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取影像: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.target_size, self.target_size))

        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        label_path = self.label_dir / img_path.name
        label_img = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if label_img is None:
            raise FileNotFoundError(f"无法读取标签: {label_path}")

        if label_img.ndim == 3:
            label_gray = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
        else:
            label_gray = label_img

        label_resized = cv2.resize(
            label_gray, (self.target_size, self.target_size),
            interpolation=cv2.INTER_NEAREST,
        )
        mask_bin = (label_resized > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0)

        boundary_bin = self._extract_boundary(mask_bin)
        boundary_tensor = torch.from_numpy(boundary_bin).unsqueeze(0)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "boundary": boundary_tensor,
            "name": stem,
        }

    @staticmethod
    def _extract_boundary(mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), np.uint8)
        mask_u8 = (mask * 255).astype(np.uint8)
        eroded = cv2.erode(mask_u8, kernel, iterations=1)
        boundary = cv2.subtract(mask_u8, eroded)
        return (boundary > 0).astype(np.float32)


# ==========================================================================
# 模块 4: 学术指标计算
#     IoU / F1-score / Boundary IoU (d=5)
# ==========================================================================

def _to_numpy_batch(t: torch.Tensor) -> np.ndarray:
    """ [B,1,H,W] float → [B,H,W] bool numpy """
    if t.is_cuda:
        t = t.cpu()
    return t.squeeze(1).numpy()


def compute_iou(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
    """
    计算 mIoU。

    参数:
        pred_binary: [N, H, W] bool, 预测二值掩膜
        gt_binary:    [N, H, W] bool, 真值二值掩膜
    返回:
        mIoU (float, 0~1)
    """
    iou_per_sample: List[float] = []
    for i in range(pred_binary.shape[0]):
        p = pred_binary[i].ravel()
        g = gt_binary[i].ravel()
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        iou_per_sample.append((inter + 1e-7) / (union + 1e-7))
    return float(np.mean(iou_per_sample))


def compute_f1(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
    """
    计算 F1-score。

    参数:
        pred_binary: [N, H, W] bool
        gt_binary:    [N, H, W] bool
    返回:
        F1-score (float, 0~1)
    """
    f1_per_sample: List[float] = []
    for i in range(pred_binary.shape[0]):
        p = pred_binary[i].ravel()
        g = gt_binary[i].ravel()
        tp = np.logical_and(p, g).sum()
        fp = np.logical_and(p, np.logical_not(g)).sum()
        fn = np.logical_and(np.logical_not(p), g).sum()
        precision = (tp + 1e-7) / (tp + fp + 1e-7)
        recall = (tp + 1e-7) / (tp + fn + 1e-7)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-7)
        f1_per_sample.append(f1)
    return float(np.mean(f1_per_sample))


def compute_boundary_iou(
    pred_binary: np.ndarray,
    gt_binary: np.ndarray,
    d: int = BOUNDARY_D,
) -> float:
    """
    计算 Boundary IoU (Remote Sensing 边界评测标准)。

    算法:
      1. 对 GT 和 Pred 分别提取物理轮廓 (形态学梯度)
      2. 用 d×d 核膨胀轮廓, 得到边界区域
      3. 在合并的边界区域内计算 IoU

    参数:
        pred_binary: [N, H, W] bool
        gt_binary:    [N, H, W] bool
        d:            边界厚度 (像素), 默认 5
    返回:
        Boundary IoU (float, 0~1)
    """
    kernel_contour = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))

    biou_per_sample: List[float] = []
    for i in range(pred_binary.shape[0]):
        p = pred_binary[i].astype(np.uint8)
        g = gt_binary[i].astype(np.uint8)

        p_contour = cv2.morphologyEx(p, cv2.MORPH_GRADIENT, kernel_contour)
        g_contour = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, kernel_contour)

        p_boundary_region = cv2.dilate(p_contour, kernel_dilate, iterations=1)
        g_boundary_region = cv2.dilate(g_contour, kernel_dilate, iterations=1)

        combined_boundary = np.logical_or(
            p_boundary_region > 0, g_boundary_region > 0
        )

        p_boundary = np.logical_and(p > 0, combined_boundary)
        g_boundary = np.logical_and(g > 0, combined_boundary)

        inter = np.logical_and(p_boundary, g_boundary).sum()
        union = np.logical_or(p_boundary, g_boundary).sum()
        biou_per_sample.append((inter + 1e-7) / (union + 1e-7))

    return float(np.mean(biou_per_sample))


def evaluate_predictions(
    preds: np.ndarray,
    gts: np.ndarray,
) -> Dict[str, float]:
    pred_bool = preds > 0.5
    gt_bool = gts > 0.5

    return {
        "mIoU": compute_iou(pred_bool, gt_bool),
        "F1": compute_f1(pred_bool, gt_bool),
        "Boundary_IoU": compute_boundary_iou(pred_bool, gt_bool, d=BOUNDARY_D),
    }


# ==========================================================================
# 模块 5: 辅助工具
# ==========================================================================

def count_trainable_params(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def load_model_weights(model: nn.Module, weight_path: str, key: str = "model_state_dict") -> None:
    checkpoint = torch.load(weight_path, map_location=DEVICE, weights_only=False)
    if key in checkpoint:
        state_dict = checkpoint[key]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    print(f"  权重已加载: {Path(weight_path).name}")


# ==========================================================================
# 模块 6: UNet 全监督评估
# ==========================================================================

@torch.no_grad()
def evaluate_unet(dataset: ValDemoDataset) -> Tuple[Dict[str, float], float]:
    print(f"\n{'='*55}")
    print(f"  评估: UNet (ResNet34)")
    print(f"{'='*55}")

    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(DEVICE)
    model.eval()
    load_model_weights(model, UNET_WEIGHT, key="model_state_dict")

    trainable_m, total_m = count_trainable_params(model)
    print(f"  可训练参数: {trainable_m:,} / {total_m:,} ({trainable_m/1e6:.2f} M)")

    all_preds: List[np.ndarray] = []
    all_gts: List[np.ndarray] = []

    for idx in tqdm(range(len(dataset)), desc="  UNet", unit="sample", ncols=100):
        sample = dataset[idx]
        img = sample["image"].unsqueeze(0).to(DEVICE)
        mask = sample["mask"].numpy()

        logits = model(img)
        pred = torch.sigmoid(logits).cpu().numpy()

        all_preds.append(pred.squeeze(0))
        all_gts.append(mask.squeeze(0))

    preds_arr = np.stack(all_preds, axis=0)
    gts_arr = np.stack(all_gts, axis=0)

    metrics = evaluate_predictions(preds_arr, gts_arr)
    metrics["Params_M"] = trainable_m / 1e6

    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return metrics, trainable_m / 1e6


# ==========================================================================
# 模块 7: SegFormer 全监督评估
# ==========================================================================

@torch.no_grad()
def evaluate_segformer(dataset: ValDemoDataset) -> Tuple[Dict[str, float], float]:
    print(f"\n{'='*55}")
    print(f"  评估: SegFormer (mit-b2)")
    print(f"{'='*55}")

    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        num_labels=1,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)
    model.eval()
    load_model_weights(model, SEGFORMER_WEIGHT, key="model_state_dict")

    trainable_m, total_m = count_trainable_params(model)
    print(f"  可训练参数: {trainable_m:,} / {total_m:,} ({trainable_m/1e6:.2f} M)")

    all_preds: List[np.ndarray] = []
    all_gts: List[np.ndarray] = []

    for idx in tqdm(range(len(dataset)), desc="  SegFormer", unit="sample", ncols=100):
        sample = dataset[idx]
        img = sample["image"].unsqueeze(0).to(DEVICE)
        mask = sample["mask"].numpy()
        _, _, h, w = img.shape

        outputs = model(pixel_values=img)
        logits = F.interpolate(
            outputs.logits, size=(h, w), mode="bilinear", align_corners=False,
        )
        pred = torch.sigmoid(logits).cpu().numpy()

        all_preds.append(pred.squeeze(0))
        all_gts.append(mask.squeeze(0))

    preds_arr = np.stack(all_preds, axis=0)
    gts_arr = np.stack(all_gts, axis=0)

    metrics = evaluate_predictions(preds_arr, gts_arr)
    metrics["Params_M"] = trainable_m / 1e6

    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return metrics, trainable_m / 1e6


# ==========================================================================
# 模块 8: SAM3 零样本评估
#     使用 Sam3Processor Text-only 推理, prompt="buildings"
#     聚合所有实例掩膜 → 单张二值预测图
# ==========================================================================

@torch.no_grad()
def evaluate_sam3_zeroshot(dataset: ValDemoDataset) -> Tuple[Dict[str, float], float]:
    print(f"\n{'='*55}")
    print(f"  评估: SAM3 Zero-shot (Text: 'buildings')")
    print(f"{'='*55}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    raw_model = build_sam3_image_model(checkpoint_path=CHECKPOINT_PATH)
    processor = Sam3Processor(raw_model, confidence_threshold=SAM3_CONFIDENCE_THRESHOLD)
    print(f"  SAM3 模型加载完成 (Zero-shot, 无需训练)")

    trainable_m = 0.0

    all_preds: List[np.ndarray] = []
    all_gts: List[np.ndarray] = []

    for idx in tqdm(range(len(dataset)), desc="  SAM3 Zero-shot", unit="sample", ncols=100):
        sample = dataset[idx]
        img_tensor = sample["image"]
        mask = sample["mask"].squeeze(0).numpy()

        img_uint8 = (img_tensor * 255.0).clamp(0, 255).to(torch.uint8)
        img_np = img_uint8.permute(1, 2, 0).cpu().numpy()
        pil_image = Image.fromarray(img_np, mode="RGB")

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(pil_image)
            state = processor.set_text_prompt(state=state, prompt="buildings")

        masks_out = state.get("masks", None)

        if masks_out is None or masks_out.numel() == 0:
            pred_bin = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
        else:
            if isinstance(masks_out, torch.Tensor):
                masks_np = masks_out.cpu().numpy()
            else:
                masks_np = np.array(masks_out)

            if masks_np.ndim == 2:
                masks_np = masks_np[np.newaxis, ...]

            aggregated = np.any(masks_np > 0.5, axis=0).astype(np.float32)

            aggregated = np.squeeze(aggregated)
            if aggregated.ndim != 2:
                aggregated = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)

            aggregated = cv2.resize(
                aggregated, (TARGET_SIZE, TARGET_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )
            pred_bin = aggregated

        all_preds.append(pred_bin)
        all_gts.append(mask)

    preds_arr = np.stack(all_preds, axis=0)
    gts_arr = np.stack(all_gts, axis=0)

    metrics = evaluate_predictions(preds_arr, gts_arr)
    metrics["Params_M"] = 0.0

    del raw_model, processor
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return metrics, 0.0


# ==========================================================================
# 模块 9: SAM3 特征提取器 (评估专用)
#     仅 extract() 路径 (no_grad + detach)
# ==========================================================================

class SAM3EvalFeatureExtractor:
    def __init__(self, checkpoint_path: str, device: str = DEVICE):
        self.device = device
        self.model = build_sam3_image_model(checkpoint_path=checkpoint_path)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(SAM3_INPUT_SIZE, SAM3_INPUT_SIZE)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @torch.no_grad()
    def extract(self, image_tensor: torch.Tensor) -> torch.Tensor:
        img_uint8 = (image_tensor * 255.0).clamp(0, 255).to(torch.uint8)
        img_processed = self.transform(img_uint8).unsqueeze(0).to(self.device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = self.model.backbone.forward_image(img_processed)
        features = backbone_out["vision_features"].float()

        if features.shape[-2:] != FEATURE_SIZE:
            features = F.interpolate(
                features, size=FEATURE_SIZE,
                mode="bilinear", align_corners=False,
            )
        return features.detach()


# ==========================================================================
# 模块 10: SAM3 Decoder-Only 评估
# ==========================================================================

@torch.no_grad()
def evaluate_sam3_decoder_only(dataset: ValDemoDataset) -> Tuple[Dict[str, float], float]:
    print(f"\n{'='*55}")
    print(f"  评估: SAM3 Decoder-Only")
    print(f"{'='*55}")

    print("  加载 SAM3 模型 ...")
    extractor = SAM3EvalFeatureExtractor(CHECKPOINT_PATH, DEVICE)

    decoder = LightweightMaskDecoder().to(DEVICE)
    decoder.eval()
    load_model_weights(decoder, DECODER_ONLY_WEIGHT, key="decoder")

    trainable_m = sum(p.numel() for p in decoder.parameters())
    print(f"  可训练参数 (Decoder): {trainable_m:,} ({trainable_m/1e6:.2f} M)")

    all_preds: List[np.ndarray] = []
    all_gts: List[np.ndarray] = []

    for idx in tqdm(range(len(dataset)), desc="  Decoder-Only", unit="sample", ncols=100):
        sample = dataset[idx]
        img = sample["image"]
        mask = sample["mask"].numpy()

        features = extractor.extract(img)

        logits = decoder(features)
        logits = F.interpolate(
            logits, size=(TARGET_SIZE, TARGET_SIZE),
            mode="bilinear", align_corners=False,
        )
        pred = torch.sigmoid(logits).cpu().numpy()

        all_preds.append(pred.squeeze(0))
        all_gts.append(mask.squeeze(0))

    preds_arr = np.stack(all_preds, axis=0)
    gts_arr = np.stack(all_gts, axis=0)

    metrics = evaluate_predictions(preds_arr, gts_arr)
    metrics["Params_M"] = trainable_m / 1e6

    del extractor, decoder
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return metrics, trainable_m / 1e6


# ==========================================================================
# 模块 11: SAM3 LoRA PEFT 评估
# ==========================================================================

def _patch_vit_mlp_for_grad_compat(vit_peft_model) -> None:
    import types as _types

    n_patched: int = 0
    for module in vit_peft_model.modules():
        if type(module).__name__ != "Mlp":
            continue

        def _grad_safe_mlp_forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop1(x)
            x = self.norm(x)
            x = self.fc2(x)
            x = self.drop2(x)
            return x

        module.forward = _types.MethodType(_grad_safe_mlp_forward, module)
        n_patched += 1

    if n_patched > 0:
        print(f"  MLP 梯度兼容补丁: 已修补 {n_patched} 个 Mlp 模块")


def inject_lora_for_eval(model, rank: int = 8, alpha: int = 16, dropout: float = 0.05) -> None:
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

    _patch_vit_mlp_for_grad_compat(model.backbone.vision_backbone.trunk)

    trainable_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_count = sum(p.numel() for p in model.parameters())
    print(f"  LoRA 注入完成: 可训练 {trainable_count:,} / {total_count:,} "
          f"({100 * trainable_count / total_count:.2f}%)")


@torch.no_grad()
def evaluate_sam3_lora(dataset: ValDemoDataset) -> Tuple[Dict[str, float], float]:
    print(f"\n{'='*55}")
    print(f"  评估: SAM3 LoRA PEFT")
    print(f"{'='*55}")

    print("  加载 SAM3 模型 + LoRA 注入 ...")
    extractor = SAM3EvalFeatureExtractor(CHECKPOINT_PATH, DEVICE)
    inject_lora_for_eval(extractor.model, rank=8, alpha=16, dropout=0.05)

    lora_checkpoint = torch.load(LORA_WEIGHT, map_location=DEVICE, weights_only=False)

    if "lora_params" in lora_checkpoint:
        print("  加载 LoRA adapter 权重 ...")
        lora_state = lora_checkpoint["lora_params"]
        model_state = extractor.model.state_dict()
        for key, value in lora_state.items():
            if key in model_state:
                model_state[key].copy_(value.to(DEVICE))
            else:
                print(f"  [警告] LoRA key 在模型中未找到: {key}")

    decoder = LightweightMaskDecoder().to(DEVICE)
    decoder.eval()
    load_model_weights(decoder, LORA_WEIGHT, key="decoder")

    lora_trainable = sum(
        p.numel() for p in extractor.model.parameters() if p.requires_grad
    )
    decoder_trainable = sum(p.numel() for p in decoder.parameters())
    total_trainable = lora_trainable + decoder_trainable
    print(f"  可训练参数 (LoRA + Decoder): {total_trainable:,} ({total_trainable/1e6:.2f} M)")

    all_preds: List[np.ndarray] = []
    all_gts: List[np.ndarray] = []

    for idx in tqdm(range(len(dataset)), desc="  LoRA", unit="sample", ncols=100):
        sample = dataset[idx]
        img = sample["image"]
        mask = sample["mask"].numpy()

        features = extractor.extract(img)

        logits = decoder(features)
        logits = F.interpolate(
            logits, size=(TARGET_SIZE, TARGET_SIZE),
            mode="bilinear", align_corners=False,
        )
        pred = torch.sigmoid(logits).cpu().numpy()

        all_preds.append(pred.squeeze(0))
        all_gts.append(mask.squeeze(0))

    preds_arr = np.stack(all_preds, axis=0)
    gts_arr = np.stack(all_gts, axis=0)

    metrics = evaluate_predictions(preds_arr, gts_arr)
    metrics["Params_M"] = total_trainable / 1e6

    del extractor, decoder
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return metrics, total_trainable / 1e6


# ==========================================================================
# 模块 12: Markdown 表格输出
# ==========================================================================

def _best(values: List[float], maximize: bool = True) -> int:
    if maximize:
        return int(np.argmax(values))
    else:
        return int(np.argmin(values))


def format_markdown_table(results: List[Dict]) -> None:
    headers = [
        "Model",
        "Trainable Params (M)",
        "mIoU (%)",
        "F1-score (%)",
        "Boundary IoU (d=5) (%)",
    ]

    col_keys = ["Model", "Params_M", "mIoU", "F1", "Boundary_IoU"]
    numeric_keys = ["Params_M", "mIoU", "F1", "Boundary_IoU"]

    best_indices: Dict[str, int] = {}
    for key in numeric_keys:
        values = [r[key] for r in results]
        best_indices[key] = _best(values, maximize=True)

    def fmt_value(row: Dict, key: str) -> str:
        val = row[key]
        if key == "Params_M":
            s = f"{val:.2f}"
        else:
            s = f"{val * 100:.2f}"
        if best_indices.get(key) == results.index(row):
            return f"**{s}**"
        return s

    col_widths = [
        max(len(headers[0]), max(len(r["Model"]) for r in results)) + 2,
        max(len(headers[1]), 20) + 2,
        max(len(headers[2]), 10) + 2,
        max(len(headers[3]), 10) + 2,
        max(len(headers[4]), 10) + 2,
    ]

    def pad(s: str, w: int) -> str:
        return f" {s} ".ljust(w)

    separator = "|" + "|".join("-" * w for w in col_widths) + "|"

    print(f"\n{'='*80}")
    print(f"  多基线统一评估对比表格 (Remote Sensing 期刊规范)")
    print(f"{'='*80}\n")

    header_line = "|" + "|".join(
        pad(headers[i], col_widths[i]) for i in range(len(headers))
    ) + "|"
    print(header_line)
    print(separator)

    for row in results:
        cols = [
            pad(row["Model"], col_widths[0]),
            pad(fmt_value(row, "Params_M"), col_widths[1]),
            pad(fmt_value(row, "mIoU"), col_widths[2]),
            pad(fmt_value(row, "F1"), col_widths[3]),
            pad(fmt_value(row, "Boundary_IoU"), col_widths[4]),
        ]
        print("|" + "|".join(cols) + "|")

    print(f"\n{'='*80}")
    print(f"  注: 最高指标已加粗标注 (**bold**), Boundary IoU 边界厚度 d={BOUNDARY_D} px")
    print(f"{'='*80}\n")


def export_markdown_table(results: List[Dict], output_path: str) -> None:
    headers = ["Model", "Trainable Params (M)", "mIoU (%)", "F1-score (%)", f"Boundary IoU (d={BOUNDARY_D}) (%)"]
    numeric_keys = ["Params_M", "mIoU", "F1", "Boundary_IoU"]

    best_indices: Dict[str, int] = {}
    for key in numeric_keys:
        values = [float(r[key]) for r in results]
        best_indices[key] = int(np.argmax(values))

    lines: List[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join([" --- " for _ in headers]) + "|")

    for idx, row in enumerate(results):
        vals: List[str] = []
        vals.append(row["Model"])
        for key in numeric_keys:
            val = float(row[key])
            if key == "Params_M":
                s = f"{val:.2f}"
            else:
                s = f"{val * 100:.2f}"
            if best_indices.get(key) == idx:
                s = f"**{s}**"
            vals.append(s)
        lines.append("| " + " | ".join(vals) + " |")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"表格已导出至: {out_path}")


# ==========================================================================
# 模块 13: 主函数
# ==========================================================================

def main() -> None:
    global DEVICE
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("警告: 未检测到 CUDA GPU, 将使用 CPU (极慢)")
        DEVICE = "cpu"

    print(f"\n{'='*60}")
    print(f"  多基线统一评估与对比指标生成")
    print(f"  设备: {DEVICE}")
    print(f"  边界厚度: d={BOUNDARY_D} px")
    print(f"{'='*60}")

    print("\n[数据加载]")
    dataset = ValDemoDataset(
        image_dir=VAL_IMAGE_DIR,
        label_dir=VAL_LABEL_DIR,
        target_size=TARGET_SIZE,
    )
    print(f"  测试样本数: {len(dataset)}")

    results: List[Dict] = []

    # ------------------------------------------------------------------
    # 模型 1: UNet 全监督
    # ------------------------------------------------------------------
    try:
        metrics_1, params_1 = evaluate_unet(dataset)
        results.append({
            "Model": "UNet (ResNet34)",
            "Params_M": params_1,
            "mIoU": metrics_1["mIoU"],
            "F1": metrics_1["F1"],
            "Boundary_IoU": metrics_1["Boundary_IoU"],
        })
        print(f"  UNet → mIoU={metrics_1['mIoU']:.4f}  F1={metrics_1['F1']:.4f}  "
              f"BIoU={metrics_1['Boundary_IoU']:.4f}")
    except FileNotFoundError as e:
        print(f"  [跳过] UNet 权重文件未找到: {e}")
    except Exception as e:
        print(f"  [错误] UNet 评估失败: {e}")
        import traceback
        traceback.print_exc()

    # ------------------------------------------------------------------
    # 模型 2: SegFormer 全监督
    # ------------------------------------------------------------------
    try:
        metrics_2, params_2 = evaluate_segformer(dataset)
        results.append({
            "Model": "SegFormer (mit-b2)",
            "Params_M": params_2,
            "mIoU": metrics_2["mIoU"],
            "F1": metrics_2["F1"],
            "Boundary_IoU": metrics_2["Boundary_IoU"],
        })
        print(f"  SegFormer → mIoU={metrics_2['mIoU']:.4f}  F1={metrics_2['F1']:.4f}  "
              f"BIoU={metrics_2['Boundary_IoU']:.4f}")
    except FileNotFoundError as e:
        print(f"  [跳过] SegFormer 权重文件未找到: {e}")
    except Exception as e:
        print(f"  [错误] SegFormer 评估失败: {e}")
        import traceback
        traceback.print_exc()

    # ------------------------------------------------------------------
    # 模型 3: SAM3 Zero-shot
    # ------------------------------------------------------------------
    try:
        metrics_3, params_3 = evaluate_sam3_zeroshot(dataset)
        results.append({
            "Model": "SAM3 Zero-shot",
            "Params_M": params_3,
            "mIoU": metrics_3["mIoU"],
            "F1": metrics_3["F1"],
            "Boundary_IoU": metrics_3["Boundary_IoU"],
        })
        print(f"  SAM3 Zero-shot → mIoU={metrics_3['mIoU']:.4f}  F1={metrics_3['F1']:.4f}  "
              f"BIoU={metrics_3['Boundary_IoU']:.4f}")
    except FileNotFoundError as e:
        print(f"  [跳过] SAM3 checkpoint 未找到: {e}")
    except Exception as e:
        print(f"  [错误] SAM3 Zero-shot 评估失败: {e}")
        import traceback
        traceback.print_exc()

    # ------------------------------------------------------------------
    # 模型 4: SAM3 Decoder-Only
    # ------------------------------------------------------------------
    try:
        metrics_4, params_4 = evaluate_sam3_decoder_only(dataset)
        results.append({
            "Model": "SAM3 Decoder-Only",
            "Params_M": params_4,
            "mIoU": metrics_4["mIoU"],
            "F1": metrics_4["F1"],
            "Boundary_IoU": metrics_4["Boundary_IoU"],
        })
        print(f"  SAM3 Decoder-Only → mIoU={metrics_4['mIoU']:.4f}  F1={metrics_4['F1']:.4f}  "
              f"BIoU={metrics_4['Boundary_IoU']:.4f}")
    except FileNotFoundError as e:
        print(f"  [跳过] Decoder-Only 权重文件未找到: {e}")
    except Exception as e:
        print(f"  [错误] SAM3 Decoder-Only 评估失败: {e}")
        import traceback
        traceback.print_exc()

    # ------------------------------------------------------------------
    # 模型 5: SAM3 LoRA
    # ------------------------------------------------------------------
    try:
        metrics_5, params_5 = evaluate_sam3_lora(dataset)
        results.append({
            "Model": "SAM3 LoRA",
            "Params_M": params_5,
            "mIoU": metrics_5["mIoU"],
            "F1": metrics_5["F1"],
            "Boundary_IoU": metrics_5["Boundary_IoU"],
        })
        print(f"  SAM3 LoRA → mIoU={metrics_5['mIoU']:.4f}  F1={metrics_5['F1']:.4f}  "
              f"BIoU={metrics_5['Boundary_IoU']:.4f}")
    except FileNotFoundError as e:
        print(f"  [跳过] LoRA 权重文件未找到: {e}")
    except Exception as e:
        print(f"  [错误] SAM3 LoRA 评估失败: {e}")
        import traceback
        traceback.print_exc()

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------
    if not results:
        print("\n[错误] 没有任何模型成功评估, 无法生成表格。")
        return

    format_markdown_table(results)

    table_output = Path(_CUSTOM_TUNING).parent.parent / "results" / "baseline_comparison.md"
    export_markdown_table(results, str(table_output))


if __name__ == "__main__":
    main()
