"""
=============================================================================
generate_mask_predictions.py  ---  WHU-Mix 全量测试集推理与掩膜保存
=============================================================================
功能说明:
  Model 0 (SAM3 Text Zero-shot): 原始 SAM3 + text prompt "building"
  Model 2 (GBG-SAM3 20-shot):    E6 目标域 20-shot GBG 权重推理

输出:
  data/raw/whu_mix_full_test/preds/sam3_zeroshot/*.png   (Model 0)
  data/raw/whu_mix_full_test/preds/gbg_sam3/*.png        (Model 2)

用法:
  python src/generate_mask_predictions.py
  python src/generate_mask_predictions.py --models 0      # 仅 Model 0
  python src/generate_mask_predictions.py --models 2      # 仅 Model 2
  python src/generate_mask_predictions.py --models 0,2    # 两个都跑
  python src/generate_mask_predictions.py --gbg_seed 123  # 指定 GBG 权重种子
=============================================================================
"""

# ==========================================================================
# 模块 1: 导入与常量
# ==========================================================================
import argparse
import gc
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_SAM3_SRC = _SRC / "models" / "sam3"
if str(_SAM3_SRC) not in sys.path:
    sys.path.insert(0, str(_SAM3_SRC))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_SIZE: int = 512
FEATURE_SIZE: Tuple[int, int] = (64, 64)
SAM3_INPUT_SIZE: int = 1008
FEATURE_CHANNELS: int = 256

LORA_RANK: int = 8
LORA_ALPHA: int = 16
LORA_DROPOUT: float = 0.05

SAM3_CONFIDENCE_THRESHOLD: float = 0.20

# 输入与输出路径
PROJECT_ROOT: Path = Path(_SRC).parent if str(_SRC).endswith("src") else Path(__file__).resolve().parent.parent
IMAGE_DIR: Path = PROJECT_ROOT / "data" / "raw" / "whu_mix_full_test" / "test" / "image"
PREDS_SAM3_DIR: Path = PROJECT_ROOT / "data" / "raw" / "whu_mix_full_test" / "preds" / "sam3_zeroshot"
PREDS_GBG_DIR: Path = PROJECT_ROOT / "data" / "raw" / "whu_mix_full_test" / "preds" / "gbg_sam3"

CHECKPOINT_PATH: str = str(PROJECT_ROOT / "weights" / "sam3.pt")

IMAGE_EXTS: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


# ==========================================================================
# 模块 2: GBG-SAM3 模型组件 (复刻 experiment_runner_gbg.py, 仅推理用)
# ==========================================================================

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class GatedBoundaryAdapter(nn.Module):
    def __init__(self, in_channels: int = 3, feat_channels: int = FEATURE_CHANNELS):
        super().__init__()
        self.edge_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.alpha_gate = nn.Parameter(torch.zeros(feat_channels, 1, 1))

    def forward(self, rgb_image: torch.Tensor, sam_features: torch.Tensor) -> torch.Tensor:
        edge_features = self.edge_extractor(rgb_image)
        return sam_features + edge_features * self.alpha_gate


class EdgeUncertaintyHead(nn.Module):
    def __init__(self, feat_channels: int = FEATURE_CHANNELS):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(feat_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        x = self.conv1(fused_features)
        return F.interpolate(
            self.conv2(x),
            size=(TARGET_SIZE, TARGET_SIZE),
            mode="bilinear", align_corners=False,
        )


class GBG_SAM3_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.boundary_adapter = GatedBoundaryAdapter()
        self.decoder = LightweightMaskDecoder()
        self.eu_head = EdgeUncertaintyHead()

    def forward(self, sam_features: torch.Tensor, rgb_images: torch.Tensor):
        fused = self.boundary_adapter(rgb_images, sam_features)
        logits = self.decoder(fused)
        return logits  # 推理时只需 logits


# ==========================================================================
# 模块 3: SAM3 特征提取器 (纯推理)
# ==========================================================================

class SAM3InferenceExtractor:
    def __init__(self, checkpoint_path: str):
        self.device = DEVICE
        self.model = build_sam3_image_model(checkpoint_path=checkpoint_path)
        self.model.to(DEVICE)
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
# 模块 4: LoRA 注入与 MLP 补丁
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


def inject_lora_to_vit(model, rank: int = LORA_RANK,
                       alpha: int = LORA_ALPHA,
                       dropout: float = LORA_DROPOUT) -> None:
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


def load_lora_params(model, lora_state: dict) -> int:
    model_state = model.state_dict()
    loaded = 0
    for key, value in lora_state.items():
        if key in model_state:
            model_state[key].copy_(value.to(DEVICE))
            loaded += 1
    return loaded


# ==========================================================================
# 模块 5: 图像预处理
# ==========================================================================

def load_and_preprocess_image(img_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    import cv2

    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取影像: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (TARGET_SIZE, TARGET_SIZE))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

    img_uint8 = (img_tensor * 255.0).clamp(0, 255).to(torch.uint8)
    img_np = img_uint8.permute(1, 2, 0).cpu().numpy()
    pil_image = Image.fromarray(img_np, mode="RGB")

    return img_tensor, pil_image


# ==========================================================================
# 模块 6: Model 0 --- SAM3 Text Zero-shot 推理
# ==========================================================================

@torch.no_grad()
def run_sam3_zeroshot() -> None:
    print(f"\n{'='*60}")
    print(f"  Model 0: SAM3 Text Zero-shot (prompt='building')")
    print(f"  输入: {IMAGE_DIR}")
    print(f"  输出: {PREDS_SAM3_DIR}")
    print(f"{'='*60}")

    if not IMAGE_DIR.exists():
        print(f"  [错误] 输入目录不存在: {IMAGE_DIR}")
        return

    PREDS_SAM3_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )

    if not image_paths:
        print(f"  [错误] 目录下未找到图像文件: {IMAGE_DIR}")
        return

    print(f"  共发现 {len(image_paths)} 张图像")

    if not Path(CHECKPOINT_PATH).exists():
        print(f"  [错误] SAM3 checkpoint 不存在: {CHECKPOINT_PATH}")
        return

    print(f"  加载 SAM3 模型 ...")
    raw_model = build_sam3_image_model(checkpoint_path=CHECKPOINT_PATH)
    processor = Sam3Processor(raw_model, confidence_threshold=SAM3_CONFIDENCE_THRESHOLD)
    print(f"  SAM3 模型加载完成")

    count = 0
    for img_path in tqdm(image_paths, desc="  SAM3 Zero-shot", unit="img", ncols=100):
        out_path = PREDS_SAM3_DIR / f"{img_path.stem}.png"

        _, pil_image = load_and_preprocess_image(img_path)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(pil_image)
            state = processor.set_text_prompt(state=state, prompt="building")

        masks_out = state.get("masks", None)
        if masks_out is None or masks_out.numel() == 0:
            pred_bin = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
        else:
            masks_np = masks_out.cpu().numpy() if isinstance(masks_out, torch.Tensor) else np.array(masks_out)
            if masks_np.ndim == 2:
                masks_np = masks_np[np.newaxis, ...]
            aggregated = np.any(masks_np > 0.5, axis=0).astype(np.uint8) * 255
            aggregated = np.squeeze(aggregated)
            if aggregated.ndim != 2:
                aggregated = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
            pred_bin = aggregated

        pred_img = Image.fromarray(pred_bin, mode="L")
        pred_img.save(out_path)
        count += 1

    del raw_model, processor
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"\n  Model 0 完成! 共保存 {count} 张掩膜 → {PREDS_SAM3_DIR}")


# ==========================================================================
# 模块 7: Model 2 --- GBG-SAM3 20-shot 推理
# ==========================================================================

@torch.no_grad()
def run_gbg_sam3(gbg_seed: int = 42) -> None:
    print(f"\n{'='*60}")
    print(f"  Model 2: GBG-SAM3 20-shot (目标域, seed={gbg_seed})")
    print(f"  输入: {IMAGE_DIR}")
    print(f"  输出: {PREDS_GBG_DIR}")
    print(f"{'='*60}")

    if not IMAGE_DIR.exists():
        print(f"  [错误] 输入目录不存在: {IMAGE_DIR}")
        return

    PREDS_GBG_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )

    if not image_paths:
        print(f"  [错误] 目录下未找到图像文件: {IMAGE_DIR}")
        return

    print(f"  共发现 {len(image_paths)} 张图像")

    if not Path(CHECKPOINT_PATH).exists():
        print(f"  [错误] SAM3 checkpoint 不存在: {CHECKPOINT_PATH}")
        return

    gbg_weight_path = PROJECT_ROOT / "weights" / f"gbg_tgt_20shot_seed{gbg_seed}.pth"
    if not gbg_weight_path.exists():
        print(f"  [错误] GBG 权重不存在: {gbg_weight_path}")
        print(f"  请先运行 E6 目标域 20-shot 训练: "
              f"python src/run_experiments_gbg.py --domain target --shots 20 --seeds {gbg_seed}")
        return

    print(f"  加载 SAM3 Backbone + LoRA 注入 ...")
    extractor = SAM3InferenceExtractor(CHECKPOINT_PATH)
    inject_lora_to_vit(extractor.model)

    print(f"  加载 GBG 权重: {gbg_weight_path.name}")
    checkpoint = torch.load(str(gbg_weight_path), map_location=DEVICE, weights_only=False)

    if "lora_params" in checkpoint:
        loaded = load_lora_params(extractor.model, checkpoint["lora_params"])
        print(f"    已载入 {loaded} 个 LoRA 参数")

    gbg_model = GBG_SAM3_Model().to(DEVICE)
    gbg_model.eval()

    if "gbg_model" in checkpoint:
        gbg_model.load_state_dict(checkpoint["gbg_model"], strict=True)
        print(f"    已载入 GBG 模型权重 (decoder + adapter + EU-head)")
    else:
        print(f"  [错误] 未在 checkpoint 中找到 'gbg_model' key")
        return

    del checkpoint

    count = 0
    for img_path in tqdm(image_paths, desc="  GBG-SAM3", unit="img", ncols=100):
        out_path = PREDS_GBG_DIR / f"{img_path.stem}.png"

        img_tensor, _ = load_and_preprocess_image(img_path)

        features = extractor.extract(img_tensor)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = gbg_model(features, img_tensor.unsqueeze(0).to(DEVICE))

        logits = F.interpolate(
            logits.float(), size=(TARGET_SIZE, TARGET_SIZE),
            mode="bilinear", align_corners=False,
        )
        probs = torch.sigmoid(logits)
        pred_bin = (probs[0, 0] > 0.5).cpu().numpy().astype(np.uint8) * 255

        pred_img = Image.fromarray(pred_bin, mode="L")
        pred_img.save(out_path)
        count += 1

        del features, logits, probs

    del extractor, gbg_model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"\n  Model 2 完成! 共保存 {count} 张掩膜 → {PREDS_GBG_DIR}")


# ==========================================================================
# 模块 8: 命令行参数与主入口
# ==========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WHU-Mix 全量测试集推理与掩膜保存",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/generate_mask_predictions.py                     # 两个模型都跑
  python src/generate_mask_predictions.py --models 0          # 仅 Model 0
  python src/generate_mask_predictions.py --models 2          # 仅 Model 2
  python src/generate_mask_predictions.py --gbg_seed 123      # 指定 GBG seed
        """,
    )
    parser.add_argument(
        "--models", type=str, default="0,2",
        help="要运行的模型, 逗号分隔: 0 (SAM3 Zero-shot), 2 (GBG-SAM3). 默认: 0,2",
    )
    parser.add_argument(
        "--gbg_seed", type=int, default=42,
        help="GBG-SAM3 权重种子 (默认: 42)",
    )
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        print("\n  警告: 未检测到 CUDA GPU!")

    args = parse_args()
    models = set(int(m.strip()) for m in args.models.split(","))

    print(f"\n{'='*60}")
    print(f"  WHU-Mix 全量测试集掩膜推理")
    print(f"  设备: {DEVICE}")
    print(f"  运行模型: {sorted(models)}")
    print(f"  输入目录: {IMAGE_DIR}")
    print(f"{'='*60}")

    if 0 in models:
        run_sam3_zeroshot()

    if 2 in models:
        run_gbg_sam3(gbg_seed=args.gbg_seed)

    print(f"\n{'='*60}")
    print(f"  全部推理完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
