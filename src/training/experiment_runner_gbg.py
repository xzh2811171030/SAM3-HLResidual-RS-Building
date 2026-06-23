"""
=============================================================================
experiment_runner_gbg.py  ---  GBG-SAM3 (Gated Boundary-Guided SAM3) (E6)
=============================================================================
核心创新:
  1. GatedBoundaryAdapter: 浅层边缘特征提取 + 可学习通道门控 (零初始化)
  2. EdgeUncertaintyHead (EU-Head): 边界不确定性估计
  3. 联合损失: seg + unc + reg + gate_reg
  4. UG-DP 后处理 (Uncertainty-Guided Douglas-Peucker, d=5)
  5. LoRA 作为 backbone PEFT, 门控适配器 + EU-Head 叠加

用法:
  from training.experiment_runner_gbg import run_single_experiment_gbg
  result = run_single_experiment_gbg(
      domain="source", num_shots=5, seed=42, full_test=False,
      weight_save_path="weights/gbg_src_5shot_seed42.pth",
  )
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
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_SAM3_SRC = _SRC / "models" / "sam3"
if str(_SAM3_SRC) not in sys.path:
    sys.path.insert(0, str(_SAM3_SRC))

from utils.cloud_paths import get_domain_paths, get_platform_name
from data.dataset import FewShotRSIDDataset, ValDataset
from evaluation.eval_metrics import compute_iou as compute_iou_np
from evaluation.eval_metrics import compute_f1 as compute_f1_np
from sam3.model_builder import build_sam3_image_model

EVAL_BATCH_SIZE: int = 32
NUM_EPOCHS: int = 30
LEARNING_RATE: float = 3e-4
NUM_WORKERS: int = 1
UGDP_D: int = 5


def get_train_batch_size(num_shots: Optional[int]) -> int:
    if num_shots is None:
        return 16
    if num_shots <= 5:
        return 2
    if num_shots <= 10:
        return 4
    if num_shots <= 20:
        return 4
    return 8


FEATURE_SIZE: Tuple[int, int] = (64, 64)
SAM3_INPUT_SIZE: int = 1008
TARGET_SIZE: int = 512
FEATURE_CHANNELS: int = 256

LORA_RANK: int = 8
LORA_ALPHA: int = 16
LORA_DROPOUT: float = 0.05
LORA_TARGET_MODULES: Tuple[str, ...] = ("qkv",)

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================================================
# 模块 2: 轻量掩膜解码器 (LightweightMaskDecoder)
#     输入 [B,256,64,64] 上采样 → [B,1,512,512]
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


# ==========================================================================
# 模块 3: Dice 损失函数
# ==========================================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        batch_size = probs.shape[0]
        probs_flat = probs.view(batch_size, -1)
        target_flat = target.view(batch_size, -1)
        intersection = (probs_flat * target_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()


# ==========================================================================
# 模块 4: 门控边界适配器 (GatedBoundaryAdapter)
#
#     从 RGB 原图提取浅层边缘特征, 通过可学习门控与 SAM3 特征融合
#     公式: fused = F_sam + F_edge * alpha_gate
#     其中 alpha_gate 零初始化 (nn.Parameter(torch.zeros(...)))
# ==========================================================================
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
            nn.Conv2d(128, FEATURE_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(FEATURE_CHANNELS),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.alpha_gate = nn.Parameter(torch.zeros(FEATURE_CHANNELS, 1, 1))

    def forward(self, rgb_image: torch.Tensor, sam_features: torch.Tensor) -> torch.Tensor:
        edge_features = self.edge_extractor(rgb_image)
        return sam_features + edge_features * self.alpha_gate


# ==========================================================================
# 模块 5: 边界不确定性头 (EdgeUncertaintyHead / EU-Head)
#
#     输入融合特征 [B,256,64,64] → 两层 3x3 Conv + Sigmoid → [B,1,64,64]
#     上采样至 [B,1,512,512] 作为不确定性图
# ==========================================================================
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
        unc_map = self.conv2(x)
        unc_map = F.interpolate(
            unc_map, size=(TARGET_SIZE, TARGET_SIZE),
            mode="bilinear", align_corners=False,
        )
        return unc_map


# ==========================================================================
# 模块 6: MLP 梯度兼容补丁
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
    else:
        print("  [警告] MLP 梯度兼容补丁: 未找到任何 Mlp 模块")


# ==========================================================================
# 模块 7: LoRA 注入函数
# ==========================================================================
def inject_lora_to_vit(model, rank: int = 8, alpha: int = 16, dropout: float = 0.05) -> None:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("未安装 peft, 正在安装 ...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "peft"])
        from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=list(LORA_TARGET_MODULES), bias="none",
    )
    vit_trunk = model.backbone.vision_backbone.trunk
    model.backbone.vision_backbone.trunk = get_peft_model(vit_trunk, lora_config)
    _patch_vit_mlp_for_grad_compat(model.backbone.vision_backbone.trunk)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"  LoRA 注入完成: 可训练 {trainable_count:,} / {total_count:,} "
          f"({100 * trainable_count / total_count:.2f}%)")


# ==========================================================================
# 模块 8: GBG-SAM3 完整模型
#
#     forward 输出: (logits [B,1,512,512], unc_map [B,1,512,512])
#     数据流:
#       RGB → SAM3 Backbone (LoRA) → sam_features [B,256,64,64]
#       RGB → GatedBoundaryAdapter → fused [B,256,64,64]
#       fused → LightweightMaskDecoder → logits [B,1,512,512]
#       fused → EdgeUncertaintyHead → unc_map [B,1,512,512]
# ==========================================================================
class GBG_SAM3_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.boundary_adapter = GatedBoundaryAdapter()
        self.decoder = LightweightMaskDecoder()
        self.eu_head = EdgeUncertaintyHead()

    def forward(
        self,
        sam_features: torch.Tensor,
        rgb_images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fused = self.boundary_adapter(rgb_images, sam_features)
        logits = self.decoder(fused)
        unc_map = self.eu_head(fused)
        return logits, unc_map


# ==========================================================================
# 模块 9: GBG 联合损失函数
#
#     seg_loss  = BCEWithLogitsLoss(logits, mask) + DiceLoss(logits, mask)
#     unc_loss  = BCEWithLogitsLoss(unc_map, boundary_tensor)
#     reg_loss  = mean(sigmoid(unc_map) * |sigmoid(logits) - mask|.detach())
#     gate_reg  = 0.01 * ||alpha_gate||_2
#     total     = seg_loss + lambda_unc*unc_loss + lambda_reg*reg_loss + gate_reg
#     (lambda_unc/lambda_reg 支持两阶段 warm-up: Phase1=0, Phase2=1.0/0.1)
# ==========================================================================
def compute_gbg_loss(
    logits: torch.Tensor,
    unc_map: torch.Tensor,
    mask: torch.Tensor,
    boundary_tensor: torch.Tensor,
    adapter: GatedBoundaryAdapter,
    bce_fn: nn.Module,
    dice_fn: nn.Module,
    lambda_unc: float = 1.0,
    lambda_reg: float = 0.1,
) -> Dict[str, torch.Tensor]:
    seg_loss = bce_fn(logits, mask) + dice_fn(logits, mask)
    unc_loss = bce_fn(unc_map, boundary_tensor)
    reg_loss = (
        torch.sigmoid(unc_map)
        * torch.abs(torch.sigmoid(logits) - mask).detach()
    ).mean()
    gate_reg = 0.01 * torch.norm(adapter.alpha_gate, p=2)
    total_loss = seg_loss + lambda_unc * unc_loss + lambda_reg * reg_loss + gate_reg

    return {
        "total": total_loss,
        "seg": seg_loss,
        "unc": unc_loss,
        "reg": reg_loss,
        "gate": gate_reg,
    }


# ==========================================================================
# 模块 10: SAM3 云端特征提取器 (带梯度, 用于训练时 LoRA 反传)
# ==========================================================================
class SAM3CloudExtractor:
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

    def _preprocess(self, image_tensor: torch.Tensor) -> torch.Tensor:
        img_uint8 = (image_tensor * 255.0).clamp(0, 255).to(torch.uint8)
        return self.transform(img_uint8).unsqueeze(0).to(self.device)

    def _forward_backbone(self, img_processed: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = self.model.backbone.forward_image(img_processed)
        features = backbone_out["vision_features"].float()
        if features.shape[-2:] != FEATURE_SIZE:
            features = F.interpolate(
                features, size=FEATURE_SIZE,
                mode="bilinear", align_corners=False,
            )
        return features

    @torch.no_grad()
    def extract(self, image_tensor: torch.Tensor) -> torch.Tensor:
        img_processed = self._preprocess(image_tensor)
        return self._forward_backbone(img_processed).detach()

    def extract_with_grad(self, image_tensor: torch.Tensor) -> torch.Tensor:
        img_processed = self._preprocess(image_tensor)
        return self._forward_backbone(img_processed)


# ==========================================================================
# 模块 11: SAM3 评估用特征提取器 (纯推理, 无梯度)
# ==========================================================================
class SAM3EvalExtractor:
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
                feat = F.interpolate(
                    feat, size=FEATURE_SIZE,
                    mode="bilinear", align_corners=False,
                )
            features_list.append(feat.detach())
        return torch.cat(features_list, dim=0)


# ==========================================================================
# 模块 12: 预计算 SAM3 冻结特征缓存
# ==========================================================================
def precache_features(
    dataset,
    extractor: SAM3CloudExtractor,
) -> Dict[str, Dict[str, torch.Tensor]]:
    cached: Dict[str, Dict[str, torch.Tensor]] = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        name = sample["name"]
        feat = extractor.extract(sample["image"]).cpu()
        cached[name] = {
            "image": sample["image"],
            "mask": sample["mask"],
            "boundary": sample["boundary"],
            "sam_features": feat,
        }
    return cached


# ==========================================================================
# 模块 13: 样本级 IoU 计算
# ==========================================================================
@torch.no_grad()
def compute_iou(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    pred_bin = (probs > threshold).float()
    batch_size = pred_bin.shape[0]
    pred_flat = pred_bin.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    iou_per_sample = (intersection + 1e-7) / (union + 1e-7)
    return iou_per_sample.mean().item()


# ==========================================================================
# 模块 14: GBG-SAM3 训练函数 (train_gbg_sam3) — v2.0 重构
#
#     v2.0 核心重构:
#       1. 废除训练集特征预缓存: 每 batch 直接从 dataset 读取 RGB,
#          通过 extract_with_grad 实时提取带梯度的 sam_features,
#          修复 LoRA 梯度断裂问题
#       2. 两阶段 Warm-up:
#          Phase 1 (epoch 1~10): 冻结 EU-Head + alpha_gate,
#                                仅训练主分割任务 (lambda_unc/reg = 0)
#          Phase 2 (epoch 11~):  解冻全部, 联合微调 (lambda_unc=1.0, lambda_reg=0.1)
#       3. 规范化 Optimizer: 仅包含 requires_grad=True 的参数 (gbg_model + extractor.model)
#       4. OOM 防御: Backbone + GBG 前向全包裹 autocast, 每 epoch 验证后 empty_cache()
#       5. 验证集仍使用预缓存(sam_features)加速, 不做梯度计算
# ==========================================================================
def train_gbg_sam3(
    gbg_model: GBG_SAM3_Model,
    train_dataset,
    val_keys: list,
    cached_val: Dict[str, Dict[str, torch.Tensor]],
    weight_save_path: str,
    model_name: str,
    feat_extractor: SAM3CloudExtractor,
    batch_size: int = 4,
) -> Dict:
    print(f"\n{'='*55}")
    print(f"  训练 GBG-SAM3: {model_name}")
    print(f"  TrainBatch={batch_size}  LR={LEARNING_RATE}  "
          f"Workers={NUM_WORKERS}  Train={len(train_dataset)}  Val={len(val_keys)}")
    print(f"{'='*55}")

    gbg_model = gbg_model.to(DEVICE)
    gbg_model.train()

    # === 规范化 Optimizer: 仅 requires_grad=True 的参数 ===
    def _build_trainable_list() -> list:
        trainable = [p for p in gbg_model.parameters() if p.requires_grad]
        for p in feat_extractor.model.parameters():
            if p.requires_grad:
                trainable.append(p)
        return trainable

    trainable = _build_trainable_list()
    total_trainable = sum(p.numel() for p in trainable)
    print(f"  总可训练参数量: {total_trainable / 1e6:.2f} M")

    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None
    bce_fn = nn.BCEWithLogitsLoss()
    dice_fn = DiceLoss()

    best_val_iou = 0.0
    warmup_epochs = 10

    # === PHASE 1: 冻结 EU-Head 和 alpha_gate，仅训练主分割任务 ===
    for param in gbg_model.eu_head.parameters():
        param.requires_grad = False
    gbg_model.boundary_adapter.alpha_gate.requires_grad = False

    # 重建 optimizer (排除刚冻结的参数)
    trainable = _build_trainable_list()
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)

    phase1_trainable = sum(p.numel() for p in trainable)
    print(f"  [Phase 1 Warm-up] EU-Head & alpha_gate 已冻结，仅训练主分割 (epoch 1~{warmup_epochs})")
    print(f"  [Phase 1] 可训练参数量: {phase1_trainable / 1e6:.2f} M")

    pbar = tqdm(range(1, NUM_EPOCHS + 1), desc=f"  [{model_name}]",
                unit="epoch", ncols=100)

    for epoch in pbar:
        # === Phase 1 → Phase 2 切换 (epoch warmup_epochs+1) ===
        if epoch == warmup_epochs + 1:
            for param in gbg_model.eu_head.parameters():
                param.requires_grad = True
            gbg_model.boundary_adapter.alpha_gate.requires_grad = True

            trainable = _build_trainable_list()
            optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)

            phase2_trainable = sum(p.numel() for p in trainable)
            print(f"\n  [Phase 2 Joint] EU-Head & alpha_gate 已解冻，联合微调 (epoch {warmup_epochs + 1}~{NUM_EPOCHS})")
            print(f"  [Phase 2] 可训练参数量: {phase2_trainable / 1e6:.2f} M")

        # === 动态 λ 系数 ===
        if epoch <= warmup_epochs:
            lambda_unc = 0.0
            lambda_reg = 0.0
        else:
            lambda_unc = 1.0
            lambda_reg = 0.1

        # === 训练循环 ===
        gbg_model.train()
        epoch_total_loss = 0.0
        num_batches = 0

        for b in range(0, len(train_dataset), batch_size):
            optimizer.zero_grad(set_to_none=True)

            sam_feat_list = []
            rgb_list = []
            mask_list = []
            boundary_list = []

            # 直接从 train_dataset 逐样本读取, extract_with_grad 保持梯度流
            for idx in range(b, min(b + batch_size, len(train_dataset))):
                sample = train_dataset[idx]
                rgb_list.append(sample["image"])
                mask_list.append(sample["mask"])
                boundary_list.append(sample["boundary"])
                sam_feat_list.append(
                    feat_extractor.extract_with_grad(sample["image"])
                )

            sam_feat_t = torch.stack(sam_feat_list).squeeze(1)
            rgb_t = torch.stack(rgb_list).to(DEVICE)
            mask_t = torch.stack(mask_list).to(DEVICE)
            boundary_t = torch.stack(boundary_list).to(DEVICE)

            # Backbone 提取 + GBG 前向全包裹在 autocast 中 (OOM 防御)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, unc_map = gbg_model(sam_feat_t, rgb_t)

            loss_dict = compute_gbg_loss(
                logits.float(), unc_map.float(),
                mask_t, boundary_t,
                gbg_model.boundary_adapter,
                bce_fn, dice_fn,
                lambda_unc=lambda_unc,
                lambda_reg=lambda_reg,
            )
            loss = loss_dict["total"]

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_total_loss += loss.item()
            num_batches += 1

        epoch_total_loss /= num_batches

        # === 验证 (使用预缓存 sam_features 加速) ===
        gbg_model.eval()
        val_iou_sum = 0.0

        with torch.no_grad():
            for key in val_keys:
                sample = cached_val[key]
                rgb_t = sample["image"].unsqueeze(0).to(DEVICE)
                gt_mask_t = sample["mask"].unsqueeze(0).to(DEVICE)
                sam_feat_t = sample["sam_features"].to(DEVICE)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = gbg_model(sam_feat_t, rgb_t)
                val_iou_sum += compute_iou(logits.float(), gt_mask_t)

        val_iou = val_iou_sum / len(val_keys) if val_keys else 0.0

        pbar.set_postfix_str(
            f"loss={epoch_total_loss:.4f}  val_iou={val_iou:.4f}  best={best_val_iou:.4f}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_dict: dict = {
                "gbg_model": gbg_model.state_dict(),
                "val_iou": best_val_iou,
            }
            lora_params = {
                n: p for n, p in feat_extractor.model.named_parameters()
                if p.requires_grad
            }
            save_dict["lora_params"] = {n: p.detach().cpu() for n, p in lora_params.items()}
            Path(weight_save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(save_dict, weight_save_path)
            pbar.set_postfix_str(
                f"loss={epoch_total_loss:.4f}  val_iou={val_iou:.4f}  ★ best={best_val_iou:.4f}"
            )

        # OOM 防御: 每 epoch 验证后显式清理显存
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"  训练完成: best Val IoU = {best_val_iou:.4f}")
    print(f"  权重已保存: {weight_save_path}")
    return {"best_val_iou": best_val_iou, "model_name": model_name}


# ==========================================================================
# 模块 14b: 源域 20-shot 权重查找与加载 (GBG 跨域自适应)
#
#     在目标域少样本训练前, 从 E5 源域 20-shot LoRA 权重中继承:
#       - decoder 权重 → gbg_model.decoder (同为 LightweightMaskDecoder)
#       - lora_params  → extractor.model 对应 LoRA 参数
#     以此实现与 E5 完全对齐的控制变量域迁移设定
# ==========================================================================
def _find_source_20shot_weight_gbg(
    seed: int,
    weights_dir: str,
) -> Optional[Path]:
    weight_name = f"sam3_lora_src_20shot_seed{seed}.pth"
    weight_path = Path(weights_dir) / weight_name
    if weight_path.exists():
        return weight_path
    return None


def _load_source_pretrained_for_gbg_adaptation(
    gbg_model: GBG_SAM3_Model,
    extractor,
    seed: int,
    weights_dir: str,
) -> None:
    weight_path = _find_source_20shot_weight_gbg(seed, weights_dir)
    if weight_path is None:
        print(f"  [警告] 未找到对应 seed={seed} 的源域 20-shot LoRA 权重 "
              f"(sam3_lora_src_20shot_seed{seed}.pth)，"
              f"目标域 GBG-SAM3 将从头进行少样本训练。")
        return

    print(f"\n  [Domain Transfer] 载入源域 20-shot 预训练权重作为目标域 GBG 自适应初始化: "
          f"{weight_path.name}")

    checkpoint = torch.load(str(weight_path), map_location=DEVICE, weights_only=False)

    if "decoder" in checkpoint:
        gbg_model.decoder.load_state_dict(checkpoint["decoder"], strict=True)
        print(f"    已载入 decoder 权重 → gbg_model.decoder")

    if "lora_params" in checkpoint:
        lora_state = checkpoint["lora_params"]
        model_state = extractor.model.state_dict()
        loaded_count = 0
        for key, value in lora_state.items():
            if key in model_state:
                model_state[key].copy_(value.to(DEVICE))
                loaded_count += 1
        print(f"    已载入 {loaded_count} 个 LoRA 参数到 extractor.model")

    del checkpoint


# ==========================================================================
# 模块 15: UG-DP 后处理 (Uncertainty-Guided Douglas-Peucker, d=5)
#
#     对二值预测掩膜提取轮廓, 按不确定性均值确定动态容差,
#     用 approxPolyDP 简化轮廓后重新绘制, 计算 Boundary IoU
# ==========================================================================
def ugdp_postprocess(
    pred_prob: np.ndarray,
    unc_map: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    pred_bin = (pred_prob > threshold).astype(np.uint8)

    contours, _ = cv2.findContours(
        pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

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
# 模块 16: UG-DP 增强的测试集评估
#
#     对每张测试图:
#       1. SAM3 提取特征 + GBG forward → logits, unc_map
#       2. 标准预测: prob > 0.5
#       3. UG-DP 预测: ugdp_postprocess(prob, unc_map)
#       4. 计算标准 mIoU/F1 + UG-DP 增强的 Boundary IoU
# ==========================================================================
@torch.no_grad()
def evaluate_with_ugdp(
    weight_path: str,
    domain_paths: Dict[str, str],
) -> Dict[str, float]:
    checkpoint_path = domain_paths["sam3_checkpoint"]
    test_image_dir = domain_paths["test_image_dir"]
    test_dual_dir = domain_paths["test_dual_dir"]

    print(f"\n  加载 GBG-SAM3 评估器 (测试集: {test_image_dir}) ...")
    sam3_extractor = SAM3EvalExtractor(checkpoint_path, DEVICE)
    inject_lora_to_vit(sam3_extractor.model, rank=LORA_RANK, alpha=LORA_ALPHA,
                       dropout=LORA_DROPOUT)

    checkpoint = torch.load(weight_path, map_location=DEVICE, weights_only=False)

    if "lora_params" in checkpoint:
        lora_state = checkpoint["lora_params"]
        model_state = sam3_extractor.model.state_dict()
        for key, value in lora_state.items():
            if key in model_state:
                model_state[key].copy_(value.to(DEVICE))
        print(f"    已载入 LoRA 参数")

    gbg_model = GBG_SAM3_Model().to(DEVICE)
    gbg_model.eval()
    if "gbg_model" in checkpoint:
        gbg_model.load_state_dict(checkpoint["gbg_model"])
        print(f"    已载入 GBG 模型权重")

    test_dataset = ValDataset(
        image_dir=test_image_dir,
        dual_label_dir=test_dual_dir,
        target_size=TARGET_SIZE,
    )

    def collate_fn(batch):
        return {
            "image": torch.stack([item["image"] for item in batch]),
            "mask": torch.stack([item["mask"] for item in batch]),
            "boundary": torch.stack([item["boundary"] for item in batch]),
            "name": [item["name"] for item in batch],
        }

    test_loader = DataLoader(
        test_dataset, batch_size=EVAL_BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    all_std_preds: List[np.ndarray] = []
    all_ugdp_preds: List[np.ndarray] = []
    all_gts: List[np.ndarray] = []

    for batch in tqdm(test_loader, desc=f"  评估 GBG-SAM3 (BS={EVAL_BATCH_SIZE})",
                      unit="batch", ncols=100):
        img_batch = batch["image"]
        mask_batch = batch["mask"]
        B = img_batch.shape[0]

        features_batch = sam3_extractor.extract_batch(img_batch)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, unc_maps = gbg_model(features_batch, img_batch.to(DEVICE))
        logits = F.interpolate(
            logits, size=(TARGET_SIZE, TARGET_SIZE),
            mode="bilinear", align_corners=False,
        )
        unc_maps = F.interpolate(
            unc_maps, size=(TARGET_SIZE, TARGET_SIZE),
            mode="bilinear", align_corners=False,
        )
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        uncs = torch.sigmoid(unc_maps.float()).cpu().numpy()

        for i in range(B):
            pred_prob = probs[i, 0]
            unc_prob = uncs[i, 0]
            gt = mask_batch[i, 0].numpy()

            std_pred = (pred_prob > 0.5).astype(np.float32)
            ugdp_pred = ugdp_postprocess(pred_prob, unc_prob)

            all_std_preds.append(std_pred)
            all_ugdp_preds.append(ugdp_pred)
            all_gts.append(gt)

    std_preds_arr = np.stack(all_std_preds, axis=0).astype(bool)
    ugdp_preds_arr = np.stack(all_ugdp_preds, axis=0).astype(bool)
    gts_arr = np.stack(all_gts, axis=0).astype(bool)

    std_miou = compute_iou_np(std_preds_arr, gts_arr)
    std_f1 = compute_f1_np(std_preds_arr, gts_arr)
    std_biou = _compute_boundary_iou_ugdp(std_preds_arr, gts_arr, d=UGDP_D)

    ugdp_miou = compute_iou_np(ugdp_preds_arr, gts_arr)
    ugdp_f1 = compute_f1_np(ugdp_preds_arr, gts_arr)
    ugdp_biou = _compute_boundary_iou_ugdp(ugdp_preds_arr, gts_arr, d=UGDP_D)

    del sam3_extractor, gbg_model, test_dataset
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return {
        "std_mIoU": std_miou,
        "std_F1": std_f1,
        "std_Boundary_IoU": std_biou,
        "ugdp_mIoU": ugdp_miou,
        "ugdp_F1": ugdp_f1,
        "ugdp_Boundary_IoU": ugdp_biou,
    }


# ==========================================================================
# 模块 17: Boundary IoU 计算 (复用 cv2 形态学方法, 兼容 UG-DP)
# ==========================================================================
def _compute_boundary_iou_ugdp(
    pred_binary: np.ndarray,
    gt_binary: np.ndarray,
    d: int = UGDP_D,
) -> float:
    kernel_contour = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))

    biou_per_sample: List[float] = []
    for i in range(pred_binary.shape[0]):
        p = pred_binary[i].astype(np.uint8)
        g = gt_binary[i].astype(np.uint8)

        p_contour = cv2.morphologyEx(p, cv2.MORPH_GRADIENT, kernel_contour)
        g_contour = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, kernel_contour)

        p_boundary = cv2.dilate(p_contour, kernel_dilate)
        g_boundary = cv2.dilate(g_contour, kernel_dilate)

        union_boundary = np.logical_or(p_boundary, g_boundary)
        if union_boundary.sum() == 0:
            biou_per_sample.append(1.0)
            continue

        intersection = np.logical_and(p, g)[union_boundary].sum()
        union = np.logical_or(p, g)[union_boundary].sum()
        biou = (intersection + 1e-7) / (union + 1e-7)
        biou_per_sample.append(biou)

    return float(np.mean(biou_per_sample))


# ==========================================================================
# 模块 18: 单次 GBG-SAM3 实验运行入口
# ==========================================================================
def run_single_experiment_gbg(
    domain: str,
    num_shots: Optional[int],
    seed: int,
    full_test: bool = False,
    weight_save_path: str = "",
) -> Dict[str, float]:
    shots_label = f"{num_shots}-shot" if num_shots is not None else "full"
    domain_label = "源域(source_whu)" if domain == "source" else "目标域(target_whu_mix)"
    test_label = "全量8402" if full_test else "瘦身测试"

    print(f"\n{'='*60}")
    print(f"  GBG-SAM3 实验 | {shots_label} | seed={seed}")
    print(f"  域: {domain_label} | 测试: {test_label}")
    print(f"  平台: {get_platform_name()} | 设备: {DEVICE}")
    print(f"{'='*60}")

    dp = get_domain_paths(domain, full_test=full_test)

    print("\n[数据加载]")
    train_dataset = FewShotRSIDDataset(
        image_dir=dp["train_image_dir"],
        dual_label_dir=dp["train_label_dir"],
        bbox_json_path=dp["train_bbox_json"],
        num_shots=num_shots,
        seed=seed,
    )
    val_dataset = ValDataset(
        image_dir=dp["val_image_dir"],
        dual_label_dir=dp["val_label_dir"],
    )
    print(f"  训练集: {len(train_dataset)}  验证集: {len(val_dataset)}")

    print("\n[特征预缓存 (仅验证集, 训练集实时提取保持梯度流)]")
    extractor = SAM3CloudExtractor(dp["sam3_checkpoint"], DEVICE)
    cached_val = precache_features(val_dataset, extractor)
    print(f"  缓存 val={len(cached_val)} 张特征完成")

    val_keys = [val_dataset[i]["name"] for i in range(len(val_dataset))]

    train_batch_size = get_train_batch_size(num_shots)
    print(f"\n  自动选择训练 batch_size: {train_batch_size} (num_shots={num_shots})")

    inject_lora_to_vit(extractor.model, rank=LORA_RANK, alpha=LORA_ALPHA,
                       dropout=LORA_DROPOUT)

    gbg_model = GBG_SAM3_Model()

    if domain == "target" and num_shots is not None:
        weights_dir_for_pretrain = dp.get(
            "weights_dir",
            str(Path(dp["project_root"]) / "weights"),
        )
        _load_source_pretrained_for_gbg_adaptation(
            gbg_model=gbg_model,
            extractor=extractor,
            seed=seed,
            weights_dir=weights_dir_for_pretrain,
        )

    train_result = train_gbg_sam3(
        gbg_model=gbg_model,
        train_dataset=train_dataset,
        val_keys=val_keys,
        cached_val=cached_val,
        weight_save_path=weight_save_path,
        model_name=f"GBG-SAM3 ({shots_label}, {domain_label}, seed={seed})",
        feat_extractor=extractor,
        batch_size=train_batch_size,
    )

    del gbg_model, extractor, cached_val
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"\n[最终评估 ({test_label})]")
    eval_metrics = evaluate_with_ugdp(
        weight_path=weight_save_path,
        domain_paths=dp,
    )

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    result = {
        "domain": domain,
        "num_shots": num_shots,
        "seed": seed,
        "full_test": full_test,
        "best_val_iou": train_result["best_val_iou"],
        "std_mIoU": eval_metrics["std_mIoU"],
        "std_F1": eval_metrics["std_F1"],
        "std_Boundary_IoU": eval_metrics["std_Boundary_IoU"],
        "ugdp_mIoU": eval_metrics["ugdp_mIoU"],
        "ugdp_F1": eval_metrics["ugdp_F1"],
        "ugdp_Boundary_IoU": eval_metrics["ugdp_Boundary_IoU"],
    }

    print(f"\n  GBG-SAM3 实验完成: {shots_label} | {domain_label} | seed={seed}")
    print(f"    best_val_iou (val)     = {result['best_val_iou']:.4f}")
    print(f"    标准 mIoU               = {result['std_mIoU']:.4f}")
    print(f"    标准 F1                 = {result['std_F1']:.4f}")
    print(f"    标准 Boundary IoU       = {result['std_Boundary_IoU']:.4f}")
    print(f"    UG-DP mIoU              = {result['ugdp_mIoU']:.4f}")
    print(f"    UG-DP F1                = {result['ugdp_F1']:.4f}")
    print(f"    UG-DP Boundary IoU (d=5)= {result['ugdp_Boundary_IoU']:.4f}")

    return result


# ==========================================================================
# 模块 19: 自测入口
# ==========================================================================
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("警告: 未检测到 CUDA GPU, 将使用 CPU (极慢)")

    dp = get_domain_paths("source", full_test=False)
    Path(dp["weights_dir"]).mkdir(parents=True, exist_ok=True)

    result = run_single_experiment_gbg(
        domain="source",
        num_shots=5,
        seed=42,
        full_test=False,
        weight_save_path=str(Path(dp["weights_dir"]) / "gbg_src_5shot_seed42.pth"),
    )
    print(f"\n最终结果: {result}")
