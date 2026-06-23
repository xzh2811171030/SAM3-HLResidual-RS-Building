"""
=============================================================================
experiment_runner.py  ---  云端全量数据多随机种子核心训练与评估模块 (v2.0)
=============================================================================
功能说明:
  1. 提供云端 SAM3 特征提取器 (SAM3CloudExtractor)
     - 彻底移除 Activation Checkpointing (use_act_checkpoint=False)
     - 移除 object.__setattr__(base_vit, 'training', True) 等 8G 显存妥协 hack
     - 使用标准 PyTorch 前向传播梯度流, 适配 24GB 显存
  2. 提供 Decoder-Only 与 LoRA 两种基线的训练函数
     - 训练 batch_size 根据 num_shots 自动选择: 5-shot→2, 10/20-shot→4, full→16
     - 评估 batch_size 固定 32 (EVAL_BATCH_SIZE)
     - BCEWithLogitsLoss + DiceLoss 混合损失, AMP 混合精度
  3. 提供验证/测试集评估函数 (mIoU / F1-score / Boundary IoU)
  4. 提供单次实验运行入口 run_single_experiment()

v2.0 核心变更:
  - 废除 random_split: 验证集严格绑定到独立目录 (地理绝对隔离)
  - 新增 domain 参数: source (源域) vs target (目标域)
  - 新增 full_test 参数: 瘦身测试集 vs 全量 8402 张测试集
  - E5 跨域自适应: target 域少样本训练时自动加载源域 20-shot 权重初始化
  - E4 零样本跨域: run_zero_shot_cross_domain_eval() 跳过训练直接评估
  - 所有路径通过 cloud_paths.get_domain_paths() 获取

设计原则:
  - 云端 48GB 显存: 无梯度检查点, batch_size 10/20-shot→8, 5-shot→2
  - num_workers=8: 最大化 CPU 利用率

用法:
  from experiment_runner import run_single_experiment
  metrics = run_single_experiment(
      mode="lora",
      domain="source",
      num_shots=10,
      seed=42,
      full_test=False,
      weight_save_path="weights/sam3_lora_10shot_seed42_source.pth",
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
from evaluation.eval_metrics import evaluate_predictions
from sam3.model_builder import build_sam3_image_model

EVAL_BATCH_SIZE: int = 32
NUM_EPOCHS: int = 30
LEARNING_RATE: float = 3e-4
NUM_WORKERS: int = 8


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
# 模块 4: SAM3 云端特征提取器 (SAM3CloudExtractor)
#     24GB 显存模式: 无 Activation Checkpointing, 标准梯度流
# ==========================================================================
class SAM3CloudExtractor:
    def __init__(self, checkpoint_path: str, device: str = DEVICE):
        print("  加载 SAM3 云端模型 (24GB 模式: 无 Checkpointing) ...")
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
        img_processed = self.transform(img_uint8).unsqueeze(0).to(self.device)
        return img_processed

    def _preprocess_batch(self, image_batch: torch.Tensor) -> torch.Tensor:
        B = image_batch.shape[0]
        processed_list = []
        for i in range(B):
            img_uint8 = (image_batch[i] * 255.0).clamp(0, 255).to(torch.uint8)
            processed = self.transform(img_uint8)
            processed_list.append(processed)
        return torch.stack(processed_list, dim=0).to(self.device)

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
    def extract_batch(self, image_batch: torch.Tensor) -> torch.Tensor:
        B = image_batch.shape[0]
        features_list = []
        for i in range(B):
            img_uint8 = (image_batch[i] * 255.0).clamp(0, 255).to(torch.uint8)
            img_processed = self.transform(img_uint8).unsqueeze(0).to(self.device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                backbone_out = self.model.backbone.forward_image(img_processed)
            features = backbone_out["vision_features"].float()

            if features.shape[-2:] != FEATURE_SIZE:
                features = F.interpolate(
                    features, size=FEATURE_SIZE,
                    mode="bilinear", align_corners=False,
                )
            features_list.append(features.detach())
        return torch.cat(features_list, dim=0)

    @torch.no_grad()
    def extract(self, image_tensor: torch.Tensor) -> torch.Tensor:
        img_processed = self._preprocess(image_tensor)
        features = self._forward_backbone(img_processed)
        return features.detach()

    def extract_with_grad(self, image_tensor: torch.Tensor) -> torch.Tensor:
        img_processed = self._preprocess(image_tensor)
        features = self._forward_backbone(img_processed)
        return features


# ==========================================================================
# 模块 5: MLP 梯度兼容补丁 (_patch_vit_mlp_for_grad_compat)
#     绕过 addmm_act 融合算子, 使用标准 PyTorch 路径
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
        print(f"  MLP 梯度兼容补丁: 已修补 {n_patched} 个 Mlp 模块 "
              f"(绕过 addmm_act 融合算子)")
    else:
        print("  [警告] MLP 梯度兼容补丁: 未找到任何 Mlp 模块, 请检查 ViT 结构")


# ==========================================================================
# 模块 6: LoRA 注入函数 (inject_lora_to_vit)
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
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
    )

    vit_trunk = model.backbone.vision_backbone.trunk
    model.backbone.vision_backbone.trunk = get_peft_model(vit_trunk, lora_config)

    _patch_vit_mlp_for_grad_compat(model.backbone.vision_backbone.trunk)

    # 确保只有 LoRA 参数可训练（防御 get_peft_model 的副作用）
    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False

    trainable_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_count = sum(p.numel() for p in model.parameters())
    print(f"  LoRA 注入完成: 可训练 {trainable_count:,} / {total_count:,} "
          f"({100 * trainable_count / total_count:.2f}%)")


# ==========================================================================
# 模块 7: 预计算 SAM3 冻结特征缓存
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
# 模块 8: 单次训练函数 (train_single_baseline)
#     v2.0: train_keys 和 val_keys 从独立目录加载, 不再 random_split
# ==========================================================================
def train_single_baseline(
    decoder: nn.Module,
    train_keys: list,
    val_keys: list,
    cached: Dict[str, Dict[str, torch.Tensor]],
    weight_save_path: str,
    model_name: str,
    mode: str,
    feat_extractor: Optional[SAM3CloudExtractor] = None,
    batch_size: int = 4,
) -> Dict:
    print(f"\n{'='*55}")
    print(f"  训练: {model_name}")
    print(f"  模式: {mode}")
    print(f"  TrainBatch={batch_size}  LR={LEARNING_RATE}  "
          f"Workers={NUM_WORKERS}  Train={len(train_keys)}  Val={len(val_keys)}")
    print(f"{'='*55}")

    decoder = decoder.to(DEVICE)
    decoder.train()

    if mode == "decoder_only":
        trainable = list(decoder.parameters())
    elif mode == "lora":
        trainable = list(decoder.parameters())
        for n, p in feat_extractor.model.named_parameters():
            if p.requires_grad:
                trainable.append(p)
    else:
        raise ValueError(f"未知 mode: {mode}")

    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None
    bce_loss_fn = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss()

    best_val_iou = 0.0
    num_batches = max(1, (len(train_keys) + batch_size - 1) // batch_size)

    pbar = tqdm(range(1, NUM_EPOCHS + 1), desc=f"  [{model_name}]",
                unit="epoch", ncols=100)

    for epoch in pbar:
        decoder.train()
        epoch_loss = 0.0

        for b in range(0, len(train_keys), batch_size):
            batch_keys = train_keys[b:b + batch_size]

            optimizer.zero_grad(set_to_none=True)

            if mode == "decoder_only":
                sam_feat_list = [cached[k]["sam_features"].squeeze(0) for k in batch_keys]
                sam_feat_t = torch.stack(sam_feat_list).to(DEVICE)
                gt_mask_list = [cached[k]["mask"] for k in batch_keys]
                gt_mask_t = torch.stack(gt_mask_list).to(DEVICE)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = decoder(sam_feat_t)

            else:
                sam_feat_list = []
                gt_mask_list = []
                for key in batch_keys:
                    sample = cached[key]
                    gt_mask_list.append(sample["mask"])
                    sam_feat_list.append(
                        feat_extractor.extract_with_grad(sample["image"])
                    )
                sam_feat_t = torch.stack(sam_feat_list).squeeze(1)
                gt_mask_t = torch.stack(gt_mask_list).to(DEVICE)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = decoder(sam_feat_t)

            loss_bce = bce_loss_fn(logits.float(), gt_mask_t)
            loss_dice = dice_loss_fn(logits.float(), gt_mask_t)
            loss = loss_bce + loss_dice

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= num_batches

        decoder.eval()
        val_iou_sum = 0.0

        with torch.no_grad():
            for key in val_keys:
                sample = cached[key]
                img_t = sample["image"].unsqueeze(0).to(DEVICE)
                gt_mask_t = sample["mask"].unsqueeze(0).to(DEVICE)
                sam_feat_cpu = sample["sam_features"]

                if mode == "lora":
                    sam_feat_t = feat_extractor.extract(img_t.squeeze(0)).to(DEVICE)
                else:
                    sam_feat_t = sam_feat_cpu.to(DEVICE)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = decoder(sam_feat_t)
                val_iou_sum += compute_iou(logits.float(), gt_mask_t)

        val_iou = val_iou_sum / len(val_keys) if val_keys else 0.0

        pbar.set_postfix_str(
            f"loss={epoch_loss:.4f}  val_iou={val_iou:.4f}  best={best_val_iou:.4f}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_dict: dict = {"decoder": decoder.state_dict(), "val_iou": best_val_iou}
            if mode == "lora":
                lora_params = {
                    n: p for n, p in feat_extractor.model.named_parameters()
                    if p.requires_grad
                }
                save_dict["lora_params"] = {n: p.detach().cpu() for n, p in lora_params.items()}
            Path(weight_save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(save_dict, weight_save_path)
            pbar.set_postfix_str(
                f"loss={epoch_loss:.4f}  val_iou={val_iou:.4f}  ★ best={best_val_iou:.4f}"
            )

    print(f"  训练完成: best Val IoU = {best_val_iou:.4f}")
    print(f"  权重已保存: {weight_save_path}")
    return {"best_val_iou": best_val_iou, "model_name": model_name}


# ==========================================================================
# 模块 9: 样本级 IoU 计算
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
# 模块 10: 模型权重加载
# ==========================================================================
def load_model_weights_for_eval(model: nn.Module, weight_path: str, key: str = "decoder") -> None:
    checkpoint = torch.load(weight_path, map_location=DEVICE, weights_only=False)
    if key in checkpoint:
        state_dict = checkpoint[key]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)


# ==========================================================================
# 模块 10b: 源域权重查找与加载 (E4/E5 跨域实验专用)
# ==========================================================================
def _find_source_20shot_weight(
    mode: str,
    seed: int,
    weights_dir: str,
) -> Optional[Path]:
    mode_short = "dec" if mode == "decoder_only" else "lora"
    weight_name = f"sam3_{mode_short}_src_20shot_seed{seed}.pth"
    weight_path = Path(weights_dir) / weight_name
    if weight_path.exists():
        return weight_path
    return None


def _load_source_pretrained_for_adaptation(
    decoder: nn.Module,
    extractor,
    mode: str,
    seed: int,
    weights_dir: str,
) -> None:
    mode_short = "dec" if mode == "decoder_only" else "lora"
    weight_path = _find_source_20shot_weight(mode, seed, weights_dir)
    if weight_path is None:
        print(f"  [警告] 未找到对应 seed={seed} 的源域 20-shot 权重 "
              f"(sam3_{mode_short}_src_20shot_seed{seed}.pth)，"
              f"目标域将从头进行少样本训练。")
        return

    print(f"\n  [Domain Transfer] 载入源域 20-shot 预训练权重作为目标域自适应初始化: "
          f"{weight_path.name}")

    checkpoint = torch.load(str(weight_path), map_location=DEVICE, weights_only=False)

    if "decoder" in checkpoint:
        decoder.load_state_dict(checkpoint["decoder"])
        print(f"    已载入 decoder 权重")

    if mode == "lora" and "lora_params" in checkpoint:
        lora_state = checkpoint["lora_params"]
        model_state = extractor.model.state_dict()
        loaded_count = 0
        for key, value in lora_state.items():
            if key in model_state:
                model_state[key].copy_(value.to(DEVICE))
                loaded_count += 1
        print(f"    已载入 {loaded_count} 个 LoRA 参数到 extractor.model")


# ==========================================================================
# 模块 11: SAM3 评估用特征提取器
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
# 模块 12: 测试集全量评估 (mIoU + F1-score + Boundary IoU)
#     v2.0: 使用 domain_paths 获取测试路径, 支持双轨制
# ==========================================================================
@torch.no_grad()
def evaluate_on_test_set(
    weight_path: str,
    mode: str,
    domain_paths: Dict[str, str],
) -> Dict[str, float]:
    checkpoint_path = domain_paths["sam3_checkpoint"]
    test_image_dir = domain_paths["test_image_dir"]
    test_dual_dir = domain_paths["test_dual_dir"]

    print(f"\n  加载评估提取器 (测试集: {test_image_dir}) ...")
    extractor = SAM3EvalExtractor(checkpoint_path, DEVICE)

    if mode == "lora":
        inject_lora_to_vit(extractor.model, rank=8, alpha=16, dropout=0.05)
        lora_checkpoint = torch.load(weight_path, map_location=DEVICE, weights_only=False)
        if "lora_params" in lora_checkpoint:
            lora_state = lora_checkpoint["lora_params"]
            model_state = extractor.model.state_dict()
            for key, value in lora_state.items():
                if key in model_state:
                    model_state[key].copy_(value.to(DEVICE))

    decoder = LightweightMaskDecoder().to(DEVICE)
    decoder.eval()
    load_model_weights_for_eval(decoder, weight_path, key="decoder")

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

    all_preds: List[np.ndarray] = []
    all_gts: List[np.ndarray] = []

    for batch in tqdm(test_loader, desc=f"  评估 {mode} (BS={EVAL_BATCH_SIZE})",
                      unit="batch", ncols=100):
        img_batch = batch["image"]
        mask_batch = batch["mask"]
        B = img_batch.shape[0]

        features_batch = extractor.extract_batch(img_batch)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = decoder(features_batch)
        logits = F.interpolate(
            logits, size=(TARGET_SIZE, TARGET_SIZE),
            mode="bilinear", align_corners=False,
        )
        preds = torch.sigmoid(logits.float()).cpu().numpy()

        for i in range(B):
            all_preds.append(preds[i, 0])
            all_gts.append(mask_batch[i, 0].numpy())

    preds_arr = np.stack(all_preds, axis=0)
    gts_arr = np.stack(all_gts, axis=0)

    metrics = evaluate_predictions(preds_arr, gts_arr)

    del extractor, decoder, test_dataset
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return metrics


# ==========================================================================
# 模块 13: 单次实验运行入口 (run_single_experiment) --- v2.0 核心重构
#
#     完整的训练 + 评估流程:
#       1. 通过 get_domain_paths() 获取 domain 对应的所有路径
#       2. 使用 FewShotRSIDDataset 按 num_shots + seed 采样训练数据
#       3. 验证集从独立 val 目录加载 (ValDataset), 废除 random_split
#       4. 预计算 SAM3 冻结特征
#       5. 训练 Decoder-Only 或 LoRA (含 MLP 补丁)
#       6. 在独立测试集上做最终评估 (瘦身或全量, 取决于 full_test)
#       7. 返回 {mIoU, F1, Boundary_IoU, best_val_iou}
#
#     参数:
#       mode:            "decoder_only" 或 "lora"
#       domain:           "source" (源域) 或 "target" (目标域)
#       num_shots:        5 / 10 / 20 / None (全量)
#       seed:            随机种子 (42 / 123 / 456)
#       full_test:       是否使用全量 8402 张测试集
#       weight_save_path: 最优权重保存路径
#
#     返回:
#       Dict 包含 mIoU, F1, Boundary_IoU, best_val_iou
# ==========================================================================
def run_single_experiment(
    mode: str,
    domain: str,
    num_shots: Optional[int],
    seed: int,
    full_test: bool = False,
    weight_save_path: str = "",
) -> Dict[str, float]:
    mode_label = "Decoder-Only" if mode == "decoder_only" else "LoRA"
    shots_label = f"{num_shots}-shot" if num_shots is not None else "full"
    domain_label = "源域(source_whu)" if domain == "source" else "目标域(target_whu_mix)"
    test_label = "全量8402" if full_test else "瘦身测试"

    print(f"\n{'='*60}")
    print(f"  实验: {mode_label} | {shots_label} | seed={seed}")
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

    print("\n[特征预缓存]")
    extractor = SAM3CloudExtractor(dp["sam3_checkpoint"], DEVICE)

    cached_train = precache_features(train_dataset, extractor)
    cached_val = precache_features(val_dataset, extractor)
    print(f"  缓存 train={len(cached_train)} / val={len(cached_val)} 张特征完成")

    train_keys = [train_dataset[i]["name"] for i in range(len(train_dataset))]
    val_keys = [val_dataset[i]["name"] for i in range(len(val_dataset))]

    combined_cache = {**cached_train, **cached_val}

    train_batch_size = get_train_batch_size(num_shots)
    print(f"\n  自动选择训练 batch_size: {train_batch_size} "
          f"(num_shots={num_shots})")

    decoder = LightweightMaskDecoder()

    if mode == "lora":
        inject_lora_to_vit(extractor.model, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)

        # ========== 修复开始 ==========
        # 强制冻结所有参数（包括可能被 LoRA 注入意外解冻的主干参数）
        for param in extractor.model.parameters():
            param.requires_grad = False

        # 仅解冻 LoRA 参数（名称包含 "lora"）
        for name, param in extractor.model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True

        # 解冻官方 Mask Decoder（必须微调以适配遥感建筑分割）
        for param in extractor.model.mask_decoder.parameters():
            param.requires_grad = True

        # 解冻官方 Prompt Encoder（处理 bbox 输入，跨域自适应必需）
        for param in extractor.model.prompt_encoder.parameters():
            param.requires_grad = True

        # 打印修正后的可训练参数量（应 < 1%）
        total = sum(p.numel() for p in extractor.model.parameters())
        trainable = sum(p.numel() for p in extractor.model.parameters() if p.requires_grad)
        print(f"  [修复后] 可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        # ========== 修复结束 ==========

    if domain == "target" and num_shots is not None:
        weights_dir_for_pretrain = dp.get(
            "weights_dir",
            str(Path(dp["project_root"]) / "weights"),
        )
        _load_source_pretrained_for_adaptation(
            decoder=decoder,
            extractor=extractor,
            mode=mode,
            seed=seed,
            weights_dir=weights_dir_for_pretrain,
        )

    train_result = train_single_baseline(
        decoder=decoder,
        train_keys=train_keys,
        val_keys=val_keys,
        cached=combined_cache,
        weight_save_path=weight_save_path,
        model_name=f"{mode_label} ({shots_label}, {domain_label}, seed={seed})",
        mode=mode,
        feat_extractor=extractor if mode == "lora" else None,
        batch_size=train_batch_size,
    )

    del decoder, extractor, cached_train, cached_val, combined_cache
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"\n[最终评估 ({test_label})]")
    eval_metrics = evaluate_on_test_set(
        weight_path=weight_save_path,
        mode=mode,
        domain_paths=dp,
    )

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    

    result = {
        "mode": mode,
        "domain": domain,
        "num_shots": num_shots,
        "full_test": full_test,
        "seed": seed,
        "best_val_iou": train_result["best_val_iou"],
        "mIoU": eval_metrics.get("mIoU", 0.0),
        "F1": eval_metrics.get("F1", 0.0),
        "Boundary_IoU": eval_metrics.get("Boundary_IoU", 0.0),
    }

    print(f"\n  实验完成: {mode_label} | {shots_label} | {domain_label} | seed={seed}")
    print(f"    best_val_iou = {result['best_val_iou']:.4f}")
    print(f"    mIoU         = {result['mIoU']:.4f}")
    print(f"    F1           = {result['F1']:.4f}")
    print(f"    Boundary IoU = {result['Boundary_IoU']:.4f}")

    return result


# ==========================================================================
# 模块 14: E4 零样本跨域评估 (Zero-Shot Cross-Domain Evaluation)
#
#     跳过训练, 直接用源域 20-shot 预训练权重评估目标域测试集
#
#     参数:
#       mode:            "decoder_only" 或 "lora"
#       seed:            随机种子 (42 / 123 / 456)
#       full_test:       是否使用全量 8402 张测试集
#       weights_dir:     源域权重所在目录
#
#     返回:
#       Dict 包含 mIoU, F1, Boundary_IoU
# ==========================================================================
def run_zero_shot_cross_domain_eval(
    mode: str,
    seed: int,
    full_test: bool = False,
    weights_dir: str = "",
) -> Dict[str, float]:
    mode_label = "Decoder-Only" if mode == "decoder_only" else "LoRA"
    test_label = "全量8402" if full_test else "瘦身测试"

    print(f"\n{'='*60}")
    print(f"  E4 零样本跨域评估: {mode_label} | seed={seed}")
    print(f"  源域(source_whu) 20-shot → 目标域(target_whu_mix) 测试")
    print(f"  测试集: {test_label} | 设备: {DEVICE}")
    print(f"{'='*60}")

    source_weight_path = _find_source_20shot_weight(mode, seed, weights_dir)
    if source_weight_path is None:
        mode_short = "dec" if mode == "decoder_only" else "lora"
        raise FileNotFoundError(
            f"未找到源域 20-shot 权重: "
            f"sam3_{mode_short}_src_20shot_seed{seed}.pth\n"
            f"  请先在 source_whu 上完成 20-shot 训练:\n"
            f"  python src/run_all_experiments.py --domain source --shots 20"
        )

    print(f"  源域权重: {source_weight_path.name}")

    dp_target = get_domain_paths("target", full_test=full_test)

    print(f"  目标域测试集: {dp_target['test_image_dir']}")

    eval_metrics = evaluate_on_test_set(
        weight_path=str(source_weight_path),
        mode=mode,
        domain_paths=dp_target,
    )

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    result = {
        "mode": mode,
        "domain": "zero_shot_cross",
        "num_shots": 20,
        "full_test": full_test,
        "seed": seed,
        "source_weight": source_weight_path.name,
        "best_val_iou": 0.0,
        "mIoU": eval_metrics.get("mIoU", 0.0),
        "F1": eval_metrics.get("F1", 0.0),
        "Boundary_IoU": eval_metrics.get("Boundary_IoU", 0.0),
    }

    print(f"\n  E4 零样本跨域评估完成: {mode_label} | seed={seed}")
    print(f"    mIoU         = {result['mIoU']:.4f}")
    print(f"    F1           = {result['F1']:.4f}")
    print(f"    Boundary IoU = {result['Boundary_IoU']:.4f}")

    return result


# ==========================================================================
# 模块 15: 自测入口
# ==========================================================================
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("警告: 未检测到 CUDA GPU, 将使用 CPU (极慢)")

    dp = get_domain_paths("source", full_test=False)
    Path(dp["weights_dir"]).mkdir(parents=True, exist_ok=True)

    result = run_single_experiment(
        mode="decoder_only",
        domain="source",
        num_shots=5,
        seed=42,
        full_test=False,
        weight_save_path=str(Path(dp["weights_dir"]) / "sam3_dec_5shot_seed42_source.pth"),
    )
    print(f"\n最终结果: {result}")
