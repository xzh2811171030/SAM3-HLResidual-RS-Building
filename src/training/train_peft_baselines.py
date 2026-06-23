"""
=============================================================================
train_peft_baselines.py  ---  SAM3 高效微调基线 (Decoder-Only + LoRA)
=============================================================================
功能说明:
  1. 使用 FewShotRSIDDataset 加载全部 demo 数据 (num_shots=None)
  2. 训练两个 SAM3 微调基线作为 GBG-SAM3 的参照系:
     - 基线 A (Decoder-Only):  冻结 SAM3 Image Encoder, 仅训练轻量掩膜解码器
     - 基线 B (LoRA PEFT):     在 ViT attention qkv 层注入 LoRA, 联合训练解码器
  3. BCEWithLogitsLoss + DiceLoss 混合损失, 混合精度训练
  4. 每 Epoch 记录 Loss 与 IoU, 按验证 IoU 保存最优权重

输出:
  weights/sam3_decoder_only.pth    (基线 A: 解码器权重)
  weights/sam3_lora.pth            (基线 B: LoRA adapter + 解码器权重)

用法:
  python src/training/train_peft_baselines.py

设计说明:
  - SAM3 ViT 使用组合投影 self.qkv (而非独立的 q_proj/v_proj),
    LoRA 注入目标层设为 "qkv", 等效覆盖 Q/K/V 三个投影通道.
  - 三个基线 (Decoder-Only / LoRA / GBG-SAM3) 使用相同解码器架构,
    唯一变量是编码器端的适配策略, 构成严格的科学对照实验.
=============================================================================
"""

# ==========================================================================
# 模块 1: 导入与全局配置
# ==========================================================================
import gc
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_SAM3_SRC = _SRC / "models" / "sam3"
if str(_SAM3_SRC) not in sys.path:
    sys.path.insert(0, str(_SAM3_SRC))

from data.dataset import FewShotRSIDDataset
from sam3.model_builder import build_sam3_image_model

RAW_IMAGE_DIR: str = r"/path/to/project\data\raw\train_demo\image"
DUAL_LABEL_DIR: str = r"/path/to/project\data\processed\dual_channel_demo"
BBOX_JSON: str = r"/path/to/project\data\processed\bbox_demo.json"
CHECKPOINT_PATH: str = r"/path/to/project\weights\sam3.pt"

DECODER_ONLY_WEIGHT: str = r"/path/to/project\weights\sam3_decoder_only.pth"
LORA_WEIGHT: str = r"/path/to/project\weights\sam3_lora.pth"

BATCH_SIZE: int = 2
NUM_EPOCHS: int = 30
LEARNING_RATE: float = 1e-4
VAL_SPLIT: float = 0.2
FEATURE_SIZE: Tuple[int, int] = (64, 64)
SAM3_INPUT_SIZE: int = 1008

LORA_RANK: int = 8
LORA_ALPHA: int = 16
LORA_DROPOUT: float = 0.05
LORA_TARGET_MODULES: Tuple[str, ...] = ("qkv",)

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================================================
# 模块 2: 轻量掩膜解码器 (LightweightMaskDecoder)
#     与 GBG-SAM3 实验中的 SimulatedMaskDecoder 完全一致
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
# 模块 3: SAM3 冻结特征提取器
#     提供两条路径:
#       extract()              → @torch.no_grad() + .detach(), 用于预缓存/验证/推理
#       extract_for_training() → 保留计算图 + 激活 checkpointing (节省显存)
#
#     预处理: uint8 → Resize(1008) → Normalize(0.5, 0.5) → backbone
#     autocast(bf16) 包裹 backbone forward, .float() 转回 float32
#
#     activation checkpointing 补丁 (extract_for_training 内):
#       - ViT forward 中 checkpoint 条件: if self.use_act_checkpoint and self.training
#       - 但 model.eval() 导致 ViT 的 self.training=False → 32 层激活全部驻留显存
#       - 修复: object.__setattr__(base_vit, 'training', True)
#         仅设置 ViT 自身的 training 属性, 不递归传播到子模块 (Block)
#       - 结果: ViT 层 activate checkpointing → O(1) 显存
#               Block 保持 eval 模式 → Dropout/DropPath 禁用 → 确定性 → checkpoint 一致
#       - 冻结参数不消耗梯度显存, 仅 LoRA adapter 参与计算图
# ==========================================================================
class SAM3FeatureExtractor:
    def __init__(self, checkpoint_path: str, device: str = DEVICE):
        print("  加载 SAM3 模型 ...")
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
        features = self._forward_backbone(img_processed)
        return features.detach()

    def extract_for_training(self, image_tensor: torch.Tensor) -> torch.Tensor:
        img_processed = self._preprocess(image_tensor)

        peft_vit = self.model.backbone.vision_backbone.trunk
        base_vit = peft_vit.get_base_model()
        saved_training = base_vit.training

        try:
            if not saved_training:
                object.__setattr__(base_vit, 'training', True)

            features = self._forward_backbone(img_processed)
        finally:
            if not saved_training:
                object.__setattr__(base_vit, 'training', False)

        return features


# ==========================================================================
# 模块 4: Dice 损失函数
#     Dice = 1 - (2*|P∩G| + smooth) / (|P|+|G| + smooth)
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
# 模块 5: IoU 评估函数
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
# 模块 6: 预计算 + 缓存冻结特征
#     对全量数据集一次性提取 SAM3 backbone 特征, 避免每 Epoch 重复编码
# ==========================================================================
def precache_features(
    dataset: FewShotRSIDDataset,
    extractor: SAM3FeatureExtractor,
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
# 模块 6.5: LoRA 梯度流验证辅助函数
#     在首次 backward() 后检查 ViT 内 LoRA 参数的 .grad 是否非空
#     若全部为 None → 计算图断裂, 报 RuntimeError 阻止静默失败
# ==========================================================================
def _verify_lora_gradient_flow(model) -> bool:
    lora_grad_none: list[str] = []
    lora_grad_ok: list[str] = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            lora_grad_none.append(n)
        else:
            lora_grad_ok.append(n)

    n_ok = len(lora_grad_ok)
    n_none = len(lora_grad_none)
    n_total = n_ok + n_none

    if n_total == 0:
        print("  [梯度验证] 警告: 未找到任何 requires_grad=True 的 LoRA 参数")
        return False

    if n_none > 0:
        print(f"  [梯度验证] 错误: 以下 {n_none}/{n_total} 个 LoRA 参数的 .grad=None (计算图断裂):")
        for name in lora_grad_none[:10]:
            print(f"    - {name}")
        if n_none > 10:
            print(f"    ... 及其他 {n_none - 10} 个")
        raise RuntimeError(
            f"LoRA 梯度断裂: {n_none}/{n_total} 个可训练参数的 .grad=None。"
            f"请检查 extract_for_training() 是否正确保留了计算图。"
        )

    print(f"  [梯度验证] ✓ LoRA 梯度流正常: {n_ok}/{n_total} 个可训练参数均有有效梯度")
    return True


# ==========================================================================
# 模块 7: 通用训练 + 验证循环
#     支持 Decoder-Only 和 LoRA 两种模式
#     模式 "decoder_only": 仅优化 decoder 参数
#     模式 "lora":         冻结所有原始参数, 仅优化 LoRA + decoder 参数
# ==========================================================================
def train_baseline(
    decoder: nn.Module,
    train_keys: list,
    val_keys: list,
    cached: Dict[str, Dict[str, torch.Tensor]],
    weight_save_path: str,
    model_name: str,
    mode: str,
    feat_extractor: Optional[SAM3FeatureExtractor] = None,
) -> Dict:
    print(f"\n{'='*55}")
    print(f"  训练: {model_name}")
    print(f"  模式: {mode}")
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

    _lora_grad_verified = False

    pbar = tqdm(range(1, NUM_EPOCHS + 1), desc=f"  [{model_name}]", unit="epoch", ncols=100)

    for epoch in pbar:
        # ----------------------------------------------------------------
        # 训练阶段
        # ----------------------------------------------------------------
        decoder.train()
        epoch_loss = 0.0

        for key in train_keys:
            sample = cached[key]
            img_t = sample["image"].unsqueeze(0).to(DEVICE)
            gt_mask_t = sample["mask"].unsqueeze(0).to(DEVICE)
            sam_feat_cpu = sample["sam_features"]

            optimizer.zero_grad(set_to_none=True)

            if mode == "lora":
                sam_feat_t = feat_extractor.extract_for_training(img_t.squeeze(0))
            else:
                sam_feat_t = sam_feat_cpu.to(DEVICE)

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

            if mode == "lora" and not _lora_grad_verified:
                _lora_grad_verified = _verify_lora_gradient_flow(feat_extractor.model)

            epoch_loss += loss.item()

        epoch_loss /= len(train_keys)

        # ----------------------------------------------------------------
        # 验证阶段
        # ----------------------------------------------------------------
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

        val_iou = val_iou_sum / len(val_keys)

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
# 模块 8: LoRA 注入 + ViT 梯度兼容补丁
#
#     SAM3 结构:
#       model.backbone (SAM3VLBackbone)
#         └── .vision_backbone (Sam3DualViTDetNeck)
#               └── .trunk (ViT)    ← LoRA 注入目标
#
#     梯度兼容补丁 (_patch_vit_mlp_for_grad_compat):
#       - 问题 1: ViT Mlp.forward() 调用 sam3 的 addmm_act 融合算子,
#         该算子在 torch.is_grad_enabled()==True 时直接报 RuntimeError
#       - 问题 2: 融合算子内部调用 .detach() 导致梯度断裂
#       - 修复: 将每个 ViT Block 内 Mlp 的 forward 替换为标准 PyTorch
#         fc1 + activation 路径, 跳过 addmm_act
#       - 注意: 不能用 torch.is_grad_enabled() 分支 (融合 vs 标准),
#         因为 torch.utils.checkpoint 在 forward 阶段处于 no_grad 上下文,
#         但 checkpoint 重计算时 grad 启用 → 两次调用路径不同 → 输出张量数
#         不一致 → CheckpointError. 所以统一使用标准算子.
#
#     激活检查点补丁 (由 extract_for_training 临时触发):
#       - ViT forward 中 activation checkpointing 的条件是:
#           if self.use_act_checkpoint and self.training:
#       - 但 model.eval() 导致 self.training=False → 32 层激活全部驻留显存
#       - 修复: extract_for_training() 临时设置 base_vit.training=True,
#         激活 checkpointing (Dropout 照常工作, checkpoint 保证 RNG 一致性)
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
              f"(统一使用 fc1+act, 绕过 addmm_act 融合算子)")
    else:
        print("  [警告] MLP 梯度兼容补丁: 未找到任何 Mlp 模块, 请检查 ViT 结构")


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

    trainable_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_count = sum(p.numel() for p in model.parameters())
    print(f"  LoRA 注入完成: 可训练 {trainable_count:,} / {total_count:,} "
          f"({100 * trainable_count / total_count:.2f}%)")


# ==========================================================================
# 模块 9: 主函数
#     数据加载 → 特征预缓存 → 基线 A (Decoder-Only) → 基线 B (LoRA) → 汇总
# ==========================================================================
def main() -> None:
    print(f"\n{'='*60}")
    print(f"  SAM3 高效微调基线实验")
    print(f"  设备: {DEVICE}")
    print(f"  Batch: {BATCH_SIZE}  |  Epoch: {NUM_EPOCHS}  |  LR: {LEARNING_RATE}")
    print(f"  LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Step 1: 加载数据, 划分训练/验证集
    # ------------------------------------------------------------------
    print("\n[数据加载]")
    full_dataset = FewShotRSIDDataset(
        image_dir=RAW_IMAGE_DIR,
        dual_label_dir=DUAL_LABEL_DIR,
        bbox_json_path=BBOX_JSON,
        num_shots=None,
    )
    n_val = max(1, int(len(full_dataset) * VAL_SPLIT))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  训练集: {len(train_ds)}  验证集: {len(val_ds)}")

    # ------------------------------------------------------------------
    # Step 2: 预计算 frozen 特征缓存
    # ------------------------------------------------------------------
    print("\n[特征预缓存]")
    extractor = SAM3FeatureExtractor(CHECKPOINT_PATH, DEVICE)
    cached = precache_features(full_dataset, extractor)
    print(f"  缓存 {len(cached)} 张影像的 SAM3 特征完成")

    train_keys = [full_dataset[i]["name"] for i in train_ds.indices]
    val_keys = [full_dataset[i]["name"] for i in val_ds.indices]

    results: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # 基线 A: Decoder-Only
    # ------------------------------------------------------------------
    print("\n[基线 A] Decoder-Only: 冻结 Encoder, 仅训练 Decoder")
    decoder_a = LightweightMaskDecoder()
    results["decoder_only"] = train_baseline(
        decoder=decoder_a,
        train_keys=train_keys,
        val_keys=val_keys,
        cached=cached,
        weight_save_path=DECODER_ONLY_WEIGHT,
        model_name="Decoder-Only",
        mode="decoder_only",
        feat_extractor=None,
    )
    del decoder_a
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 基线 B: LoRA
    # ------------------------------------------------------------------
    print("\n[基线 B] LoRA PEFT: 冻结 Encoder + 注入 LoRA → 训练 LoRA + Decoder")
    inject_lora_to_vit(extractor.model, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)

    decoder_b = LightweightMaskDecoder()
    results["lora"] = train_baseline(
        decoder=decoder_b,
        train_keys=train_keys,
        val_keys=val_keys,
        cached=cached,
        weight_save_path=LORA_WEIGHT,
        model_name="LoRA",
        mode="lora",
        feat_extractor=extractor,
    )
    del decoder_b
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  训练汇总")
    print(f"{'='*60}")
    for name, res in results.items():
        print(f"  {name:>15s}:  Best Val IoU = {res['best_val_iou']:.4f}")
    print(f"{'='*60}")


# ==========================================================================
# 模块 10: 主入口
# ==========================================================================
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("警告: 未检测到 CUDA GPU, 将使用 CPU (极慢)")
    main()
