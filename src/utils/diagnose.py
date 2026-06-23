"""
=============================================================================
diagnose.py  ---  GBG-SAM3 过拟合与可行性诊断脚本
=============================================================================
功能说明:
  1. 使用 2 张 demo 图像, 50 Epoch 快速过拟合测试
  2. 诊断一: 验证 alpha_gate 严格的零初始化
  3. 诊断二: 验证 Loss 在 50 Epoch 内平滑收敛至 0.05 以下
  4. 诊断三: 验证门控范数自 Epoch 10 起逐渐偏离 0 增长
  5. 诊断四: 在 Epoch 10/30/50 导出对比图 (原图/GT/预测/不确定性)

用法:
  python src/utils/diagnose.py

依赖:
  - sam3 权重: weights/sam3.pt
  - FewShotRSIDDataset, GBG_SAM3_Module (同目录)
  - torch, matplotlib, PIL, numpy, tqdm
=============================================================================
"""

# ==========================================================================
# 模块 1: 导入与全局路径配置
# ==========================================================================
import gc
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2
from tqdm import tqdm

_SYS_INSERTED = False
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
    _SYS_INSERTED = True

_SAM3_SRC = _SRC / "models" / "sam3"
if str(_SAM3_SRC) not in sys.path:
    sys.path.insert(0, str(_SAM3_SRC))

from data.dataset import FewShotRSIDDataset
from models.model import GBG_SAM3_Module
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

CHECKPOINT_PATH: str = r"/path/to/project\weights\sam3.pt"
NUM_SHOTS: int = 2
NUM_EPOCHS: int = 50
LEARNING_RATE: float = 1e-4
EXPORT_EPOCHS: Tuple[int, ...] = (10, 30, 50)
OUTPUT_DIR: Path = Path(r"/path/to/project\data\processed\diagnostic_outputs")
FEATURE_SIZE: Tuple[int, int] = (64, 64)
SAM3_RESOLUTION: int = 1008

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================================================
# 模块 2: 模拟掩膜解码器 (SimulatedMaskDecoder)
#     将 fused_features [B,256,64,64] 上采样为预测掩膜 [B,1,512,512]
# ==========================================================================
class SimulatedMaskDecoder(nn.Module):
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
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


# ==========================================================================
# 模块 3: SAM3 特征提取辅助函数
#     从冻结的 SAM3 Image Encoder 批量提取中间特征 [B,256,64,64]
#     使用 PIL 作为中间格式 (与 Sam3Processor 接口对齐), 预计算并缓存
# ==========================================================================
class SAM3FeatureExtractor:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        print("加载 SAM3 模型 (仅用于冻结特征提取) ...")
        self.device = device
        self.model = build_sam3_image_model(checkpoint_path=checkpoint_path)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(SAM3_RESOLUTION, SAM3_RESOLUTION)),
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
# 模块 4: 可视化导出函数
#     将原图/GT掩膜/预测掩膜/不确定性图合并为 2×2 对比图并保存
# ==========================================================================
def save_diagnostic_image(
    image_np: np.ndarray,
    gt_mask_np: np.ndarray,
    pred_mask_np: np.ndarray,
    unc_map_np: np.ndarray,
    save_path: Path,
    epoch: int,
    image_name: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor="white")
    axes = axes.flatten()

    axes[0].imshow(image_np)
    axes[0].set_title("Raw Image (RGB)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(gt_mask_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT Mask", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(pred_mask_np, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Predicted Mask", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(unc_map_np, cmap="inferno", vmin=0, vmax=1)
    axes[3].set_title("Uncertainty Map", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    fig.suptitle(
        f"GBG-SAM3 Diagnostic  |  Epoch {epoch}  |  {image_name}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ==========================================================================
# 模块 5: 诊断主函数 (run_diagnosis)
#     数据加载 → 零初始化验证 → 50 Epoch 过拟合 → 梯度验证 → 导出
# ==========================================================================
def run_diagnosis() -> None:
    print(f"{'='*60}")
    print(f"  GBG-SAM3 过拟合可行性诊断")
    print(f"  设备: {DEVICE}")
    print(f"  样本数: {NUM_SHOTS}  |  Epoch: {NUM_EPOCHS}  |  LR: {LEARNING_RATE}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Step 1: 加载数据
    # ------------------------------------------------------------------
    print("[Step 1] 加载 FewShotRSIDDataset ...")
    dataset = FewShotRSIDDataset(num_shots=NUM_SHOTS, seed=42)
    print(f"  数据集大小: {len(dataset)}\n")

    # ------------------------------------------------------------------
    # Step 2: 预计算 SAM3 冻结特征
    # ------------------------------------------------------------------
    print("[Step 2] 预计算 SAM3 冻结特征 ...")
    extractor = SAM3FeatureExtractor(CHECKPOINT_PATH, DEVICE)
    cached_samples: Dict[str, Dict[str, torch.Tensor]] = {}

    for idx in range(len(dataset)):
        sample = dataset[idx]
        name = sample["name"]
        img_i = sample["image"]
        sam_feat = extractor.extract(img_i).cpu()
        cached_samples[name] = {
            "image": img_i,
            "mask": sample["mask"],
            "boundary": sample["boundary"],
            "sam_features": sam_feat,
        }
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  缓存 {len(cached_samples)} 张影像的 SAM3 特征完成\n")

    # ------------------------------------------------------------------
    # Step 3: 构建可训练模块
    # ------------------------------------------------------------------
    print("[Step 3] 构建 GBG-SAM3 可训练模块 ...")
    gbg_module = GBG_SAM3_Module().to(DEVICE)
    decoder = SimulatedMaskDecoder().to(DEVICE)

    trainable_params = list(gbg_module.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None

    print(f"  GBG Module 参数: {sum(p.numel() for p in gbg_module.parameters()):,}")
    print(f"  Decoder 参数:    {sum(p.numel() for p in decoder.parameters()):,}")
    print(f"  优化器: AdamW (lr={LEARNING_RATE})\n")

    # ------------------------------------------------------------------
    # 诊断一: 零初始化验证
    # ------------------------------------------------------------------
    print("[诊断一] 零初始化验证 ...")
    gate_norm_zero = torch.norm(gbg_module.adapter.alpha_gate).item()
    assert gate_norm_zero == 0.0, (
        f"alpha_gate 未零初始化! norm = {gate_norm_zero:.10f}"
    )
    print(f"  ✅ alpha_gate 零初始化通过 (norm = {gate_norm_zero})\n")

    # ------------------------------------------------------------------
    # Step 4: 训练循环 (50 Epoch)
    # ------------------------------------------------------------------
    print(f"{'='*60}")
    print(f"  [诊断二/三] 开始 {NUM_EPOCHS} Epoch 过拟合训练")
    print(f"{'='*60}\n")

    sample_keys = list(cached_samples.keys())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gate_norms: Dict[int, float] = {}

    pbar = tqdm(range(1, NUM_EPOCHS + 1), desc="训练进度", unit="epoch", ncols=100)
    for epoch in pbar:
        gbg_module.train()
        decoder.train()
        epoch_loss = 0.0

        for key in sample_keys:
            sample = cached_samples[key]

            img_t = sample["image"].unsqueeze(0).to(DEVICE)
            gt_mask_t = sample["mask"].unsqueeze(0).to(DEVICE)
            gt_bound_t = sample["boundary"].unsqueeze(0).to(DEVICE)
            sam_feat_t = sample["sam_features"].to(DEVICE)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                fused_features, unc_map = gbg_module(img_t, sam_feat_t)
                pred_mask = decoder(fused_features)

            loss_bce_mask = F.binary_cross_entropy(pred_mask.float(), gt_mask_t)
            loss_bce_boundary = F.binary_cross_entropy(unc_map.float(), gt_bound_t)
            loss_unc = (unc_map.float() * torch.abs(pred_mask.float() - gt_mask_t).detach()).mean()
            loss = loss_bce_mask + loss_bce_boundary + loss_unc

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(sample_keys)
        gate_norm = torch.norm(gbg_module.adapter.alpha_gate).item()
        gate_norms[epoch] = gate_norm

        pbar.set_postfix_str(
            f"loss={epoch_loss:.5f}  |α|={gate_norm:.4f}"
        )

        # --------------------------------------------------------------
        # 诊断四: 在指定 Epoch 导出对比图
        # --------------------------------------------------------------
        if epoch in EXPORT_EPOCHS:
            gbg_module.eval()
            decoder.eval()
            with torch.no_grad():
                for ki, key in enumerate(sample_keys):
                    sample = cached_samples[key]
                    img_t = sample["image"].unsqueeze(0).to(DEVICE)
                    gt_mask_t = sample["mask"].unsqueeze(0).to(DEVICE)
                    sam_feat_t = sample["sam_features"].to(DEVICE)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        fused_features, unc_map = gbg_module(img_t, sam_feat_t)
                        pred_mask = decoder(fused_features)

                    img_np = sample["image"].permute(1, 2, 0).numpy()
                    gt_np = gt_mask_t.squeeze().cpu().numpy()
                    pred_np = pred_mask.squeeze().float().cpu().numpy()
                    unc_np = unc_map.squeeze().float().cpu().numpy()

                    save_path = OUTPUT_DIR / f"diagnostic_epoch_{epoch:03d}_{key}.png"
                    save_diagnostic_image(
                        img_np, gt_np, pred_np, unc_np,
                        save_path, epoch, key,
                    )

    # ------------------------------------------------------------------
    # 诊断二: 最终 Loss 验证
    # ------------------------------------------------------------------
    final_loss = list(gate_norms.keys())
    print(f"\n{'='*60}")
    print(f"  [诊断总结]")
    print(f"{'='*60}")

    last_keys = sorted(gate_norms.keys())[-1] if gate_norms else 0
    if epoch_loss < 0.05:
        print(f"  ✅ 诊断二 (收敛): 最终 Loss = {epoch_loss:.5f} < 0.05")
    else:
        print(f"  ⚠️  诊断二 (收敛): 最终 Loss = {epoch_loss:.5f}, 建议增加 Epoch 或调 LR")

    # ------------------------------------------------------------------
    # 诊断三: 门控更新验证
    # ------------------------------------------------------------------
    norm_early = gate_norms.get(1, 0)
    norm_10 = gate_norms.get(10, norm_early)
    norm_last = gate_norms.get(NUM_EPOCHS, norm_early)
    if norm_last > norm_10 * 1.5:
        print(f"  ✅ 诊断三 (门控更新): |α| 从 {norm_10:.4f} → {norm_last:.4f} (增长 > 1.5×)")
    else:
        print(f"  ⚠️  诊断三 (门控更新): |α| {norm_10:.4f} → {norm_last:.4f}")

    print(f"\n  门控范数变化曲线 (每 5 Epoch):")
    for ep in sorted(gate_norms.keys()):
        if ep % 5 == 1 or ep in (10, 30, 50):
            print(f"    Epoch {ep:3d}: |α| = {gate_norms[ep]:.6f}")

    print(f"\n  诊断对比图已保存至: {OUTPUT_DIR}")
    print(f"  文件: diagnostic_epoch_010_*.png / _030_*.png / _050_*.png")
    print(f"{'='*60}")


# ==========================================================================
# 模块 6: 主入口
# ==========================================================================
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("警告: 未检测到 CUDA GPU, 将使用 CPU (速度较慢)")

    run_diagnosis()
