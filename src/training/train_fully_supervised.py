"""
=============================================================================
train_fully_supervised.py  ---  全监督基线训练 v3.0 (UNet + SegFormer)
=============================================================================
v3.0 重构:
  1. 多随机种子循环 (--seeds 42,123,456) → 自动计算 Mean ± Std
  2. 路径自适应: cloud_paths.get_domain_paths("source") 消除硬编码
  3. 双轨测试: --full_test 切换瘦身/全量 8,402 测试集
  4. 学术全指标: evaluate_predictions → mIoU + F1 + Boundary IoU (d=5)
  5. 性能升级: BS=8, num_workers=8, AMP bfloat16

输出:
  weights/unet_best_seed{seed}.pth
  weights/segformer_best_seed{seed}.pth

用法:
  python src/training/train_fully_supervised.py
  python src/training/train_fully_supervised.py --full_test
  python src/training/train_fully_supervised.py --seeds 42,123,456
  python src/training/train_fully_supervised.py --epochs 50
=============================================================================
"""

# ==========================================================================
# 模块 1: 导入与全局配置
# ==========================================================================
import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.cloud_paths import get_domain_paths, get_platform_name
from data.dataset import FewShotRSIDDataset, ValDataset
from evaluation.eval_metrics import evaluate_predictions

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE: int = 8
NUM_EPOCHS: int = 30
LEARNING_RATE: float = 1e-4
NUM_WORKERS: int = 8
TARGET_SIZE: int = 512


# ==========================================================================
# 模块 2: Dice 损失函数
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
# 模块 3: 模型构建函数
# ==========================================================================
def build_unet() -> nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "segmentation-models-pytorch"])
        import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    print("  UNet (ResNet34) 已构建")
    return model


def build_segformer():
    from transformers import SegformerForSemanticSegmentation
    import os
    # 尝试使用镜像
    original_endpoint = os.environ.get("HF_ENDPOINT", "")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    try:
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2", num_labels=1, ignore_mismatched_sizes=True
        )
    except Exception as e:
        print(f"镜像下载失败: {e}, 尝试官方源...")
        if original_endpoint:
            os.environ["HF_ENDPOINT"] = original_endpoint
        else:
            del os.environ["HF_ENDPOINT"]
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2", num_labels=1, ignore_mismatched_sizes=True
        )
    return model


# ==========================================================================
# 模块 4: DataLoader 辅助 (collate, 忽略 boxes 字段)
# ==========================================================================
def _collate_ignore_boxes(batch):
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "boundary": torch.stack([item["boundary"] for item in batch]),
        "name": [item["name"] for item in batch],
    }


# ==========================================================================
# 模块 5: 单模型训练 + 验证循环 (Per-Seed)
# ==========================================================================
def train_one_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    weight_save_path: str,
) -> Dict[str, float]:
    print(f"\n{'='*55}")
    print(f"  训练 {model_name.upper()}")
    print(f"  TrainBatch={BATCH_SIZE}  LR={LEARNING_RATE}  "
          f"Epoch={NUM_EPOCHS}  Workers={NUM_WORKERS}")
    print(f"{'='*55}")

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None
    bce_loss_fn = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss()

    best_val_iou = 0.0

    pbar = tqdm(range(1, NUM_EPOCHS + 1), desc=f"  [{model_name}]",
                unit="epoch", ncols=100)

    for epoch in pbar:
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if model_name == "segformer":
                    outputs = model(pixel_values=images)
                    logits = F.interpolate(
                        outputs.logits, size=masks.shape[-2:],
                        mode="bilinear", align_corners=False,
                    )
                else:
                    logits = model(images)

                loss = bce_loss_fn(logits.float(), masks) + dice_loss_fn(logits, masks)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        model.eval()
        val_iou_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    if model_name == "segformer":
                        outputs = model(pixel_values=images)
                        logits = F.interpolate(
                            outputs.logits, size=masks.shape[-2:],
                            mode="bilinear", align_corners=False,
                        )
                    else:
                        logits = model(images)

                probs = torch.sigmoid(logits.float())
                pred_bin = (probs > 0.5).float()
                batch_size = pred_bin.shape[0]
                pred_flat = pred_bin.view(batch_size, -1)
                mask_flat = masks.view(batch_size, -1)
                inter = (pred_flat * mask_flat).sum(dim=1)
                union = pred_flat.sum(dim=1) + mask_flat.sum(dim=1) - inter
                val_iou_sum += ((inter + 1e-7) / (union + 1e-7)).mean().item()

        val_iou = val_iou_sum / len(val_loader)

        postfix = f"loss={epoch_loss:.4f}  val_iou={val_iou:.4f}  best={best_val_iou:.4f}"
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            Path(weight_save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), weight_save_path)
            postfix = f"loss={epoch_loss:.4f}  val_iou={val_iou:.4f}  ★ best={best_val_iou:.4f}"

        pbar.set_postfix_str(postfix)

    print(f"  训练完成: best Val IoU = {best_val_iou:.4f}")
    print(f"  权重已保存: {weight_save_path}")
    return {"best_val_iou": best_val_iou}


# ==========================================================================
# 模块 6: 测试集全量评估 (学术全指标)
# ==========================================================================
@torch.no_grad()
def evaluate_on_test_set(
    model_builder: callable,
    model_name: str,
    weight_path: str,
    test_loader: DataLoader,
) -> Dict[str, float]:
    print(f"\n  [测试评估] {model_name.upper()}")
    print(f"    权重: {Path(weight_path).name}")

    model = model_builder().to(DEVICE)
    model.eval()

    state_dict = torch.load(weight_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)

    all_preds: List[np.ndarray] = []
    all_gts: List[np.ndarray] = []

    for batch in tqdm(test_loader, desc=f"  测试 {model_name}",
                      unit="batch", ncols=100):
        images = batch["image"].to(DEVICE)
        masks = batch["mask"]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if model_name == "segformer":
                outputs = model(pixel_values=images)
                logits = F.interpolate(
                    outputs.logits, size=masks.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
            else:
                logits = model(images)

        preds = torch.sigmoid(logits.float()).cpu().numpy()
        B = preds.shape[0]
        for i in range(B):
            all_preds.append(preds[i, 0])
            all_gts.append(masks[i, 0].numpy())

    preds_arr = np.stack(all_preds, axis=0)
    gts_arr = np.stack(all_gts, axis=0)

    metrics = evaluate_predictions(preds_arr, gts_arr)

    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return {"mIoU": metrics["mIoU"], "F1": metrics["F1"],
            "Boundary_IoU": metrics["Boundary_IoU"]}


# ==========================================================================
# 模块 7: 多随机种子实验主函数
# ==========================================================================
def run_fully_supervised_experiment(
    seeds: List[int],
    full_test: bool,
    epochs: int,
) -> Dict:
    global NUM_EPOCHS
    NUM_EPOCHS = epochs

    test_label = "全量 8,402" if full_test else "瘦身测试"
    print(f"\n{'='*70}")
    print(f"  全监督基线训练 v3.0 (UNet + SegFormer)")
    print(f"  平台: {get_platform_name()}  |  设备: {DEVICE}")
    print(f"  测试集: {test_label}  |  种子: {seeds}")
    print(f"  Batch={BATCH_SIZE}  Epochs={NUM_EPOCHS}  "
          f"LR={LEARNING_RATE}  Workers={NUM_WORKERS}")
    print(f"{'='*70}")

    dp = get_domain_paths("source", full_test=full_test)
    dp_train = get_domain_paths("source", full_test=False)

    weights_dir = dp["weights_dir"]

    print("\n[数据加载]")
    train_dataset = FewShotRSIDDataset(
        image_dir=dp_train["train_image_dir"],
        dual_label_dir=dp_train["train_label_dir"],
        bbox_json_path=dp_train["train_bbox_json"],
        num_shots=None,
        seed=42,
    )
    val_dataset = ValDataset(
        image_dir=dp_train["val_image_dir"],
        dual_label_dir=dp_train["val_label_dir"],
        target_size=TARGET_SIZE,
    )
    test_dataset = ValDataset(
        image_dir=dp["test_image_dir"],
        dual_label_dir=dp["test_dual_dir"],
        target_size=TARGET_SIZE,
    )

    print(f"  训练集: {len(train_dataset)}  验证集: {len(val_dataset)}  "
          f"测试集: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=_collate_ignore_boxes, num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=_collate_ignore_boxes, num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=_collate_ignore_boxes, num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    seed_metrics: Dict[str, List[Dict[str, float]]] = {
        "unet": [],
        "segformer": [],
    }

    for idx, seed in enumerate(seeds):
        print(f"\n{'#'*60}")
        print(f"  [{idx+1}/{len(seeds)}] Seed = {seed}")
        print(f"{'#'*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # UNet
        try:
            unet = build_unet()
            unet_result = train_one_model(
                model=unet,
                model_name="unet",
                train_loader=train_loader,
                val_loader=val_loader,
                weight_save_path=str(
                    Path(weights_dir) / f"unet_best_seed{seed}.pth"
                ),
            )
            del unet
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            test_metrics_unet = evaluate_on_test_set(
                model_builder=build_unet,
                model_name="unet",
                weight_path=str(Path(weights_dir) / f"unet_best_seed{seed}.pth"),
                test_loader=test_loader,
            )
            test_metrics_unet["best_val_iou"] = unet_result["best_val_iou"]
            seed_metrics["unet"].append(test_metrics_unet)
            print(f"  UNet seed={seed}: mIoU={test_metrics_unet['mIoU']:.4f}  "
                  f"F1={test_metrics_unet['F1']:.4f}  "
                  f"BIoU={test_metrics_unet['Boundary_IoU']:.4f}")

        except Exception as e:
            print(f"  [错误] UNet seed={seed} 失败: {e}")
            import traceback
            traceback.print_exc()

        # SegFormer
        try:
            segformer = build_segformer()
            seg_result = train_one_model(
                model=segformer,
                model_name="segformer",
                train_loader=train_loader,
                val_loader=val_loader,
                weight_save_path=str(
                    Path(weights_dir) / f"segformer_best_seed{seed}.pth"
                ),
            )
            del segformer
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            test_metrics_seg = evaluate_on_test_set(
                model_builder=build_segformer,
                model_name="segformer",
                weight_path=str(
                    Path(weights_dir) / f"segformer_best_seed{seed}.pth"
                ),
                test_loader=test_loader,
            )
            test_metrics_seg["best_val_iou"] = seg_result["best_val_iou"]
            seed_metrics["segformer"].append(test_metrics_seg)
            print(f"  SegFormer seed={seed}: mIoU={test_metrics_seg['mIoU']:.4f}  "
                  f"F1={test_metrics_seg['F1']:.4f}  "
                  f"BIoU={test_metrics_seg['Boundary_IoU']:.4f}")

        except Exception as e:
            print(f"  [错误] SegFormer seed={seed} 失败: {e}")
            import traceback
            traceback.print_exc()

    return {"seed_metrics": seed_metrics, "seeds": seeds,
            "full_test": full_test, "num_epochs": NUM_EPOCHS}


# ==========================================================================
# 模块 8: 统计聚合与 Markdown 表格输出
# ==========================================================================
def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.array(values)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def print_summary_table(
    seed_metrics: Dict[str, List[Dict[str, float]]],
    seeds: List[int],
    full_test: bool,
) -> Dict:
    test_label = "全量 8,402" if full_test else "瘦身测试"
    N = len(seeds)

    print(f"\n{'='*85}")
    print(f"  全监督基线 学术指标汇总 | 测试集: {test_label} | 种子: {seeds}")
    print(f"{'='*85}")

    print(f"\n  | {'Model':<14} | {'mIoU (%)':>20} | "
          f"{'F1 (%)':>18} | {'Boundary IoU (%)':>22} |")
    print(f"  |{'-'*16}|{'-'*22}|{'-'*20}|{'-'*24}|")

    summary: Dict[str, Dict] = {}

    for model_name in ["unet", "segformer"]:
        entries = seed_metrics.get(model_name, [])
        if not entries:
            continue

        mious = [e["mIoU"] for e in entries]
        f1s = [e["F1"] for e in entries]
        bious = [e["Boundary_IoU"] for e in entries]

        miou_mean, miou_std = _mean_std(mious)
        f1_mean, f1_std = _mean_std(f1s)
        biou_mean, biou_std = _mean_std(bious)

        display_name = "UNet (ResNet34)" if model_name == "unet" else "SegFormer (mit-b2)"

        miou_str = f"{miou_mean*100:.2f} ± {miou_std*100:.2f}"
        f1_str = f"{f1_mean*100:.2f} ± {f1_std*100:.2f}"
        biou_str = f"{biou_mean*100:.2f} ± {biou_std*100:.2f}"

        print(f"  | {display_name:<14} | {miou_str:>20} | "
              f"{f1_str:>18} | {biou_str:>22} |")

        summary[model_name] = {
            "display_name": display_name,
            "mIoU_mean": miou_mean, "mIoU_std": miou_std,
            "F1_mean": f1_mean, "F1_std": f1_std,
            "Boundary_IoU_mean": biou_mean, "Boundary_IoU_std": biou_std,
            "per_seed": entries,
        }

    print(f"\n  N = {N} 随机种子  |  Mean ± Std")

    print(f"\n{'='*85}")
    print(f"  逐种子详细结果")
    print(f"{'='*85}")

    for model_name in ["unet", "segformer"]:
        entries = seed_metrics.get(model_name, [])
        if not entries:
            continue
        display_name = "UNet (ResNet34)" if model_name == "unet" else "SegFormer (mit-b2)"
        print(f"\n  {display_name}:")
        print(f"  {'Seed':<8}{'mIoU':>10}{'F1':>10}{'Boundary IoU':>16}{'Val IoU':>12}")
        print(f"  {'─'*8}{'─'*10}{'─'*10}{'─'*16}{'─'*12}")
        for e, s in zip(entries, seeds):
            print(f"  {s:<8}{e['mIoU']*100:>10.2f}{e['F1']*100:>10.2f}"
                  f"{e['Boundary_IoU']*100:>16.2f}{e['best_val_iou']*100:>12.2f}")

    return summary


# ==========================================================================
# 模块 9: 命令行参数解析
# ==========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="全监督基线训练 v3.0 (UNet + SegFormer) --- 多种子 + 全指标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/training/train_fully_supervised.py
  python src/training/train_fully_supervised.py --full_test
  python src/training/train_fully_supervised.py --seeds 42
  python src/training/train_fully_supervised.py --seeds 42,123,456 --epochs 50 --full_test
        """,
    )
    parser.add_argument(
        "--seeds", type=str, default="42,123,456",
        help="随机种子列表, 逗号分隔 (默认: 42,123,456)",
    )
    parser.add_argument(
        "--full_test", action="store_true",
        help="使用全量 8,402 张测试集 (默认: 瘦身测试集)",
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="训练轮数 (默认: 30)",
    )
    return parser.parse_args()


# ==========================================================================
# 模块 10: 主入口
# ==========================================================================
def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if not torch.cuda.is_available():
        print("\n  警告: 未检测到 CUDA GPU, 将使用 CPU (极慢)")

    result = run_fully_supervised_experiment(
        seeds=seeds,
        full_test=args.full_test,
        epochs=args.epochs,
    )

    summary = print_summary_table(
        result["seed_metrics"],
        result["seeds"],
        result["full_test"],
    )

    print(f"\n{'='*70}")
    print(f"  全监督基线训练完成!")
    print(f"  权重目录: {get_domain_paths('source')['weights_dir']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
