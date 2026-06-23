"""
=============================================================================
eval_segformer.py  ---  SegFormer 纯评估脚本 (单次推理，防卡死)
=============================================================================
问题分析:
  - 原脚本 build_segformer() 在 evaluate_on_test_set 内部调用，每种子触发一次
    HuggingFace from_pretrained，刷出大量 MISSING/UNEXPECTED 日志，极其耗时
  - BATCH_SIZE=8 评估 8,402 张图 → 1,051 个 batch，I/O 开销大
  - num_workers=8 可能触发 SegFormer + DataLoader 多进程死锁

改进:
  - 模型只加载一次 (build_segformer 调用一次)
  - EVAL_BATCH_SIZE=32 (评估用大 batch)
  - num_workers=4 (更安全)
  - 禁用 transformers 日志 (避免 MISSING/UNEXPECTED 刷屏)
  - AMP bfloat16 加速推理

用法:
  python src/training/eval_segformer.py
  python src/training/eval_segformer.py --weight weights/segformer_best_seed123.pth
  python src/training/eval_segformer.py --full_test
  python src/training/eval_segformer.py --seeds 42,123,456
=============================================================================
"""

# ==========================================================================
# 模块 1: 导入与全局配置
# ==========================================================================
import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.cloud_paths import get_domain_paths, get_platform_name
from data.dataset import ValDataset
from evaluation.eval_metrics import evaluate_predictions

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

EVAL_BATCH_SIZE: int = 32
NUM_WORKERS: int = 2
TARGET_SIZE: int = 512


# ==========================================================================
# 模块 2: SegFormer 模型构建 (只调用一次)
# ==========================================================================
def build_segformer():
    from transformers import SegformerForSemanticSegmentation
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()

    original_endpoint = os.environ.get("HF_ENDPOINT", "")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    try:
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2", num_labels=1, ignore_mismatched_sizes=True
        )
    except Exception:
        if original_endpoint:
            os.environ["HF_ENDPOINT"] = original_endpoint
        else:
            del os.environ["HF_ENDPOINT"]
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2", num_labels=1, ignore_mismatched_sizes=True
        )
    return model


# ==========================================================================
# 模块 3: DataLoader collate
# ==========================================================================
def _collate_ignore_boxes(batch):
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "boundary": torch.stack([item["boundary"] for item in batch]),
        "name": [item["name"] for item in batch],
    }


# ==========================================================================
# 模块 4: 单权重评估
# ==========================================================================
@torch.no_grad()
def _eval_one_weight(
    model,
    test_loader: DataLoader,
    weight_path: str,
) -> Dict[str, float]:
    checkpoint = torch.load(weight_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint, strict=True)

    all_preds: List[np.ndarray] = []
    all_gts: List[np.ndarray] = []

    for batch in tqdm(test_loader, desc="  评估", unit="batch", ncols=100):
        images = batch["image"].to(DEVICE)
        masks = batch["mask"]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(pixel_values=images)
            logits = F.interpolate(
                outputs.logits, size=masks.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        preds = torch.sigmoid(logits.float()).cpu().numpy()
        B = preds.shape[0]
        for i in range(B):
            all_preds.append(preds[i, 0])
            all_gts.append(masks[i, 0].numpy())

    preds_arr = np.stack(all_preds, axis=0)
    gts_arr = np.stack(all_gts, axis=0)

    metrics = evaluate_predictions(preds_arr, gts_arr)
    return {"mIoU": metrics["mIoU"], "F1": metrics["F1"],
            "Boundary_IoU": metrics["Boundary_IoU"]}


# ==========================================================================
# 模块 5: 多种子评估主函数
# ==========================================================================
def run_evaluation(
    weight_paths: List[str],
    full_test: bool,
) -> Dict:
    test_label = "全量 8,402" if full_test else "瘦身测试"

    print(f"\n{'='*70}")
    print(f"  SegFormer 纯评估")
    print(f"  平台: {get_platform_name()}  |  设备: {DEVICE}")
    print(f"  测试集: {test_label}  |  Batch={EVAL_BATCH_SIZE}  Workers={NUM_WORKERS}")
    print(f"  待评估权重: {len(weight_paths)} 个")
    print(f"{'='*70}")

    print("\n[1/3] 构建 SegFormer 模型 (只加载一次) ...")
    model = build_segformer().to(DEVICE)
    model.eval()
    print("  完成")

    dp = get_domain_paths("source", full_test=full_test)

    print("\n[2/3] 加载测试集 ...")
    test_dataset = ValDataset(
        image_dir=dp["test_image_dir"],
        dual_label_dir=dp["test_dual_dir"],
        target_size=TARGET_SIZE,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        collate_fn=_collate_ignore_boxes, num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    print(f"  测试集: {len(test_dataset)} 张  |  "
          f"{len(test_loader)} 个 batch (BS={EVAL_BATCH_SIZE})")

    results: List[Dict] = []
    print(f"\n[3/3] 开始评估...")

    for idx, wpath in enumerate(weight_paths):
        wname = Path(wpath).name
        print(f"\n  [{idx+1}/{len(weight_paths)}] {wname}")

        try:
            metrics = _eval_one_weight(model, test_loader, wpath)
            metrics["weight"] = wname
            metrics["seed"] = _extract_seed(wname)
            results.append(metrics)

            print(f"    mIoU         = {metrics['mIoU']*100:.2f}%")
            print(f"    F1           = {metrics['F1']*100:.2f}%")
            print(f"    Boundary IoU = {metrics['Boundary_IoU']*100:.2f}%")

        except Exception as e:
            print(f"    [错误] {e}")
            import traceback
            traceback.print_exc()

    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print_summary(results)

    save_results(results, full_test)

    return {"results": results}


# ==========================================================================
# 模块 6: 辅助函数
# ==========================================================================
def _extract_seed(weight_name: str) -> int:
    name = Path(weight_name).stem
    for token in name.split("_"):
        if token.startswith("seed"):
            return int(token.replace("seed", ""))
    try:
        parts = name.split("seed")
        return int(parts[-1].replace(".pth", ""))
    except Exception:
        return 0


def _mean_std(values: List[float]):
    if not values:
        return 0.0, 0.0
    arr = np.array(values)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


# ==========================================================================
# 模块 7: 结果输出
# ==========================================================================
def print_summary(results: List[Dict]) -> None:
    N = len(results)

    print(f"\n{'='*80}")
    print(f"  SegFormer (mit-b2) 全监督基线评估结果 | N = {N} 个权重")
    print(f"{'='*80}")

    mious = [r["mIoU"] for r in results]
    f1s = [r["F1"] for r in results]
    bious = [r["Boundary_IoU"] for r in results]

    miou_m, miou_s = _mean_std(mious)
    f1_m, f1_s = _mean_std(f1s)
    biou_m, biou_s = _mean_std(bious)

    print(f"\n  ### 汇总 (Mean ± Std)")
    print(f"  mIoU (%)         = {miou_m*100:.2f} ± {miou_s*100:.2f}")
    print(f"  F1 (%)           = {f1_m*100:.2f} ± {f1_s*100:.2f}")
    print(f"  Boundary IoU (%) = {biou_m*100:.2f} ± {biou_s*100:.2f}")

    if N > 1:
        print(f"\n  ### 逐权重详细结果")
        print(f"  {'权重':<45}{'mIoU':>10}{'F1':>10}{'BIoU':>10}")
        print(f"  {'-'*45}{'-'*10}{'-'*10}{'-'*10}")
        for r in results:
            print(f"  {r['weight']:<45}"
                  f"{r['mIoU']*100:>10.2f}{r['F1']*100:>10.2f}"
                  f"{r['Boundary_IoU']*100:>10.2f}")

    print(f"{'='*80}")


# ==========================================================================
# 模块 8: 结果持久化
# ==========================================================================
def save_results(results: List[Dict], full_test: bool) -> None:
    dp = get_domain_paths("source", full_test=full_test)
    results_dir = dp.get("results_dir", str(Path(dp["project_root"]) / "results"))
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_label = "full" if full_test else "slim"
    filepath = str(Path(results_dir) / f"segformer_eval_{test_label}_{timestamp}.json")

    output = {
        "model": "SegFormer (mit-b2)",
        "eval_batch_size": EVAL_BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "full_test": full_test,
        "timestamp": timestamp,
        "results": results,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  结果已保存: {filepath}")


# ==========================================================================
# 模块 9: 命令行参数
# ==========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SegFormer 纯评估脚本 (单次模型加载, 防卡死)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/training/eval_segformer.py
  python src/training/eval_segformer.py --weight weights/segformer_best_seed123.pth
  python src/training/eval_segformer.py --seeds 42,123,456
  python src/training/eval_segformer.py --full_test
  python src/training/eval_segformer.py --seeds 123 --full_test --batch_size 64
        """,
    )
    parser.add_argument(
        "--weight", type=str, default=None,
        help="指定单个权重文件路径 (与 --seeds 互斥)",
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="随机种子列表, 逗号分隔 (默认: 42,123,456), "
             "自动拼接为 segformer_best_seed{seed}.pth",
    )
    parser.add_argument(
        "--full_test", action="store_true",
        help="使用全量 8,402 张测试集 (默认: 瘦身测试集)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=EVAL_BATCH_SIZE,
        help=f"评估 batch_size (默认: {EVAL_BATCH_SIZE})",
    )
    return parser.parse_args()


# ==========================================================================
# 模块 10: 主入口
# ==========================================================================
def main() -> None:
    args = parse_args()

    global EVAL_BATCH_SIZE
    EVAL_BATCH_SIZE = args.batch_size

    if not torch.cuda.is_available():
        print("\n  警告: 未检测到 CUDA GPU")

    dp = get_domain_paths("source", full_test=False)
    weights_dir = dp["weights_dir"]

    if args.weight:
        weight_paths = [args.weight]
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",")] if args.seeds else [42, 123, 456]
        weight_paths = [str(Path(weights_dir) / f"segformer_best_seed{seed}.pth")
                        for seed in seeds]

    existing = [w for w in weight_paths if Path(w).exists()]
    missing = [w for w in weight_paths if not Path(w).exists()]

    if missing:
        print(f"\n  以下 {len(missing)} 个权重文件不存在 (将跳过):")
        for w in missing:
            print(f"    - {w}")

    if not existing:
        print("\n  错误: 没有找到任何可评估的权重文件")
        sys.exit(1)

    run_evaluation(existing, args.full_test)

    print(f"\n{'='*70}")
    print(f"  评估完成!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
