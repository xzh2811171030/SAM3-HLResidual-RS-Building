"""
=============================================================================
eval_baselines_full.py  ---  全量 8,402 测试集统一评估脚本
=============================================================================
功能说明:
  1. 自动加载 target 域的所有已训练基线权重 (Decoder-Only / LoRA, 5/10/20-shot, 3 种子)
  2. 在 whu_mix_full_test (8,402 张) 上逐一评估 mIoU / F1 / Boundary IoU
  3. 计算每个 (shot, mode) 组合的 Mean ± Std
  4. 输出 Remote Sensing 期刊规范的 Markdown 对比表格
  5. 结果自动保存至 results/baselines_full_evaluation.json

用法:
  python src/custom_tuning/eval_baselines_full.py
  python src/custom_tuning/eval_baselines_full.py --shots 5,10,20
  python src/custom_tuning/eval_baselines_full.py --modes lora --seeds 42
  python src/custom_tuning/eval_baselines_full.py --dry-run
=============================================================================
"""

# ==========================================================================
# 模块 1: 导入与全局配置
# ==========================================================================
import argparse
import gc
import json
import sys
from datetime import datetime
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

from utils.cloud_paths import get_domain_paths, get_paths, get_platform_name
from data.dataset import ValDataset
from evaluation.eval_metrics import evaluate_predictions
from sam3.model_builder import build_sam3_image_model

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

EVAL_BATCH_SIZE: int = 32
NUM_WORKERS: int = 2
TARGET_SIZE: int = 512
SAM3_INPUT_SIZE: int = 1008
FEATURE_SIZE: Tuple[int, int] = (64, 64)
FEATURE_CHANNELS: int = 256

LORA_RANK: int = 8
LORA_ALPHA: int = 16
LORA_DROPOUT: float = 0.05
LORA_TARGET_MODULES: Tuple[str, ...] = ("qkv",)

ALL_SHOTS: List[int] = [5, 10, 20]
ALL_MODES: List[str] = ["decoder_only", "lora"]
ALL_SEEDS: List[int] = [42, 123, 456]

_paths = get_paths()
WEIGHTS_DIR: str = _paths["weights_dir"]
RESULTS_DIR: str = _paths.get("results_dir", str(Path(_paths["project_root"]) / "results"))


# ==========================================================================
# 模块 2: 轻量掩膜解码器 (与训练时结构一致)
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
# 模块 3: MLP 梯度兼容补丁 (LoRA 模式下需要)
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
        print(f"    MLP 补丁: {n_patched} 个模块")


# ==========================================================================
# 模块 4: LoRA 注入
# ==========================================================================
def _inject_lora(vit_model) -> None:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "peft"])
        from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=list(LORA_TARGET_MODULES), bias="none",
    )
    trunk = vit_model.backbone.vision_backbone.trunk
    vit_model.backbone.vision_backbone.trunk = get_peft_model(trunk, lora_config)
    _patch_vit_mlp_for_grad_compat(vit_model.backbone.vision_backbone.trunk)

    # 打印 LoRA 注入后可训练参数量
    trainable_count = sum(
        p.numel() for p in vit_model.parameters() if p.requires_grad
    )
    total_count = sum(p.numel() for p in vit_model.parameters())
    print(f"    LoRA 注入完成: 可训练 {trainable_count:,} / {total_count:,} "
          f"({100 * trainable_count / total_count:.2f}%)")


# ==========================================================================
# 模块 5: SAM3 推理用特征提取器
# ==========================================================================
class Sam3InferenceExtractor:
    def __init__(self, checkpoint_path: str):
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
    def extract_batch(self, image_batch: torch.Tensor) -> torch.Tensor:
        B = image_batch.shape[0]
        features_list = []
        for i in range(B):
            img_uint8 = (image_batch[i] * 255.0).clamp(0, 255).to(torch.uint8)
            img_processed = self.transform(img_uint8).unsqueeze(0).to(DEVICE)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                backbone_out = self.model.backbone.forward_image(img_processed)
            feat = backbone_out["vision_features"].float()
            if feat.shape[-2:] != FEATURE_SIZE:
                feat = F.interpolate(
                    feat, size=FEATURE_SIZE, mode="bilinear", align_corners=False,
                )
            features_list.append(feat.detach())
        return torch.cat(features_list, dim=0)


# ==========================================================================
# 模块 6: 单个权重文件的全量评估
# ==========================================================================
@torch.no_grad()
def _evaluate_single_weight(
    extractor: Sam3InferenceExtractor,
    decoder: LightweightMaskDecoder,
    test_loader: DataLoader,
    mode: str,
    weight_name: str,
) -> Dict[str, float]:
    print(f"    评估: {weight_name}")

    all_preds: List[np.ndarray] = []
    all_gts: List[np.ndarray] = []

    for batch in tqdm(test_loader, desc=f"      ({mode})",
                      unit="batch", ncols=100, leave=False):
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
    return {"mIoU": metrics["mIoU"], "F1": metrics["F1"],
            "Boundary_IoU": metrics["Boundary_IoU"]}


# ==========================================================================
# 模块 7: 按模式批量评估 (共享 SAM3 加载开销)
# ==========================================================================
def _evaluate_mode_weights(
    mode: str,
    test_loader: DataLoader,
    sam3_checkpoint: str,
    weight_paths: Dict[str, str],
) -> List[Dict]:
    print(f"\n{'='*55}")
    print(f"  模式: {mode.upper()}")
    print(f"{'='*55}")

    extractor = Sam3InferenceExtractor(sam3_checkpoint)

    if mode == "lora":
        _inject_lora(extractor.model)
        lora_params_ref: Optional[Dict] = None

    decoder = LightweightMaskDecoder().to(DEVICE)
    decoder.eval()

    results: List[Dict] = []

    for label, wpath in weight_paths.items():
        if not Path(wpath).exists():
            print(f"    [跳过] 权重不存在: {wpath}")
            results.append({
                "label": label, "mode": mode,
                "mIoU": 0.0, "F1": 0.0, "Boundary_IoU": 0.0,
                "weight_found": False,
            })
            continue

        checkpoint = torch.load(wpath, map_location=DEVICE, weights_only=False)

        if mode == "lora" and "lora_params" in checkpoint:
            model_state = extractor.model.state_dict()
            for key, value in checkpoint["lora_params"].items():
                if key in model_state:
                    model_state[key].copy_(value.to(DEVICE))

        if "decoder" in checkpoint:
            decoder.load_state_dict(checkpoint["decoder"])
        else:
            decoder.load_state_dict(checkpoint)

        metrics = _evaluate_single_weight(
            extractor, decoder, test_loader, mode, Path(wpath).name,
        )
        metrics["label"] = label
        metrics["mode"] = mode
        metrics["shot"] = int(label.replace("shot", "").split("_")[2]) if "full" not in label else 0
        metrics["seed"] = int(label.split("seed")[1])
        metrics["weight_found"] = True
        results.append(metrics)

        print(f"      mIoU={metrics['mIoU']*100:.2f}%  "
              f"F1={metrics['F1']*100:.2f}%  "
              f"BIoU={metrics['Boundary_IoU']*100:.2f}%")

        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    del extractor, decoder
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return results


# ==========================================================================
# 模块 8: 构建权重文件索引
# ==========================================================================
def _build_weight_index(
    shots: List[int],
    modes: List[str],
    seeds: List[int],
) -> Dict[str, Dict[str, str]]:
    index: Dict[str, Dict[str, str]] = {}

    for mode in modes:
        mode_short = "dec" if mode == "decoder_only" else "lora"
        index[mode] = {}
        for shot in shots:
            for seed in seeds:
                label = f"{mode_short}_tgt_{shot}shot_seed{seed}"
                fname = f"sam3_{mode_short}_tgt_{shot}shot_seed{seed}.pth"
                index[mode][label] = str(Path(WEIGHTS_DIR) / fname)

    return index


# ==========================================================================
# 模块 9: 主函数
# ==========================================================================
def main() -> None:
    args = parse_args()

    shots = [int(s.strip()) for s in args.shots.split(",")] if args.shots else ALL_SHOTS
    modes = [m.strip() for m in args.modes.split(",")] if args.modes else ALL_MODES
    seeds = [int(s.strip()) for s in args.seeds.split(",")] if args.seeds else ALL_SEEDS

    total = len(shots) * len(modes) * len(seeds)
    print(f"\n{'='*70}")
    print(f"  基线模型全量 8,402 测试集统一评估")
    print(f"  平台: {get_platform_name()}  |  设备: {DEVICE}")
    print(f"  Shots: {shots}  |  Modes: {modes}  |  Seeds: {seeds}")
    print(f"  待评估权重: {total} 个  |  Batch={EVAL_BATCH_SIZE}  Workers={NUM_WORKERS}")
    print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    if not torch.cuda.is_available():
        print("\n  警告: 未检测到 CUDA GPU")

    if args.dry_run:
        print("\n  [DRY-RUN] 检查权重文件是否存在...")
        weight_index = _build_weight_index(shots, modes, seeds)
        found, missing = 0, 0
        for mode in modes:
            for label, wpath in weight_index[mode].items():
                if Path(wpath).exists():
                    found += 1
                else:
                    missing += 1
                    print(f"    [缺失] {Path(wpath).name}")
        print(f"\n  存在: {found}/{total}  缺失: {missing}/{total}")
        return

    dp = get_domain_paths("target", full_test=True)
    sam3_checkpoint = dp["sam3_checkpoint"]
    test_image_dir = dp["test_image_dir"]
    test_dual_dir = dp["test_dual_dir"]

    print(f"\n[测试集] {test_image_dir}")

    test_dataset = ValDataset(
        image_dir=test_image_dir,
        dual_label_dir=test_dual_dir,
        target_size=TARGET_SIZE,
    )
    print(f"  样本数: {len(test_dataset)}")

    def collate_fn(batch):
        return {
            "image": torch.stack([item["image"] for item in batch]),
            "mask": torch.stack([item["mask"] for item in batch]),
            "boundary": torch.stack([item["boundary"] for item in batch]),
            "name": [item["name"] for item in batch],
        }

    test_loader = DataLoader(
        test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    weight_index = _build_weight_index(shots, modes, seeds)

    all_results: List[Dict] = []
    start_time = datetime.now()

    for mode in modes:
        mode_results = _evaluate_mode_weights(
            mode=mode,
            test_loader=test_loader,
            sam3_checkpoint=sam3_checkpoint,
            weight_paths=weight_index[mode],
        )
        all_results.extend(mode_results)

    elapsed = datetime.now() - start_time
    print(f"\n{'='*70}")
    print(f"  评估完成!  总耗时: {elapsed}")
    print(f"{'='*70}")

    summary = _aggregate(all_results, shots, modes, seeds)

    _print_markdown_table(summary, shots, modes)

    _save_results_json(all_results, summary, shots, modes, seeds)

    print(f"\n{'='*70}")
    print(f"  全量评估完成!")
    print(f"{'='*70}")


# ==========================================================================
# 模块 10: Mean ± Std 聚合
# ==========================================================================
def _aggregate(
    all_results: List[Dict],
    shots: List[int],
    modes: List[str],
    seeds: List[int],
) -> Dict:
    summary: Dict = {}
    for shot in shots:
        for mode in modes:
            key = f"{mode}_{shot}shot"
            group = [
                r for r in all_results
                if r.get("shot") == shot
                and r.get("mode") == mode
                and r.get("weight_found", False)
            ]
            if not group:
                continue
            summary[key] = {
                "shot": shot,
                "mode": mode,
                "count": len(group),
                "mIoU_mean": float(np.mean([r["mIoU"] for r in group])),
                "mIoU_std": float(np.std([r["mIoU"] for r in group], ddof=1)),
                "F1_mean": float(np.mean([r["F1"] for r in group])),
                "F1_std": float(np.std([r["F1"] for r in group], ddof=1)),
                "BIoU_mean": float(np.mean([r["Boundary_IoU"] for r in group])),
                "BIoU_std": float(np.std([r["Boundary_IoU"] for r in group], ddof=1)),
            }
    return summary


# ==========================================================================
# 模块 11: Markdown 表格打印
# ==========================================================================
def _print_markdown_table(
    summary: Dict,
    shots: List[int],
    modes: List[str],
) -> None:
    mode_display = {
        "decoder_only": "Decoder-Only",
        "lora": "LoRA",
    }

    N = max(s.get("count", 0) for s in summary.values())

    print(f"\n{'='*100}")
    print(f"  Remote Sensing 期刊规范  ---  基线模型全量测试集 (8,402 张) 评估结果")
    print(f"  目标域 (target_whu_mix)  |  N = {N} 随机种子  |  Mean ± Std")
    print(f"{'='*100}")

    print(f"\n  | {{:<22}} | {{:>26}} | {{:>26}} | {{:>10}} |".format("方法", "5-shot mIoU (%)", "10-shot mIoU (%)", "20-shot mIoU (%)"))
    print(f"  |{{:-<24}}|{{:-<28}}|{{:-<28}}|{{:-<12}}|".format("", "", "", ""))

    for mode_key in modes:
        display_name = mode_display.get(mode_key, mode_key)
        cols = []
        for shot in shots:
            key = f"{mode_key}_{shot}shot"
            if key in summary:
                s = summary[key]
                cols.append(f"{s['mIoU_mean']*100:.2f} ± {s['mIoU_std']*100:.2f}")
            else:
                cols.append("-")
        print(
            f"  | {display_name:<22} | "
            f"{cols[0]:>26} | {cols[1]:>26} | {cols[2]:>10} |"
        )

    print(f"\n{'='*100}")

    print(f"\n  ### F1-score")
    print(f"\n  | {{:<22}} | {{:>26}} | {{:>26}} | {{:>10}} |".format("方法", "5-shot F1 (%)", "10-shot F1 (%)", "20-shot F1 (%)"))
    print(f"  |{{:-<24}}|{{:-<28}}|{{:-<28}}|{{:-<12}}|".format("", "", "", ""))

    for mode_key in modes:
        display_name = mode_display.get(mode_key, mode_key)
        cols = []
        for shot in shots:
            key = f"{mode_key}_{shot}shot"
            if key in summary:
                s = summary[key]
                cols.append(f"{s['F1_mean']*100:.2f} ± {s['F1_std']*100:.2f}")
            else:
                cols.append("-")
        print(
            f"  | {display_name:<22} | "
            f"{cols[0]:>26} | {cols[1]:>26} | {cols[2]:>10} |"
        )

    print(f"\n{'='*100}")

    print(f"\n  ### Boundary IoU (d=5)")
    print(f"\n  | {{:<22}} | {{:>26}} | {{:>26}} | {{:>10}} |".format("方法", "5-shot BIoU (%)", "10-shot BIoU (%)", "20-shot BIoU (%)"))
    print(f"  |{{:-<24}}|{{:-<28}}|{{:-<28}}|{{:-<12}}|".format("", "", "", ""))

    for mode_key in modes:
        display_name = mode_display.get(mode_key, mode_key)
        cols = []
        for shot in shots:
            key = f"{mode_key}_{shot}shot"
            if key in summary:
                s = summary[key]
                cols.append(f"{s['BIoU_mean']*100:.2f} ± {s['BIoU_std']*100:.2f}")
            else:
                cols.append("-")
        print(
            f"  | {display_name:<22} | "
            f"{cols[0]:>26} | {cols[1]:>26} | {cols[2]:>10} |"
        )

    print(f"\n{'='*100}")


# ==========================================================================
# 模块 12: 结果持久化
# ==========================================================================
def _save_results_json(
    all_results: List[Dict],
    summary: Dict,
    shots: List[int],
    modes: List[str],
    seeds: List[int],
) -> str:
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"baselines_full_evaluation_{timestamp}.json"
    filepath = str(Path(RESULTS_DIR) / filename)

    serializable_summary = {}
    for k, v in summary.items():
        serializable_summary[k] = {
            k2: (float(v2) if isinstance(v2, (np.floating, np.integer)) else v2)
            for k2, v2 in v.items()
        }

    output = {
        "experiment": "Baselines_Full_Test_Evaluation",
        "metadata": {
            "platform": get_platform_name(),
            "device": DEVICE,
            "timestamp": timestamp,
            "test_set": "whu_mix_full_test (8,402 images)",
            "domain": "target (target_whu_mix)",
            "shots": shots,
            "modes": modes,
            "seeds": seeds,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "num_workers": NUM_WORKERS,
        },
        "aggregated_summary": serializable_summary,
        "individual_results": all_results,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  结果已保存: {filepath}")
    return filepath


# ==========================================================================
# 模块 13: 命令行参数解析
# ==========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基线模型全量 8,402 测试集统一评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/custom_tuning/eval_baselines_full.py
  python src/custom_tuning/eval_baselines_full.py --shots 5,10,20
  python src/custom_tuning/eval_baselines_full.py --modes lora --seeds 42
  python src/custom_tuning/eval_baselines_full.py --dry-run
        """,
    )
    parser.add_argument(
        "--shots", type=str, default=None,
        help="评估的 few-shot 规模, 逗号分隔 (默认: 5,10,20)",
    )
    parser.add_argument(
        "--modes", type=str, default=None,
        help="评估模式, 逗号分隔: decoder_only,lora (默认: 全量)",
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="随机种子, 逗号分隔 (默认: 42,123,456)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅检查权重文件是否存在, 不执行评估",
    )
    return parser.parse_args()


# ==========================================================================
# 模块 14: 主入口
# ==========================================================================
if __name__ == "__main__":
    main()
