# -*- coding: utf-8 -*-
"""
run_e7b_image_corruption.py
=============================================================================
E7b: Image Corruption Robustness

目的：
  评估 prompt-free LoRA 与 HL-Residual 在图像输入扰动下的鲁棒性。

比较：
  1. Prompt-free LoRA
  2. HL-Residual (Ours)

默认：
  test = target_pilot_test_500.txt
  val = target_val.txt
  seeds = 42,123,456
  TTA + validation-calibrated postprocess
=============================================================================
"""

import argparse
import gc
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 复用 E7a 中已经定义好的模型、指标、后处理和加载函数
from run_e7a_prompt_drift import (
    DEVICE,
    TARGET_SIZE,
    PROJECT_ROOT_DEFAULT,
    ManifestDataset,
    collate_fn,
    make_loader,
    load_promptfree_model,
    predict_promptfree_probs,
    grid_search_postprocess,
    eval_with_cfg,
    eval_arrays,
)


# =============================================================================
# Corruptions
# =============================================================================

def _to_numpy_img(img_t: torch.Tensor) -> np.ndarray:
    img = (img_t.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return img


def _to_tensor_img(img_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0


def corrupt_one_image(
    img_t: torch.Tensor,
    corruption: str,
    severity: float,
    rng: np.random.RandomState,
) -> torch.Tensor:
    img_t = img_t.clone()

    if corruption == "clean":
        return img_t

    if corruption == "gaussian":
        sigma = float(severity) / 255.0
        return (img_t + torch.randn_like(img_t) * sigma).clamp(0, 1)

    if corruption == "brightness":
        factor = float(severity)
        return (img_t * factor).clamp(0, 1)

    if corruption == "contrast":
        factor = float(severity)
        return ((img_t - 0.5) * factor + 0.5).clamp(0, 1)

    img_np = _to_numpy_img(img_t)

    if corruption == "blur":
        k = int(severity)
        if k % 2 == 0:
            k += 1
        k = max(3, k)
        out = cv2.GaussianBlur(img_np, (k, k), 0)
        return _to_tensor_img(out)

    if corruption == "downsample":
        scale = float(severity)
        h, w = img_np.shape[:2]
        small_w = max(8, int(w * scale))
        small_h = max(8, int(h * scale))
        small = cv2.resize(img_np, (small_w, small_h), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return _to_tensor_img(out)

    if corruption == "occlusion":
        ratio = float(severity)
        h, w = img_np.shape[:2]
        area = h * w * ratio
        occ_h = int(np.sqrt(area))
        occ_w = int(np.sqrt(area))
        occ_h = max(8, min(h - 1, occ_h))
        occ_w = max(8, min(w - 1, occ_w))
        y = rng.randint(0, h - occ_h)
        x = rng.randint(0, w - occ_w)
        out = img_np.copy()
        out[y:y + occ_h, x:x + occ_w, :] = 127
        return _to_tensor_img(out)

    raise ValueError(f"Unknown corruption: {corruption}")


def corrupt_batch(
    imgs: torch.Tensor,
    corruption: str,
    severity: float,
    seed: int,
    start_index: int,
) -> torch.Tensor:
    out = []
    for i in range(imgs.shape[0]):
        rng = np.random.RandomState(seed + start_index + i)
        out.append(corrupt_one_image(imgs[i], corruption, severity, rng))
    return torch.stack(out, dim=0)


def parse_corruption_specs(preset: str) -> List[Tuple[str, float]]:
    if preset == "fast":
        return [
            ("clean", 0),
            ("gaussian", 10),
            ("blur", 3),
            ("brightness", 0.7),
            ("contrast", 0.7),
            ("downsample", 0.5),
            ("occlusion", 0.10),
        ]

    if preset == "paper":
        return [
            ("clean", 0),
            ("gaussian", 10),
            ("gaussian", 20),
            ("blur", 3),
            ("blur", 5),
            ("brightness", 0.7),
            ("brightness", 1.3),
            ("contrast", 0.7),
            ("contrast", 1.3),
            ("downsample", 0.5),
            ("downsample", 0.25),
            ("occlusion", 0.10),
            ("occlusion", 0.20),
        ]

    raise ValueError(preset)


# =============================================================================
# Prediction with corruption
# =============================================================================

@torch.no_grad()
def predict_promptfree_probs_corrupted(
    extractor,
    model,
    loader,
    corruption: str,
    severity: float,
    use_tta: bool,
    seed: int,
):
    probs_all = []
    gts_all = []
    seen = 0

    for batch in tqdm(loader, desc=f"  predict {corruption}:{severity}", ncols=100):
        imgs = batch["image"]
        gts = batch["mask"].numpy()[:, 0]

        imgs_corr = corrupt_batch(imgs, corruption, severity, seed=seed, start_index=seen)
        seen += imgs.shape[0]

        # 这里复用 E7a 的 predict 逻辑，但为了传入 corrupted imgs，直接手写一版
        if not use_tta:
            feat = extractor.extract_batch(imgs_corr)
            logits = model(feat.to(DEVICE), imgs_corr.to(DEVICE))
            probs = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]
        else:
            probs_list = []

            feat = extractor.extract_batch(imgs_corr)
            logits = model(feat.to(DEVICE), imgs_corr.to(DEVICE))
            probs_list.append(torch.sigmoid(logits.float()).cpu())

            imgs_h = torch.flip(imgs_corr, dims=[3])
            feat_h = extractor.extract_batch(imgs_h)
            logits_h = model(feat_h.to(DEVICE), imgs_h.to(DEVICE))
            ph = torch.sigmoid(logits_h.float()).cpu()
            ph = torch.flip(ph, dims=[3])
            probs_list.append(ph)

            imgs_v = torch.flip(imgs_corr, dims=[2])
            feat_v = extractor.extract_batch(imgs_v)
            logits_v = model(feat_v.to(DEVICE), imgs_v.to(DEVICE))
            pv = torch.sigmoid(logits_v.float()).cpu()
            pv = torch.flip(pv, dims=[2])
            probs_list.append(pv)

            probs = torch.mean(torch.stack(probs_list, dim=0), dim=0).numpy()[:, 0]

        probs_all.append(probs)
        gts_all.append(gts)

    return np.concatenate(probs_all), np.concatenate(gts_all)


# =============================================================================
# Plot
# =============================================================================

def plot_corruption_bars(results: Dict, out_path: Path):
    model_names = ["Prompt-free LoRA", "HL-Residual (Ours)"]

    labels = []
    lora_vals = []
    ours_vals = []

    for corr_key, model_dict in results.items():
        labels.append(corr_key)
        lora_vals.append(model_dict.get("Prompt-free LoRA", {}).get("mIoU", 0) * 100)
        ours_vals.append(model_dict.get("HL-Residual (Ours)", {}).get("mIoU", 0) * 100)

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 5.5))
    ax.bar(x - width / 2, lora_vals, width, label="Prompt-free LoRA")
    ax.bar(x + width / 2, ours_vals, width, label="HL-Residual (Ours)")

    ax.set_ylabel("mIoU (%)")
    ax.set_title("Image Corruption Robustness")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def aggregate_seed_metrics(seed_metrics: List[Dict[str, float]]):
    out = {}
    for k in ["mIoU", "F1", "Boundary_IoU"]:
        vals = [m[k] for m in seed_metrics]
        out[k] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


def parse_args():
    parser = argparse.ArgumentParser("E7b image corruption robustness")

    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT_DEFAULT))
    parser.add_argument("--manifest_dir", type=str, default=None)
    parser.add_argument("--test_manifest_name", type=str, default="target_pilot_test_500.txt")
    parser.add_argument("--test_eval_limit", type=int, default=500)
    parser.add_argument("--val_manifest_name", type=str, default="target_val.txt")
    parser.add_argument("--val_eval_limit", type=int, default=500)

    parser.add_argument("--sam3_checkpoint", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--preset", type=str, default="fast", choices=["fast", "paper"])
    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(args.project_root)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else project_root / "data/splits/e0_manifest"
    sam3_ckpt = Path(args.sam3_checkpoint) if args.sam3_checkpoint else project_root / "weights/sam3.pt"
    out_dir = Path(args.out_dir) if args.out_dir else project_root / "results/e7b_image_corruption"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    corruption_specs = parse_corruption_specs(args.preset)

    test_limit = args.test_eval_limit
    if test_limit is not None and test_limit <= 0:
        test_limit = None

    val_limit = args.val_eval_limit
    if val_limit is not None and val_limit <= 0:
        val_limit = None

    print("=" * 90)
    print("E7b Image Corruption Robustness")
    print("=" * 90)
    print(f"test_manifest: {args.test_manifest_name}")
    print(f"val_manifest : {args.val_manifest_name}")
    print(f"seeds        : {seeds}")
    print(f"preset       : {args.preset}")
    print(f"corruptions  : {corruption_specs}")
    print(f"use_tta      : {args.use_tta}")
    print(f"device       : {DEVICE}")
    print("=" * 90)

    test_ds = ManifestDataset(manifest_dir / args.test_manifest_name, limit=test_limit)
    val_ds = ManifestDataset(manifest_dir / args.val_manifest_name, limit=val_limit)
    test_loader = make_loader(test_ds, batch_size=4, num_workers=args.num_workers)
    val_loader = make_loader(val_ds, batch_size=4, num_workers=args.num_workers)

    results = {}
    raw = defaultdict(lambda: defaultdict(list))

    for corruption, severity in corruption_specs:
        corr_key = f"{corruption}_{severity}"
        print("\n" + "=" * 90)
        print(f"Corruption: {corr_key}")
        print("=" * 90)

        results[corr_key] = {}

        for model_key, model_label in [
            ("promptfree_lora", "Prompt-free LoRA"),
            ("hl_residual", "HL-Residual (Ours)"),
        ]:
            seed_metrics = []

            for seed in seeds:
                try:
                    print(f"\n[Model] {model_label} seed={seed}: load")
                    extractor, model, ckpt_path = load_promptfree_model(
                        model_key,
                        seed,
                        project_root,
                        sam3_ckpt,
                    )

                    # clean val calibration
                    print(f"[Model] {model_label} seed={seed}: clean val calibration")
                    val_probs, val_gts = predict_promptfree_probs(
                        extractor,
                        model,
                        val_loader,
                        use_tta=args.use_tta,
                    )
                    search = grid_search_postprocess(val_probs, val_gts)
                    best_cfg = search["best"]

                    print(f"[Model] {model_label} seed={seed}: corrupted test {corr_key}")
                    test_probs, test_gts = predict_promptfree_probs_corrupted(
                        extractor=extractor,
                        model=model,
                        loader=test_loader,
                        corruption=corruption,
                        severity=severity,
                        use_tta=args.use_tta,
                        seed=seed,
                    )

                    m = eval_with_cfg(test_probs, test_gts, best_cfg)
                    m["checkpoint"] = str(ckpt_path)
                    m["val_best_cfg"] = best_cfg
                    m["corruption"] = corruption
                    m["severity"] = severity
                    seed_metrics.append(m)

                    del extractor, model
                    gc.collect()
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"[Skip] model={model_label}, seed={seed}, corr={corr_key}, error={e}")
                    import traceback
                    traceback.print_exc()

            if seed_metrics:
                results[corr_key][model_label] = aggregate_seed_metrics(seed_metrics)
                raw[corr_key][model_label] = seed_metrics

                mm = results[corr_key][model_label]
                print(
                    f"[Result] {corr_key} | {model_label} "
                    f"mIoU={mm['mIoU']*100:.2f} "
                    f"F1={mm['F1']*100:.2f} "
                    f"BIoU={mm['Boundary_IoU']*100:.2f}"
                )

    summary = {
        "experiment": "E7b_image_corruption",
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "test_manifest": args.test_manifest_name,
            "val_manifest": args.val_manifest_name,
            "seeds": seeds,
            "preset": args.preset,
            "corruptions": corruption_specs,
            "use_tta": args.use_tta,
            "note": "Validation calibration is performed on clean target_val; corrupted test is evaluated using the clean-val-selected postprocess config.",
        },
        "results": results,
        "raw_per_seed": raw,
    }

    json_path = out_dir / "e7b_image_corruption_metrics.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fig_path = out_dir / "e7b_image_corruption_miou.png"
    plot_corruption_bars(results, fig_path)

    print("\n" + "=" * 90)
    print("E7b finished")
    print("=" * 90)
    print(f"JSON: {json_path}")
    print(f"FIG : {fig_path}")


if __name__ == "__main__":
    main()