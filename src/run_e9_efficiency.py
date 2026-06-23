# -*- coding: utf-8 -*-
"""
run_e9_efficiency.py
=============================================================================
E9: Efficiency / Parameters / Deployment Cost

目的：
  统计 Prompt-free LoRA 与 HL-Residual 的参数量、额外参数、推理速度。
  同时报告 single-pass 与 TTA 推理耗时。

依赖：
  src/run_e7a_prompt_drift.py

推荐 smoke test：
  python -u src/run_e9_efficiency.py \
    --models promptfree_lora,hl_residual \
    --seeds 42 \
    --test_manifest_name target_pilot_test_500.txt \
    --test_eval_limit 50 \
    --batch_size 4 \
    --num_workers 2 \
    --out_dir results/e9_efficiency_smoke

正式：
  python -u src/run_e9_efficiency.py \
    --models promptfree_lora,hl_residual \
    --seeds 42,123,456 \
    --test_manifest_name target_pilot_test_500.txt \
    --test_eval_limit 500 \
    --batch_size 4 \
    --num_workers 2 \
    --out_dir results/e9_efficiency_pilot500
=============================================================================
"""

import argparse
import csv
import gc
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_e7a_prompt_drift import (
    DEVICE,
    PROJECT_ROOT_DEFAULT,
    ManifestDataset,
    collate_fn,
    load_promptfree_model,
)


# =============================================================================
# Loader
# =============================================================================

def make_loader(ds, batch_size: int, num_workers: int):
    kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


# =============================================================================
# Parameter counting
# =============================================================================

def count_params_module(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def count_trainable_params_module(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def count_lora_params(extractor) -> int:
    total = 0
    for name, p in extractor.model.named_parameters():
        if "lora" in name.lower():
            total += p.numel()
    return int(total)


def build_param_record(model_key: str, model_label: str, seed: int, extractor, model, ckpt_path: str):
    sam_total = count_params_module(extractor.model)
    lora_params = count_lora_params(extractor)

    model_total = count_params_module(model)
    model_trainable_now = count_trainable_params_module(model)

    residual_params = 0
    decoder_params = 0

    if model_key == "hl_residual":
        if hasattr(model, "residual"):
            residual_params = count_params_module(model.residual)
        if hasattr(model, "base_decoder"):
            decoder_params = count_params_module(model.base_decoder)
    else:
        decoder_params = count_params_module(model)

    # 训练阶段可训练参数的合理估计：
    # promptfree_lora: LoRA + decoder
    # hl_residual:     LoRA + decoder + residual，若论文描述为两阶段 residual，可额外报告 residual only
    if model_key == "promptfree_lora":
        trainable_training = lora_params + decoder_params
        extra_params_vs_lora = 0
    elif model_key == "hl_residual":
        trainable_training = lora_params + decoder_params + residual_params
        extra_params_vs_lora = residual_params
    else:
        trainable_training = lora_params + model_total
        extra_params_vs_lora = 0

    return {
        "model": model_label,
        "model_key": model_key,
        "seed": seed,
        "checkpoint": str(ckpt_path),
        "sam_total_params": sam_total,
        "lora_params": lora_params,
        "decoder_params": decoder_params,
        "residual_params": residual_params,
        "model_head_total_params": model_total,
        "trainable_params_estimated": trainable_training,
        "extra_params_vs_lora": extra_params_vs_lora,
        "trainable_ratio_percent": 100.0 * trainable_training / max(1, sam_total + model_total),
    }


# =============================================================================
# Timing
# =============================================================================

@torch.no_grad()
def forward_once(extractor, model, imgs: torch.Tensor, use_tta: bool):
    if not use_tta:
        feat = extractor.extract_batch(imgs)
        logits = model(feat.to(DEVICE), imgs.to(DEVICE))
        return logits

    outputs = []

    feat = extractor.extract_batch(imgs)
    logits = model(feat.to(DEVICE), imgs.to(DEVICE))
    outputs.append(logits)

    imgs_h = torch.flip(imgs, dims=[3])
    feat_h = extractor.extract_batch(imgs_h)
    logits_h = model(feat_h.to(DEVICE), imgs_h.to(DEVICE))
    logits_h = torch.flip(logits_h, dims=[3])
    outputs.append(logits_h)

    imgs_v = torch.flip(imgs, dims=[2])
    feat_v = extractor.extract_batch(imgs_v)
    logits_v = model(feat_v.to(DEVICE), imgs_v.to(DEVICE))
    logits_v = torch.flip(logits_v, dims=[2])
    outputs.append(logits_v)

    return torch.mean(torch.stack(outputs, dim=0), dim=0)


@torch.no_grad()
def benchmark_model(
    extractor,
    model,
    loader,
    use_tta: bool,
    warmup_batches: int,
    max_batches: int,
):
    model.eval()
    extractor.model.eval()

    times = []
    n_images = 0
    n_batches = 0

    # warmup
    print(f"[Warmup] use_tta={use_tta}, warmup_batches={warmup_batches}", flush=True)
    for i, batch in enumerate(loader):
        if i >= warmup_batches:
            break
        imgs = batch["image"]
        _ = forward_once(extractor, model, imgs, use_tta=use_tta)
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    print(f"[Benchmark] use_tta={use_tta}, max_batches={max_batches}", flush=True)

    for i, batch in enumerate(tqdm(loader, desc=f"  timing tta={use_tta}", ncols=100)):
        if max_batches > 0 and i >= max_batches:
            break

        imgs = batch["image"]

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        _ = forward_once(extractor, model, imgs, use_tta=use_tta)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        dt = t1 - t0
        times.append(dt)
        n_images += imgs.shape[0]
        n_batches += 1

    total_time = float(np.sum(times))
    avg_batch_time = float(np.mean(times)) if times else 0.0
    avg_ms_per_image = 1000.0 * total_time / max(1, n_images)
    fps = n_images / max(1e-9, total_time)

    return {
        "use_tta": bool(use_tta),
        "num_images": int(n_images),
        "num_batches": int(n_batches),
        "total_time_sec": total_time,
        "avg_batch_time_sec": avg_batch_time,
        "avg_ms_per_image": avg_ms_per_image,
        "fps": fps,
    }


# =============================================================================
# Save
# =============================================================================

def write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(vals):
    arr = np.array(vals, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    if len(arr) == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def summarize_timing(rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for r in rows:
        key = (r["model"], r["use_tta"])
        grouped[key].append(r)

    out = []
    for (model, use_tta), items in sorted(grouped.items()):
        ms_mean, ms_std = mean_std([x["avg_ms_per_image"] for x in items])
        fps_mean, fps_std = mean_std([x["fps"] for x in items])
        out.append({
            "model": model,
            "use_tta": use_tta,
            "num_seeds": len(items),
            "avg_ms_per_image_mean": ms_mean,
            "avg_ms_per_image_std": ms_std,
            "fps_mean": fps_mean,
            "fps_std": fps_std,
        })
    return out


def summarize_params(rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["model"]].append(r)

    out = []
    for model, items in sorted(grouped.items()):
        row = {"model": model, "num_seeds": len(items)}
        for key in [
            "sam_total_params",
            "lora_params",
            "decoder_params",
            "residual_params",
            "model_head_total_params",
            "trainable_params_estimated",
            "extra_params_vs_lora",
            "trainable_ratio_percent",
        ]:
            mean, std = mean_std([x[key] for x in items])
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
        out.append(row)
    return out


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser("E9 efficiency and parameter benchmark")

    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT_DEFAULT))
    parser.add_argument("--manifest_dir", type=str, default=None)
    parser.add_argument("--test_manifest_name", type=str, default="target_pilot_test_500.txt")
    parser.add_argument("--test_eval_limit", type=int, default=500)

    parser.add_argument("--sam3_checkpoint", type=str, default=None)
    parser.add_argument("--models", type=str, default="promptfree_lora,hl_residual")
    parser.add_argument("--seeds", type=str, default="42,123,456")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--warmup_batches", type=int, default=5)
    parser.add_argument("--max_batches", type=int, default=0, help="0 表示跑完整个 test_eval_limit")
    parser.add_argument("--include_tta", action="store_true", help="同时测试 TTA 速度")

    parser.add_argument("--out_dir", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(args.project_root)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else project_root / "data/splits/e0_manifest"
    sam3_ckpt = Path(args.sam3_checkpoint) if args.sam3_checkpoint else project_root / "weights/sam3.pt"
    out_dir = Path(args.out_dir) if args.out_dir else project_root / "results/e9_efficiency"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    test_limit = args.test_eval_limit
    if test_limit is not None and test_limit <= 0:
        test_limit = None

    model_label_map = {
        "promptfree_lora": "Prompt-free LoRA",
        "hl_residual": "HL-Residual (Ours)",
    }

    print("=" * 90)
    print("E9 Efficiency Benchmark")
    print("=" * 90)
    print(f"test_manifest: {args.test_manifest_name}")
    print(f"test_limit   : {test_limit}")
    print(f"models       : {models}")
    print(f"seeds        : {seeds}")
    print(f"batch_size   : {args.batch_size}")
    print(f"num_workers  : {args.num_workers}")
    print(f"include_tta  : {args.include_tta}")
    print(f"device       : {DEVICE}")
    print(f"out_dir      : {out_dir}")
    print("=" * 90)

    ds = ManifestDataset(manifest_dir / args.test_manifest_name, limit=test_limit)
    loader = make_loader(ds, batch_size=args.batch_size, num_workers=args.num_workers)

    param_rows = []
    timing_rows = []

    for model_key in models:
        model_label = model_label_map.get(model_key, model_key)

        for seed in seeds:
            print("\n" + "=" * 90)
            print(f"[E9] model={model_label} seed={seed}")
            print("=" * 90)

            extractor, model, ckpt_path = load_promptfree_model(
                model_key,
                seed,
                project_root,
                sam3_ckpt,
            )

            p_row = build_param_record(
                model_key=model_key,
                model_label=model_label,
                seed=seed,
                extractor=extractor,
                model=model,
                ckpt_path=str(ckpt_path),
            )
            param_rows.append(p_row)

            # single-pass
            t_single = benchmark_model(
                extractor=extractor,
                model=model,
                loader=loader,
                use_tta=False,
                warmup_batches=args.warmup_batches,
                max_batches=args.max_batches,
            )
            t_single.update({
                "model": model_label,
                "model_key": model_key,
                "seed": seed,
                "checkpoint": str(ckpt_path),
                "batch_size": args.batch_size,
            })
            timing_rows.append(t_single)

            # TTA
            if args.include_tta:
                t_tta = benchmark_model(
                    extractor=extractor,
                    model=model,
                    loader=loader,
                    use_tta=True,
                    warmup_batches=args.warmup_batches,
                    max_batches=args.max_batches,
                )
                t_tta.update({
                    "model": model_label,
                    "model_key": model_key,
                    "seed": seed,
                    "checkpoint": str(ckpt_path),
                    "batch_size": args.batch_size,
                })
                timing_rows.append(t_tta)

            write_csv(out_dir / "e9_params_partial.csv", param_rows)
            write_csv(out_dir / "e9_timing_partial.csv", timing_rows)

            del extractor, model
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    param_summary = summarize_params(param_rows)
    timing_summary = summarize_timing(timing_rows)

    write_csv(out_dir / "e9_params_by_seed.csv", param_rows)
    write_csv(out_dir / "e9_timing_by_seed.csv", timing_rows)
    write_csv(out_dir / "e9_params_summary.csv", param_summary)
    write_csv(out_dir / "e9_timing_summary.csv", timing_summary)

    summary = {
        "experiment": "E9_efficiency_params",
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "test_manifest": args.test_manifest_name,
            "test_eval_limit": args.test_eval_limit,
            "models": models,
            "seeds": seeds,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "include_tta": args.include_tta,
            "device": DEVICE,
            "note": "Timing excludes disk loading as much as possible but includes SAM feature extraction and decoder inference. TTA uses original + horizontal flip + vertical flip.",
        },
        "params_summary": param_summary,
        "timing_summary": timing_summary,
    }

    json_path = out_dir / "e9_efficiency_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("E9 finished")
    print("=" * 90)
    print(f"Params by seed : {out_dir / 'e9_params_by_seed.csv'}")
    print(f"Timing by seed : {out_dir / 'e9_timing_by_seed.csv'}")
    print(f"Params summary : {out_dir / 'e9_params_summary.csv'}")
    print(f"Timing summary : {out_dir / 'e9_timing_summary.csv'}")
    print(f"JSON summary   : {json_path}")

    print("\nTiming summary:")
    for r in timing_summary:
        print(
            f"{r['model']:<22} "
            f"TTA={r['use_tta']} "
            f"ms/img={r['avg_ms_per_image_mean']:.2f}±{r['avg_ms_per_image_std']:.2f} "
            f"FPS={r['fps_mean']:.2f}"
        )

    print("\nParams summary:")
    for r in param_summary:
        print(
            f"{r['model']:<22} "
            f"trainable={r['trainable_params_estimated_mean'] / 1e6:.2f}M "
            f"extra_residual={r['extra_params_vs_lora_mean'] / 1e6:.2f}M "
            f"ratio={r['trainable_ratio_percent_mean']:.3f}%"
        )


if __name__ == "__main__":
    main()