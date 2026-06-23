# -*- coding: utf-8 -*-
"""
run_e7b_resume_select.py
=============================================================================
只补跑 E7b 指定 corruption + 指定 model，用于断点续跑。

典型用途：
  前半部分已经有 Prompt-free LoRA + HL-Residual；
  后半部分只跑出了 HL-Residual；
  现在只补跑 Prompt-free LoRA 的后半部分。

运行示例：
  python -u src/run_e7b_resume_select.py \
    --models promptfree_lora \
    --corruptions brightness:1.3,contrast:0.7,contrast:1.3,downsample:0.5,downsample:0.25,occlusion:0.1,occlusion:0.2 \
    --use_tta \
    --test_manifest_name target_pilot_test_500.txt \
    --test_eval_limit 0 \
    --val_eval_limit 500 \
    --seeds 42,123,456 \
    --out_dir results/e7b_image_corruption_resume_lora_missing
=============================================================================
"""

import argparse
import gc
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_e7a_prompt_drift import (
    DEVICE,
    PROJECT_ROOT_DEFAULT,
    ManifestDataset,
    make_loader,
    load_promptfree_model,
    predict_promptfree_probs,
    grid_search_postprocess,
    eval_with_cfg,
)

from run_e7b_image_corruption import (
    predict_promptfree_probs_corrupted,
)


def parse_corruptions(s: str) -> List[Tuple[str, float]]:
    """
    输入格式:
      brightness:1.3,contrast:0.7,downsample:0.25,occlusion:0.1
    """
    out = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        name, sev = item.split(":")
        out.append((name.strip(), float(sev)))
    return out


def aggregate_seed_metrics(seed_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    out = {}
    for k in ["mIoU", "F1", "Boundary_IoU"]:
        vals = [m[k] for m in seed_metrics]
        out[k] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


def parse_args():
    parser = argparse.ArgumentParser("Resume selected E7b corruption/model runs")

    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT_DEFAULT))
    parser.add_argument("--manifest_dir", type=str, default=None)
    parser.add_argument("--test_manifest_name", type=str, default="target_pilot_test_500.txt")
    parser.add_argument("--test_eval_limit", type=int, default=500)
    parser.add_argument("--val_manifest_name", type=str, default="target_val.txt")
    parser.add_argument("--val_eval_limit", type=int, default=500)

    parser.add_argument("--sam3_checkpoint", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument(
        "--models",
        type=str,
        default="promptfree_lora",
        help="可选: promptfree_lora,hl_residual，用逗号分隔",
    )
    parser.add_argument(
        "--corruptions",
        type=str,
        required=True,
        help="格式: brightness:1.3,contrast:0.7,downsample:0.25",
    )
    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(args.project_root)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else project_root / "data/splits/e0_manifest"
    sam3_ckpt = Path(args.sam3_checkpoint) if args.sam3_checkpoint else project_root / "weights/sam3.pt"
    out_dir = Path(args.out_dir) if args.out_dir else project_root / "results/e7b_resume_select"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    corruption_specs = parse_corruptions(args.corruptions)

    test_limit = args.test_eval_limit
    if test_limit is not None and test_limit <= 0:
        test_limit = None

    val_limit = args.val_eval_limit
    if val_limit is not None and val_limit <= 0:
        val_limit = None

    print("=" * 90)
    print("E7b Resume Selected Runs")
    print("=" * 90)
    print(f"test_manifest: {args.test_manifest_name}")
    print(f"val_manifest : {args.val_manifest_name}")
    print(f"models       : {models}")
    print(f"corruptions  : {corruption_specs}")
    print(f"seeds        : {seeds}")
    print(f"use_tta      : {args.use_tta}")
    print(f"device       : {DEVICE}")
    print("=" * 90)

    test_ds = ManifestDataset(manifest_dir / args.test_manifest_name, limit=test_limit)
    val_ds = ManifestDataset(manifest_dir / args.val_manifest_name, limit=val_limit)
    test_loader = make_loader(test_ds, batch_size=4, num_workers=args.num_workers)
    val_loader = make_loader(val_ds, batch_size=4, num_workers=args.num_workers)

    model_label_map = {
        "promptfree_lora": "Prompt-free LoRA",
        "hl_residual": "HL-Residual (Ours)",
    }

    results = {}
    raw = defaultdict(lambda: defaultdict(list))

    for corruption, severity in corruption_specs:
        corr_key = f"{corruption}_{severity:g}"
        print("\n" + "=" * 90)
        print(f"Corruption: {corr_key}")
        print("=" * 90)

        results[corr_key] = {}

        for model_key in models:
            model_label = model_label_map[model_key]
            seed_metrics = []

            for seed in seeds:
                try:
                    print(f"\n[Load] {model_label} seed={seed}")
                    extractor, model, ckpt_path = load_promptfree_model(
                        model_key,
                        seed,
                        project_root,
                        sam3_ckpt,
                    )

                    print(f"[Calibrate clean val] {model_label} seed={seed}")
                    val_probs, val_gts = predict_promptfree_probs(
                        extractor,
                        model,
                        val_loader,
                        use_tta=args.use_tta,
                    )
                    search = grid_search_postprocess(val_probs, val_gts)
                    best_cfg = search["best"]

                    print(f"[Corrupted test] {model_label} seed={seed} {corr_key}")
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

                r = results[corr_key][model_label]
                print(
                    f"[Result] {corr_key} | {model_label} "
                    f"mIoU={r['mIoU']*100:.2f} "
                    f"F1={r['F1']*100:.2f} "
                    f"BIoU={r['Boundary_IoU']*100:.2f}"
                )

    summary = {
        "experiment": "E7b_image_corruption_resume_select",
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "test_manifest": args.test_manifest_name,
            "val_manifest": args.val_manifest_name,
            "models": models,
            "seeds": seeds,
            "corruptions": corruption_specs,
            "use_tta": args.use_tta,
            "note": "Validation calibration is performed on clean target_val; corrupted test is evaluated using the clean-val-selected postprocess config.",
        },
        "results": results,
        "raw_per_seed": raw,
    }

    out_path = out_dir / "e7b_resume_select_metrics.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("E7b resume select finished")
    print("=" * 90)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()