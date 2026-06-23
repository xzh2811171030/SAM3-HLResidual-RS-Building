#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_three_model_coverage_eval.py

Unified wrapper for the paper's coverage diagnostic.

Purpose
-------
Evaluate and save per-image TP/FP/FN coverage CSVs for three model families:

  1) Prompt-free LoRA        -> evaluated by run_calibrated_fulltest_save_masks_coverage.py
  2) HL-Residual             -> evaluated by run_calibrated_fulltest_save_masks_coverage.py
  3) UNetFormer-style        -> evaluated by run_unetformer_matched_baseline_save_confusion.py

Then merge all generated CSVs into one held-out non-pilot coverage table.

Important
---------
This is intentionally a wrapper instead of merging all model definitions into one
huge script. The SAM3-LoRA/HL models and UNetFormer-style model use different
architectures and checkpoint-loading logic. Wrapping the two already-tested
evaluators is safer and easier to debug.

The final coverage table is computed on:
  target_final_test_8402.txt minus target_pilot_test_500.txt = held-out non-pilot 7902.

No prediction masks are saved by default. Only per-image confusion CSVs are saved.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def fmt_template(tpl: Optional[str], seed: int) -> Optional[str]:
    if tpl is None or str(tpl).strip() == "":
        return None
    return str(tpl).format(seed=seed)


def run_cmd(cmd: List[str], cwd: Path, dry_run: bool = False) -> None:
    print("\n" + "=" * 100)
    print("[CMD]")
    print(" ".join(shlex.quote(x) for x in cmd))
    print("=" * 100, flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def require_file(path: Path, msg: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{msg}: {path}")


def expected_sam_csvs(sam_out_root: Path, seeds: List[int], sam_models: List[str]) -> List[Path]:
    out = []
    for seed in seeds:
        seed_tag = f"seed{seed}"
        for model in sam_models:
            out.append(sam_out_root / seed_tag / "per_image_confusion" / f"{model}_{seed_tag}_calibrated_confusion.csv")
    return out


def check_unet_ckpts(unet_output_dir: Path, encoder: str, seeds: List[int]) -> None:
    for seed in seeds:
        ckpt = unet_output_dir / f"seed{seed}" / f"unetformer_{encoder}_source_target20_seed{seed}_best.pth"
        require_file(ckpt, "Required UNetFormer-style target checkpoint not found")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Unified three-model coverage diagnostic evaluator")

    p.add_argument("--project_root", type=str, default="./data")
    p.add_argument("--manifest_dir", type=str, default="data/splits/e0_manifest")

    p.add_argument("--sam_script", type=str, default="src/run_calibrated_fulltest_save_masks_coverage.py")
    p.add_argument("--unet_script", type=str, default="src/run_unetformer_matched_baseline_save_confusion.py")

    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])

    # SAM3-LoRA / HL-Residual part
    p.add_argument("--sam_models", type=str, default="lora_light,hl_residual_v2",
                   help="Comma-separated SAM-side models. Recommended: lora_light,hl_residual_v2.")
    p.add_argument("--lora_light_ckpt_template", type=str, default=None,
                   help="Optional checkpoint template for lora_light, e.g. results/.../seed{seed}/lora_light_best.pth")
    p.add_argument("--hl_residual_v2_ckpt_template", type=str, default=None,
                   help="Optional checkpoint template for hl_residual_v2, e.g. results/.../seed{seed}/hl_residual_best.pth")
    p.add_argument("--hl_residual_old_ckpt_template", type=str, default=None,
                   help="Optional checkpoint template for hl_residual_old if used.")
    p.add_argument("--hl_lite_ckpt_template", type=str, default=None,
                   help="Optional checkpoint template for hl_lite if used.")

    p.add_argument("--sam_out_root", type=str, default="results/three_model_coverage/sam")
    p.add_argument("--sam_grid_mode", type=str, default="full", choices=["fast", "full"])
    p.add_argument("--val_eval_limit", type=int, default=500)
    p.add_argument("--test_manifest_name", type=str, default="target_final_test_8402.txt")
    p.add_argument("--num_workers", type=int, default=2)

    # UNetFormer-style part
    p.add_argument("--unet_output_dir", type=str, default="results/unetformer_matched")
    p.add_argument("--unet_source_ckpt", type=str, default="results/unetformer_matched/unetformer_resnet34_source_best.pth")
    p.add_argument("--unet_encoder", type=str, default="resnet34", choices=["resnet34", "resnet50"])
    p.add_argument("--unet_method_name", type=str, default="UNetFormer-style ResNet34")
    p.add_argument("--unet_eval_batch_size", type=int, default=16)
    p.add_argument("--unet_num_workers", type=int, default=4)
    p.add_argument("--require_existing_unet_ckpts", action="store_true", default=True,
                   help="Fail if UNetFormer-style target checkpoints do not exist. Default: true.")
    p.add_argument("--allow_unet_training_if_missing", action="store_true",
                   help="Dangerous/time-consuming: allow UNet script to train if target checkpoints are missing.")

    # Common
    p.add_argument("--use_tta", action="store_true",
                   help="Use TTA for all three models, matching the manuscript main setting.")
    p.add_argument("--coverage_output_dir", type=str, default="results/three_model_coverage/coverage_nonpilot")
    p.add_argument("--run_sam", action="store_true", default=True)
    p.add_argument("--run_unet", action="store_true", default=True)
    p.add_argument("--only_merge_existing", action="store_true",
                   help="Do not run inference. Only merge existing CSVs from expected paths.")
    p.add_argument("--dry_run", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    manifest_dir = args.manifest_dir
    sam_script = Path(args.sam_script)
    unet_script = Path(args.unet_script)

    if not sam_script.is_absolute():
        sam_script = project_root / sam_script
    if not unet_script.is_absolute():
        unet_script = project_root / unet_script

    require_file(sam_script, "SAM calibrated evaluator script not found")
    require_file(unet_script, "UNetFormer-style evaluator script not found")
    require_file(project_root / manifest_dir / "target_final_test_8402.txt", "Final-test manifest not found")
    require_file(project_root / manifest_dir / "target_pilot_test_500.txt", "Pilot manifest not found")

    seeds = list(args.seeds)
    sam_models = [m.strip() for m in args.sam_models.split(",") if m.strip()]
    sam_out_root = (project_root / args.sam_out_root).resolve() if not os.path.isabs(args.sam_out_root) else Path(args.sam_out_root).resolve()
    unet_output_dir = (project_root / args.unet_output_dir).resolve() if not os.path.isabs(args.unet_output_dir) else Path(args.unet_output_dir).resolve()
    coverage_output_dir = (project_root / args.coverage_output_dir).resolve() if not os.path.isabs(args.coverage_output_dir) else Path(args.coverage_output_dir).resolve()

    if args.allow_unet_training_if_missing:
        args.require_existing_unet_ckpts = False

    if args.require_existing_unet_ckpts and not args.only_merge_existing:
        source_ckpt = Path(args.unet_source_ckpt)
        if not source_ckpt.is_absolute():
            source_ckpt = project_root / source_ckpt
        require_file(source_ckpt, "Required UNetFormer-style source checkpoint not found")
        check_unet_ckpts(unet_output_dir, args.unet_encoder, seeds)

    # -------------------------------------------------------------------------
    # 1) Run SAM3-LoRA / HL-Residual evaluator per seed.
    # -------------------------------------------------------------------------
    if args.run_sam and not args.only_merge_existing:
        for seed in seeds:
            seed_tag = f"seed{seed}"
            out_dir = sam_out_root / seed_tag

            cmd = [
                sys.executable, str(sam_script),
                "--project_root", str(project_root),
                "--manifest_dir", manifest_dir,
                "--models", ",".join(sam_models),
                "--val_eval_limit", str(args.val_eval_limit),
                "--test_manifest_name", args.test_manifest_name,
                "--grid_mode", args.sam_grid_mode,
                "--run_seed", str(seed),
                "--num_workers", str(args.num_workers),
                "--save_per_image_confusion",
                "--out_dir", str(out_dir),
            ]
            if args.use_tta:
                cmd.append("--use_tta")

            # Optional seed-specific checkpoint templates.
            ckpt_opts = {
                "lora_light": ("--lora_light_ckpt", args.lora_light_ckpt_template),
                "hl_lite": ("--hl_lite_ckpt", args.hl_lite_ckpt_template),
                "hl_residual_old": ("--hl_residual_old_ckpt", args.hl_residual_old_ckpt_template),
                "hl_residual_v2": ("--hl_residual_v2_ckpt", args.hl_residual_v2_ckpt_template),
            }
            for model_name in sam_models:
                opt_and_tpl = ckpt_opts.get(model_name)
                if opt_and_tpl is None:
                    continue
                opt, tpl = opt_and_tpl
                ckpt = fmt_template(tpl, seed)
                if ckpt:
                    cmd.extend([opt, ckpt])

            run_cmd(cmd, cwd=project_root, dry_run=args.dry_run)

    sam_csvs = expected_sam_csvs(sam_out_root, seeds, sam_models)

    # -------------------------------------------------------------------------
    # 2) Run UNetFormer-style evaluator and merge SAM CSVs as extra inputs.
    # -------------------------------------------------------------------------
    if args.run_unet and not args.only_merge_existing:
        cmd = [
            sys.executable, str(unet_script),
            "--project_root", str(project_root),
            "--manifest_dir", manifest_dir,
            "--output_dir", str(unet_output_dir),
            "--encoder", args.unet_encoder,
            "--seeds", *[str(s) for s in seeds],
            "--eval_batch_size", str(args.unet_eval_batch_size),
            "--num_workers", str(args.unet_num_workers),
            "--skip_source_pretrain",
            "--source_ckpt", args.unet_source_ckpt,
            "--skip_target_finetune_if_exists",
            "--save_confusion_csv",
            "--method_name", args.unet_method_name,
            "--make_coverage_table",
            "--coverage_output_dir", str(coverage_output_dir),
            "--extra_confusion_csv", *[str(p) for p in sam_csvs],
        ]
        if args.use_tta:
            cmd.append("--use_tta")

        # If the user explicitly allows training, remove skip_target behavior.
        if args.allow_unet_training_if_missing:
            cmd = [x for x in cmd if x != "--skip_target_finetune_if_exists"]

        run_cmd(cmd, cwd=project_root, dry_run=args.dry_run)

    # -------------------------------------------------------------------------
    # 3) Basic output checks.
    # -------------------------------------------------------------------------
    if not args.dry_run:
        missing = [p for p in sam_csvs if not p.exists()]
        if missing:
            print("[WARN] Some expected SAM CSVs are missing:")
            for p in missing:
                print("  -", p)

        coverage_tex = coverage_output_dir / "coverage_table.tex"
        if coverage_tex.exists():
            print("\n[DONE] Three-model coverage table:")
            print(coverage_tex)
        else:
            print("\n[WARN] Coverage table was not found at:")
            print(coverage_tex)
            print("Check the logs above. If only SAM CSVs were generated, rerun with --run_unet or merge manually.")


if __name__ == "__main__":
    main()
