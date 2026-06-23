# -*- coding: utf-8 -*-
"""
paired_significance_analysis.py

Purpose:
  Compute paired image-level statistics for Prompt-free LoRA vs HL-Residual
  using e8_per_image_metrics.csv.

Input CSV columns required:
  image_name, region_prefix, seed, model, mIoU, F1, Boundary_IoU

Recommended run:
  python paired_significance_analysis.py \
    --input e8_per_image_metrics.csv \
    --out_dir results/paired_significance

Outputs:
  paired_significance_summary.json
  paired_significance_table.csv
  paired_significance_table_latex.txt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_ci(arr: np.ndarray, n_boot: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, n)]
        means[i] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def try_pvalues(x, y):
    """Return paired t-test and Wilcoxon one-sided p-values if scipy is available."""
    try:
        from scipy import stats
        t = stats.ttest_rel(x, y, alternative="greater")
        w = stats.wilcoxon(x - y, alternative="greater", zero_method="wilcox")
        return {
            "paired_t_p": float(t.pvalue),
            "wilcoxon_p": float(w.pvalue),
        }
    except Exception:
        return {
            "paired_t_p": None,
            "wilcoxon_p": None,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to e8_per_image_metrics.csv")
    parser.add_argument("--out_dir", default="results/paired_significance")
    parser.add_argument("--lora_name", default="Prompt-free LoRA")
    parser.add_argument("--hl_name", default="HL-Residual (Ours)")
    parser.add_argument("--n_boot", type=int, default=5000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    required = {"image_name", "seed", "model", "mIoU", "F1", "Boundary_IoU"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Average the same image over seeds before paired comparison.
    group_cols = ["image_name", "model"]
    if "region_prefix" in df.columns:
        group_cols = ["image_name", "region_prefix", "model"]

    img_avg = (
        df.groupby(group_cols)[["mIoU", "F1", "Boundary_IoU"]]
        .mean()
        .reset_index()
    )

    index_cols = ["image_name"] + (["region_prefix"] if "region_prefix" in img_avg.columns else [])
    pivot = img_avg.pivot_table(
        index=index_cols,
        columns="model",
        values=["mIoU", "F1", "Boundary_IoU"],
        aggfunc="first",
    )

    rows = []
    summary = {
        "n_images": int(len(pivot)),
        "lora_name": args.lora_name,
        "hl_name": args.hl_name,
        "metrics": {},
        "protocol": (
            "Each image is first averaged over random seeds, and then HL-Residual "
            "is paired with Prompt-free LoRA on the same test image."
        ),
    }

    for metric in ["mIoU", "F1", "Boundary_IoU"]:
        hl = pivot[(metric, args.hl_name)].to_numpy(dtype=float)
        lo = pivot[(metric, args.lora_name)].to_numpy(dtype=float)
        diff = hl - lo
        ci_lo, ci_hi = bootstrap_ci(diff, n_boot=args.n_boot, seed=0)
        pvals = try_pvalues(hl, lo)

        item = {
            "metric": metric,
            "mean_gain_pp": float(diff.mean() * 100.0),
            "median_gain_pp": float(np.median(diff) * 100.0),
            "ci95_low_pp": float(ci_lo * 100.0),
            "ci95_high_pp": float(ci_hi * 100.0),
            "image_win_rate_percent": float((diff > 0).mean() * 100.0),
            "image_nonnegative_rate_percent": float((diff >= 0).mean() * 100.0),
            **pvals,
        }
        summary["metrics"][metric] = item
        rows.append(item)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "paired_significance_table.csv", index=False)

    with open(out_dir / "paired_significance_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    def p_str(p):
        if p is None:
            return "--"
        if p < 1e-6:
            return "$<10^{-6}$"
        return f"{p:.2e}"

    latex_lines = [
        r"\begin{table}[H]",
        r"\caption{Paired image-level analysis on the full WHU-Mix test set. Metrics are computed by averaging each image over the three random seeds before comparing HL-Residual with prompt-free LoRA. Confidence intervals are estimated by bootstrap resampling over the 8,402 test images.}",
        r"\label{tab:paired_significance}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{7.0pt}",
        r"\begin{tabular}{l c c c c}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Mean gain (pp)} & \textbf{Median gain (pp)} & \textbf{95\% CI (pp)} & \textbf{Image win rate} \\",
        r"\midrule",
    ]

    name_map = {"mIoU": "mIoU", "F1": "F1", "Boundary_IoU": "Boundary IoU"}
    for _, row in table.iterrows():
        latex_lines.append(
            f"{name_map[row['metric']]} & "
            f"{row['mean_gain_pp']:+.2f} & "
            f"{row['median_gain_pp']:+.2f} & "
            f"[{row['ci95_low_pp']:.2f}, {row['ci95_high_pp']:.2f}] & "
            f"{row['image_win_rate_percent']:.2f}\\% \\\\"
        )

    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    with open(out_dir / "paired_significance_table_latex.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
