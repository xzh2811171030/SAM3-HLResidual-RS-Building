# -*- coding: utf-8 -*-
"""
make_paper_tables.py

把当前论文需要的核心结果汇总为 markdown/csv。
需要你手动填部分旧 E1-E5 数字；新 E7/E9 可从 JSON 读取。
"""

import argparse
import csv
import json
from pathlib import Path


def read_json(path):
    if not path or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_md(path, title, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\nNo rows.\n", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("|" + "|".join(["---"] * len(fields)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(f, "")) for f in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e7a_json", type=str, default="")
    parser.add_argument("--e7b_json", type=str, default="")
    parser.add_argument("--e8_json", type=str, default="")
    parser.add_argument("--e9_json", type=str, default="")
    parser.add_argument("--foundation_json", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="paper_tables")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Table 1: old + final main results
    main_rows = [
        {"Exp": "E1", "Method": "UNet", "Test": "Target full 8402", "mIoU (%)": "34.71", "F1 (%)": "-", "BIoU (%)": "-", "Note": "traditional baseline"},
        {"Exp": "E1", "Method": "SegFormer", "Test": "Target full 8402", "mIoU (%)": "46.82", "F1 (%)": "-", "BIoU (%)": "-", "Note": "traditional baseline"},
        {"Exp": "E4", "Method": "Source-only LoRA", "Test": "Target full 8402", "mIoU (%)": "61.04±1.15", "F1 (%)": "-", "BIoU (%)": "-", "Note": "domain shift"},
        {"Exp": "E5", "Method": "Prompt-free LoRA", "Test": "Target full 8402", "mIoU (%)": "73.99", "F1 (%)": "填full JSON", "BIoU (%)": "43.74", "Note": "20-shot baseline"},
        {"Exp": "E6", "Method": "HL-Residual", "Test": "Target full 8402", "mIoU (%)": "75.08", "F1 (%)": "填full JSON", "BIoU (%)": "46.26", "Note": "ours"},
    ]

    write_csv(out_dir / "table_main_results.csv", main_rows)
    write_md(out_dir / "table_main_results.md", "Main Results", main_rows)

    # Low-shot sensitivity
    shot_rows = [
        {"Shot": "5", "Test": "pilot500", "Method": "Prompt-free LoRA", "mIoU (%)": "62.45", "F1 (%)": "71.36", "BIoU (%)": "32.64"},
        {"Shot": "5", "Test": "pilot500", "Method": "HL-Residual", "mIoU (%)": "62.17", "F1 (%)": "71.10", "BIoU (%)": "32.25"},
        {"Shot": "10", "Test": "pilot500", "Method": "Prompt-free LoRA", "mIoU (%)": "67.30", "F1 (%)": "75.93", "BIoU (%)": "36.16"},
        {"Shot": "10", "Test": "pilot500", "Method": "HL-Residual", "mIoU (%)": "67.05", "F1 (%)": "75.69", "BIoU (%)": "35.91"},
        {"Shot": "20", "Test": "full8402", "Method": "Prompt-free LoRA", "mIoU (%)": "73.99", "F1 (%)": "填full JSON", "BIoU (%)": "43.74"},
        {"Shot": "20", "Test": "full8402", "Method": "HL-Residual", "mIoU (%)": "75.08", "F1 (%)": "填full JSON", "BIoU (%)": "46.26"},
    ]
    write_csv(out_dir / "table_lowshot_sensitivity.csv", shot_rows)
    write_md(out_dir / "table_lowshot_sensitivity.md", "Low-shot Sensitivity", shot_rows)

    # E9
    e9 = read_json(args.e9_json)
    if e9:
        e9_rows = []
        for r in e9.get("timing_summary", []):
            e9_rows.append({
                "Method": r["model"],
                "TTA": r["use_tta"],
                "ms/img": f"{r['avg_ms_per_image_mean']:.2f}",
                "FPS": f"{r['fps_mean']:.2f}",
            })
        write_csv(out_dir / "table_e9_timing.csv", e9_rows)
        write_md(out_dir / "table_e9_timing.md", "E9 Timing", e9_rows)

        param_rows = []
        for r in e9.get("params_summary", []):
            param_rows.append({
                "Method": r["model"],
                "Trainable Params (M)": f"{r['trainable_params_estimated_mean']/1e6:.2f}",
                "Extra Residual Params (M)": f"{r['extra_params_vs_lora_mean']/1e6:.2f}",
                "Trainable Ratio (%)": f"{r['trainable_ratio_percent_mean']:.3f}",
            })
        write_csv(out_dir / "table_e9_params.csv", param_rows)
        write_md(out_dir / "table_e9_params.md", "E9 Parameters", param_rows)

    print(f"Saved paper tables to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()