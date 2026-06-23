# -*- coding: utf-8 -*-
"""
merge_e7_json_parts.py
=============================================================================
合并 E7a / E7b 分段 JSON。

可用于：
  1. E7a: box+LoRA 文件 + HL-Residual 文件
  2. E7b: 前半部分文件 + 后半部分 HL 文件 + 后半部分 LoRA 补跑文件

运行示例：
  python src/merge_e7_json_parts.py \
    --inputs results/e7a_prompt_drift_pilot500/e7a_prompt_drift_metrics.json,results/e7a_prompt_drift_residual/e7a_prompt_drift_metrics.json \
    --out results/e7a_prompt_drift_merged/e7a_merged_metrics.json
=============================================================================
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def deep_merge_results(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并 results:
      E7a 格式: results[model][sigma]
      E7b 格式: results[corruption][model]
    两种都可以递归合并。
    """
    for k, v in src.items():
        if k not in dst:
            dst[k] = v
        else:
            if isinstance(dst[k], dict) and isinstance(v, dict):
                deep_merge_results(dst[k], v)
            else:
                dst[k] = v
    return dst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, required=True, help="多个 json 路径，用逗号分隔")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    input_paths = [Path(x.strip()) for x in args.inputs.split(",") if x.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    merged = {
        "experiment": "merged_e7_results",
        "metadata": {
            "source_files": [str(p) for p in input_paths],
        },
        "results": {},
        "raw_per_seed": {},
    }

    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(p)

        data = json.loads(p.read_text(encoding="utf-8"))

        merged["metadata"][p.name] = data.get("metadata", {})
        merged["results"] = deep_merge_results(
            merged["results"],
            data.get("results", {}),
        )
        merged["raw_per_seed"] = deep_merge_results(
            merged["raw_per_seed"],
            data.get("raw_per_seed", {}),
        )

    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved merged json: {out_path}")

    print("\nQuick summary:")
    for k, v in merged["results"].items():
        print(f"- {k}: {list(v.keys())[:10]}")


if __name__ == "__main__":
    main()