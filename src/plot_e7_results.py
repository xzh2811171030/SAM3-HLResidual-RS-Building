"""
=============================================================================
plot_e7_results.py  ---  E7 鲁棒性曲线独立可视化工具
=============================================================================
功能说明:
  1. 读取 run_e7_robustness(_revised).py 生成的 JSON 结果文件
  2. 支持同时加载多个 JSON 文件，按模型名合并结果
  3. 生成与原脚本一致的 1x2 双子图 (mIoU + Boundary IoU)
  4. 可选输出合并后的 JSON 文件

用法:
  # 基本用法: 喂入一个结果 JSON
  python src/plot_e7_results.py --json results/merged_e7_results.json

  # 合并两个 JSON (如分开跑的 Model 1 + Model 2)
  python src/plot_e7_results.py \
      --json results/robustness_metrics_m1.json \
      --json results/robustness_metrics_m2.json

  # 指定输出路径
  python src/plot_e7_results.py \
      --json results/merged_e7_results.json \
      --output results/e7_curve_merged.png

  # 同时输出合并后的 JSON
  python src/plot_e7_results.py \
      --json results/merged_e7_results.json \
      --output_json results/merged_e7_results.json
=============================================================================
"""

# ==========================================================================
# 模块 1: 导入与全局配置
# ==========================================================================
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 颜色/标记/线型配置（与原脚本完全一致）
MODEL_COLORS: Dict[str, str] = {
    "E5-FT (Tgt LoRA)": "#4CAF50",
    "GBG-SAM3 (Ours)": "#E91E63",
}

MODEL_MARKERS: Dict[str, str] = {
    "E5-FT (Tgt LoRA)": "^",
    "GBG-SAM3 (Ours)": "*",
}

MODEL_LINESTYLES: Dict[str, str] = {
    "E5-FT (Tgt LoRA)": "-",
    "GBG-SAM3 (Ours)": "-",
}

# 模型在图中出现的顺序
MODEL_ORDER: List[str] = [
    "E5-FT (Tgt LoRA)",
    "GBG-SAM3 (Ours)",
]


# ==========================================================================
# 模块 2: JSON 加载与合并
# ==========================================================================
def load_json_results(json_path: str) -> Dict:
    """加载单个 JSON 文件，返回 {'results': ..., 'sigmas': ..., ...}"""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {json_path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results_raw = data.get("results", {})
    # JSON key 是字符串 (如 "0", "2", "5")，转回 int
    results_int = {}
    for model_name, sigma_dict in results_raw.items():
        results_int[model_name] = {}
        for sigma_str, metrics in sigma_dict.items():
            results_int[model_name][int(sigma_str) if sigma_str.lstrip("-").isdigit() else sigma_str] = metrics

    return {
        "results": results_int,
        "metadata": data.get("metadata", {}),
    }


def merge_results(json_paths: List[str]) -> Tuple[Dict, List[int]]:
    """合并多个 JSON 的结果，返回 (merged_results, sorted_sigmas)"""
    merged: Dict[str, Dict[int, Dict[str, float]]] = {}
    all_sigmas: set = set()

    for jp in json_paths:
        loaded = load_json_results(jp)
        for model_name, sigma_dict in loaded["results"].items():
            if model_name not in merged:
                merged[model_name] = {}
            for sigma, metrics in sigma_dict.items():
                merged[model_name][sigma] = metrics
                all_sigmas.add(sigma)

    sorted_sigmas = sorted(all_sigmas)
    print(f"\n  已加载 {len(json_paths)} 个 JSON 文件")
    print(f"  模型: {list(merged.keys())}")
    print(f"  Sigma 级别: {sorted_sigmas}")

    return merged, sorted_sigmas


# ==========================================================================
# 模块 3: 学术制图 (1x2 双子图)
# ==========================================================================
def plot_robustness_curve(
    results: Dict[str, Dict[int, Dict[str, float]]],
    sigmas: List[int],
    output_path: str,
) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "legend.fontsize": 10,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    # 按 MODEL_ORDER 排序，未在 ORDER 中的模型追加末尾
    ordered = [m for m in MODEL_ORDER if m in results]
    for m in results:
        if m not in ordered:
            ordered.append(m)

    # === 1x2 双子图: 左=mIoU, 右=Boundary_IoU ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

    for name in ordered:
        model_data = results.get(name, {})
        x_vals: List[int] = []
        y_miou: List[float] = []
        y_biou: List[float] = []

        for sigma in sigmas:
            if sigma in model_data:
                x_vals.append(sigma)
                y_miou.append(model_data[sigma]["mIoU"] * 100.0)
                y_biou.append(model_data[sigma]["Boundary_IoU"] * 100.0)

        if not x_vals:
            print(f"  [跳过] {name}: 无数据")
            continue

        color = MODEL_COLORS.get(name, "#000000")
        marker = MODEL_MARKERS.get(name, "o")
        linestyle = MODEL_LINESTYLES.get(name, "-")
        marker_kw = dict(
            color=color, marker=marker, linestyle=linestyle,
            linewidth=2.2, markersize=9, markeredgewidth=1.2,
            markeredgecolor="white" if marker != "*" else color,
            label=name, zorder=3,
        )

        # --- 左子图: mIoU ---
        ax1.plot(x_vals, y_miou, **marker_kw)
        for x, y in zip(x_vals, y_miou):
            ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=7, color=color, alpha=0.85)

        # --- 右子图: Boundary IoU ---
        ax2.plot(x_vals, y_biou, **marker_kw)
        for x, y in zip(x_vals, y_biou):
            ax2.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=7, color=color, alpha=0.85)

    # === 左子图设置 ===
    ax1.set_xlabel("Noise Intensity $\\sigma$ (pixels)")
    ax1.set_ylabel("Test mIoU (%)")
    ax1.set_xticks(sigmas)
    ax1.set_xticklabels([str(s) for s in sigmas])
    ax1.set_xlim(sigmas[0] - 0.5, sigmas[-1] + 1.5)
    ax1.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    ax1.set_title("Robustness: mIoU", fontweight="bold", pad=10)

    # === 右子图设置 ===
    ax2.set_xlabel("Noise Intensity $\\sigma$ (pixels)")
    ax2.set_ylabel("Test Boundary IoU (%)")
    ax2.set_xticks(sigmas)
    ax2.set_xticklabels([str(s) for s in sigmas])
    ax2.set_xlim(sigmas[0] - 0.5, sigmas[-1] + 1.5)
    ax2.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    ax2.set_title("Robustness: Boundary IoU (UG-DP d=5)", fontweight="bold", pad=10)

    # === 共享图例 (置于 Figure 底部) ===
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=2,
        framealpha=0.92,
        edgecolor="gray",
        fancybox=True,
        shadow=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Robustness to Noise: Box Jitter Perturbation",
        fontweight="bold", fontsize=17, y=1.02,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"\n  鲁棒性曲线已保存: {output_path}")


# ==========================================================================
# 模块 4: 命令行参数解析
# ==========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E7 鲁棒性曲线独立可视化工具（无需重跑评估）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/plot_e7_results.py --json results/robustness_metrics_slim_20250101.json
  python src/plot_e7_results.py --json m1.json --json m2.json
  python src/plot_e7_results.py --json results/robustness_metrics.json --output my_curve.png
        """,
    )
    parser.add_argument(
        "--json", type=str, action="append", required=True,
        dest="json_paths",
        help="结果 JSON 文件路径, 可重复使用以合并多个文件 (如 --json m1.json --json m2.json)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出 PNG 路径 (默认: 与第一个输入 JSON 同目录的 robustness_curve.png)",
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="合并后的 JSON 输出路径 (可选)",
    )
    return parser.parse_args()


# ==========================================================================
# 模块 5: 主函数
# ==========================================================================
def main() -> None:
    args = parse_args()

    # 1) 加载并合并 JSON
    merged_results, sigmas = merge_results(args.json_paths)

    # 2) 确定输出路径
    if args.output:
        output_png = args.output
    else:
        first_dir = Path(args.json_paths[0]).parent
        output_png = str(first_dir / "robustness_curve.png")

    # 3) 绘制曲线
    plot_robustness_curve(merged_results, sigmas, output_png)

    # 4) 可选输出合并 JSON
    if args.output_json:
        from datetime import datetime
        output_json = {
            "experiment": "E7_Robustness_Analysis_Merged",
            "metadata": {
                "source_files": args.json_paths,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "results": {
                model: {str(s): m for s, m in sigma_dict.items()}
                for model, sigma_dict in merged_results.items()
            },
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)
        print(f"  合并后的 JSON 已保存: {args.output_json}")

    print(f"\n  完成!")


if __name__ == "__main__":
    main()
