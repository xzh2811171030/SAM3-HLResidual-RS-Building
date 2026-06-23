"""
=============================================================================
cross_city_variance_analysis.py  ---  跨城市地理偏见与性能方差分析
=============================================================================
功能说明:
  1. 读取 WHU-Mix 全量测试集的 GT 标签和两个模型的二值化预测掩膜
  2. 逐图像计算 IoU，按城市 (文件名前缀) 聚合
  3. 输出 Markdown 统计表和顶刊级 1x2 学术可视化图

模型:
  - Model 0: SAM3 Text Zero-shot  (文本提示 "building")
  - Model 2: GBG-SAM3 20-shot     (Ours)

用法:
  python src/cross_city_variance_analysis.py
  python src/cross_city_variance_analysis.py --gt_dir path/to/labels \
      --m0_pred_dir path/to/m0_preds --m2_pred_dir path/to/m2_preds \
      --output_dir results/cross_city_m0m2

依赖: numpy, pandas, matplotlib, seaborn, tqdm, Pillow/tifffile
=============================================================================
"""

# ==========================================================================
# 模块 1: 导入与全局配置
# ==========================================================================
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm

# ── 默认路径（可配置） ──
_default_project = Path(__file__).resolve().parents[1]
# WHU-Mix 全量测试集 (8402 张)
_DEFAULT_FULL_TEST = _default_project / "data" / "raw" / "whu_mix_full_test"
_DEFAULT_GT_DIR = str(_DEFAULT_FULL_TEST / "test" / "label")          # GT 存放位置
_DEFAULT_M0_DIR = str(_DEFAULT_FULL_TEST / "preds" / "sam3_zeroshot") # Model 0 预测
_DEFAULT_M2_DIR = str(_DEFAULT_FULL_TEST / "preds" / "gbg_sam3")      # Model 2 预测
_DEFAULT_OUTPUT_DIR = str(_default_project / "results" / "cross_city")

TARGET_SIZE: int = 512


# ==========================================================================
# 模块 2: 图像读取与二值化
# ==========================================================================
def load_binary_mask(path: str, target_size: int = TARGET_SIZE) -> np.ndarray:
    """加载单张掩膜并转为 bool 数组 (True=建筑, False=背景)。

    支持 .tif (单通道) 和 .png 格式。
    自动处理灰度图 / RGBA 转灰度 / 首通道提取。
    """
    img = Image.open(path)

    # 转为灰度 numpy 数组
    if img.mode in ("RGBA", "RGB", "P"):
        img = img.convert("L")
    arr = np.array(img, dtype=np.float32)

    # 归一化到 [0, 1]
    if arr.max() > 1.0:
        arr = arr / 255.0

    # 调整大小 (如果不是 512x512)
    if arr.shape[0] != target_size or arr.shape[1] != target_size:
        arr_resized = np.array(Image.fromarray((arr * 255).astype(np.uint8)).resize(
            (target_size, target_size), Image.NEAREST), dtype=np.float32) / 255.0
        arr = arr_resized

    # 二值化: >0.5 → True
    return arr > 0.5


def compute_iou(pred_bool: np.ndarray, gt_bool: np.ndarray) -> float:
    """计算两个 bool 数组的 IoU (Jaccard Index)。

    IoU = |P ∩ G| / |P ∪ G|

    边界情况处理:
      - 若 GT 全 0 且 Pred 全 0 → 返回 1.0 (两者一致, 都认为是背景)
      - 若 GT 全 0 但 Pred 有预测 → 返回 0.0 (误报)
      - 若 GT 有建筑但 Pred 全 0 → 返回 0.0 (漏检)
    """
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()

    if union == 0:
        # GT 和 Pred 都是全黑, 视为完美一致
        return 1.0
    return float(intersection) / float(union)


# ==========================================================================
# 模块 3: 逐图像 IoU 计算
# ==========================================================================
def compute_per_image_iou(
    gt_dir: str,
    m0_pred_dir: str,
    m2_pred_dir: str,
) -> pd.DataFrame:
    """遍历 GT 目录, 逐图像计算两个模型的 IoU, 返回 DataFrame。

    列: Filename, City, M0_IoU, M2_IoU
    """
    gt_path = Path(gt_dir)
    m0_path = Path(m0_pred_dir)
    m2_path = Path(m2_pred_dir)

    if not gt_path.exists():
        raise FileNotFoundError(f"GT 目录不存在: {gt_dir}")
    if not m0_path.exists():
        raise FileNotFoundError(f"Model 0 预测目录不存在: {m0_pred_dir}")
    if not m2_path.exists():
        raise FileNotFoundError(f"Model 2 预测目录不存在: {m2_pred_dir}")

    # 收集所有 GT 文件
    gt_files = sorted(gt_path.glob("*.tif"))
    if not gt_files:
        gt_files = sorted(gt_path.glob("*.png"))
    if not gt_files:
        raise FileNotFoundError(f"GT 目录中没有 .tif 或 .png 文件: {gt_dir}")

    print(f"\n  找到 {len(gt_files)} 个 GT 文件")
    print(f"  GT 目录:    {gt_dir}")
    print(f"  M0 Pred 目录: {m0_pred_dir}")
    print(f"  M2 Pred 目录: {m2_pred_dir}")

    records: List[Dict] = []
    skipped_m0: int = 0
    skipped_m2: int = 0

    for gt_file in tqdm(gt_files, desc="  逐图像计算 IoU", unit="img", ncols=100):
        fname = gt_file.name

        # ── 提取城市名 (如 "Christchurch_001.tif" → "Christchurch") ──
        # 部分文件可能有多级前缀, 取第一个 '_' 之前的部分
        city = fname.split("_")[0]

        # ── 加载 GT ──
        try:
            gt_mask = load_binary_mask(str(gt_file))
        except Exception as e:
            tqdm.write(f"  [警告] GT 读取失败: {fname}: {e}")
            continue

        # ── 加载 M0 预测 ──
        m0_iou = np.nan
        m0_file = m0_path / fname
        if not m0_file.exists():
            # 尝试 .png 扩展名
            m0_file = m0_path / (gt_file.stem + ".png")
        if m0_file.exists():
            try:
                m0_mask = load_binary_mask(str(m0_file))
                m0_iou = compute_iou(m0_mask, gt_mask)
            except Exception as e:
                tqdm.write(f"  [警告] M0 预测读取失败: {fname}: {e}")
                skipped_m0 += 1
        else:
            skipped_m0 += 1

        # ── 加载 M2 预测 ──
        m2_iou = np.nan
        m2_file = m2_path / fname
        if not m2_file.exists():
            m2_file = m2_path / (gt_file.stem + ".png")
        if m2_file.exists():
            try:
                m2_mask = load_binary_mask(str(m2_file))
                m2_iou = compute_iou(m2_mask, gt_mask)
            except Exception as e:
                tqdm.write(f"  [警告] M2 预测读取失败: {fname}: {e}")
                skipped_m2 += 1
        else:
            skipped_m2 += 1

        records.append({
            "Filename": fname,
            "City": city,
            "M0_IoU": m0_iou,
            "M2_IoU": m2_iou,
        })

    df = pd.DataFrame(records)

    if skipped_m0 > 0:
        print(f"\n  M0 预测缺失: {skipped_m0}/{len(gt_files)} 张")
    if skipped_m2 > 0:
        print(f"  M2 预测缺失: {skipped_m2}/{len(gt_files)} 张")
    print(f"  有效记录: {len(df)} 条\n")

    return df


# ==========================================================================
# 模块 4: 统计聚合与 Markdown 表格
# ==========================================================================
def aggregate_by_city(df: pd.DataFrame) -> pd.DataFrame:
    """按 City 聚合, 计算 M0/M2 的 Mean, Variance, Std。

    返回 DataFrame (index=City, columns=M0_Mean, M0_Std, M2_Mean, M2_Std, ...)
    """
    agg = df.groupby("City").agg(
        M0_Mean=("M0_IoU", lambda x: np.nanmean(x)),
        M0_Std=("M0_IoU", lambda x: np.nanstd(x, ddof=1)),
        M0_Var=("M0_IoU", lambda x: np.nanvar(x, ddof=1)),
        M2_Mean=("M2_IoU", lambda x: np.nanmean(x)),
        M2_Std=("M2_IoU", lambda x: np.nanstd(x, ddof=1)),
        M2_Var=("M2_IoU", lambda x: np.nanvar(x, ddof=1)),
        N=("M0_IoU", "count"),
    ).reset_index()
    # 按 M2 Mean 降序排列 (让表现好的城市在前)
    agg = agg.sort_values("M2_Mean", ascending=False).reset_index(drop=True)
    return agg


def print_markdown_table(agg_df: pd.DataFrame, output_path: Optional[str] = None) -> str:
    """打印并可选保存 Markdown 格式的统计对比表。

    返回 Markdown 字符串。
    """
    lines: List[str] = []
    lines.append("## 跨城市性能对比 (Cross-City Performance Comparison)")
    lines.append("")
    lines.append("")
    lines.append("| City | N | M0 Mean IoU | M0 Std | M2 Mean IoU | M2 Std | Δ Mean |")
    lines.append("|------|---|-------------|--------|-------------|--------|--------|")

    for _, row in agg_df.iterrows():
        city = row["City"]
        n = int(row["N"])
        m0_m = row["M0_Mean"]
        m0_s = row["M0_Std"]
        m2_m = row["M2_Mean"]
        m2_s = row["M2_Std"]
        delta = m2_m - m0_m

        lines.append(
            f"| {city} | {n} | {m0_m:.4f} | {m0_s:.4f} | "
            f"{m2_m:.4f} | {m2_s:.4f} | {delta:+.4f} |"
        )

    # ── 全局行 ──
    all_m0_mean = np.nanmean([r["M0_Mean"] for _, r in agg_df.iterrows()])
    all_m0_std = np.nanmean([r["M0_Std"] for _, r in agg_df.iterrows()])
    all_m2_mean = np.nanmean([r["M2_Mean"] for _, r in agg_df.iterrows()])
    all_m2_std = np.nanmean([r["M2_Std"] for _, r in agg_df.iterrows()])
    total_n = sum(int(r["N"]) for _, r in agg_df.iterrows())

    lines.append(
        f"| **All Cities** | {total_n} | {all_m0_mean:.4f} | {all_m0_std:.4f} | "
        f"{all_m2_mean:.4f} | {all_m2_std:.4f} | {all_m2_mean - all_m0_mean:+.4f} |"
    )

    md = "\n".join(lines)

    print(f"\n{md}\n")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"  Markdown 表格已保存: {output_path}")

    return md


# ==========================================================================
# 模块 5: 学术可视化 (1x2 双子图)
# ==========================================================================
def plot_cross_city_variance(
    df: pd.DataFrame,
    agg_df: pd.DataFrame,
    sigmas: int,
    output_path: str,
) -> None:
    """绘制跨城市方差分析的 1x2 双子图。

    左图: 分组条形图 (Mean IoU + Error Bar)
    右图: 小提琴图 (Per-image IoU 分布)
    """
    sns.set_style("whitegrid")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    # ── 按 M2 Mean 排序, 取前 12 个城市 ──
    cities_sorted = agg_df.sort_values("M2_Mean", ascending=False)["City"].tolist()
    # 如果城市太多(超过15个), 只显示前15个避免拥挤, 其余归入 "Others"
    if len(cities_sorted) > 15:
        display_cities = cities_sorted[:15]
    else:
        display_cities = cities_sorted

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.0))

    # ================================================================
    # 左图: 分组条形图 (Mean IoU ± Std)
    # ================================================================
    n_cities = len(display_cities)
    x = np.arange(n_cities)
    bar_width = 0.35

    m0_means = []
    m0_stds = []
    m2_means = []
    m2_stds = []

    city_lookup = dict(zip(agg_df["City"], agg_df.index))
    for city in display_cities:
        if city in city_lookup:
            row = agg_df.iloc[city_lookup[city]]
            m0_means.append(row["M0_Mean"] * 100.0)
            m0_stds.append(row["M0_Std"] * 100.0)
            m2_means.append(row["M2_Mean"] * 100.0)
            m2_stds.append(row["M2_Std"] * 100.0)
        else:
            m0_means.append(0)
            m0_stds.append(0)
            m2_means.append(0)
            m2_stds.append(0)

    bars_m0 = ax1.bar(
        x - bar_width / 2, m0_means, bar_width,
        yerr=m0_stds, capsize=3,
        color="#2196F3", alpha=0.85, edgecolor="white", linewidth=0.5,
        label="Model 0: SAM3 Zero-shot",
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )
    bars_m2 = ax1.bar(
        x + bar_width / 2, m2_means, bar_width,
        yerr=m2_stds, capsize=3,
        color="#E91E63", alpha=0.85, edgecolor="white", linewidth=0.5,
        label="Model 2: GBG-SAM3 (Ours)",
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(display_cities, rotation=35, ha="right", fontsize=10)
    ax1.set_ylabel("Mean IoU (%)")
    ax1.set_title("Per-City Mean IoU (± Std)", fontweight="bold", pad=10)
    ax1.legend(loc="lower right", framealpha=0.9, edgecolor="gray")
    ax1.set_ylim(bottom=0)

    # 在 M2 的柱子上方标注具体数值
    for bar, val in zip(bars_m2, m2_means):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f"{val:.1f}", ha="center", va="bottom", fontsize=7,
            color="#E91E63", fontweight="bold",
        )

    # ================================================================
    # 右图: 小提琴图 (Per-image IoU 分布)
    # ================================================================
    # 筛选出 display_cities 中的行
    df_display = df[df["City"].isin(display_cities)].copy()
    # 将 City 转为有序 categorical, 维持排序
    df_display["City"] = pd.Categorical(
        df_display["City"], categories=display_cities, ordered=True,
    )

    # Melt 为长格式以便 seaborn hue
    df_melt = df_display.melt(
        id_vars=["Filename", "City"],
        value_vars=["M0_IoU", "M2_IoU"],
        var_name="Model",
        value_name="IoU",
    )
    df_melt["IoU_pct"] = df_melt["IoU"] * 100.0
    df_melt["Model"] = df_melt["Model"].replace({
        "M0_IoU": "Model 0: SAM3 Zero-shot",
        "M2_IoU": "Model 2: GBG-SAM3",
    })

    palette = {
        "Model 0: SAM3 Zero-shot": "#2196F3",
        "Model 2: GBG-SAM3": "#E91E63",
    }

    sns.violinplot(
        data=df_melt, x="City", y="IoU_pct", hue="Model",
        palette=palette, split=False, inner="quartile",
        linewidth=0.8, cut=0, ax=ax2,
        density_norm="width",  # 每个 violin 宽度归一化, 便于比较
    )

    ax2.set_xticklabels(display_cities, rotation=35, ha="right", fontsize=10)
    ax2.set_ylabel("Per-Image IoU (%)")
    ax2.set_title("Per-Image IoU Distribution", fontweight="bold", pad=10)
    ax2.legend(loc="lower right", framealpha=0.9, edgecolor="gray", fontsize=10)

    # ── 全局标题 ──
    fig.suptitle(
        "Cross-City Geographic Bias & Variance Analysis",
        fontweight="bold", fontsize=17, y=1.02,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  可视化图表已保存: {output_path}")


# ==========================================================================
# 模块 6: 命令行参数解析
# ==========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="跨城市地理偏见与性能方差分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/cross_city_variance_analysis.py
  python src/cross_city_variance_analysis.py \
      --gt_dir data/raw/whu_mix_full_test/test/label \
      --m0_pred_dir data/raw/whu_mix_full_test/preds/sam3_zeroshot \
      --m2_pred_dir data/raw/whu_mix_full_test/preds/gbg_sam3 \
      --output_dir results/cross_city
        """,
    )
    parser.add_argument(
        "--gt_dir", type=str, default=_DEFAULT_GT_DIR,
        help=f"GT 标签目录 (默认: {_DEFAULT_GT_DIR})",
    )
    parser.add_argument(
        "--m0_pred_dir", type=str, default=_DEFAULT_M0_DIR,
        help=f"Model 0 预测掩膜目录 (默认: {_DEFAULT_M0_DIR})",
    )
    parser.add_argument(
        "--m2_pred_dir", type=str, default=_DEFAULT_M2_DIR,
        help=f"Model 2 预测掩膜目录 (默认: {_DEFAULT_M2_DIR})",
    )
    parser.add_argument(
        "--output_dir", type=str, default=_DEFAULT_OUTPUT_DIR,
        help=f"输出目录 (默认: {_DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


# ==========================================================================
# 模块 7: 主函数
# ==========================================================================
def main() -> None:
    args = parse_args()

    print(f"\n{'='*70}")
    print(f"  跨城市地理偏见 & 性能方差分析")
    print(f"  GT 目录:    {args.gt_dir}")
    print(f"  M0 Pred 目录: {args.m0_pred_dir}")
    print(f"  M2 Pred 目录: {args.m2_pred_dir}")
    print(f"  输出目录:   {args.output_dir}")
    print(f"{'='*70}")

    # ── 1) 逐图像计算 IoU ──
    df = compute_per_image_iou(
        gt_dir=args.gt_dir,
        m0_pred_dir=args.m0_pred_dir,
        m2_pred_dir=args.m2_pred_dir,
    )

    # ── 2) 聚合统计 ──
    agg_df = aggregate_by_city(df)
    cities_found = agg_df["City"].tolist()
    print(f"  识别到的城市 ({len(cities_found)} 个): {', '.join(cities_found)}")

    # ── 3) Markdown 表格 ──
    md_path = str(Path(args.output_dir) / "cross_city_stats.md")
    print_markdown_table(agg_df, output_path=md_path)

    # ── 4) 学术可视化 ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = str(Path(args.output_dir) / f"cross_city_variance_{timestamp}.png")
    plot_cross_city_variance(df, agg_df, sigmas=len(cities_found), output_path=png_path)

    # ── 5) 保存完整 IoU DataFrame 到 CSV ──
    csv_path = str(Path(args.output_dir) / f"per_image_iou_{timestamp}.csv")
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  逐图像 IoU CSV 已保存: {csv_path}")

    # ── 6) 快捷链接 ──
    symlink_path = str(Path(args.output_dir) / "cross_city_variance.png")
    if Path(symlink_path).exists() or Path(symlink_path).is_symlink():
        Path(symlink_path).unlink(missing_ok=True)
    try:
        Path(symlink_path).symlink_to(Path(png_path).name)
    except OSError:
        import shutil
        shutil.copy2(png_path, symlink_path)

    print(f"\n{'='*70}")
    print(f"  分析全部完成!")
    print(f"  图表:      {png_path}")
    print(f"  表格:      {md_path}")
    print(f"  逐图像 IoU: {csv_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
