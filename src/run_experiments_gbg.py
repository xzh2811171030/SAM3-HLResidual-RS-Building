"""
=============================================================================
run_experiments_gbg.py  ---  GBG-SAM3 E6 实验总控脚本
=============================================================================
功能说明:
  1. 按顺序运行 GBG-SAM3 消融实验矩阵:
     - few-shot 规模: 5-shot / 10-shot / 20-shot / full_data
     - 域: source (源域) / target (目标域)
     - 随机种子: 42 / 123 / 456
     每个 (num_shots, domain) 组合在 3 个种子下各运行一次

  2. 支持 "双轨制" 测试:
     - 瘦身测试集 (默认)
     - 全量测试集 (--full_test)

  3. 自动聚合 Mean ± Std, 输出标准 + UG-DP 增强的 mIoU / F1 / Boundary IoU

用法:
  # 源域 5-shot 单种子
  python src/run_experiments_gbg.py --domain source --shots 5 --seeds 42

  # 目标域全量数据
  python src/run_experiments_gbg.py --domain target --full_data

  # 全量测试集
  python src/run_experiments_gbg.py --domain source --shots 10 --full_test

  # 仅打印实验矩阵
  python src/run_experiments_gbg.py --domain source --shots 5 --dry-run

命令行参数:
  --domain    域: source 或 target, 默认 target
  --full_test 使用全量 8402 张测试集
  --full_data 全量训练数据 (num_shots=None)
  --shots     指定 few-shot 规模, 逗号分隔: 5,10,20
  --seeds     指定随机种子, 逗号分隔: 42,123,456
  --dry-run   仅打印实验矩阵, 不实际运行
=============================================================================
"""

# ==========================================================================
# 模块 1: 导入与云端路径配置
# ==========================================================================
import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.cloud_paths import get_paths, get_platform_name

_paths = get_paths()

WEIGHTS_DIR: str = _paths["weights_dir"]
RESULTS_DIR: str = _paths.get("results_dir", str(Path(_paths["project_root"]) / "results"))

ALL_SHOTS: List[int] = [5, 10, 20]
ALL_SEEDS: List[int] = [42, 123, 456]

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================================================
# 模块 2: 实验矩阵生成
# ==========================================================================
def build_gbg_matrix(
    shots: Optional[List[Optional[int]]] = None,
    domains: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
) -> List[Dict]:
    shots = shots if shots is not None else ALL_SHOTS
    domains = domains or ["target"]
    seeds = seeds or ALL_SEEDS

    matrix: List[Dict] = []
    for domain in domains:
        for shot in shots:
            for seed in seeds:
                domain_short = "src" if domain == "source" else "tgt"
                shot_suffix = "full" if shot is None else f"{shot}shot"
                weight_name = f"gbg_{domain_short}_{shot_suffix}_seed{seed}.pth"
                weight_path = str(Path(WEIGHTS_DIR) / weight_name)
                matrix.append({
                    "domain": domain,
                    "num_shots": shot,
                    "seed": seed,
                    "weight_save_path": weight_path,
                })
    return matrix


# ==========================================================================
# 模块 3: 统计聚合函数
# ==========================================================================
def aggregate_gbg_results(
    all_results: List[Dict],
    shots: List[Optional[int]],
    domains: List[str],
) -> Dict:
    summary: Dict = {}

    for domain in domains:
        for shot in shots:
            shot_suffix = "full" if shot is None else f"{shot}shot"
            key = f"{domain}_{shot_suffix}"
            group = [
                r for r in all_results
                if r.get("domain") == domain
                and r["num_shots"] == shot
            ]
            if not group:
                continue

            for metric_name in ["ugdp_mIoU", "ugdp_F1", "ugdp_Boundary_IoU",
                                "std_mIoU", "std_F1", "std_Boundary_IoU"]:
                values = np.array([r.get(metric_name, 0.0) for r in group])
                summary[f"{key}__{metric_name}"] = {
                    "domain": domain,
                    "num_shots": shot,
                    "metric": metric_name,
                    "num_runs": len(group),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }

    return summary


# ==========================================================================
# 模块 4: 表格输出
# ==========================================================================
def print_gbg_table(matrix: List[Dict]) -> None:
    print(f"\n{'='*90}")
    print(f"  GBG-SAM3 E6 实验矩阵 ({len(matrix)} 次独立训练)")
    print(f"{'='*90}")
    print(f"  {'#':<4} {'Domain':<10} {'Few-Shot':<10} {'Seed':<6} {'权重文件'}")
    print(f"  {'-'*4} {'-'*10} {'-'*10} {'-'*6} {'-'*45}")
    for i, exp in enumerate(matrix, 1):
        domain_label = "source" if exp["domain"] == "source" else "target"
        shot_label = "full" if exp["num_shots"] is None else f"{exp['num_shots']}-shot"
        print(f"  {i:<4} {domain_label:<10} {shot_label:<10} {exp['seed']:<6} "
              f"{Path(exp['weight_save_path']).name}")
    print(f"{'='*90}")


def print_gbg_summary(summary: Dict, domains: List[str], shots: List[Optional[int]]) -> None:
    print(f"\n{'='*110}")
    print(f"  GBG-SAM3 消融实验汇总 (Mean ± Std)")
    print(f"  平台: {get_platform_name()}  |  设备: {DEVICE}")
    print(f"{'='*110}")

    for domain in domains:
        print(f"\n  --- 域: {domain} ---")
        header = (
            f"  {'':<6}"
            f"{'UG-DP mIoU':<28}"
            f"{'UG-DP F1':<28}"
            f"{'UG-DP BIoU':<28}"
        )
        print(header)
        print(f"  {'':<6}{'-'*28}{'-'*28}{'-'*28}")

        for shot in shots:
            shot_label = "full" if shot is None else f"{shot}-shot"
            shot_suffix = "full" if shot is None else f"{shot}shot"

            miou_str = _fmt_metric(summary, f"{domain}_{shot_suffix}__ugdp_mIoU")
            f1_str = _fmt_metric(summary, f"{domain}_{shot_suffix}__ugdp_F1")
            biou_str = _fmt_metric(summary, f"{domain}_{shot_suffix}__ugdp_Boundary_IoU")
            print(f"  {shot_label:<6}{miou_str:<28}{f1_str:<28}{biou_str:<28}")

    print(f"{'='*110}")


def _fmt_metric(summary: Dict, key: str) -> str:
    if key in summary:
        s = summary[key]
        return f"{s['mean']*100:.2f} ± {s['std']*100:.2f}"
    return "-"


# ==========================================================================
# 模块 5: 结果持久化
# ==========================================================================
def save_gbg_results_json(
    all_results: List[Dict],
    summary: Dict,
    results_dir: str,
    full_test: bool,
) -> str:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_label = "full" if full_test else "slim"
    filename = f"gbg_results_{test_label}_{timestamp}.json"
    filepath = str(Path(results_dir) / filename)

    output = {
        "experiment": "E6_GBG_SAM3",
        "metadata": {
            "platform": get_platform_name(),
            "device": DEVICE,
            "timestamp": timestamp,
            "full_test": full_test,
            "model": "GBG-SAM3 (LoRA + GatedBoundaryAdapter + EU-Head + UG-DP)",
            "batch_size_desc": "train: auto(2/4/16), eval: 32",
            "num_epochs": 30,
            "num_workers": 8,
            "lora_rank": 8,
            "ugdp_d": 5,
            "loss_weights": "seg=1.0, unc=1.0, reg=0.1, gate_norm=0.01",
        },
        "individual_results": all_results,
        "aggregated_summary": summary,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  结果已保存: {filepath}")
    return filepath


# ==========================================================================
# 模块 6: 命令行参数解析
# ==========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GBG-SAM3 E6 消融实验总控",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/run_experiments_gbg.py --domain source --shots 5 --seeds 42
  python src/run_experiments_gbg.py --domain target --full_data
  python src/run_experiments_gbg.py --domain source --shots 10 --full_test
  python src/run_experiments_gbg.py --domain source --shots 5 --dry-run
        """,
    )
    parser.add_argument(
        "--domain", type=str, default="target",
        choices=["source", "target"],
        help="域: source (源域) 或 target (目标域), 默认: target",
    )
    parser.add_argument(
        "--full_test", action="store_true",
        help="使用全量 8402 张测试集",
    )
    parser.add_argument(
        "--full_data", action="store_true",
        help="全量训练数据 (num_shots=None, 不采样)",
    )
    parser.add_argument(
        "--shots", type=str, default=None,
        help="few-shot 规模, 逗号分隔: 5,10,20 (默认全量)",
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="随机种子, 逗号分隔: 42,123,456 (默认全量)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅打印实验矩阵, 不运行",
    )
    return parser.parse_args()


# ==========================================================================
# 模块 7: 主函数
# ==========================================================================
def main() -> None:
    args = parse_args()

    if args.full_data:
        shots: Optional[List[Optional[int]]] = [None]
        print("\n  [全量数据模式] 将使用训练目录下全部影像")
    else:
        shots = [int(s.strip()) for s in args.shots.split(",")] if args.shots else None

    seeds = [int(s.strip()) for s in args.seeds.split(",")] if args.seeds else None
    domains = [args.domain]

    if shots is not None and not args.full_data:
        invalid_shots = [s for s in shots if s not in ALL_SHOTS]
        if invalid_shots:
            print(f"错误: 无效 shots 值 {invalid_shots}, 可选: {ALL_SHOTS}")
            sys.exit(1)
    if seeds is not None:
        invalid_seeds = [s for s in seeds if s not in ALL_SEEDS]
        if invalid_seeds:
            print(f"错误: 无效 seeds 值 {invalid_seeds}, 可选: {ALL_SEEDS}")
            sys.exit(1)

    matrix = build_gbg_matrix(shots, domains, seeds)

    test_label = "全量8402" if args.full_test else "瘦身测试"
    domain_label = "源域(source_whu)" if args.domain == "source" else "目标域(target_whu_mix)"

    print(f"\n{'='*60}")
    print(f"  GBG-SAM3 E6 消融实验总控")
    print(f"  平台: {get_platform_name()}  |  设备: {DEVICE}")
    print(f"  域: {domain_label}  |  测试: {test_label}")
    print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    if not torch.cuda.is_available():
        print("\n  警告: 未检测到 CUDA GPU")

    print_gbg_table(matrix)

    if args.dry_run:
        print("\n  [DRY-RUN 模式] 实验矩阵已打印, 不执行实际训练.")
        return

    from training.experiment_runner_gbg import run_single_experiment_gbg

    Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []
    total = len(matrix)
    start_time = datetime.now()

    for idx, exp in enumerate(matrix, 1):
        domain_label_cur = "源域" if exp["domain"] == "source" else "目标域"
        shot_label = "full" if exp["num_shots"] is None else f"{exp['num_shots']}-shot"
        print(f"\n{'#'*60}")
        print(f"  [{idx}/{total}] GBG-SAM3 | {domain_label_cur} | {shot_label} | seed={exp['seed']}")
        print(f"  已用时间: {datetime.now() - start_time}")
        print(f"{'#'*60}")

        try:
            result = run_single_experiment_gbg(
                domain=exp["domain"],
                num_shots=exp["num_shots"],
                seed=exp["seed"],
                full_test=args.full_test,
                weight_save_path=exp["weight_save_path"],
            )
            all_results.append(result)

        except FileNotFoundError as e:
            print(f"\n  [跳过] 数据未就绪: {e}")
            all_results.append({
                "domain": exp["domain"],
                "num_shots": exp["num_shots"],
                "seed": exp["seed"],
                "full_test": args.full_test,
                "best_val_iou": 0.0,
                "std_mIoU": 0.0, "std_F1": 0.0, "std_Boundary_IoU": 0.0,
                "ugdp_mIoU": 0.0, "ugdp_F1": 0.0, "ugdp_Boundary_IoU": 0.0,
                "error": str(e),
            })

        except Exception as e:
            print(f"\n  [错误] 实验失败: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "domain": exp["domain"],
                "num_shots": exp["num_shots"],
                "seed": exp["seed"],
                "full_test": args.full_test,
                "best_val_iou": 0.0,
                "std_mIoU": 0.0, "std_F1": 0.0, "std_Boundary_IoU": 0.0,
                "ugdp_mIoU": 0.0, "ugdp_F1": 0.0, "ugdp_Boundary_IoU": 0.0,
                "error": str(e),
            })

        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"  GBG-SAM3 E6 实验完成!  总耗时: {elapsed}")
    success_count = len([r for r in all_results if 'error' not in r])
    print(f"  成功: {success_count}/{total}")
    errors = [r for r in all_results if 'error' in r]
    if errors:
        print(f"  失败/跳过: {len(errors)} 次")
        for e in errors:
            print(f"    - {e.get('domain','?')} {e.get('num_shots','?')}-shot "
                  f"seed={e.get('seed','?')}: {e.get('error','?')}")
    print(f"{'='*60}")

    effective_shots = shots or ALL_SHOTS
    effective_domains = domains
    summary = aggregate_gbg_results(all_results, effective_shots, effective_domains)

    print_gbg_summary(summary, effective_domains, effective_shots)

    save_gbg_results_json(all_results, summary, RESULTS_DIR, args.full_test)


# ==========================================================================
# 模块 8: 主入口
# ==========================================================================
if __name__ == "__main__":
    main()
