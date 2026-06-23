"""
=============================================================================
run_all_experiments.py  ---  全量数据多随机种子自动训练与评估总控脚本 (v2.0)
=============================================================================
功能说明:
  1. 按顺序运行核心消融与策略对比实验矩阵:
     - few-shot 规模: 5-shot / 10-shot / 20-shot
     - 微调策略: Decoder-Only / LoRA
     - 域: source (源域) / target (目标域)
     - 随机种子: 42 / 123 / 456
     共计 2 × 2 × 3 × 3 = 36 次独立训练 + 评估

  2. 支持 "双轨制" 测试:
     - 瘦身测试集 (默认): target_whu_mix/test 或 source_whu/test
     - 全量测试集 (--full_test): whu_mix_full_test/ (8402 张)

  3. 对于每个 (num_shots, mode, domain) 组合, 在 3 个种子下各运行一次,
     计算 mIoU 和 F1-score 的 Mean ± Std,
     确保论文在 Remote Sensing 一审时的数理严谨度

  4. 自动检测运行环境 (Windows 本地 vs Linux 云端),
     自适应数据集路径与权重保存路径

  5. 所有最优模型权重自动保存至 weights/ 目录,
     文件命名规则: sam3_{mode}_{domain}_{N}shot_seed{seed}.pth

  6. 所有实验结果自动保存至 results/ 目录

v2.0 核心变更:
  - 新增 --domain 参数: 一键切换源域/目标域
  - 新增 --full_test 参数: 瘦身测试集 vs 全量 8402 张测试集
  - 传递 domain + full_test 到 experiment_runner

输出:
  results/experiment_results.json   (结构化原始数据, 含均值 ± 标准差)
  weights/sam3_dec_source_5shot_seed42.pth   (36 个最优权重文件)
  ...

用法:
  python src/run_all_experiments.py
  python src/run_all_experiments.py --domain source
  python src/run_all_experiments.py --domain target --full_test
  python src/run_all_experiments.py --shots 5 --mode lora --domain source
  python src/run_all_experiments.py --domain source --full_data        (全量数据, 不采样)
  python src/run_all_experiments.py --dry-run
  python src/run_all_experiments.py --domain source --shots 5 --dry-run
  python src/run_all_experiments.py --domain source --full_test --dry-run

  # E4 零样本跨域评估 (源域20-shot权重 → 目标域测试集, 跳过训练)
  python src/run_all_experiments.py --eval_zero_shot_cross_domain
  python src/run_all_experiments.py --eval_zero_shot_cross_domain --full_test

  # E5 跨域自适应微调 (源域20-shot初始化 → 目标域少样本训练)
  python src/run_all_experiments.py --domain target --shots 5 --mode lora --seeds 42

命令行参数:
  --domain                      域: source (源域) 或 target (目标域), 默认 source
  --full_test                   使用全量 8402 张测试集 (默认使用瘦身测试集)
  --full_data                   全量训练数据: num_shots=None, 不进行 few-shot 采样
  --eval_zero_shot_cross_domain E4: 源域20-shot零样本跨域评估 (跳过训练)
  --shots                       指定 few-shot 规模, 逗号分隔: 5,10,20 (默认: 全量)
  --mode                        指定微调策略: decoder_only,lora (默认: 全量)
  --seeds                       指定随机种子, 逗号分隔: 42,123,456 (默认: 全量)
  --dry-run                     仅打印实验矩阵, 不实际运行
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.cloud_paths import get_domain_paths, get_paths, get_platform_name
from training.experiment_runner import run_single_experiment

_paths = get_paths()

WEIGHTS_DIR: str = _paths["weights_dir"]
RESULTS_DIR: str = _paths.get("results_dir", str(Path(_paths["project_root"]) / "results"))

ALL_SHOTS: List[int] = [5, 10, 20]
ALL_MODES: List[str] = ["decoder_only", "lora"]
ALL_DOMAINS: List[str] = ["source", "target"]
ALL_SEEDS: List[int] = [42, 123, 456]

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================================================
# 模块 2: 实验矩阵生成
#     根据命令行参数过滤要运行的实验组合
# ==========================================================================
def build_experiment_matrix(
    shots: Optional[List[Optional[int]]] = None,
    modes: Optional[List[str]] = None,
    domains: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
) -> List[Dict]:
    shots = shots if shots is not None else ALL_SHOTS
    modes = modes or ALL_MODES
    domains = domains or ALL_DOMAINS
    seeds = seeds or ALL_SEEDS

    matrix: List[Dict] = []
    for domain in domains:
        for shot in shots:
            for mode in modes:
                for seed in seeds:
                    mode_short = "dec" if mode == "decoder_only" else "lora"
                    domain_short = "src" if domain == "source" else "tgt"
                    shot_suffix = "full" if shot is None else f"{shot}shot"
                    weight_name = f"sam3_{mode_short}_{domain_short}_{shot_suffix}_seed{seed}.pth"
                    weight_path = str(Path(WEIGHTS_DIR) / weight_name)
                    matrix.append({
                        "domain": domain,
                        "num_shots": shot,
                        "mode": mode,
                        "seed": seed,
                        "weight_save_path": weight_path,
                    })
    return matrix


# ==========================================================================
# 模块 3: 统计聚合函数
#     对同 (domain, num_shots, mode) 下不同种子的结果计算 Mean ± Std
# ==========================================================================
def aggregate_results(
    all_results: List[Dict],
    shots: List[Optional[int]],
    modes: List[str],
    domains: List[str],
) -> Dict:
    summary: Dict = {}

    for domain in domains:
        for shot in shots:
            for mode in modes:
                shot_suffix = "full" if shot is None else f"{shot}shot"
                key = f"{domain}_{mode}_{shot_suffix}"
                group = [
                    r for r in all_results
                    if r.get("domain") == domain
                    and r["num_shots"] == shot
                    and r["mode"] == mode
                ]

                if not group:
                    continue

                mious = np.array([r["mIoU"] for r in group])
                f1s = np.array([r["F1"] for r in group])
                bious = np.array([r["Boundary_IoU"] for r in group])

                summary[key] = {
                    "domain": domain,
                    "num_shots": shot,
                    "mode": mode,
                    "num_runs": len(group),
                    "mIoU_mean": float(np.mean(mious)),
                    "mIoU_std": float(np.std(mious, ddof=1)) if len(mious) > 1 else 0.0,
                    "F1_mean": float(np.mean(f1s)),
                    "F1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
                    "Boundary_IoU_mean": float(np.mean(bious)),
                    "Boundary_IoU_std": float(np.std(bious, ddof=1)) if len(bious) > 1 else 0.0,
                    "seeds": [r["seed"] for r in group],
                    "individual_mIoU": [float(r["mIoU"]) for r in group],
                    "individual_F1": [float(r["F1"]) for r in group],
                    "individual_Boundary_IoU": [float(r["Boundary_IoU"]) for r in group],
                }

    return summary


# ==========================================================================
# 模块 4: 控制台格式化输出
#     输出规范的 Mean ± Std 对比表格
# ==========================================================================
def print_experiment_table(matrix: List[Dict]) -> None:
    print(f"\n{'='*95}")
    print(f"  实验矩阵 ({len(matrix)} 次独立训练)")
    print(f"{'='*95}")
    print(f"  {'#':<4} {'Domain':<10} {'Few-Shot':<10} {'Mode':<16} {'Seed':<6} {'权重文件'}")
    print(f"  {'-'*4} {'-'*10} {'-'*10} {'-'*16} {'-'*6} {'-'*45}")
    for i, exp in enumerate(matrix, 1):
        mode_label = "Decoder-Only" if exp["mode"] == "decoder_only" else "LoRA"
        domain_label = "source" if exp["domain"] == "source" else "target"
        print(f"  {i:<4} {domain_label:<10} {exp['num_shots']}-shot{'':>3} "
              f"{mode_label:<16} {exp['seed']:<6} "
              f"{Path(exp['weight_save_path']).name}")
    print(f"{'='*95}")


def print_summary_table(summary: Dict, domains: List[str]) -> None:
    print(f"\n{'='*100}")
    print(f"  多随机种子消融实验汇总 (Mean ± Std)")
    print(f"  平台: {get_platform_name()}  |  设备: {DEVICE}")
    print(f"  种子: {ALL_SEEDS}")
    print(f"{'='*100}")

    for domain in domains:
        print(f"\n  --- 域: {domain} ---")
        header = (
            f"  {'':<5}"
            f"{'Decoder-Only mIoU':<32}"
            f"{'LoRA mIoU':<32}"
        )
        print(header)
        print(f"  {'':<5}{'-'*32}{'-'*32}")

        shots_in_domain = sorted(
            (s["num_shots"] for s in summary.values()
             if s.get("domain") == domain),
            key=lambda x: (x is None, x or 0),
        )

        for shot in shots_in_domain:
            shot_label = "full" if shot is None else f"{shot}-shot"
            shot_suffix = "full" if shot is None else f"{shot}shot"
            dec_key = f"{domain}_decoder_only_{shot_suffix}"
            lora_key = f"{domain}_lora_{shot_suffix}"

            dec_str = ""
            lora_str = ""

            if dec_key in summary:
                s = summary[dec_key]
                dec_str = f"{s['mIoU_mean']*100:.2f} ± {s['mIoU_std']*100:.2f}"

            if lora_key in summary:
                s = summary[lora_key]
                lora_str = f"{s['mIoU_mean']*100:.2f} ± {s['mIoU_std']*100:.2f}"

            print(f"  {shot_label:<6}{dec_str:<32}{lora_str:<32}")

    print(f"{'='*100}")


# ==========================================================================
# 模块 5: 结果持久化
# ==========================================================================
def save_results_json(
    all_results: List[Dict],
    summary: Dict,
    results_dir: str,
    full_test: bool,
) -> str:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_label = "full" if full_test else "slim"
    filename = f"experiment_results_{test_label}_{timestamp}.json"
    filepath = str(Path(results_dir) / filename)

    output = {
        "metadata": {
            "platform": get_platform_name(),
            "device": DEVICE,
            "timestamp": timestamp,
            "full_test": full_test,
            "batch_size_desc": "train: auto(2/4/16 by num_shots), eval: 32",
            "learning_rate": 3e-4,
            "num_epochs": 30,
            "num_workers": 8,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
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
        description="SAM3 GeoAI 全量数据多随机种子自动训练与评估 (v2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 源域, 瘦身测试 (调试)
  python src/run_all_experiments.py --domain source

  # 目标域, 全量测试 (论文最终结果)
  python src/run_all_experiments.py --domain target --full_test

  # 快速测试
  python src/run_all_experiments.py --shots 5 --mode lora --domain source --seeds 42

  # 仅打印实验矩阵
  python src/run_all_experiments.py --domain target --dry-run
        """,
    )
    parser.add_argument(
        "--domain", type=str, default="source",
        choices=["source", "target"],
        help="域选择: source (源域/source_whu) 或 target (目标域/target_whu_mix), 默认: source",
    )
    parser.add_argument(
        "--full_test", action="store_true",
        help="使用全量 8402 张测试集 (whu_mix_full_test); "
             "不设置则使用瘦身测试集",
    )
    parser.add_argument(
        "--shots", type=str, default=None,
        help="指定 few-shot 规模, 逗号分隔: 5,10,20 (默认: 全量)",
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        help="指定微调策略: decoder_only,lora (默认: 全量)",
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="指定随机种子, 逗号分隔: 42,123,456 (默认: 全量)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅打印实验矩阵, 不实际运行训练",
    )
    parser.add_argument(
        "--full_data", action="store_true",
        help="全量数据模式: num_shots=None, 使用训练目录下全部影像 (不进行 few-shot 采样)",
    )
    parser.add_argument(
        "--eval_zero_shot_cross_domain", action="store_true",
        help="E4 零样本跨域评估: 用源域 20-shot 预训练权重直接评估目标域测试集 (跳过训练)",
    )
    return parser.parse_args()


# ==========================================================================
# 模块 7: 主函数
#     解析参数 → 构建矩阵 → 逐实验运行 → 聚合统计 → 保存结果
# ==========================================================================
def main() -> None:
    args = parse_args()

    if args.full_data:
        shots: Optional[List[Optional[int]]] = [None]
        print("\n  [全量数据模式] 将使用训练目录下全部影像, 不进行 few-shot 采样")
    else:
        shots = [int(s.strip()) for s in args.shots.split(",")] if args.shots else None

    modes = [m.strip() for m in args.mode.split(",")] if args.mode else None
    seeds = [int(s.strip()) for s in args.seeds.split(",")] if args.seeds else None
    domains = [args.domain]

    if shots is not None and not args.full_data:
        invalid_shots = [s for s in shots if s not in ALL_SHOTS]
        if invalid_shots:
            print(f"错误: 无效的 shots 值 {invalid_shots}, 可选: {ALL_SHOTS}")
            sys.exit(1)
    if modes is not None:
        invalid_modes = [m for m in modes if m not in ALL_MODES]
        if invalid_modes:
            print(f"错误: 无效的 mode 值 {invalid_modes}, 可选: {ALL_MODES}")
            sys.exit(1)
    if seeds is not None:
        invalid_seeds = [s for s in seeds if s not in ALL_SEEDS]
        if invalid_seeds:
            print(f"错误: 无效的 seeds 值 {invalid_seeds}, 可选: {ALL_SEEDS}")
            sys.exit(1)

    matrix = build_experiment_matrix(shots, modes, domains, seeds)

    test_label = "全量8402" if args.full_test else "瘦身测试"
    domain_label = "源域(source_whu)" if args.domain == "source" else "目标域(target_whu_mix)"

    print(f"\n{'='*60}")
    print(f"  SAM3 GeoAI 多随机种子消融实验总控 (v2.0)")
    print(f"  平台: {get_platform_name()}")
    print(f"  设备: {DEVICE}")
    print(f"  域: {domain_label}  |  测试: {test_label}")
    print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    if args.eval_zero_shot_cross_domain:
        from training.experiment_runner import run_zero_shot_cross_domain_eval

        effective_modes = modes or ALL_MODES
        effective_seeds = seeds or ALL_SEEDS
        total = len(effective_modes) * len(effective_seeds)

        print(f"\n{'='*60}")
        print(f"  E4 零样本跨域评估模式")
        print(f"  源域(source_whu) 20-shot → 目标域(target_whu_mix) 测试集")
        print(f"  模式: {effective_modes}  |  种子: {effective_seeds}")
        print(f"  测试集: {test_label}  |  共 {total} 次评估")
        print(f"{'='*60}")

        if args.dry_run:
            print("\n  [DRY-RUN 模式] 零样本跨域评估矩阵已打印, 不执行.")
            return

        zero_shot_results: List[Dict] = []
        start_time = datetime.now()
        idx = 0

        for mode in effective_modes:
            for seed in effective_seeds:
                idx += 1
                mode_label = "Decoder-Only" if mode == "decoder_only" else "LoRA"
                print(f"\n{'#'*60}")
                print(f"  [{idx}/{total}] E4 Zero-Shot: {mode_label} | seed={seed}")
                print(f"  已用时间: {datetime.now() - start_time}")
                print(f"{'#'*60}")

                try:
                    result = run_zero_shot_cross_domain_eval(
                        mode=mode,
                        seed=seed,
                        full_test=args.full_test,
                        weights_dir=WEIGHTS_DIR,
                    )
                    zero_shot_results.append(result)

                except FileNotFoundError as e:
                    print(f"\n  [跳过] {e}")
                    zero_shot_results.append({
                        "mode": mode,
                        "domain": "zero_shot_cross",
                        "num_shots": 20,
                        "full_test": args.full_test,
                        "seed": seed,
                        "best_val_iou": 0.0,
                        "mIoU": 0.0,
                        "F1": 0.0,
                        "Boundary_IoU": 0.0,
                        "error": str(e),
                    })

                except Exception as e:
                    print(f"\n  [错误] E4 评估失败: {e}")
                    import traceback
                    traceback.print_exc()
                    zero_shot_results.append({
                        "mode": mode,
                        "domain": "zero_shot_cross",
                        "num_shots": 20,
                        "full_test": args.full_test,
                        "seed": seed,
                        "best_val_iou": 0.0,
                        "mIoU": 0.0,
                        "F1": 0.0,
                        "Boundary_IoU": 0.0,
                        "error": str(e),
                    })

                gc.collect()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

        elapsed = datetime.now() - start_time
        print(f"\n{'='*60}")
        print(f"  E4 零样本跨域评估完成!  总耗时: {elapsed}")
        print(f"{'='*60}")

        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_label_short = "full" if args.full_test else "slim"
        result_path = str(Path(RESULTS_DIR) / f"cross_domain_zero_shot_{test_label_short}_{timestamp}.json")

        output = {
            "experiment": "E4_Zero_Shot_Cross_Domain",
            "metadata": {
                "platform": get_platform_name(),
                "device": DEVICE,
                "timestamp": timestamp,
                "full_test": args.full_test,
                "source_domain": "source_whu (20-shot)",
                "target_domain": "target_whu_mix",
                "eval_batch_size": 32,
                "num_workers": 8,
            },
            "individual_results": zero_shot_results,
        }
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n  结果已保存: {result_path}")
        return

    if not torch.cuda.is_available():
        print("\n  警告: 未检测到 CUDA GPU, 将使用 CPU (极慢, 不推荐)")

    print_experiment_table(matrix)

    if args.dry_run:
        print("\n  [DRY-RUN 模式] 实验矩阵已打印, 不执行实际训练.")
        return

    from training.experiment_runner import run_single_experiment

    Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []
    total = len(matrix)
    start_time = datetime.now()

    for idx, exp in enumerate(matrix, 1):
        mode_label = "Decoder-Only" if exp["mode"] == "decoder_only" else "LoRA"
        domain_label_cur = "源域" if exp["domain"] == "source" else "目标域"
        print(f"\n{'#'*60}")
        print(f"  [{idx}/{total}] {domain_label_cur} | {mode_label} | "
              f"{exp['num_shots']}-shot | seed={exp['seed']}")
        print(f"  已用时间: {datetime.now() - start_time}")
        print(f"{'#'*60}")

        try:
            result = run_single_experiment(
                mode=exp["mode"],
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
                "mode": exp["mode"],
                "domain": exp["domain"],
                "num_shots": exp["num_shots"],
                "full_test": args.full_test,
                "seed": exp["seed"],
                "best_val_iou": 0.0,
                "mIoU": 0.0,
                "F1": 0.0,
                "Boundary_IoU": 0.0,
                "error": str(e),
            })

        except Exception as e:
            print(f"\n  [错误] 实验失败: {e}")
            print(f"  配置: domain={exp['domain']}, mode={exp['mode']}, "
                  f"shots={exp['num_shots']}, seed={exp['seed']}")
            import traceback
            traceback.print_exc()

            all_results.append({
                "mode": exp["mode"],
                "domain": exp["domain"],
                "num_shots": exp["num_shots"],
                "full_test": args.full_test,
                "seed": exp["seed"],
                "best_val_iou": 0.0,
                "mIoU": 0.0,
                "F1": 0.0,
                "Boundary_IoU": 0.0,
                "error": str(e),
            })

        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"  所有实验完成!")
    print(f"  总耗时: {elapsed}")
    success_count = len([r for r in all_results if 'error' not in r])
    print(f"  成功: {success_count}/{total}")
    errors = [r for r in all_results if 'error' in r]
    if errors:
        print(f"  失败/跳过: {len(errors)} 次")
        for e in errors:
            print(f"    - domain={e.get('domain','?')} {e['mode']} "
                  f"{e['num_shots']}-shot seed={e['seed']}: {e['error']}")
    print(f"{'='*60}")

    effective_shots = shots or ALL_SHOTS
    effective_modes = modes or ALL_MODES
    effective_domains = domains
    summary = aggregate_results(all_results, effective_shots, effective_modes, effective_domains)

    print_summary_table(summary, effective_domains)

    save_results_json(all_results, summary, RESULTS_DIR, args.full_test)


# ==========================================================================
# 模块 8: 主入口
# ==========================================================================
if __name__ == "__main__":
    main()
