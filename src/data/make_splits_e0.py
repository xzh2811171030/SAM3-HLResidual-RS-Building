"""
=============================================================================
make_splits_e0.py  ---  E0 数据划分与泄露审计脚本
=============================================================================

【功能说明】
  1. 作为所有后续 E1–E8 实验的【唯一数据划分入口】
  2. 生成统一的 manifest 文件目录 (data/splits/e0_manifest/)
  3. 保证 few-shot support、validation、pilot test、final test 的:
     - 严谨性 (城市分层抽样)
     - 可复现性 (seed 控制)
     - 零交集 (文件名级审计)
  4. 输出审计报告 (JSON) + 城市分布报告 + README

【生成的核心文件】
  data/splits/e0_manifest/
    source_train.txt          源域训练集
    source_val.txt            源域验证集
    source_test.txt           源域测试集
    target_train_pool_all.txt 目标域全部可用 train 候选池
    target_val.txt            目标域验证集 (paper-ready 或 pilot-only)
    target_support_5_seed42.txt   ... (9个support文件)
    target_support_10_seed42.txt
    target_support_20_seed42.txt
    target_support_5_seed123.txt
    ... (同样 seed123, seed456)
    target_pilot_test_500.txt    试点快速评估集
    target_pilot_test_1000.txt   试点快速评估集
    target_final_test_8402.txt   最终论文评估集
    split_audit_report.json      完整审计报告
    city_distribution_report.json 城市分布统计
    README_E0_SPLITS.md          使用说明

【命令行参数】
  --project_root        项目根目录 (默认自动检测)
  --manifest_dir        输出 manifest 目录 (默认 data/splits/e0_manifest)
  --target_val_size     target val 大小 (默认 500)
  --pilot_test_sizes    pilot test 大小, 逗号分隔 (默认 500,1000)
  --support_shots       support shot 值, 逗号分隔 (默认 5,10,20)
  --seeds               随机种子, 逗号分隔 (默认 42,123,456)
  --expand_source       是否扩展 source split
  --source_test_size    source test 大小 (默认 500)
  --dry_run             只打印计划, 不写文件
  --strict              严格模式: 正式 target trainval 不存在则报错退出

【推荐运行命令】
  pilot 模式:       python src/data/make_splits_e0.py --dry_run
  pilot 模式写盘:   python src/data/make_splits_e0.py
  严格 paper-ready: python src/data/make_splits_e0.py --strict
  expand_source:    python src/data/make_splits_e0.py --expand_source --target_val_size 1000

=============================================================================
"""

# ==========================================================================
# 模块 1: 导入与常量
# ==========================================================================
import argparse
import json
import tempfile
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

IMAGE_EXTS: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
LABEL_EXTS: Tuple[str, ...] = IMAGE_EXTS
DEFAULT_MIN_SUPPORT_POSITIVE_PIXELS: int = 100

# 论文核心城市名 (WHU-Mix 5 城 + 训练池常见城市)
CORE_CITIES: Set[str] = {
    "wuxi", "potsdam", "dunedin", "kitsap", "khartoum",
    "asia", "hangzhou", "chongqing", "christchurch",
    "hubei", "tyrol", "vienna", "tianjin",
}


# ==========================================================================
# 模块 2: 路径探测 (Probe Paths)
#     探测项目内所有可能的数据目录
# ==========================================================================

def probe_project_paths(project_root: Path) -> Dict[str, str]:
    """探测所有可能存在的数据目录, 返回路径字典."""
    p = project_root

    paths = {
        "project_root": str(p),
        "slim_root": "",
        "whu_mix_train_image": "",
        "whu_mix_train_label": "",
        "whu_mix_val_image": "",
        "whu_mix_val_label": "",
        "whu_mix_full_test_image": "",
        "whu_mix_full_test_label": "",
        "whu_mix_full_test_bbox": "",
        "aerial_train_image": "",
        "aerial_train_label": "",
        "aerial_val_image": "",
        "aerial_val_label": "",
        "aerial_test_image": "",
        "aerial_test_label": "",
        # slim 子路径 (从 processed_slim)
        "source_train_image": "",
        "source_train_label": "",
        "source_val_image": "",
        "source_val_label": "",
        "source_test_image": "",
        "source_test_label": "",
        "target_train_pool_image": "",
        "target_train_pool_label": "",
        "target_test_slim_image": "",
        "target_test_slim_label": "",
    }

    # --- slim_root ---
    for slim_cand in [p / "data" / "raw" / "processed_slim", p / "data" / "processed_slim"]:
        if slim_cand.is_dir():
            paths["slim_root"] = str(slim_cand)
            break

    if paths["slim_root"]:
        s = Path(paths["slim_root"])
        # Source slim
        for sub in [("source_train_image", "source_whu/train/images"),
                     ("source_train_label", "source_whu/train/dual_channel_labels"),
                     ("source_val_image",   "source_whu/val/images"),
                     ("source_val_label",   "source_whu/val/dual_channel_labels"),
                     ("source_test_image",  "source_whu/test/images"),
                     ("source_test_label",  "source_whu/test/dual_channel_labels")]:
            cand = s / sub[1]
            if cand.is_dir():
                paths[sub[0]] = str(cand)
        # Target slim
        for sub in [("target_train_pool_image", "target_whu_mix/train_pool/images"),
                     ("target_train_pool_label", "target_whu_mix/train_pool/dual_channel_labels"),
                     ("target_test_slim_image",  "target_whu_mix/test/images"),
                     ("target_test_slim_label",  "target_whu_mix/test/dual_channel_labels")]:
            cand = s / sub[1]
            if cand.is_dir():
                paths[sub[0]] = str(cand)

    # --- WHU-Mix 官方 train ---
    for train_cand, label_cand in [
        (p / "data" / "raw" / "whu_mix" / "train" / "image",   p / "data" / "raw" / "whu_mix" / "train" / "label"),
        (p / "data" / "raw" / "WHU-Mix" / "train" / "image",   p / "data" / "raw" / "WHU-Mix" / "train" / "label"),
        (p / "data" / "raw" / "whu_mix" / "trainval" / "image", p / "data" / "raw" / "whu_mix" / "trainval" / "label"),
    ]:
        if train_cand.is_dir():
            paths["whu_mix_train_image"] = str(train_cand)
            if label_cand.is_dir():
                paths["whu_mix_train_label"] = str(label_cand)
            break

    # --- WHU-Mix 官方 val ---
    for val_cand, label_cand in [
        (p / "data" / "raw" / "whu_mix" / "val" / "image", p / "data" / "raw" / "whu_mix" / "val" / "label"),
        (p / "data" / "raw" / "WHU-Mix" / "val" / "image", p / "data" / "raw" / "WHU-Mix" / "val" / "label"),
    ]:
        if val_cand.is_dir():
            paths["whu_mix_val_image"] = str(val_cand)
            if label_cand.is_dir():
                paths["whu_mix_val_label"] = str(label_cand)
            break

    # --- WHU-Mix full test (8402) ---
    for test_cand, label_cand, bbox_cand in [
        (p / "data" / "raw" / "whu_mix_full_test" / "test" / "image",
         p / "data" / "raw" / "whu_mix_full_test" / "dual_channel_labels",
         p / "data" / "raw" / "whu_mix_full_test" / "bbox.json"),
        (p / "data" / "raw" / "whu_mix" / "test" / "image",
         p / "data" / "raw" / "whu_mix" / "test" / "label",
         p / "data" / "raw" / "whu_mix" / "test" / "bbox.json"),
    ]:
        if test_cand.is_dir():
            paths["whu_mix_full_test_image"] = str(test_cand)
            if label_cand.is_dir():
                paths["whu_mix_full_test_label"] = str(label_cand)
            if bbox_cand.is_file():
                paths["whu_mix_full_test_bbox"] = str(bbox_cand)
            break

    # --- WHU aerial_0.3 官方 ---
    for base_cand in [
        p / "data" / "raw" / "aerial_0.3",
        p / "data" / "raw" / "WHU_Aerial",
    ]:
        if base_cand.is_dir():
            for split, key_img, key_lbl in [
                ("train", "aerial_train_image", "aerial_train_label"),
                ("val",   "aerial_val_image",   "aerial_val_label"),
                ("test",  "aerial_test_image",  "aerial_test_label"),
            ]:
                img_d = base_cand / split / "image"
                lbl_d = base_cand / split / "label"
                if img_d.is_dir():
                    paths[key_img] = str(img_d)
                if lbl_d.is_dir():
                    paths[key_lbl] = str(lbl_d)
            break

    return paths


# ==========================================================================
# 模块 3: 城市解析 (City Inference)
#     从文件名推断所属城市, 支持多种命名约定
# ==========================================================================

def infer_city_from_filename(filename: str) -> str:
    """
    从文件名推断城市名。
    支持格式: dunedin_1006, wuxi_200, Potsdam_1, asia_train_2526 等。
    返回小写城市名, 无法解析返回 "unknown"。
    """
    stem = Path(filename).stem.lower().replace("-", "_")

    # 1. 精确匹配核心城市名
    for city in sorted(CORE_CITIES, key=len, reverse=True):
        if stem.startswith(city + "_") or stem.startswith(city + "."):
            return city

    # 2. 前缀匹配 (文件名以城市名开头)
    parts = stem.split("_")
    if parts and parts[0] in CORE_CITIES:
        return parts[0]

    # 3. 模糊匹配 (城市名出现在文件名中)
    for city in CORE_CITIES:
        if city in stem:
            return city

    return "unknown"


# ==========================================================================
# 模块 4: 文件扫描与配对 (File Scanner)
#     扫描 image 目录, 与 dual_channel_labels 配对
# ==========================================================================

def scan_image_dir(
    image_dir: str,
    label_dir: str,
    required_label: bool = True,
) -> List[str]:
    """
    扫描 image 目录, 返回与 label 配对的文件名 stem 列表。

    required_label=True 时, label_dir 必须存在且每张 image 必须有对应标签；
    required_label=False 时, 只扫描 image。
    """
    img_path = Path(image_dir)
    lbl_path = Path(label_dir) if label_dir else Path("__NON_EXISTENT_LABEL_DIR__")

    if not img_path.is_dir():
        return []

    if required_label and not lbl_path.is_dir():
        return []

    result: List[str] = []
    for ext in IMAGE_EXTS:
        for f in img_path.glob(f"*{ext}"):
            stem = f.stem

            if required_label:
                has_label = False
                for lext in (".png", ".tif", ".tiff", ".jpg", ".jpeg", ext):
                    if (lbl_path / f"{stem}{lext}").exists():
                        has_label = True
                        break
                if not has_label:
                    continue

            result.append(stem)

    return sorted(set(result))


def scan_image_dir_absolute(image_dir: str) -> List[str]:
    """扫描 image 目录, 返回文件绝对路径列表 (仅 stem 不含扩展名, 用绝对路径)。"""
    img_path = Path(image_dir)
    if not img_path.is_dir():
        return []
    result: List[str] = []
    for ext in IMAGE_EXTS:
        for f in img_path.glob(f"*{ext}"):
            result.append(str(f.resolve()).replace("\\", "/"))
    return sorted(result)

# ==========================================================================
# 模块 4b: 正样本检测 (Positive Building Filter)
#     用于确保 few-shot support 不包含空建筑切片
# ==========================================================================

def find_label_for_stem(label_dir: str, stem: str) -> Optional[Path]:
    """
    在 label_dir 中查找与 stem 对应的标签文件。
    支持 tif/png/jpg 等常见扩展名。
    """
    lbl_dir = Path(label_dir)
    if not lbl_dir.is_dir():
        return None

    for ext in LABEL_EXTS:
        cand = lbl_dir / f"{stem}{ext}"
        if cand.exists():
            return cand

    return None


def count_foreground_pixels(label_path: Path) -> int:
    """
    读取 label 并统计建筑前景像素数。

    兼容：
      1. 官方二值 label: 单通道或 RGB 白色建筑
      2. dual_channel_labels: B/G/R 三通道，其中任意通道 > 0 都说明存在建筑或边界

    注意：
      这里只用于判定 support 是否为空建筑切片，不用于生成最终训练标签。
    """
    img = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return 0

    if img.ndim == 2:
        mask = img > 0
    elif img.ndim == 3:
        # 对官方 RGB label 和 dual-channel label 都稳健：
        # 只要任一通道有前景，就认为该切片含建筑相关标注。
        mask = np.max(img, axis=2) > 0
    else:
        return 0

    return int(mask.sum())


def filter_positive_stems(
    stems: List[str],
    label_dir: str,
    min_positive_pixels: int = DEFAULT_MIN_SUPPORT_POSITIVE_PIXELS,
) -> Tuple[List[str], Dict[str, int], List[str], List[str]]:
    """
    从候选 stems 中筛出有建筑前景的样本。

    返回：
      positive_stems: 满足前景像素阈值的样本
      fg_count_map: 每个 stem 的前景像素数
      empty_stems: 有 label 但前景不足的样本
      missing_label_stems: 找不到 label 的样本
    """
    positive_stems: List[str] = []
    empty_stems: List[str] = []
    missing_label_stems: List[str] = []
    fg_count_map: Dict[str, int] = {}

    for stem in sorted(set(stems)):
        label_path = find_label_for_stem(label_dir, stem)
        if label_path is None:
            missing_label_stems.append(stem)
            fg_count_map[stem] = 0
            continue

        fg = count_foreground_pixels(label_path)
        fg_count_map[stem] = fg

        if fg >= min_positive_pixels:
            positive_stems.append(stem)
        else:
            empty_stems.append(stem)

    return positive_stems, fg_count_map, empty_stems, missing_label_stems

# ==========================================================================
# 模块 5: 分层抽样引擎 (Stratified Sampling Engine)
#     city_stratified_sample: 按城市分层抽样
#     nested_sample:          嵌套采样 (5 ⊂ 10 ⊂ 20)
# ==========================================================================

def _build_city_buckets(
    stems: List[str],
    num_per_city: Optional[int] = None,
) -> Dict[str, List[str]]:
    """将 stems 按城市分桶, 并可选地对每桶进行随机采样。"""
    buckets: Dict[str, List[str]] = defaultdict(list)
    for s in stems:
        city = infer_city_from_filename(s)
        buckets[city].append(s)

    if num_per_city is not None and num_per_city > 0:
        for city in list(buckets.keys()):
            if len(buckets[city]) > num_per_city:
                # 使用稳定排序保证 seed 控制下的确定性
                buckets[city] = sorted(buckets[city])[:num_per_city]

    return dict(buckets)


def city_stratified_sample(
    stems: List[str],
    n: int,
    seed: int,
    per_city_limit: Optional[int] = None,
    exclude_set: Optional[Set[str]] = None,
) -> List[str]:
    """
    从 stems 中按城市/区域分层抽取 n 个样本。

    关键修复:
      1. 不再在函数末尾全局 sorted(), 避免 nested few-shot 取前 k 张时被字母序污染；
      2. city 顺序由 seed 控制随机打乱，避免 5-shot 永远落在 alphabetically first cities；
      3. 返回顺序保留分层抽样顺序，用于 nested sampling。
    """
    rng = random.Random(seed)

    if exclude_set:
        stems = [s for s in stems if s not in exclude_set]

    stems = sorted(set(stems))
    if len(stems) <= n:
        out = stems[:]
        rng.shuffle(out)
        return out

    buckets = _build_city_buckets(stems, num_per_city=per_city_limit)
    city_names = sorted(buckets.keys())
    rng.shuffle(city_names)

    n_cities = max(1, len(city_names))
    per_city = n // n_cities
    remainder = n % n_cities

    sampled: List[str] = []
    leftover_pool: List[str] = []

    for i, city in enumerate(city_names):
        alloc = per_city + (1 if i < remainder else 0)
        pool = sorted(buckets[city])
        rng.shuffle(pool)

        take = min(alloc, len(pool))
        sampled.extend(pool[:take])
        leftover_pool.extend(pool[take:])

    shortage = n - len(sampled)
    if shortage > 0 and leftover_pool:
        rng.shuffle(leftover_pool)
        sampled.extend(leftover_pool[:shortage])

    # 保留抽样顺序，不做全局排序
    return sampled[:n]


def nested_support_sample(
    stems: List[str],
    shots: List[int],
    seeds: List[int],
    exclude_set: Optional[Set[str]] = None,
) -> Dict[str, List[str]]:
    """
    嵌套采样: 对每个 seed, 先抽 max(shots), 再取前 k 张作为 k-shot。

    保证：
      5-shot ⊂ 10-shot ⊂ 20-shot

    注意：
      stems 应当已经经过 support positive filter；
      即 support 只从有建筑前景的切片中抽取。
    """
    result: Dict[str, List[str]] = {}
    shots_sorted = sorted(shots)
    max_shot = max(shots_sorted)

    for seed in seeds:
        max_sample_ordered = city_stratified_sample(
            stems,
            max_shot,
            seed,
            per_city_limit=None,
            exclude_set=exclude_set,
        )

        if len(max_sample_ordered) < max_shot:
            raise ValueError(
                f"可用正样本 support 不足: seed={seed}, "
                f"需要 {max_shot}, 实际只有 {len(max_sample_ordered)}。"
                f"请降低 min_support_positive_pixels 或扩大 WHU-Mix train pool。"
            )

        for shot in shots_sorted:
            key = f"target_support_{shot}_seed{seed}"
            result[key] = sorted(max_sample_ordered[:shot])

    return result

# ==========================================================================
# 模块 6: Manifest 写入 (Write Manifest)
#     manifest 中保存绝对路径, 方便 DataLoader 直接读取
# ==========================================================================

def write_manifest(
    manifest_dir: str,
    filename: str,
    stems: List[str],
    image_dir: str,
) -> str:
    """
    将 stems 写入 manifest 文件。

    manifest 格式: 每行一个绝对路径 (到 .tif/.png 影像文件)。
    自动检测文件扩展名。
    """
    out_dir = Path(manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    img_dir_path = Path(image_dir)
    lines: List[str] = []

    for stem in stems:
        # 查找实际存在的文件
        found = False
        for ext in IMAGE_EXTS:
            cand = img_dir_path / f"{stem}{ext}"
            if cand.exists():
                lines.append(str(cand.resolve()).replace("\\", "/"))
                found = True
                break
        if not found:
            # 尝试直接用 stem 作为文件名
            lines.append(str((img_dir_path / stem).resolve()).replace("\\", "/"))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return str(out_path)


# ==========================================================================
# 模块 7: Source Split 生成
#     source_train / source_val / source_test
# ==========================================================================

def build_source_splits(
    paths: Dict[str, str],
    manifest_dir: str,
    expand_source: bool,
    source_test_size: int,
) -> Dict[str, any]:
    """
    生成源域 WHU aerial 的 train/val/test manifest。

    修复点:
      1. expand_source=True 时, manifest 写入 official aerial 路径，而不是 slim 路径；
      2. source_test_size 参数真正生效；
      3. source slim 默认仍可作为 auxiliary benchmark。
    """
    info: Dict[str, any] = {
        "type": "source_auxiliary",
        "note": "",
        "splits": {},
    }

    use_slim = True

    train_img_dir = paths.get("source_train_image", "")
    val_img_dir = paths.get("source_val_image", "")
    test_img_dir = paths.get("source_test_image", "")

    if expand_source and paths.get("aerial_train_image"):
        official_train = scan_image_dir(
            paths["aerial_train_image"], paths.get("aerial_train_label", "")
        )
        official_val = scan_image_dir(
            paths.get("aerial_val_image", ""), paths.get("aerial_val_label", "")
        )
        official_test = scan_image_dir(
            paths.get("aerial_test_image", ""), paths.get("aerial_test_label", "")
        )

        if len(official_train) >= 100 and len(official_val) >= 100 and len(official_test) >= 100:
            use_slim = False
            train_stems = official_train
            val_stems = official_val
            test_stems = city_stratified_sample(
                official_test,
                min(source_test_size, len(official_test)),
                seed=42,
            )

            train_img_dir = paths["aerial_train_image"]
            val_img_dir = paths["aerial_val_image"]
            test_img_dir = paths["aerial_test_image"]

            info["note"] = "expanded from official WHU aerial_0.3 directory"
        else:
            info["note"] = (
                "expand_source requested but official WHU aerial data is insufficient; "
                "using existing slim source splits"
            )

    if use_slim:
        train_stems = scan_image_dir(paths.get("source_train_image", ""), paths.get("source_train_label", ""))
        val_stems = scan_image_dir(paths.get("source_val_image", ""), paths.get("source_val_label", ""))
        test_stems = scan_image_dir(paths.get("source_test_image", ""), paths.get("source_test_label", ""))

        train_img_dir = paths.get("source_train_image", "")
        val_img_dir = paths.get("source_val_image", "")
        test_img_dir = paths.get("source_test_image", "")

        info["note"] = (
            "using existing slim source splits; source split is auxiliary, "
            "not the main target-domain benchmark"
        )

    for key, stems, img_dir in [
        ("source_train", train_stems, train_img_dir),
        ("source_val", val_stems, val_img_dir),
        ("source_test", test_stems, test_img_dir),
    ]:
        if stems and img_dir:
            p = write_manifest(manifest_dir, f"{key}.txt", stems, img_dir)
            info["splits"][key] = {"count": len(stems), "path": p}
        else:
            info["splits"][key] = {
                "count": 0,
                "path": "",
                "warning": "directory not found or no paired labels",
            }

    return info


# ==========================================================================
# 模块 8: Target Split 生成
#     target_support / target_val / target_pilot_test / target_final_test
# ==========================================================================

def build_target_splits(
    paths: Dict[str, str],
    manifest_dir: str,
    args: argparse.Namespace,
) -> Dict[str, any]:
    """
    构建目标域所有 split。

    决策树:
      1. target_final_test: 优先 whu_mix_full_test (8402), 否则 fallback 到 slim test
      2. target_support / target_val:
         a) 如果 官方 train+label 存在 → 从官方 train 构建
         b) 否则 → 从 slim train_pool 构建 (PILOT_ONLY)
    """
    info: Dict[str, any] = {
        "type": "target",
        "paper_ready": False,
        "warnings": [],
        "splits": {},
    }

    shots = [int(s.strip()) for s in args.support_shots.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # ----------------------------------------------------------------
    # Step 1: 检测正式 target train pool (有标签)
    # ----------------------------------------------------------------
    official_train_stems: List[str] = []
    official_val_stems: List[str] = []
    has_official_train = False

    if paths["whu_mix_train_image"] and paths["whu_mix_train_label"]:
        official_train_stems = scan_image_dir(
            paths["whu_mix_train_image"], paths["whu_mix_train_label"],
        )
        if len(official_train_stems) >= 200:
            has_official_train = True

    if paths["whu_mix_val_image"] and paths["whu_mix_val_label"]:
        official_val_stems = scan_image_dir(
            paths["whu_mix_val_image"], paths["whu_mix_val_label"],
        )

    # ----------------------------------------------------------------
    # Step 2: Slim train_pool (始终作为 fallback)
    # ----------------------------------------------------------------
    slim_train_stems = scan_image_dir(
        paths["target_train_pool_image"], paths["target_train_pool_label"],
    )
    slim_test_stems = scan_image_dir(
        paths["target_test_slim_image"], paths["target_test_slim_label"],
    )

    # ----------------------------------------------------------------
    # Step 3: target_final_test
    # ----------------------------------------------------------------
    final_test_stems: List[str] = []
    final_test_image_dir = ""
    final_test_is_full_8402 = False

    if paths["whu_mix_full_test_image"]:
        raw_stems = scan_image_dir(
            paths["whu_mix_full_test_image"],
            paths.get("whu_mix_full_test_label", ""),
            required_label=False,
        )
        if raw_stems:
            final_test_stems = raw_stems
            final_test_image_dir = paths["whu_mix_full_test_image"]
            final_test_is_full_8402 = (len(final_test_stems) >= 8400)
    elif slim_test_stems:
        final_test_stems = slim_test_stems
        final_test_image_dir = paths["target_test_slim_image"]
        info["warnings"].append(
            "full 8402 test set not found; using slim test (1000 images) as fallback"
        )

    if final_test_stems and final_test_image_dir:
        p = write_manifest(manifest_dir, "target_final_test_8402.txt",
                           final_test_stems, final_test_image_dir)
        info["splits"]["target_final_test"] = {
            "count": len(final_test_stems),
            "path": p,
            "is_full_8402": final_test_is_full_8402,
        }

    # ----------------------------------------------------------------
    # Step 4: target_train_pool_all
    # ----------------------------------------------------------------
    train_pool_all: List[str] = []
    train_pool_image_dir = ""
    train_pool_label_dir = ""

    if has_official_train:
        train_pool_all = official_train_stems
        train_pool_image_dir = paths["whu_mix_train_image"]
        train_pool_label_dir = paths["whu_mix_train_label"]
        info["paper_ready"] = True
    elif slim_train_stems:
        train_pool_all = slim_train_stems
        train_pool_image_dir = paths["target_train_pool_image"]
        train_pool_label_dir = paths["target_train_pool_label"]
        info["warnings"].append(
            "Only slim target train_pool was found. "
            "These splits are for pilot/demo only unless this pool was explicitly "
            "constructed from official WHU-Mix trainval."
        )
    else:
        if args.strict:
            raise FileNotFoundError(
                "严格模式: 未找到任何目标域 train pool。"
                "请确保 WHU-Mix official train+label 或 processed_slim/target_whu_mix/train_pool 存在。"
            )
        info["warnings"].append("No target train pool found at all.")

    if train_pool_all and train_pool_image_dir:
        p = write_manifest(
            manifest_dir,
            "target_train_pool_all.txt",
            train_pool_all,
            train_pool_image_dir,
        )
        info["splits"]["target_train_pool_all"] = {
            "count": len(train_pool_all),
            "path": p,
        }

    # ----------------------------------------------------------------
    # Step 4b: support positive filter
    #     正式 few-shot support 不允许抽到空建筑切片。
    #     注意：只过滤 support 候选池，不过滤 val/test。
    # ----------------------------------------------------------------
    support_candidate_pool: List[str] = list(train_pool_all)
    support_positive_filter_report: Dict[str, any] = {
        "enabled": not args.allow_empty_support,
        "min_positive_pixels": args.min_support_positive_pixels,
        "input_candidates": len(train_pool_all),
        "positive_candidates": len(train_pool_all),
        "empty_or_too_small": 0,
        "missing_labels": 0,
        "examples_empty": [],
        "examples_missing_label": [],
    }

    if train_pool_all and train_pool_label_dir and not args.allow_empty_support:
        (
            positive_stems,
            fg_count_map,
            empty_stems,
            missing_label_stems,
        ) = filter_positive_stems(
            train_pool_all,
            train_pool_label_dir,
            min_positive_pixels=args.min_support_positive_pixels,
        )

        support_candidate_pool = positive_stems

        support_positive_filter_report.update({
            "positive_candidates": len(positive_stems),
            "empty_or_too_small": len(empty_stems),
            "missing_labels": len(missing_label_stems),
            "examples_empty": empty_stems[:20],
            "examples_missing_label": missing_label_stems[:20],
        })

        print(
            f"\n  [Support Positive Filter] "
            f"input={len(train_pool_all)}  "
            f"positive={len(positive_stems)}  "
            f"empty_or_too_small={len(empty_stems)}  "
            f"missing_label={len(missing_label_stems)}  "
            f"threshold={args.min_support_positive_pixels}"
        )

        if len(positive_stems) < max(shots):
            msg = (
                f"正样本 support 候选不足：需要至少 {max(shots)}，"
                f"但只有 {len(positive_stems)}。"
            )
            if args.strict:
                raise RuntimeError(msg)
            info["warnings"].append(msg)

    elif args.allow_empty_support:
        info["warnings"].append(
            "allow_empty_support=True: 空建筑切片允许进入 support。"
            "正式论文实验不建议开启。"
        )
    else:
        info["warnings"].append(
            "无法执行 support positive filter: train_pool_label_dir 不存在。"
        )

    info["support_positive_filter"] = support_positive_filter_report

    # ----------------------------------------------------------------
    # Step 5: target_val
    #     - target_val 不默认过滤空图，保留 false-positive 检验能力。
    #     - 若显式 --filter_val_positive_only，则只保留有建筑 val。
    # ----------------------------------------------------------------
    target_val_stems: List[str] = []
    target_val_size = args.target_val_size
    val_image_dir = ""
    val_label_dir = ""

    if official_val_stems:
        val_image_dir = paths["whu_mix_val_image"]
        val_label_dir = paths["whu_mix_val_label"]
        val_pool = official_val_stems
    elif train_pool_all:
        val_image_dir = train_pool_image_dir
        val_label_dir = train_pool_label_dir
        val_pool = train_pool_all
    else:
        val_pool = []
        val_image_dir = ""
        val_label_dir = ""

    if args.filter_val_positive_only and val_pool and val_label_dir:
        val_pool, _, val_empty, val_missing = filter_positive_stems(
            val_pool,
            val_label_dir,
            min_positive_pixels=args.min_support_positive_pixels,
        )
        info["warnings"].append(
            f"filter_val_positive_only=True: target_val 也过滤了空建筑图。"
            f"过滤掉 empty={len(val_empty)}, missing_label={len(val_missing)}。"
        )

    if val_pool:
        # val 必须从 support 候选池之外抽，防止 support/val 交集。
        # 如果 val 来自官方独立 val 目录，通常天然不会与 train 重合；
        # 如果 val 来自同一个 train_pool，则后续 support 会显式排除 target_val。
        n_val = min(target_val_size, len(val_pool))
        target_val_stems = city_stratified_sample(
            val_pool,
            n_val,
            seed=99,
        )

    if target_val_stems and val_image_dir:
        p = write_manifest(
            manifest_dir,
            "target_val.txt",
            target_val_stems,
            val_image_dir,
        )
        info["splits"]["target_val"] = {
            "count": len(target_val_stems),
            "path": p,
        }
    else:
        info["splits"]["target_val"] = {
            "count": 0,
            "path": "",
            "warning": "target_val is empty",
        }

    # ----------------------------------------------------------------
    # Step 6: target_support
    #     support 从正样本候选池抽取，并排除 target_val / final_test。
    # ----------------------------------------------------------------
    outer_exclude: Set[str] = set(target_val_stems)

    # 如果 fallback 到 slim/pilot 数据，还要防止和 final_test stem 重叠。
    if not has_official_train:
        outer_exclude.update(final_test_stems)

    support_dict = nested_support_sample(
        support_candidate_pool,
        shots,
        seeds,
        exclude_set=outer_exclude,
    )

    for key, stems in support_dict.items():
        p = write_manifest(
            manifest_dir,
            f"{key}.txt",
            stems,
            train_pool_image_dir,
        )
        info["splits"][key] = {
            "count": len(stems),
            "path": p,
        }
    # ----------------------------------------------------------------
    # Step 7: target_pilot_test
    # ----------------------------------------------------------------
    pilot_sizes = [int(s.strip()) for s in args.pilot_test_sizes.split(",")]
    for ps in pilot_sizes:
        key = f"target_pilot_test_{ps}"
        pilot_stems = city_stratified_sample(final_test_stems, ps, seed=42)
        if pilot_stems and final_test_image_dir:
            p = write_manifest(manifest_dir, f"{key}.txt",
                               pilot_stems, final_test_image_dir)
            info["splits"][key] = {"count": len(pilot_stems), "path": p}

    return info


# ==========================================================================
# 模块 9: 审计引擎 (Audit Engine)
#     检查交集、存在性、嵌套关系
# ==========================================================================

def run_audit(
    manifest_dir: str,
    source_info: Dict,
    target_info: Dict,
    paths: Dict,
) -> Dict[str, any]:
    """
    运行全部审计检查。

    修复点:
      1. 不再写死 shots=[5,10,20] 和 seeds=[42,123,456]；
      2. 自动扫描所有 target_support_*_seed*.txt；
      3. strict_paper_ready 强制要求 final_test 接近 8402；
      4. 检查 support 实际数量是否等于文件名中的 shot。
    """
    manifest_path = Path(manifest_dir)
    audit: Dict[str, any] = {
        "timestamp": datetime.now().isoformat(),
        "manifest_dir": str(manifest_path.resolve()),
        "checks": {},
        "intersections": {},
        "paper_ready": target_info.get("paper_ready", False),
        "warnings": list(target_info.get("warnings", [])),
        "support_positive_filter": target_info.get("support_positive_filter", {}),
    }

    all_manifests = sorted(manifest_path.glob("*.txt"))
    audit["checks"]["manifest_files_found"] = len(all_manifests)

    empty_manifests: List[str] = []
    missing_paths: List[str] = []

    for mf in all_manifests:
        content = mf.read_text(encoding="utf-8").strip()
        if not content:
            empty_manifests.append(mf.name)
            continue

        for line in content.split("\n"):
            line = line.strip()
            if line and not Path(line).exists():
                missing_paths.append(line)

    audit["checks"]["empty_manifests"] = empty_manifests
    audit["checks"]["missing_files"] = len(missing_paths)
    if missing_paths:
        audit["checks"]["missing_file_examples"] = missing_paths[:10]

    def load_stems(mf_name: str) -> Set[str]:
        f = manifest_path / mf_name
        if not f.exists():
            return set()
        stems = set()
        for line in f.read_text(encoding="utf-8").strip().split("\n"):
            line = line.strip()
            if line:
                stems.add(Path(line).stem)
        return stems

    target_final = load_stems("target_final_test_8402.txt")
    target_val = load_stems("target_val.txt")

    final_count = len(target_final)
    final_is_full_8402 = final_count >= 8400
    audit["checks"]["target_final_test_count"] = final_count
    audit["checks"]["target_final_test_is_full_8402"] = final_is_full_8402

    if not final_is_full_8402:
        audit["warnings"].append(
            f"target_final_test_8402 has only {final_count} images; "
            f"paper-ready final test should be the full 8402 WHU-Mix test set."
        )

    support_files = sorted(manifest_path.glob("target_support_*_seed*.txt"))

    intersections: Dict[str, Dict[str, int]] = {}
    support_by_seed: Dict[int, Dict[int, Set[str]]] = defaultdict(dict)

    for sf in support_files:
        stem_name = sf.stem  # e.g., target_support_20_seed42
        parts = stem_name.split("_")

        try:
            shot = int(parts[2])
            seed = int(parts[3].replace("seed", ""))
        except Exception:
            audit["warnings"].append(f"Cannot parse support manifest name: {sf.name}")
            continue

        sup = load_stems(sf.name)
        common_val = sup & target_val
        common_test = sup & target_final

        intersections[stem_name] = {
            "count": len(sup),
            "expected_shot": shot,
            "intersect_with_val": len(common_val),
            "intersect_with_final_test": len(common_test),
        }

        if len(sup) != shot:
            audit["warnings"].append(
                f"{stem_name} has {len(sup)} samples, expected {shot}."
            )

        if common_val:
            audit["warnings"].append(
                f"{stem_name} shares {len(common_val)} files with target_val."
            )

        if common_test:
            audit["warnings"].append(
                f"{stem_name} shares {len(common_test)} files with final_test."
            )

        support_by_seed[seed][shot] = sup

    common_val_final = target_val & target_final
    audit["checks"]["target_val_count"] = len(target_val)
    audit["checks"]["target_val_intersect_final_test"] = len(common_val_final)
    if common_val_final:
        audit["warnings"].append(
            f"target_val shares {len(common_val_final)} files with final_test."
        )

    nest_ok = True
    for seed, shot_map in support_by_seed.items():
        shot_list = sorted(shot_map.keys())
        for small, large in zip(shot_list[:-1], shot_list[1:]):
            if not shot_map[small].issubset(shot_map[large]):
                nest_ok = False
                audit["warnings"].append(
                    f"seed={seed}: {small}-shot is NOT subset of {large}-shot."
                )

    audit["checks"]["nested_support_ok"] = nest_ok
    support_positive_info = audit.get("support_positive_filter", {})
    support_positive_ok = True

    if support_positive_info.get("enabled", False):
        support_positive_ok = (
            support_positive_info.get("positive_candidates", 0)
            >= max(
                [
                    v.get("expected_shot", 0)
                    for v in intersections.values()
                ] or [0]
            )
        )

    audit["checks"]["support_positive_filter_ok"] = support_positive_ok

    all_support_no_val = all(
        v.get("intersect_with_val", 0) == 0 for v in intersections.values()
    )
    all_support_no_test = all(
        v.get("intersect_with_final_test", 0) == 0 for v in intersections.values()
    )
    all_support_count_ok = all(
        v.get("count", -1) == v.get("expected_shot", -2)
        for v in intersections.values()
    )

    audit["checks"]["support_val_intersection_ok"] = all_support_no_val
    audit["checks"]["support_test_intersection_ok"] = all_support_no_test
    audit["checks"]["support_count_ok"] = all_support_count_ok

    audit["checks"]["strict_paper_ready"] = (
        audit["paper_ready"]
        and final_is_full_8402
        and len(target_val) > 0
        and not empty_manifests
        and not missing_paths
        and nest_ok
        and all_support_no_val
        and all_support_no_test
        and all_support_count_ok
        and support_positive_ok
        and len(common_val_final) == 0
    )

    audit["intersections"] = intersections
    return audit

# ==========================================================================
# 模块 10: 城市分布报告 (City Distribution Report)
# ==========================================================================

def build_city_distribution_report(
    manifest_dir: str,
) -> Dict[str, any]:
    """扫描所有 manifest 文件, 统计每个 split 的城市分布."""
    manifest_path = Path(manifest_dir)
    report: Dict[str, any] = {}

    for mf in sorted(manifest_path.glob("*.txt")):
        cities: Dict[str, int] = defaultdict(int)
        total = 0
        for line in mf.read_text(encoding="utf-8").strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            total += 1
            city = infer_city_from_filename(line)
            cities[city] += 1

        report[mf.stem] = {
            "total": total,
            "cities": dict(sorted(cities.items(), key=lambda x: -x[1])),
        }

    return report


# ==========================================================================
# 模块 11: README 生成
# ==========================================================================

README_TEMPLATE = """# E0 Data Splits — Manifest & Audit

> 生成时间: {timestamp}
> 生成脚本: `src/data/make_splits_e0.py`
> 项目根目录: {project_root}

---

## 1. E0 的目的

E0 是项目所有后续实验 (E1–E8) 的**唯一数据划分入口**。所有实验必须使用本目录中的 manifest 文件加载数据, 以保证划分的可复现性和零交集。

## 2. Manifest 文件说明

| 文件 | 用途 | 是否 Paper-Ready |
|------|------|:---:|
| `source_train.txt` | 源域 (WHU aerial) 训练集 | ✅ auxiliary |
| `source_val.txt` | 源域验证集 | ✅ auxiliary |
| `source_test.txt` | 源域测试集 | ✅ auxiliary |
| `target_train_pool_all.txt` | 目标域全部可用训练候选池 | {paper_target_status} |
| `target_val.txt` | 目标域验证集 | {paper_target_status} |
| `target_support_*_seed*.txt` | Few-shot support (嵌套) | {paper_target_status} |
| `target_pilot_test_500.txt` | 试点 500 张测试 | ⚠️ pilot only |
| `target_pilot_test_1000.txt` | 试点 1000 张测试 | ⚠️ pilot only |
| `target_final_test_8402.txt` | 最终论文测试集 | {paper_test_status} |

## 3. 哪些 split 可用于正式论文

{paper_section}

## 4. 哪些 split 只用于 pilot

- `target_pilot_test_500.txt` 和 `target_pilot_test_1000.txt` 仅用于快速调试和模型筛选
- **禁止在 pilot test 上调阈值、调模型、选择 checkpoint**
- 最终论文主结果必须使用 `target_final_test_8402.txt`

## 5. 后续实验如何使用这些 manifest

### E5 (LoRA PEFT) / E6 (GBG-SAM3):
```python
# 读取 support
with open("data/splits/e0_manifest/target_support_10_seed42.txt") as f:
    support_files = [line.strip() for line in f if line.strip()]

# 读取 val
with open("data/splits/e0_manifest/target_val.txt") as f:
    val_files = [line.strip() for line in f if line.strip()]
```

### 注意事项:
- manifest 中每行是**绝对路径**, DataLoader 可直接使用 `cv2.imread(line)`
- 如果现有 Dataset 类不支持 manifest 输入, 需要新增构造参数 `manifest_path`

## 6. 禁止事项

- ❌ 禁止在 final_test 上调阈值
- ❌ 禁止在 final_test 上选择 checkpoint
- ❌ 禁止 support 和 val 之间有交集
- ❌ 禁止 support/val 与 final_test 之间有交集

## 7. 生成命令

{generation_commands}

---

*此文件由 `make_splits_e0.py` 自动生成, 请勿手动编辑。*
*审计报告详见 `split_audit_report.json`。*
"""


def generate_readme(
    manifest_dir: str,
    target_info: Dict,
    audit: Dict,
    project_root: str,
    args: argparse.Namespace,
) -> str:
    paper_ready = audit.get("paper_ready", False)
    strict_paper_ready = audit["checks"].get("strict_paper_ready", False)

    if paper_ready:
        paper_target_status = "✅ paper-ready"
    else:
        paper_target_status = "⚠️ pilot only"

    if target_info["splits"].get("target_final_test", {}).get("is_full_8402"):
        paper_test_status = "✅ 8402 full"
    else:
        paper_test_status = "⚠️ partial (not full 8402)"

    if strict_paper_ready:
        paper_section = (
            "- 所有 target split 均为 paper-ready\n"
            "- 通过严格审计 (零交集 + 嵌套正确 + 8402 full test)\n"
            "- 可直接用于 Remote Sensing 期刊投稿"
        )
    else:
        paper_section = (
            "- ⚠️ **当前不是 paper-ready 状态**\n"
        )
        for w in audit.get("warnings", [])[:3]:
            paper_section += f"  - {w}\n"
        paper_section += (
            "- 要成为 paper-ready, 请确保:\n"
            "  1. WHU-Mix 官方 train+label 存在\n"
            "  2. WHU-Mix 官方 val+label 存在\n"
            "  3. whu_mix_full_test 8402 张完整\n"
        )

    generation_commands = (
        "```bash\n"
        f"# 当前运行命令:\n"
        f"python src/data/make_splits_e0.py"
    )
    if args.dry_run:
        generation_commands += " --dry_run"
    if args.strict:
        generation_commands += " --strict"
    if args.expand_source:
        generation_commands += " --expand_source"
    generation_commands += "\n"
    generation_commands += (
        "\n"
        "# 严格 paper-ready 模式:\n"
        "python src/data/make_splits_e0.py --strict\n"
        "\n"
        "# 扩展 source + 1000 val:\n"
        "python src/data/make_splits_e0.py --expand_source --target_val_size 1000\n"
        "```\n"
    )

    content = README_TEMPLATE.format(
        timestamp=audit.get("timestamp", datetime.now().isoformat()),
        project_root=project_root,
        paper_target_status=paper_target_status,
        paper_test_status=paper_test_status,
        paper_section=paper_section,
        generation_commands=generation_commands,
    )

    readme_path = Path(manifest_dir) / "README_E0_SPLITS.md"
    readme_path.write_text(content, encoding="utf-8")
    return str(readme_path)


# ==========================================================================
# 模块 12: 控制台表格输出
# ==========================================================================

def print_summary_table(
    source_info: Dict,
    target_info: Dict,
    audit: Dict,
    paths: Dict,
) -> None:
    """打印清晰的汇总表格."""
    print(f"\n{'='*90}")
    print(f"  E0 数据划分与审计汇总")
    print(f"  项目根目录: {paths['project_root']}")
    print(f"  manifest 目录: {audit['manifest_dir']}")
    print(f"{'='*90}")

    print(f"\n  {'Split':<32} {'Count':<8} {'来源':<20} {'状态'}")
    print(f"  {'-'*32} {'-'*8} {'-'*20} {'-'*20}")

    # Source
    for key, info in source_info.get("splits", {}).items():
        status = "[OK] auxiliary" if info.get("count", 0) > 0 else "[X] missing"
        source_label = "slim source_whu"
        print(f"  {key:<32} {info.get('count', 0):<8} {source_label:<20} {status}")

    # Target
    for key, info in target_info.get("splits", {}).items():
        count = info.get("count", 0)
        if "support" in key:
            source_label = "train_pool (strat)"
            status = "[OK]" if count > 0 else "[X] empty"
        elif "val" in key:
            source_label = "official or train_pool"
            status = "[OK] paper" if audit.get("paper_ready") else "[!] pilot"
        elif "pilot_test" in key:
            source_label = "final_test subset"
            status = "[!] pilot only"
        elif "final_test" in key:
            source_label = "full test"
            status = "[OK] 8402" if info.get("is_full_8402") else "[!] partial"
        elif "train_pool_all" in key:
            source_label = "all train candidates"
            status = "[OK]" if count > 0 else "[X] empty"
        else:
            source_label = "-"
            status = "-"
        print(f"  {key:<32} {count:<8} {source_label:<20} {status}")

    # audit summary
    print(f"\n  --- Audit Results ---")
    print(f"  Zero-intersect (support/val/test): "
          f"{'[OK] pass' if audit['checks'].get('strict_paper_ready') else '[!] has intersect'}")
    print(f"  Nested (5<10<20):                  "
          f"{'[OK] pass' if audit['checks'].get('nested_support_ok') else '[X] fail'}")
    print(f"  File existence:                    "
          f"{'[OK] all exist' if audit['checks'].get('missing_files', 0) == 0 else '[X] missing'}")
    print(f"  Paper-Ready:                       "
          f"{'[OK] YES' if audit['checks'].get('strict_paper_ready') else '[!] NO (pilot mode)'}")
    spf = audit.get("support_positive_filter", {})
    if spf:
        print(f"\n  --- Support Positive Filter ---")
        print(f"  Enabled:                       {spf.get('enabled')}")
        print(f"  Min positive pixels:           {spf.get('min_positive_pixels')}")
        print(f"  Input candidates:              {spf.get('input_candidates')}")
        print(f"  Positive candidates:           {spf.get('positive_candidates')}")
        print(f"  Empty / too small:             {spf.get('empty_or_too_small')}")
        print(f"  Missing labels:                {spf.get('missing_labels')}")
    if audit.get("warnings"):
        print(f"\n  [!] Warnings:")
        for w in audit["warnings"]:
            print(f"     - {w}")
    print(f"{'='*90}\n")


# ==========================================================================
# 模块 13: CLI 参数解析
# ==========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E0 数据划分与泄露审计脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # pilot 模式 (dry-run)
  python src/data/make_splits_e0.py --dry_run

  # pilot 模式 (写盘)
  python src/data/make_splits_e0.py

  # 严格 paper-ready 模式
  python src/data/make_splits_e0.py --strict

  # 扩展 source split + 更大 val
  python src/data/make_splits_e0.py --expand_source --target_val_size 1000
        """,
    )
    parser.add_argument("--project_root", type=str, default=None,
                        help="项目根目录 (默认自动检测)")
    parser.add_argument("--manifest_dir", type=str,
                        default="data/splits/e0_manifest",
                        help="输出 manifest 目录")
    parser.add_argument("--target_val_size", type=int, default=500,
                        help="target val 规模 (默认 500)")
    parser.add_argument("--pilot_test_sizes", type=str, default="500,1000",
                        help="pilot test 规模, 逗号分隔 (默认 500,1000)")
    parser.add_argument("--support_shots", type=str, default="5,10,20",
                        help="support shot 值, 逗号分隔 (默认 5,10,20)")
    parser.add_argument("--seeds", type=str, default="42,123,456",
                        help="随机种子, 逗号分隔 (默认 42,123,456)")
    parser.add_argument("--expand_source", action="store_true",
                        help="尝试从 WHU aerial_0.3 官方全量目录扩展 source")
    parser.add_argument("--source_test_size", type=int, default=500,
                        help="source test 规模 (默认 500)")
    parser.add_argument("--dry_run", action="store_true",
                        help="只打印计划, 不写文件")
    parser.add_argument("--strict", action="store_true",
                        help="严格模式: 正式 target trainval 不存在则报错退出")
    parser.add_argument(
        "--min_support_positive_pixels",
        type=int,
        default=DEFAULT_MIN_SUPPORT_POSITIVE_PIXELS,
        help=(
            "few-shot support 最小建筑前景像素数。"
            "低于该阈值的切片不会进入 target support。默认 100。"
        ),
    )
    parser.add_argument(
        "--allow_empty_support",
        action="store_true",
        help=(
            "允许空建筑切片进入 support。默认不允许。"
            "正式论文实验不要开启该选项。"
        ),
    )
    parser.add_argument(
        "--filter_val_positive_only",
        action="store_true",
        help=(
            "是否也过滤 target val 中的空建筑切片。默认不启用。"
            "正式评估建议保留 val 空图，因此一般不要开启。"
        ),
    )
    return parser.parse_args()



# ==========================================================================
# 模块 14: 主函数
# ==========================================================================

def main() -> None:
    args = parse_args()

    # --- 确定 project_root ---
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        # 自动检测: make_splits_e0.py 在 src/data/, 向上两级 = project_root
        project_root = Path(__file__).resolve().parent.parent.parent

    print(f"项目根目录: {project_root}")
    print(f"manifest 目录: {args.manifest_dir}")
    print(f"模式: {'STRICT' if args.strict else 'NORMAL'}"
          f"{' (dry-run)' if args.dry_run else ''}")

    # --- 探测路径 ---
    print("\n[模块 2] 探测数据路径 ...")
    paths = probe_project_paths(project_root)

    def _ok(path_key: str) -> str:
        return "OK" if paths[path_key] else "--"

    print(f"  slim_root:              {_ok('slim_root'):>3} "
          f"{paths['slim_root'][:60] if paths['slim_root'] else 'N/A'}")
    print(f"  whu_mix_train_image:    {_ok('whu_mix_train_image'):>3} "
          f"{paths['whu_mix_train_image'][:60] if paths['whu_mix_train_image'] else 'N/A'}")
    print(f"  whu_mix_train_label:    {_ok('whu_mix_train_label'):>3}")
    print(f"  whu_mix_val_image:      {_ok('whu_mix_val_image'):>3}")
    print(f"  whu_mix_full_test:      {_ok('whu_mix_full_test_image'):>3}")
    print(f"  source_train:           {_ok('source_train_image'):>3}")
    print(f"  target_train_pool:      {_ok('target_train_pool_image'):>3}")
    print(f"  target_test_slim:       {_ok('target_test_slim_image'):>3}")

    # --- strict 检查 ---
    if args.strict:
        if not paths["whu_mix_train_image"] or not paths["whu_mix_train_label"]:
            print("\n[X] Strict mode failed: WHU-Mix official train+label not found")
            print("   Please place the official data, or use non-strict mode (remove --strict)")
            sys.exit(1)
        if not paths["whu_mix_full_test_image"]:
            print("\n[X] Strict mode failed: whu_mix_full_test (8402) not found")
            print("   Please run the full test set preprocessing script first")
            sys.exit(1)

    dry_run_tmp = None
    original_manifest_dir = args.manifest_dir

    if args.dry_run:
        print("\n  [DRY-RUN] Print plan only, no files written to the requested manifest_dir.")
        dry_run_tmp = tempfile.TemporaryDirectory()
        args.manifest_dir = dry_run_tmp.name
        print(f"  [DRY-RUN] Temporary manifest dir: {args.manifest_dir}")

    # --- 构建 split ---
    print("\n[模块 7] 构建 Source Splits ...")
    source_info = build_source_splits(paths, args.manifest_dir,
                                       args.expand_source, args.source_test_size)

    print("\n[模块 8] 构建 Target Splits ...")
    target_info = build_target_splits(paths, args.manifest_dir, args)

    if args.dry_run:
        # 打印计划
        print(f"\n  Source splits: {list(source_info['splits'].keys())}")
        for k, v in source_info["splits"].items():
            print(f"    {k}: {v.get('count', 0)} 样本")
        print(f"\n  Target splits: {list(target_info['splits'].keys())}")
        for k, v in target_info["splits"].items():
            print(f"    {k}: {v.get('count', 0)} 样本")
        print(f"  paper_ready: {target_info.get('paper_ready')}")
        if target_info.get("warnings"):
            print(f"  warnings:")
            for w in target_info["warnings"]:
                print(f"    - {w}")

        # 仍生成审计报告
        audit = run_audit(args.manifest_dir, source_info, target_info, paths)

        # 城市分布报告
        city_report = build_city_distribution_report(args.manifest_dir)
        city_path = Path(args.manifest_dir) / "city_distribution_report.json"
        # dry-run 也打印
        print(f"\n  城市分布摘要:")
        for split_key, info in sorted(city_report.items()):
            if split_key.startswith("target_support"):
                cities_str = ", ".join(
                    f"{c}={n}" for c, n in sorted(info["cities"].items())
                )
                print(f"    {split_key}: {info['total']} [{cities_str}]")

        print_summary_table(source_info, target_info, audit, paths)
        args.manifest_dir = original_manifest_dir
        if dry_run_tmp is not None:
            dry_run_tmp.cleanup()
        return

    # --- 写盘 ---
    os.makedirs(args.manifest_dir, exist_ok=True)

    print("\n[模块 9] 运行审计 ...")
    audit = run_audit(args.manifest_dir, source_info, target_info, paths)

    # 写 split_audit_report.json
    audit_path = Path(args.manifest_dir) / "split_audit_report.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"  审计报告已保存: {audit_path}")

    # 城市分布报告
    city_report = build_city_distribution_report(args.manifest_dir)
    city_path = Path(args.manifest_dir) / "city_distribution_report.json"
    with open(city_path, "w", encoding="utf-8") as f:
        json.dump(city_report, f, ensure_ascii=False, indent=2)
    print(f"  城市分布报告已保存: {city_path}")

    # README
    readme_path = generate_readme(
        args.manifest_dir, target_info, audit,
        str(project_root), args,
    )
    print(f"  README 已保存: {readme_path}")

    # 汇总表格
    print_summary_table(source_info, target_info, audit, paths)

    # 城市分布摘要
    print(f"\n  城市分布摘要 (support):")
    for key, info in sorted(city_report.items()):
        if key.startswith("target_support"):
            cities_str = ", ".join(
                f"{c}={n}" for c, n in sorted(info["cities"].items())
            )
            print(f"    {key}: {info['total']} [{cities_str}]")


# ==========================================================================
# 模块 15: 主入口
# ==========================================================================
if __name__ == "__main__":
    main()
