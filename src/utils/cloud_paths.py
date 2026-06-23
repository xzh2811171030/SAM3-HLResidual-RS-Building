"""
=============================================================================
cloud_paths.py  ---  云端 / 本地路径自适应模块 (v2.0 双轨制重构)
=============================================================================
功能说明:
  1. 自动检测运行环境 (Linux 云端 vs Windows 本地)
  2. 支持 "双轨制" 数据路径:
     - 瘦身版调试 (processed_slim): 本地 2GB, 用于快速调试
     - 全量测试集 (whu_mix_full_test): 云端 8402 张, 用于论文最终评估
  3. 支持域切换: source_whu (源域) vs target_whu_mix (目标域)
  4. 所有下游脚本通过统一的 get_paths() / get_domain_paths() 获取路径

路径结构:
  {PROJECT_ROOT}/data/raw/processed_slim/
    source_whu/
      train/images/          训练影像
      train/dual_channel_labels/  双通道标签
      train/bboxes.json           建筑物 bbox 标注
      val/images/            验证影像 (地理绝对隔离)
      val/dual_channel_labels/
      val/bboxes.json
      test/images/           瘦身测试集
      test/dual_channel_labels/
      test/bboxes.json
    target_whu_mix/
      train_pool/images/     Few-shot 候选池
      train_pool/dual_channel_labels/
      train_pool/bboxes.json
      test/images/           目标域瘦身测试集
      test/dual_channel_labels/
      test/bboxes.json

  {PROJECT_ROOT}/data/raw/whu_mix_full_test/
    test/image/              全量测试影像
    test/label/              全量测试标签
    dual_channel_labels/     生成的双通道标签
    bbox.json                建筑物 bbox 标注

用法:
  from cloud_paths import get_paths, get_domain_paths, get_platform_name

  paths = get_paths()
  domain_paths = get_domain_paths("source", full_test=False)
  print(domain_paths["train_image_dir"])
=============================================================================
"""

# ==========================================================================
# 模块 1: 导入与常量定义
# ==========================================================================
import platform
from pathlib import Path
from typing import Dict, Optional

_WINDOWS_PROJECT_ROOT: str = r"/path/to/project"
_LINUX_PROJECT_ROOT: str = /path/to/project"

_paths_cache: Dict[str, str] = {}
_domain_paths_cache: Dict[str, Dict[str, str]] = {}


# ==========================================================================
# 模块 2: 基础路径计算函数
# ==========================================================================
def _get_project_root() -> Path:
    is_linux = platform.system() == "Linux"
    return Path(_LINUX_PROJECT_ROOT if is_linux else _WINDOWS_PROJECT_ROOT)


def _detect_slim_root(project_root: Path) -> Path:
    slim_candidates = [
        project_root / "data" / "raw" / "processed_slim",
        project_root / "data" / "processed_slim",
    ]
    for candidate in slim_candidates:
        if candidate.exists():
            return candidate
    return slim_candidates[0]


def _detect_full_test_root(project_root: Path) -> Path:
    candidates = [
        project_root / "data" / "raw" / "whu_mix_full_test",
        project_root / "data" / "whu_mix_full_test",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


# ==========================================================================
# 模块 3: 全局路径计算 (兼容旧接口)
# ==========================================================================
def _compute_paths() -> Dict[str, str]:
    root = _get_project_root()
    slim_root = _detect_slim_root(root)
    full_test_root = _detect_full_test_root(root)

    return {
        "project_root": str(root),

        "raw_image_dir": str(root / "data" / "raw" / "train_demo" / "image"),
        "dual_label_dir": str(root / "data" / "processed" / "dual_channel_demo"),
        "bbox_json_path": str(root / "data" / "processed" / "bbox_demo.json"),
        "val_image_dir": str(root / "data" / "raw" / "val_demo" / "image"),
        "val_label_dir": str(root / "data" / "raw" / "val_demo" / "label"),

        "slim_root": str(slim_root),
        "full_test_root": str(full_test_root),

        "weights_dir": str(root / "weights"),
        "sam3_checkpoint": str(root / "weights" / "sam3.pt"),
        "results_dir": str(root / "results"),

        "is_cloud": platform.system() == "Linux",
    }


# ==========================================================================
# 模块 4: 域相关路径计算 (核心新接口)
# ==========================================================================
def _compute_domain_paths(domain: str, full_test: bool = False) -> Dict[str, str]:
    root = _get_project_root()
    slim_root = _detect_slim_root(root)
    full_test_root = _detect_full_test_root(root)

    if domain == "source":
        domain_dir_name = "source_whu"
        train_subdir = "train"
        val_subdir = "val"
        test_subdir = "test"
    elif domain == "target":
        domain_dir_name = "target_whu_mix"
        train_subdir = "train_pool"
        val_subdir = "train_pool"
        test_subdir = "test"
    else:
        raise ValueError(f"未知 domain: '{domain}', 可选: 'source' 或 'target'")

    domain_base = slim_root / domain_dir_name

    train_root = domain_base / train_subdir
    val_root = domain_base / val_subdir
    test_root = domain_base / test_subdir

    train_image_dir = str(train_root / "images")
    train_label_dir = str(train_root / "dual_channel_labels")
    train_bbox_json = str(train_root / "bboxes.json")
    val_image_dir = str(val_root / "images")
    val_label_dir = str(val_root / "dual_channel_labels")
    val_bbox_json = str(val_root / "bboxes.json")

    if full_test:
        test_image_dir = str(full_test_root / "test" / "image")
        test_label_dir = str(full_test_root / "test" / "label")
        test_dual_dir = str(full_test_root / "dual_channel_labels")
        test_bbox_json = str(full_test_root / "bbox.json")

        if not Path(test_image_dir).exists():
            raise FileNotFoundError(
                f"全量测试集未找到: {test_image_dir}\n"
                f"请先运行全量测试集预处理脚本:\n"
                f"  python src/data/preprocess_full_test.py"
            )
    else:
        test_image_dir = str(test_root / "images")
        test_label_dir = str(test_root / "dual_channel_labels")
        test_dual_dir = str(test_root / "dual_channel_labels")
        test_bbox_json = str(test_root / "bboxes.json")

    return {
        "domain": domain,
        "full_test": full_test,
        "domain_base": str(domain_base),

        "train_image_dir": train_image_dir,
        "train_label_dir": train_label_dir,
        "train_bbox_json": train_bbox_json,

        "val_image_dir": val_image_dir,
        "val_label_dir": val_label_dir,
        "val_bbox_json": val_bbox_json,

        "test_image_dir": test_image_dir,
        "test_label_dir": test_label_dir,
        "test_dual_dir": test_dual_dir,
        "test_bbox_json": test_bbox_json,

        "weights_dir": str(root / "weights"),
        "sam3_checkpoint": str(root / "weights" / "sam3.pt"),
        "results_dir": str(root / "results"),
        "project_root": str(root),
        "is_cloud": platform.system() == "Linux",
    }


# ==========================================================================
# 模块 5: 公开接口
# ==========================================================================
def get_paths() -> Dict[str, str]:
    global _paths_cache
    if not _paths_cache:
        _paths_cache = _compute_paths()
    return _paths_cache


def get_domain_paths(
    domain: str = "source",
    full_test: bool = False,
    force_refresh: bool = False,
) -> Dict[str, str]:
    global _domain_paths_cache
    cache_key = f"{domain}_{full_test}"
    if force_refresh or cache_key not in _domain_paths_cache:
        _domain_paths_cache[cache_key] = _compute_domain_paths(domain, full_test)
    return _domain_paths_cache[cache_key]


def get_platform_name() -> str:
    return "cloud-linux" if platform.system() == "Linux" else "local-windows"


def get_supported_domains() -> list:
    return ["source", "target"]


# ==========================================================================
# 模块 6: 自测入口
# ==========================================================================
if __name__ == "__main__":
    print(f"运行环境: {get_platform_name()}")
    print()

    print("=== 全局路径 (兼容旧接口) ===")
    paths = get_paths()
    for k, v in paths.items():
        exists = "✓" if Path(v).exists() else "✗"
        print(f"  [{exists}] {k:20s}: {v}")

    print()
    for domain in ["source", "target"]:
        for ft in [False, True]:
            if ft and domain != "source":
                continue
            label = "全量" if ft else "瘦身"
            print(f"=== 域路径: domain={domain}, {label}测试集 ===")
            try:
                dp = get_domain_paths(domain, full_test=ft)
                for k, v in dp.items():
                    if k in ("domain", "full_test", "is_cloud"):
                        continue
                    exists = "✓" if Path(v).exists() else "✗"
                    print(f"  [{exists}] {k:20s}: {v}")
            except FileNotFoundError as e:
                print(f"  [跳过] {e}")
            print()
