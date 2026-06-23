# -*- coding: utf-8 -*-
"""
export_e0_target_subset.py
=============================================================================
功能：
  读取 E0 manifest 中的 target support / target val 图像路径，
  自动复制图像，并从原始 label 生成 dual_channel_labels 与 bboxes.json。

输出目录：
  /path/to/project/data/raw/processed_slim/target_whu_mix/train_pool/
    images/
    dual_channel_labels/
    bboxes.json

  /path/to/project/data/raw/processed_slim/target_whu_mix/val/
    images/
    dual_channel_labels/
    bboxes.json

推荐运行：
  python src/data/export_e0_target_subset.py

可选：
  python src/data/export_e0_target_subset.py --clear_output
  python src/data/export_e0_target_subset.py --dry_run
=============================================================================
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import cv2
import numpy as np


# =============================================================================
# 1. 默认路径配置
# =============================================================================

DEFAULT_PROJECT_ROOT = Path(
    r"/path/to/project"
)

IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
LABEL_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


# =============================================================================
# 2. 基础工具函数
# =============================================================================

def norm_path(p: str) -> Path:
    """兼容 Windows / Unix 斜杠。"""
    return Path(p.strip().strip('"').strip("'").replace("\\", "/"))


def read_manifest(manifest_path: Path) -> List[Path]:
    """读取 manifest txt，每行一个图像路径。"""
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")

    paths: List[Path] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        paths.append(norm_path(line))

    return paths


def make_hash_suffix(path: Path, n: int = 8) -> str:
    """用于处理同名文件冲突。"""
    s = str(path.resolve()).replace("\\", "/")
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:n]


def safe_copy(src: Path, dst: Path) -> None:
    """安全复制，源和目标相同时跳过。"""
    src_r = src.resolve()
    dst_r = dst.resolve()

    if src_r == dst_r:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def ensure_clean_dir(path: Path, clear: bool = False) -> None:
    """创建目录；clear=True 时清空目录。"""
    if clear and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 3. label 查找与处理
# =============================================================================

def candidate_label_dirs_from_image(image_path: Path) -> List[Path]:
    """
    根据 image 路径推断可能的 label 目录。

    常见结构：
      train/image/xxx.tif
      train/label/xxx.tif

      train/images/xxx.tif
      train/labels/xxx.tif

      train/images/xxx.tif
      train/dual_channel_labels/xxx.png
    """
    image_path = image_path.resolve()
    parent = image_path.parent
    grand = parent.parent

    candidates: List[Path] = []

    # 1. 兄弟目录
    for dirname in [
        "label", "labels", "mask", "masks",
        "annotation", "annotations",
        "dual_channel_labels",
    ]:
        candidates.append(grand / dirname)

    # 2. 字符串替换兜底
    s = str(image_path).replace("\\", "/")

    replace_pairs = [
        ("/image/", "/label/"),
        ("/images/", "/label/"),
        ("/images/", "/labels/"),
        ("/images/", "/dual_channel_labels/"),
        ("/train/image/", "/train/label/"),
        ("/train/images/", "/train/label/"),
        ("/train/images/", "/train/labels/"),
    ]

    for old, new in replace_pairs:
        if old in s:
            candidates.append(Path(s.replace(old, new)).parent)

    # 去重，保留顺序
    uniq: List[Path] = []
    seen: Set[str] = set()
    for d in candidates:
        key = str(d).replace("\\", "/").lower()
        if key not in seen:
            seen.add(key)
            uniq.append(d)

    return uniq


def find_label_for_image(image_path: Path) -> Optional[Path]:
    """
    为一张 image 找对应 label。
    优先查找同 stem 的 label 文件。
    """
    stem = image_path.stem
    label_dirs = candidate_label_dirs_from_image(image_path)

    for d in label_dirs:
        if not d.exists():
            continue

        for ext in LABEL_EXTS:
            cand = d / f"{stem}{ext}"
            if cand.exists():
                return cand

    return None


def read_binary_mask(label_path: Path) -> np.ndarray:
    """
    读取 label 并转成 0/255 二值 mask。
    如果 label 是 RGB / dual label，也统一转灰度。
    """
    img = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取 label: {label_path}")

    if img.ndim == 3:
        # 如果是 BGR/RGB/dual-channel png，转灰度即可。
        # 对二值标签来说 >0 阈值足够稳。
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = (img > 0).astype(np.uint8) * 255
    return mask


def make_dual_channel_label(binary_mask: np.ndarray) -> np.ndarray:
    """
    生成双通道 label：
      B 通道：原始 mask
      G 通道：boundary
      R 通道：0

    OpenCV 写 PNG 使用 BGR 顺序。
    """
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    boundary = cv2.subtract(binary_mask, eroded)

    zeros = np.zeros_like(binary_mask, dtype=np.uint8)
    dual_bgr = cv2.merge([binary_mask, boundary, zeros])
    return dual_bgr


def extract_bboxes(binary_mask: np.ndarray, min_area: int = 4) -> List[List[int]]:
    """
    从二值 mask 中提取连通域 bbox。
    输出格式：
      [x1, y1, x2, y2]
    其中 x2/y2 使用 x+w/y+h 的半开区间风格。
    """
    mask_uint8 = (binary_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes: List[List[int]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            continue

        boxes.append([int(x), int(y), int(x + w), int(y + h)])

    return boxes


# =============================================================================
# 4. manifest 收集逻辑
# =============================================================================

def collect_support_paths(manifest_dir: Path) -> List[Path]:
    """
    只读取 target_support_20_seed*.txt。
    因为 5/10-shot 已经嵌套在 20-shot 内。
    """
    support_files = sorted(manifest_dir.glob("target_support_20_seed*.txt"))
    if not support_files:
        raise FileNotFoundError(
            f"未找到 target_support_20_seed*.txt: {manifest_dir}"
        )

    all_paths: List[Path] = []
    for mf in support_files:
        all_paths.extend(read_manifest(mf))

    # 按文件绝对路径去重
    uniq: Dict[str, Path] = {}
    for p in all_paths:
        key = str(p.resolve()).replace("\\", "/")
        uniq[key] = p

    return list(uniq.values())


def collect_val_paths(manifest_dir: Path) -> List[Path]:
    val_manifest = manifest_dir / "target_val.txt"
    return read_manifest(val_manifest)


def resolve_output_filename(
    src_image: Path,
    used_names: Dict[str, str],
) -> str:
    """
    默认保留原始文件名。
    如果不同源路径出现同名文件，则加 hash 后缀避免覆盖。
    """
    name = src_image.name
    src_key = str(src_image.resolve()).replace("\\", "/")

    if name not in used_names:
        used_names[name] = src_key
        return name

    if used_names[name] == src_key:
        return name

    # 同名但不同源文件，添加 hash
    suffix = make_hash_suffix(src_image)
    new_name = f"{src_image.stem}__{suffix}{src_image.suffix}"
    used_names[new_name] = src_key
    return new_name


# =============================================================================
# 5. 核心导出逻辑
# =============================================================================

def export_subset(
    image_paths: List[Path],
    out_root: Path,
    clear_output: bool = False,
    dry_run: bool = False,
    min_area: int = 4,
) -> Dict:
    """
    将 image_paths 导出到 out_root：
      out_root/images
      out_root/dual_channel_labels
      out_root/bboxes.json
    """
    out_img_dir = out_root / "images"
    out_dual_dir = out_root / "dual_channel_labels"
    out_bbox_path = out_root / "bboxes.json"

    if not dry_run:
        ensure_clean_dir(out_img_dir, clear=clear_output)
        ensure_clean_dir(out_dual_dir, clear=clear_output)
        out_root.mkdir(parents=True, exist_ok=True)

    bbox_json: Dict[str, List[List[int]]] = {}

    report = {
        "out_root": str(out_root).replace("\\", "/"),
        "requested_count": len(image_paths),
        "exported_count": 0,
        "missing_images": [],
        "missing_labels": [],
        "failed_items": [],
        "bbox_image_count": 0,
        "bbox_total_count": 0,
        "name_mapping": {},
    }

    used_names: Dict[str, str] = {}

    for idx, src_img in enumerate(image_paths, start=1):
        try:
            src_img = src_img.resolve()

            if not src_img.exists():
                report["missing_images"].append(str(src_img).replace("\\", "/"))
                continue

            src_label = find_label_for_image(src_img)
            if src_label is None:
                report["missing_labels"].append(str(src_img).replace("\\", "/"))
                continue

            out_name = resolve_output_filename(src_img, used_names)
            out_img_path = out_img_dir / out_name
            out_dual_path = out_dual_dir / f"{Path(out_name).stem}.png"

            binary_mask = read_binary_mask(src_label)
            dual_label = make_dual_channel_label(binary_mask)
            boxes = extract_bboxes(binary_mask, min_area=min_area)

            if not dry_run:
                safe_copy(src_img, out_img_path)
                ok = cv2.imwrite(str(out_dual_path), dual_label)
                if not ok:
                    raise RuntimeError(f"cv2.imwrite 失败: {out_dual_path}")

            # Dataset 使用 orig_filename = img_path.name 查 bboxes
            bbox_json[out_name] = boxes

            # 兼容一些旧脚本可能用 stem 查 bbox
            bbox_json[Path(out_name).stem] = boxes

            report["exported_count"] += 1
            report["bbox_image_count"] += 1
            report["bbox_total_count"] += len(boxes)
            report["name_mapping"][str(src_img).replace("\\", "/")] = {
                "image": str(out_img_path).replace("\\", "/"),
                "dual_label": str(out_dual_path).replace("\\", "/"),
                "source_label": str(src_label).replace("\\", "/"),
                "bbox_count": len(boxes),
            }

            if idx % 50 == 0:
                print(f"  已处理 {idx}/{len(image_paths)} ...")

        except Exception as e:
            report["failed_items"].append({
                "image": str(src_img).replace("\\", "/"),
                "error": repr(e),
            })

    if not dry_run:
        with open(out_bbox_path, "w", encoding="utf-8") as f:
            json.dump(bbox_json, f, ensure_ascii=False, indent=2)

    report["bboxes_json"] = str(out_bbox_path).replace("\\", "/")
    return report


# =============================================================================
# 6. 输出 manifest 重写文件，方便后续 AutoDL 路径替换或本地检查
# =============================================================================

def write_exported_filelists(
    project_root: Path,
    train_report: Dict,
    val_report: Dict,
) -> None:
    """
    输出复制后的 image 路径列表。
    """
    out_dir = project_root / "data" / "splits" / "e0_manifest_exported"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_lines = [
        v["image"] for v in train_report.get("name_mapping", {}).values()
    ]
    val_lines = [
        v["image"] for v in val_report.get("name_mapping", {}).values()
    ]

    (out_dir / "target_train_pool_exported.txt").write_text(
        "\n".join(train_lines) + "\n",
        encoding="utf-8",
    )
    (out_dir / "target_val_exported.txt").write_text(
        "\n".join(val_lines) + "\n",
        encoding="utf-8",
    )


# =============================================================================
# 7. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export E0 target support/val subset to processed_slim structure."
    )

    parser.add_argument(
        "--project_root",
        type=str,
        default=str(DEFAULT_PROJECT_ROOT),
        help="项目根目录",
    )
    parser.add_argument(
        "--manifest_dir",
        type=str,
        default=None,
        help="E0 manifest 目录，默认 data/splits/e0_manifest",
    )
    parser.add_argument(
        "--clear_output",
        action="store_true",
        help="清空输出 images/dual_channel_labels 后重新生成",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只检查，不复制、不写文件",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=4,
        help="bbox 连通域最小面积过滤阈值",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root)
    manifest_dir = (
        Path(args.manifest_dir)
        if args.manifest_dir is not None
        else project_root / "data" / "splits" / "e0_manifest"
    )

    out_train_root = (
        project_root
        / "data"
        / "raw"
        / "processed_slim"
        / "target_whu_mix"
        / "train_pool"
    )
    out_val_root = (
        project_root
        / "data"
        / "raw"
        / "processed_slim"
        / "target_whu_mix"
        / "val"
    )

    print("=" * 90)
    print("E0 target subset export")
    print("=" * 90)
    print(f"project_root : {project_root}")
    print(f"manifest_dir : {manifest_dir}")
    print(f"train out    : {out_train_root}")
    print(f"val out      : {out_val_root}")
    print(f"dry_run      : {args.dry_run}")
    print(f"clear_output : {args.clear_output}")
    print("=" * 90)

    print("\n[1] 读取 support manifests ...")
    support_paths = collect_support_paths(manifest_dir)
    print(f"  support union images: {len(support_paths)}")

    print("\n[2] 读取 target_val manifest ...")
    val_paths = collect_val_paths(manifest_dir)
    print(f"  val images: {len(val_paths)}")

    # 防止 support 和 val 在源路径层面交叉
    support_names = {p.name for p in support_paths}
    val_names = {p.name for p in val_paths}
    intersect_names = support_names & val_names
    if intersect_names:
        print(f"\n[警告] support 与 val 存在同名文件交集: {len(intersect_names)}")
        for n in sorted(list(intersect_names))[:10]:
            print(f"  - {n}")
        raise RuntimeError("support/val 存在同名交集，请先检查 E0 split。")

    print("\n[3] 导出 support 到 train_pool ...")
    train_report = export_subset(
        image_paths=support_paths,
        out_root=out_train_root,
        clear_output=args.clear_output,
        dry_run=args.dry_run,
        min_area=args.min_area,
    )

    print("\n[4] 导出 val ...")
    val_report = export_subset(
        image_paths=val_paths,
        out_root=out_val_root,
        clear_output=args.clear_output,
        dry_run=args.dry_run,
        min_area=args.min_area,
    )

    report = {
        "project_root": str(project_root).replace("\\", "/"),
        "manifest_dir": str(manifest_dir).replace("\\", "/"),
        "dry_run": args.dry_run,
        "clear_output": args.clear_output,
        "train_pool": train_report,
        "val": val_report,
    }

    report_path = (
        project_root
        / "data"
        / "splits"
        / "e0_manifest"
        / "export_e0_target_subset_report.json"
    )

    if not args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        write_exported_filelists(project_root, train_report, val_report)

    print("\n" + "=" * 90)
    print("导出完成")
    print("=" * 90)
    print(f"train_pool exported: {train_report['exported_count']} / {train_report['requested_count']}")
    print(f"val exported       : {val_report['exported_count']} / {val_report['requested_count']}")
    print(f"train missing img  : {len(train_report['missing_images'])}")
    print(f"train missing lbl  : {len(train_report['missing_labels'])}")
    print(f"val missing img    : {len(val_report['missing_images'])}")
    print(f"val missing lbl    : {len(val_report['missing_labels'])}")
    print(f"train bbox total   : {train_report['bbox_total_count']}")
    print(f"val bbox total     : {val_report['bbox_total_count']}")

    if not args.dry_run:
        print(f"\n报告已保存: {report_path}")

    if train_report["missing_images"] or train_report["missing_labels"] or val_report["missing_images"] or val_report["missing_labels"]:
        print("\n[警告] 存在缺失 image 或 label，请查看 report JSON 后再上传 AutoDL。")
    else:
        print("\n[OK] 所有 support/val image 与 label 均已成功处理。")


if __name__ == "__main__":
    main()