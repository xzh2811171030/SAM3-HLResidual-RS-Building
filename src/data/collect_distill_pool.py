# -*- coding: utf-8 -*-
"""
collect_distill_pool.py
=============================================================================
功能：
  读取 E0 manifest 中的 target_distill_pool_1000.txt 图像路径，
  自动复制图像，并从原始 label 生成 dual_channel_labels 与 bboxes.json。

输出目录：
  /path/to/project/data/raw/processed_slim/target_whu_mix/distill_pool/
    images/
    dual_channel_labels/
    bboxes.json

推荐运行：
  python src/data/collect_distill_pool.py
  python src/data/collect_distill_pool.py --dry_run
=============================================================================
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set

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
    """
    image_path = image_path.resolve()
    parent = image_path.parent
    grand = parent.parent

    candidates: List[Path] = []

    for dirname in [
        "label", "labels", "mask", "masks",
        "annotation", "annotations",
        "dual_channel_labels",
    ]:
        candidates.append(grand / dirname)

    s = str(image_path).replace("\\", "/")

    replace_pairs = [
        ("/image/", "/label/"),
        ("/images/", "/label/"),
        ("/images/", "/labels/"),
        ("/images/", "/dual_channel_labels/"),
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
    """为一张 image 找对应 label（同 stem）。"""
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
    """读取 label 并转成 0/255 二值 mask。"""
    img = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取 label: {label_path}")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = (img > 0).astype(np.uint8) * 255
    return mask


def make_dual_channel_label(binary_mask: np.ndarray) -> np.ndarray:
    """
    生成双通道 label：
      B 通道：原始 mask
      G 通道：boundary
      R 通道：0
    """
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    boundary = cv2.subtract(binary_mask, eroded)

    zeros = np.zeros_like(binary_mask, dtype=np.uint8)
    dual_bgr = cv2.merge([binary_mask, boundary, zeros])
    return dual_bgr


def extract_bboxes(binary_mask: np.ndarray, min_area: int = 4) -> List[List[int]]:
    """从二值 mask 中提取连通域 bbox。"""
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
# 4. 核心导出逻辑
# =============================================================================

def export_distill_pool(
    stems: List[str],
    out_root: Path,
    src_image_dir: Path,
    src_label_dir: Path,
    clear_output: bool = False,
    dry_run: bool = False,
    min_area: int = 4,
) -> Dict:
    """
    根据 stem 列表，从 src_image_dir / src_label_dir 复制并生成 dual_channel_labels。

    输出到:
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
        "requested_count": len(stems),
        "exported_count": 0,
        "missing_images": [],
        "missing_labels": [],
        "failed_items": [],
        "bbox_image_count": 0,
        "bbox_total_count": 0,
        "name_mapping": {},
    }

    for idx, stem in enumerate(stems, start=1):
        try:
            # 在源 image 目录中查找
            src_img = None
            for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
                cand = src_image_dir / f"{stem}{ext}"
                if cand.exists():
                    src_img = cand
                    break

            if src_img is None:
                report["missing_images"].append(stem)
                continue

            # 在源 label 目录中查找
            src_label = None
            for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"):
                cand = src_label_dir / f"{stem}{ext}"
                if cand.exists():
                    src_label = cand
                    break

            if src_label is None:
                report["missing_labels"].append(stem)
                continue

            out_img_path = out_img_dir / f"{stem}.tif"
            out_dual_path = out_dual_dir / f"{stem}.png"

            binary_mask = read_binary_mask(src_label)
            dual_label = make_dual_channel_label(binary_mask)
            boxes = extract_bboxes(binary_mask, min_area=min_area)

            if not dry_run:
                safe_copy(src_img, out_img_path)
                ok = cv2.imwrite(str(out_dual_path), dual_label)
                if not ok:
                    raise RuntimeError(f"cv2.imwrite 失败: {out_dual_path}")

            bbox_json[f"{stem}.tif"] = boxes
            bbox_json[stem] = boxes

            report["exported_count"] += 1
            report["bbox_image_count"] += 1
            report["bbox_total_count"] += len(boxes)
            report["name_mapping"][stem] = {
                "image": str(out_img_path).replace("\\", "/"),
                "dual_label": str(out_dual_path).replace("\\", "/"),
                "source_label": str(src_label).replace("\\", "/"),
                "bbox_count": len(boxes),
            }

            if idx % 100 == 0:
                print(f"  已处理 {idx}/{len(stems)} ...")

        except Exception as e:
            report["failed_items"].append({
                "stem": stem,
                "error": str(e),
            })

    if not dry_run:
        with open(out_bbox_path, "w", encoding="utf-8") as f:
            json.dump(bbox_json, f, ensure_ascii=False, indent=2)

    report["bboxes_json"] = str(out_bbox_path).replace("\\", "/")
    return report


# =============================================================================
# 5. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export E0 target_distill_pool_1000 to processed_slim structure."
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

    out_root = (
        project_root
        / "data"
        / "raw"
        / "processed_slim"
        / "target_whu_mix"
        / "distill_pool"
    )

    src_image_dir = project_root / "data" / "raw" / "whu_mix" / "train" / "image"
    src_label_dir = project_root / "data" / "raw" / "whu_mix" / "train" / "label"

    manifest_path = manifest_dir / "target_distill_pool_1000.txt"

    print("=" * 90)
    print("E0 distill pool (1000) export")
    print("=" * 90)
    print(f"project_root  : {project_root}")
    print(f"manifest      : {manifest_path}")
    print(f"src image dir : {src_image_dir}")
    print(f"src label dir : {src_label_dir}")
    print(f"out root      : {out_root}")
    print(f"dry_run       : {args.dry_run}")
    print(f"clear_output  : {args.clear_output}")
    print("=" * 90)

    print("\n[1] 读取 target_distill_pool_1000.txt ...")
    manifest_paths = read_manifest(manifest_path)
    # 从 manifest 路径中提取 stem, 从源目录查找
    stems = sorted(set(p.stem for p in manifest_paths))
    print(f"  manifest 行数: {len(manifest_paths)}")
    print(f"  唯一 stem 数: {len(stems)}")

    print("\n[2] 导出 distill pool 到 processed_slim ...")
    report = export_distill_pool(
        stems=stems,
        out_root=out_root,
        src_image_dir=src_image_dir,
        src_label_dir=src_label_dir,
        clear_output=args.clear_output,
        dry_run=args.dry_run,
        min_area=args.min_area,
    )

    report_path = manifest_dir / "export_distill_pool_report.json"
    if not args.dry_run:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 90)
    print("导出完成")
    print("=" * 90)
    print(f"exported     : {report['exported_count']} / {report['requested_count']}")
    print(f"missing img  : {len(report['missing_images'])}")
    print(f"missing lbl  : {len(report['missing_labels'])}")
    print(f"bbox total   : {report['bbox_total_count']}")

    if not args.dry_run:
        print(f"\n报告已保存: {report_path}")

    if report["missing_images"] or report["missing_labels"]:
        print("\n[警告] 存在缺失 image 或 label，请查看 report JSON。")
    else:
        print("\n[OK] 所有 distill pool image 与 label 均已成功处理。")


if __name__ == "__main__":
    main()
