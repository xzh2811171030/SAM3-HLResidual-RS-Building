# -*- coding: utf-8 -*-
"""
rewrite_and_check_e0_on_autodl.py
=============================================================================
功能：
  1. 将本地 E0 manifest 中的 Windows 路径重写为 AutoDL 路径
  2. 检查 support / val / final test 文件是否存在
  3. 检查 dual_channel_labels 是否存在
  4. 检查 bboxes.json 是否存在且覆盖 target support / val
  5. 检查 support / val / final_test 零交集
  6. 检查 5/10/20-shot 嵌套关系
  7. 输出 AutoDL 审计报告

推荐运行：
  cd <project_root>
  python src/data/rewrite_and_check_e0_on_autodl.py --rewrite

首次建议：
  python src/data/rewrite_and_check_e0_on_autodl.py --dry_run

严格检查：
  python src/data/rewrite_and_check_e0_on_autodl.py --rewrite --strict
=============================================================================
"""

import argparse
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
LABEL_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


# =============================================================================
# 基础工具
# =============================================================================

def normalize_line_path(line: str) -> str:
    return line.strip().strip('"').strip("'").replace("\\", "/")


def get_basename_from_any_path(line: str) -> str:
    """
    从 Windows / Linux 路径中提取文件名。
    """
    s = normalize_line_path(line)
    return Path(s).name


def get_stem_from_any_path(line: str) -> str:
    return Path(get_basename_from_any_path(line)).stem


def read_txt(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [
        normalize_line_path(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def write_txt(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_existing_by_name_or_stem(directory: Path, basename: str) -> Optional[Path]:
    """
    优先找同名文件；找不到则按 stem + 常见扩展名查找。
    """
    directory = Path(directory)
    if not directory.exists():
        return None

    direct = directory / basename
    if direct.exists():
        return direct.resolve()

    stem = Path(basename).stem
    for ext in IMAGE_EXTS:
        cand = directory / f"{stem}{ext}"
        if cand.exists():
            return cand.resolve()

    return None


def find_label_by_stem(label_dir: Path, image_basename: str) -> Optional[Path]:
    label_dir = Path(label_dir)
    if not label_dir.exists():
        return None

    stem = Path(image_basename).stem

    for ext in LABEL_EXTS:
        cand = label_dir / f"{stem}{ext}"
        if cand.exists():
            return cand.resolve()

    return None


def backup_manifest_dir(manifest_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = manifest_dir.parent / f"{manifest_dir.name}_backup_before_autodl_rewrite_{timestamp}"
    shutil.copytree(manifest_dir, backup_dir)
    return backup_dir


# =============================================================================
# split 类型判断
# =============================================================================

def classify_manifest_file(txt_name: str) -> str:
    """
    根据 manifest 文件名判断应该映射到哪个 AutoDL 图像目录。
    """
    stem = Path(txt_name).stem

    if stem.startswith("target_support_"):
        return "target_support"

    if stem == "target_val":
        return "target_val"

    if stem == "target_pilot_test_500":
        return "target_pilot_test"

    if stem == "target_pilot_test_1000":
        return "target_pilot_test_1000_skip"

    if stem == "target_final_test_8402":
        return "target_test"

    if stem == "source_train":
        return "source_train"

    if stem == "source_val":
        return "source_val"

    if stem == "source_test":
        return "source_test"

    if stem == "target_train_pool_all":
        return "target_train_pool_all"

    return "unknown"


def build_autodl_dirs(project_root: Path) -> Dict[str, Path]:
    raw = project_root / "data" / "raw"

    return {
        "target_support_image": raw / "processed_slim" / "target_whu" / "train_pool" / "images",
        "target_support_label": raw / "processed_slim" / "target_whu" / "train_pool" / "dual_channel_labels",
        "target_support_bbox": raw / "processed_slim" / "target_whu" / "train_pool" / "bboxes.json",

        "target_val_image": raw / "processed_slim" / "target_whu" / "val" / "images",
        "target_val_label": raw / "processed_slim" / "target_whu" / "val" / "dual_channel_labels",
        "target_val_bbox": raw / "processed_slim" / "target_whu" / "val" / "bboxes.json",

        "target_test_image": raw / "whu_mix_full_test" / "test" / "image",
        "target_test_label": raw / "whu_mix_full_test" / "dual_channel_labels",
        "target_test_bbox": raw / "whu_mix_full_test" / "bbox.json",

        "target_pilot_test_image": raw / "processed_slim" / "target_whu" / "test" / "images",
        "target_pilot_test_label": raw / "processed_slim" / "target_whu" / "test" / "dual_channel_labels",
        "target_pilot_test_bbox": raw / "processed_slim" / "target_whu" / "test" / "bboxes.json",

        "source_train_image": raw / "processed_slim" / "source_whu" / "train" / "images",
        "source_train_label": raw / "processed_slim" / "source_whu" / "train" / "dual_channel_labels",

        "source_val_image": raw / "processed_slim" / "source_whu" / "val" / "images",
        "source_val_label": raw / "processed_slim" / "source_whu" / "val" / "dual_channel_labels",

        "source_test_image": raw / "processed_slim" / "source_whu" / "test" / "images",
        "source_test_label": raw / "processed_slim" / "source_whu" / "test" / "dual_channel_labels",
    }


def get_image_and_label_dir(split_type: str, dirs: Dict[str, Path]) -> Tuple[Optional[Path], Optional[Path]]:
    if split_type == "target_support":
        return dirs["target_support_image"], dirs["target_support_label"]

    if split_type == "target_val":
        return dirs["target_val_image"], dirs["target_val_label"]

    if split_type == "target_test":
        return dirs["target_test_image"], dirs["target_test_label"]

    if split_type == "target_pilot_test":
        return dirs["target_pilot_test_image"], dirs["target_pilot_test_label"]

    if split_type == "source_train":
        return dirs["source_train_image"], dirs["source_train_label"]

    if split_type == "source_val":
        return dirs["source_val_image"], dirs["source_val_label"]

    if split_type == "source_test":
        return dirs["source_test_image"], dirs["source_test_label"]

    return None, None


# =============================================================================
# manifest 重写
# =============================================================================

def rewrite_one_manifest(
    txt_path: Path,
    split_type: str,
    image_dir: Path,
    dry_run: bool,
) -> Dict:
    old_lines = read_txt(txt_path)

    new_lines: List[str] = []
    missing_images: List[str] = []

    for old in old_lines:
        basename = get_basename_from_any_path(old)
        resolved = find_existing_by_name_or_stem(image_dir, basename)

        if resolved is None:
            missing_images.append(basename)
            # 即使不存在，也按预期路径写，便于 report 定位
            expected = image_dir / basename
            new_lines.append(str(expected).replace("\\", "/"))
        else:
            new_lines.append(str(resolved).replace("\\", "/"))

    if not dry_run:
        write_txt(txt_path, new_lines)

    return {
        "manifest": txt_path.name,
        "split_type": split_type,
        "count": len(old_lines),
        "rewritten_count": len(new_lines),
        "missing_images": missing_images,
        "missing_image_count": len(missing_images),
    }


def rewrite_manifests(
    manifest_dir: Path,
    dirs: Dict[str, Path],
    dry_run: bool,
    ignore_train_pool_all: bool,
) -> Dict:
    report = {
        "rewritten": [],
        "skipped": [],
    }

    txt_files = sorted(manifest_dir.glob("*.txt"))

    for txt in txt_files:
        split_type = classify_manifest_file(txt.name)

        if split_type == "target_train_pool_all" and ignore_train_pool_all:
            report["skipped"].append({
                "manifest": txt.name,
                "reason": "ignored by default because full 39346 train pool is not uploaded to AutoDL",
            })
            continue

        if split_type == "target_pilot_test_1000_skip":
            report["skipped"].append({
                "manifest": txt.name,
                "reason": "pilot_test_1000 data not uploaded; only pilot_test_500 is available",
            })
            continue

        image_dir, _ = get_image_and_label_dir(split_type, dirs)

        if image_dir is None:
            report["skipped"].append({
                "manifest": txt.name,
                "reason": f"unknown split type: {split_type}",
            })
            continue

        r = rewrite_one_manifest(txt, split_type, image_dir, dry_run=dry_run)
        report["rewritten"].append(r)

    return report


# =============================================================================
# label 与 bbox 检查
# =============================================================================

def load_bbox_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def bbox_has_key(bbox_data: Dict, image_basename: str) -> bool:
    stem = Path(image_basename).stem
    keys = [
        image_basename,
        stem,
        f"{stem}.tif",
        f"{stem}.tiff",
        f"{stem}.png",
        f"{stem}.jpg",
    ]
    return any(k in bbox_data for k in keys)


def check_one_manifest_files(
    txt_path: Path,
    split_type: str,
    dirs: Dict[str, Path],
    check_bbox: bool,
) -> Dict:
    lines = read_txt(txt_path)
    image_dir, label_dir = get_image_and_label_dir(split_type, dirs)

    missing_images: List[str] = []
    missing_labels: List[str] = []
    missing_bbox_keys: List[str] = []

    bbox_data: Dict = {}
    bbox_path: Optional[Path] = None

    if split_type == "target_support":
        bbox_path = dirs["target_support_bbox"]
    elif split_type == "target_val":
        bbox_path = dirs["target_val_bbox"]
    elif split_type == "target_pilot_test":
        bbox_path = dirs["target_pilot_test_bbox"]
    elif split_type == "target_test":
        bbox_path = dirs["target_test_bbox"]

    if bbox_path is not None:
        bbox_data = load_bbox_json(bbox_path)

    empty_bbox_keys: List[str] = []

    for line in lines:
        p = Path(line)
        basename = p.name

        if not p.exists():
            missing_images.append(str(p))
            continue

        if label_dir is not None:
            label = find_label_by_stem(label_dir, basename)
            if label is None:
                missing_labels.append(basename)

        if check_bbox and bbox_path is not None:
            if not bbox_data:
                missing_bbox_keys.append(basename)
            else:
                if not bbox_has_key(bbox_data, basename):
                    missing_bbox_keys.append(basename)
                else:
                    # 统计 bbox=[] 的情况
                    stem = Path(basename).stem
                    candidate_keys = [
                        basename,
                        stem,
                        f"{stem}.tif",
                        f"{stem}.tiff",
                        f"{stem}.png",
                        f"{stem}.jpg",
                    ]
                    val = None
                    for k in candidate_keys:
                        if k in bbox_data:
                            val = bbox_data[k]
                            break
                    if isinstance(val, list) and len(val) == 0:
                        empty_bbox_keys.append(basename)

    return {
        "manifest": txt_path.name,
        "split_type": split_type,
        "count": len(lines),
        "missing_images": missing_images[:20],
        "missing_image_count": len(missing_images),
        "missing_labels": missing_labels[:20],
        "missing_label_count": len(missing_labels),
        "bbox_path": str(bbox_path) if bbox_path else "",
        "missing_bbox_keys": missing_bbox_keys[:20],
        "missing_bbox_key_count": len(missing_bbox_keys),
        "empty_bbox_keys": empty_bbox_keys[:20],
        "empty_bbox_key_count": len(empty_bbox_keys),
    }


def check_all_files(
    manifest_dir: Path,
    dirs: Dict[str, Path],
    ignore_train_pool_all: bool,
) -> Dict:
    report = {
        "file_checks": [],
        "directory_status": {},
    }

    for k, v in dirs.items():
        if k.endswith("_bbox"):
            report["directory_status"][k] = {
                "path": str(v),
                "exists": v.exists(),
                "type": "file",
            }
        else:
            report["directory_status"][k] = {
                "path": str(v),
                "exists": v.exists(),
                "type": "dir",
            }

    for txt in sorted(manifest_dir.glob("*.txt")):
        split_type = classify_manifest_file(txt.name)

        if split_type == "target_train_pool_all" and ignore_train_pool_all:
            continue

        if split_type == "target_pilot_test_1000_skip":
            continue

        if split_type == "unknown":
            continue

        check_bbox = split_type in {"target_support", "target_val", "target_pilot_test", "target_test"}

        r = check_one_manifest_files(
            txt,
            split_type,
            dirs,
            check_bbox=check_bbox,
        )
        report["file_checks"].append(r)

    return report


# =============================================================================
# 交集与嵌套检查
# =============================================================================

def load_stems_from_manifest(path: Path) -> Set[str]:
    stems: Set[str] = set()
    for line in read_txt(path):
        stems.add(Path(line).stem)
    return stems


def check_intersections_and_nested(manifest_dir: Path) -> Dict:
    report = {
        "intersections": {},
        "nested": {},
        "summary": {},
    }

    final_stems = load_stems_from_manifest(manifest_dir / "target_final_test_8402.txt")
    val_stems = load_stems_from_manifest(manifest_dir / "target_val.txt")

    support_files = sorted(manifest_dir.glob("target_support_*_seed*.txt"))

    support_intersections = {}

    for sf in support_files:
        sup = load_stems_from_manifest(sf)
        support_intersections[sf.stem] = {
            "count": len(sup),
            "intersect_val": len(sup & val_stems),
            "intersect_final_test": len(sup & final_stems),
        }

    report["intersections"]["support"] = support_intersections
    report["intersections"]["val_final_test"] = {
        "target_val_count": len(val_stems),
        "target_final_test_count": len(final_stems),
        "intersect": len(val_stems & final_stems),
    }

    # nested check
    seed_map: Dict[int, Dict[int, Set[str]]] = defaultdict(dict)

    pattern = re.compile(r"target_support_(\d+)_seed(\d+)")
    for sf in support_files:
        m = pattern.match(sf.stem)
        if not m:
            continue
        shot = int(m.group(1))
        seed = int(m.group(2))
        seed_map[seed][shot] = load_stems_from_manifest(sf)

    nested_ok = True
    nested_detail = {}

    for seed, shot_map in seed_map.items():
        shot_list = sorted(shot_map.keys())
        checks = []
        for small, large in zip(shot_list[:-1], shot_list[1:]):
            ok = shot_map[small].issubset(shot_map[large])
            checks.append({
                "small": small,
                "large": large,
                "ok": ok,
                "small_count": len(shot_map[small]),
                "large_count": len(shot_map[large]),
            })
            if not ok:
                nested_ok = False
        nested_detail[str(seed)] = checks

    report["nested"]["ok"] = nested_ok
    report["nested"]["detail"] = nested_detail

    any_support_val_intersect = any(
        v["intersect_val"] > 0 for v in support_intersections.values()
    )
    any_support_test_intersect = any(
        v["intersect_final_test"] > 0 for v in support_intersections.values()
    )
    val_test_intersect = len(val_stems & final_stems) > 0

    report["summary"] = {
        "target_final_test_count": len(final_stems),
        "target_val_count": len(val_stems),
        "final_test_is_8402": len(final_stems) >= 8400,
        "support_val_zero_intersection": not any_support_val_intersect,
        "support_test_zero_intersection": not any_support_test_intersect,
        "val_test_zero_intersection": not val_test_intersect,
        "nested_ok": nested_ok,
    }

    return report


# =============================================================================
# 总结
# =============================================================================

def summarize_report(report: Dict) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    # rewrite missing
    for r in report.get("rewrite_report", {}).get("rewritten", []):
        if r.get("missing_image_count", 0) > 0:
            errors.append(
                f"{r['manifest']}: missing images during rewrite = {r['missing_image_count']}"
            )

    # file checks
    for r in report.get("file_report", {}).get("file_checks", []):
        if r.get("missing_image_count", 0) > 0:
            errors.append(f"{r['manifest']}: missing image = {r['missing_image_count']}")

        if r.get("missing_label_count", 0) > 0:
            errors.append(f"{r['manifest']}: missing label = {r['missing_label_count']}")

        # support / val 的 bbox key 缺失更严重
        if r.get("split_type") in {"target_support", "target_val"}:
            if r.get("missing_bbox_key_count", 0) > 0:
                errors.append(
                    f"{r['manifest']}: missing bbox key = {r['missing_bbox_key_count']}"
                )

    # split checks
    split_summary = report.get("split_report", {}).get("summary", {})
    if not split_summary.get("final_test_is_8402", False):
        errors.append("target_final_test is not full 8402")

    if not split_summary.get("support_val_zero_intersection", False):
        errors.append("support intersects with val")

    if not split_summary.get("support_test_zero_intersection", False):
        errors.append("support intersects with final test")

    if not split_summary.get("val_test_zero_intersection", False):
        errors.append("val intersects with final test")

    if not split_summary.get("nested_ok", False):
        errors.append("nested support check failed")

    return len(errors) == 0, errors


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite E0 manifest paths and check files on AutoDL."
    )

    parser.add_argument(
        "--project_root",
        type=str,
        default=str(DEFAULT_PROJECT_ROOT),
        help="AutoDL 项目根目录",
    )
    parser.add_argument(
        "--manifest_dir",
        type=str,
        default=None,
        help="E0 manifest 目录，默认 data/splits/e0_manifest",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="执行 manifest 路径重写；不加则只检查当前路径",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只模拟重写，不写回 txt",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：发现错误则 exit(1)",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="不备份 manifest 目录",
    )
    parser.add_argument(
        "--check_train_pool_all",
        action="store_true",
        help="检查 target_train_pool_all.txt。默认不检查，因为 39346 全量 train pool 通常不会上传到 AutoDL。",
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

    ignore_train_pool_all = not args.check_train_pool_all

    if not manifest_dir.exists():
        raise FileNotFoundError(f"manifest_dir 不存在: {manifest_dir}")

    dirs = build_autodl_dirs(project_root)

    print("=" * 90)
    print("AutoDL E0 manifest rewrite & check")
    print("=" * 90)
    print(f"project_root         : {project_root}")
    print(f"manifest_dir         : {manifest_dir}")
    print(f"rewrite              : {args.rewrite}")
    print(f"dry_run              : {args.dry_run}")
    print(f"strict               : {args.strict}")
    print(f"ignore_train_pool_all: {ignore_train_pool_all}")
    print("=" * 90)

    backup_dir = ""
    if args.rewrite and not args.dry_run and not args.no_backup:
        backup = backup_manifest_dir(manifest_dir)
        backup_dir = str(backup)
        print(f"\n[Backup] 已备份 manifest 到: {backup}")

    print("\n[1] Rewrite manifests ...")
    if args.rewrite:
        rewrite_report = rewrite_manifests(
            manifest_dir=manifest_dir,
            dirs=dirs,
            dry_run=args.dry_run,
            ignore_train_pool_all=ignore_train_pool_all,
        )
    else:
        rewrite_report = {
            "rewritten": [],
            "skipped": [{"reason": "rewrite disabled"}],
        }

    for r in rewrite_report.get("rewritten", []):
        status = "OK" if r["missing_image_count"] == 0 else "MISSING"
        print(
            f"  {r['manifest']:<36} {r['count']:>5} "
            f"missing={r['missing_image_count']:<5} {status}"
        )

    if rewrite_report.get("skipped"):
        print("\n  Skipped manifests:")
        for s in rewrite_report["skipped"]:
            print(f"    - {s.get('manifest', '-')}: {s.get('reason', '')}")

    print("\n[2] Check files, labels, bboxes ...")
    file_report = check_all_files(
        manifest_dir=manifest_dir,
        dirs=dirs,
        ignore_train_pool_all=ignore_train_pool_all,
    )

    for r in file_report["file_checks"]:
        print(
            f"  {r['manifest']:<36} "
            f"n={r['count']:<5} "
            f"img_missing={r['missing_image_count']:<4} "
            f"label_missing={r['missing_label_count']:<4} "
            f"bbox_missing={r['missing_bbox_key_count']:<4} "
            f"bbox_empty={r['empty_bbox_key_count']:<4}"
        )

    print("\n[3] Check intersections and nested support ...")
    split_report = check_intersections_and_nested(manifest_dir)

    s = split_report["summary"]
    print(f"  target_final_test_count       : {s['target_final_test_count']}")
    print(f"  target_val_count              : {s['target_val_count']}")
    print(f"  final_test_is_8402            : {s['final_test_is_8402']}")
    print(f"  support_val_zero_intersection : {s['support_val_zero_intersection']}")
    print(f"  support_test_zero_intersection: {s['support_test_zero_intersection']}")
    print(f"  val_test_zero_intersection    : {s['val_test_zero_intersection']}")
    print(f"  nested_ok                     : {s['nested_ok']}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "project_root": str(project_root),
        "manifest_dir": str(manifest_dir),
        "backup_dir": backup_dir,
        "rewrite_enabled": args.rewrite,
        "dry_run": args.dry_run,
        "ignore_train_pool_all": ignore_train_pool_all,
        "directory_status": file_report.get("directory_status", {}),
        "rewrite_report": rewrite_report,
        "file_report": file_report,
        "split_report": split_report,
    }

    ok, errors = summarize_report(report)
    report["ok"] = ok
    report["errors"] = errors

    report_path = manifest_dir / "autodl_e0_check_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 90)
    print("Final summary")
    print("=" * 90)
    print(f"OK: {ok}")
    print(f"Report saved: {report_path}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  - {e}")

    # 空 bbox 不一定是错误，但对 support 是强警告
    support_empty_bbox = []
    for r in file_report["file_checks"]:
        if r["split_type"] == "target_support" and r["empty_bbox_key_count"] > 0:
            support_empty_bbox.append((r["manifest"], r["empty_bbox_key_count"]))

    if support_empty_bbox:
        print("\n[Strong Warning] target support 中存在 bbox=[] 的切片：")
        for mf, cnt in support_empty_bbox:
            print(f"  - {mf}: {cnt}")
        print("  建议重新抽取 support，过滤空建筑切片。")

    print("=" * 90)

    if args.strict and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()