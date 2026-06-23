# -*- coding: utf-8 -*-
"""
run_foundation_prompt_baselines.py

评估 SAM3 原生 prompt baseline：
1. text prompt: "building"
2. oracle-box prompt: GT bbox prompt

默认使用 target_pilot_test_500.txt，作为 foundation diagnostic baseline。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SAM3_SRC = SRC / "models" / "sam3"
if str(SAM3_SRC) not in sys.path:
    sys.path.insert(0, str(SAM3_SRC))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_SIZE = 512
LABEL_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


def read_manifest(path: Path, limit: Optional[int] = None) -> List[Path]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().replace("\\", "/")
        if line:
            rows.append(Path(line))
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def find_dual_label_for_image(image_path: Path) -> Optional[Path]:
    stem = image_path.stem
    s = str(image_path).replace("\\", "/")
    candidates = []
    if "/images/" in s:
        candidates.append(Path(s.replace("/images/", "/dual_channel_labels/")).parent)
    if "/test/image/" in s:
        candidates.append(Path(s.replace("/test/image/", "/dual_channel_labels/")).parent)
    candidates.append(image_path.parent.parent / "dual_channel_labels")
    candidates.append(image_path.parent.parent / "label")
    candidates.append(image_path.parent.parent / "labels")

    seen = set()
    for d in candidates:
        key = str(d).lower()
        if key in seen:
            continue
        seen.add(key)
        if not d.exists():
            continue
        for ext in LABEL_EXTS:
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def load_mask_for_image(image_path: Path) -> np.ndarray:
    lab_path = find_dual_label_for_image(image_path)
    if lab_path is None:
        raise FileNotFoundError(f"missing label for {image_path}")
    lab = cv2.imread(str(lab_path), cv2.IMREAD_UNCHANGED)
    if lab is None:
        raise FileNotFoundError(lab_path)
    if lab.ndim == 2:
        mask = cv2.resize(lab, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.float32)
    lab = cv2.resize(lab, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)
    return (lab[:, :, 0] > 0).astype(np.float32)


def load_rgb_pil(image_path: Path):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(img)


def load_bboxes(path: Path) -> Dict[str, list]:
    if not path.exists():
        print(f"[Warning] bbox json missing: {path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def lookup_bboxes(bboxes: Dict[str, list], stem: str) -> list:
    for k in [stem, f"{stem}.tif", f"{stem}.tiff", f"{stem}.png", f"{stem}.jpg"]:
        if k in bboxes:
            return bboxes[k]
    return []


def box_to_prompt(box):
    x1, y1, x2, y2 = box[:4]
    cx = ((x1 + x2) / 2.0) / TARGET_SIZE
    cy = ((y1 + y2) / 2.0) / TARGET_SIZE
    w = abs(x2 - x1) / TARGET_SIZE
    h = abs(y2 - y1) / TARGET_SIZE
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.01, min(1.0, w))
    h = max(0.01, min(1.0, h))
    return [cx, cy, w, h]


def compute_iou(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + 1e-7) / (union + 1e-7))


def compute_f1(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return float((2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7))


def boundary_map(mask, d=5):
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask.astype(bool)
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    b = mask - eroded
    b = cv2.dilate(b, np.ones((d, d), np.uint8), iterations=1)
    return b.astype(bool)


def compute_biou(pred, gt, d=5):
    pb = boundary_map(pred, d=d)
    gb = boundary_map(gt, d=d)
    inter = np.logical_and(pb, gb).sum()
    union = np.logical_or(pb, gb).sum()
    return float((inter + 1e-7) / (union + 1e-7))


def state_to_pred(state) -> np.ndarray:
    masks = state.get("masks", None)
    if masks is None:
        return np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
    if isinstance(masks, torch.Tensor):
        if masks.numel() == 0:
            return np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
        arr = masks.detach().float().cpu().numpy()
    else:
        arr = np.array(masks)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        return np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
    pred = np.any(arr > 0.5, axis=0).astype(np.float32)
    pred = cv2.resize(pred, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)
    return pred


def eval_one_mode(paths, processor, mode: str, prompt: str, bboxes_all):
    rows = []
    preds, gts = [], []

    for img_path in tqdm(paths, desc=f"eval {mode}", ncols=100):
        pil = load_rgb_pil(img_path)
        gt = load_mask_for_image(img_path)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                state = processor.set_image(pil)

                if mode == "text":
                    state = processor.set_text_prompt(state=state, prompt=prompt)

                elif mode == "oracle_box":
                    boxes = lookup_bboxes(bboxes_all, img_path.stem)
                    for box in boxes:
                        if len(box) >= 4:
                            state = processor.add_geometric_prompt(box_to_prompt(box), True, state)

                else:
                    raise ValueError(mode)

        pred = state_to_pred(state)

        rows.append({
            "image_name": img_path.stem,
            "mode": mode,
            "mIoU": compute_iou(pred, gt),
            "F1": compute_f1(pred, gt),
            "Boundary_IoU": compute_biou(pred, gt, d=5),
        })

        preds.append(pred)
        gts.append(gt)

    summary = {
        "mode": mode,
        "mIoU": float(np.mean([r["mIoU"] for r in rows])),
        "F1": float(np.mean([r["F1"] for r in rows])),
        "Boundary_IoU": float(np.mean([r["Boundary_IoU"] for r in rows])),
        "N": len(rows),
    }
    return summary, rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--manifest_dir", type=str, default=None)
    parser.add_argument("--test_manifest_name", type=str, default="target_pilot_test_500.txt")
    parser.add_argument("--test_eval_limit", type=int, default=500)
    parser.add_argument("--sam3_checkpoint", type=str, default=None)
    parser.add_argument("--bbox_json", type=str, default=None)
    parser.add_argument("--modes", type=str, default="text,oracle_box")
    parser.add_argument("--prompt", type=str, default="building")
    parser.add_argument("--out_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else project_root / "data/splits/e0_manifest"
    sam3_ckpt = Path(args.sam3_checkpoint) if args.sam3_checkpoint else project_root / "weights/sam3.pt"
    bbox_json = Path(args.bbox_json) if args.bbox_json else project_root / "data/raw/whu_mix_full_test/bbox.json"
    out_dir = Path(args.out_dir) if args.out_dir else project_root / "results/foundation_prompt_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    limit = args.test_eval_limit
    if limit is not None and limit <= 0:
        limit = None

    paths = read_manifest(manifest_dir / args.test_manifest_name, limit=limit)
    bboxes_all = load_bboxes(bbox_json)

    model = build_sam3_image_model(checkpoint_path=str(sam3_ckpt))
    model.to(DEVICE)
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=0.20)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    summaries = []
    all_rows = []

    print("=" * 90)
    print("Foundation prompt baselines")
    print(f"manifest={args.test_manifest_name}, N={len(paths)}, modes={modes}, prompt={args.prompt}")
    print("=" * 90)

    for mode in modes:
        summary, rows = eval_one_mode(paths, processor, mode, args.prompt, bboxes_all)
        summaries.append(summary)
        all_rows.extend(rows)
        print(
            f"{mode:<12} mIoU={summary['mIoU']*100:.2f} "
            f"F1={summary['F1']*100:.2f} "
            f"BIoU={summary['Boundary_IoU']*100:.2f}"
        )

    out = {
        "metadata": {
            "test_manifest": args.test_manifest_name,
            "test_eval_limit": args.test_eval_limit,
            "prompt": args.prompt,
            "modes": modes,
        },
        "summary": summaries,
        "per_image": all_rows,
    }

    out_path = out_dir / "foundation_prompt_baselines.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()