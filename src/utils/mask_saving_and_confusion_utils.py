#!/usr/bin/env python3
"""
Utilities to add mask saving and per-image TP/FP/FN accounting to an existing
calibrated full-test inference script.

How to use inside src/run_calibrated_fulltest.py
------------------------------------------------
1) Copy this file to your project, for example:
       src/utils/mask_saving_and_confusion_utils.py

2) Add these arguments to your existing argparse parser:
       parser.add_argument("--save-mask-dir", type=str, default=None,
                           help="Directory to save final post-processed binary prediction masks.")
       parser.add_argument("--save-per-image-confusion", type=str, default=None,
                           help="CSV path to save TP/FP/FN/pred_area/gt_area per image.")

3) Import and initialize before the evaluation loop:
       from utils.mask_saving_and_confusion_utils import MaskConfusionSaver, to_numpy_mask
       saver = MaskConfusionSaver(
           mask_dir=args.save_mask_dir,
           confusion_csv=args.save_per_image_confusion,
           method=getattr(args, "model", None) or getattr(args, "variant", None),
           seed=getattr(args, "seed", None),
       )

4) Inside the final-test inference loop, AFTER validation-calibrated thresholding,
   post-processing, and TTA aggregation, call:
       saver.add(
           image_id=Path(image_path).stem,
           pred_mask=final_pred_mask,
           gt_mask=gt_mask,   # optional but recommended; pass None if unavailable
       )

   IMPORTANT: final_pred_mask must be the exact binary mask used for your reported
   final metrics: validation-calibrated + TTA + post-processing if that is your
   main setting.

5) After the loop:
       saver.close()

This produces two kinds of output:
- PNG binary prediction masks under --save-mask-dir
- a per-image CSV containing TP, FP, FN, pred_area, and gt_area

You can then summarize the CSV with coverage_from_confusion_csv.py, or compute
coverage from saved masks with coverage_diagnostic_from_masks_v2.py.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Any, Dict

import numpy as np
from PIL import Image


def to_numpy_mask(x: Any, threshold: float = 0.5) -> np.ndarray:
    """Convert torch/numpy/PIL mask or probability map to a boolean numpy mask."""
    # Torch tensor support without importing torch as a hard dependency.
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "cpu") and hasattr(x, "numpy"):
        x = x.cpu().numpy()
    elif isinstance(x, Image.Image):
        x = np.array(x)
    else:
        x = np.asarray(x)

    if x.ndim == 4:
        # Common shapes: BxCxHxW or BxHxWxC. Caller should pass one image, but
        # keep a safe fallback for B=1.
        if x.shape[0] != 1:
            raise ValueError(f"Expected a single mask, got batch shape {x.shape}")
        x = x[0]
    if x.ndim == 3:
        # CxHxW with C=1 or HxWxC. Use first channel.
        if x.shape[0] in (1, 2, 3) and x.shape[0] < x.shape[-1]:
            x = x[0]
        else:
            x = x[..., 0]
    if x.dtype == np.bool_:
        return x
    x = x.astype(np.float32)
    if x.max(initial=0) > 1.5:
        return x > 127.0
    return x >= threshold


def safe_image_id(image_id: str) -> str:
    """Create a filesystem-safe image id while preserving the useful stem."""
    image_id = Path(str(image_id)).stem
    return image_id.replace("/", "_").replace("\\", "_")


class MaskConfusionSaver:
    """Save final prediction masks and/or per-image TP/FP/FN rows."""

    def __init__(self, mask_dir: Optional[str] = None, confusion_csv: Optional[str] = None,
                 method: Optional[str] = None, seed: Optional[int] = None) -> None:
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.confusion_csv = Path(confusion_csv) if confusion_csv else None
        self.method = method
        self.seed = seed
        self._csv_fh = None
        self._writer = None

        if self.mask_dir:
            self.mask_dir.mkdir(parents=True, exist_ok=True)
        if self.confusion_csv:
            self.confusion_csv.parent.mkdir(parents=True, exist_ok=True)
            self._csv_fh = self.confusion_csv.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._csv_fh,
                fieldnames=[
                    "image_id", "method", "seed", "tp", "fp", "fn", "tn",
                    "pred_area", "gt_area", "union", "iou_fg", "precision", "recall", "f1"
                ],
            )
            self._writer.writeheader()

    def add(self, image_id: str, pred_mask: Any, gt_mask: Optional[Any] = None) -> Dict[str, float]:
        image_id = safe_image_id(image_id)
        pred = to_numpy_mask(pred_mask)

        if self.mask_dir:
            out_path = self.mask_dir / f"{image_id}.png"
            Image.fromarray(pred.astype(np.uint8) * 255).save(out_path)

        row: Dict[str, float] = {}
        if gt_mask is not None:
            gt = to_numpy_mask(gt_mask)
            if gt.shape != pred.shape:
                raise ValueError(f"Shape mismatch for {image_id}: pred={pred.shape}, gt={gt.shape}")
            tp = float(np.logical_and(pred, gt).sum())
            fp = float(np.logical_and(pred, ~gt).sum())
            fn = float(np.logical_and(~pred, gt).sum())
            tn = float(np.logical_and(~pred, ~gt).sum())
            pred_area = tp + fp
            gt_area = tp + fn
            union = tp + fp + fn
            iou = 1.0 if union == 0 else tp / union
            precision = 1.0 if pred_area == 0 and gt_area == 0 else (tp / pred_area if pred_area > 0 else 0.0)
            recall = 1.0 if gt_area == 0 and pred_area == 0 else (tp / gt_area if gt_area > 0 else 0.0)
            denom = 2.0 * tp + fp + fn
            f1 = 1.0 if denom == 0 else 2.0 * tp / denom
            row = {
                "image_id": image_id,
                "method": self.method or "",
                "seed": self.seed if self.seed is not None else "",
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "pred_area": pred_area,
                "gt_area": gt_area,
                "union": union,
                "iou_fg": iou,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            if self._writer:
                self._writer.writerow(row)
        return row

    def close(self) -> None:
        if self._csv_fh:
            self._csv_fh.close()
            self._csv_fh = None


if __name__ == "__main__":
    print(__doc__)
