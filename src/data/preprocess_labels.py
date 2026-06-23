"""
=============================================================================
preprocess_labels.py  ---  WHU-Mix 建筑掩膜预处理 (边界提取 + BBox JSON)
=============================================================================
功能说明:
  1. 读取 train_demo/label/ 下的二值建筑掩膜 (PNG/TIF)
  2. 形态学腐蚀提取物理边界, 输出双通道可视化图像
     - B 通道: original_mask (0/255)
     - G 通道: boundary_mask (0/255)
     - R 通道: 0
  3. cv2.findContours 提取连通域, 计算最小外接矩形
  4. 所有 BBox 汇聚保存为 bbox_demo.json

输入:
  data/raw/train_demo/label/          # 原始二值掩膜

输出:
  data/processed/dual_channel_demo/   # 双通道边界可视化 PNG
  data/processed/bbox_demo.json       # 建筑实例 BBox 坐标

用法:
  python src/data/preprocess_labels.py
=============================================================================
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from tqdm import tqdm

RAW_MASK_DIR: str = r"/path/to/project\data\raw\train_demo\label"
DUAL_OUT_DIR: str = r"/path/to/project\data\processed\dual_channel_demo"
BBOX_JSON_PATH: str = r"/path/to/project\data\processed\bbox_demo.json"

KERNEL_SIZE: int = 3
EXTENSIONS: tuple = (".png", ".tif", ".tiff")


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = cv2.subtract(mask, eroded)
    return boundary


def build_dual_channel(original: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    h, w = original.shape
    dual = np.zeros((h, w, 3), dtype=np.uint8)
    dual[:, :, 0] = original
    dual[:, :, 1] = boundary
    return dual


def extract_bboxes_from_mask(mask: np.ndarray) -> List[List[int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        bboxes.append([x, y, x + w_box, y + h_box])
    return bboxes


def main() -> None:
    raw_dir = Path(RAW_MASK_DIR)
    if not raw_dir.exists():
        print(f"错误: 掩膜目录不存在 → {raw_dir}")
        return

    mask_files = sorted(
        [f for f in raw_dir.iterdir() if f.suffix.lower() in EXTENSIONS]
    )
    if not mask_files:
        print(f"错误: {raw_dir} 中未找到 PNG/TIF 掩膜文件")
        return

    print(f"找到 {len(mask_files)} 张掩膜文件\n")

    dual_dir = Path(DUAL_OUT_DIR)
    os.makedirs(dual_dir, exist_ok=True)

    all_bboxes: Dict[str, List[List[int]]] = {}
    success = 0
    failed = 0

    pbar = tqdm(mask_files, desc="预处理", unit="file", ncols=100)
    for mask_path in pbar:
        fname = mask_path.name
        pbar.set_postfix_str(fname[:40])

        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"无法读取图像: {mask_path}")

            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            boundary = extract_boundary(mask_bin)

            dual_img = build_dual_channel(mask_bin, boundary)
            out_name = mask_path.stem + ".png"
            out_path = dual_dir / out_name
            cv2.imwrite(str(out_path), dual_img)

            bboxes = extract_bboxes_from_mask(mask_bin)
            all_bboxes[fname] = bboxes

            success += 1

        except Exception as e:
            failed += 1
            tqdm.write(f"  [失败] {fname}: {e}")

    pbar.close()

    json_dir = Path(BBOX_JSON_PATH).parent
    os.makedirs(json_dir, exist_ok=True)
    with open(BBOX_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_bboxes, f, ensure_ascii=False, indent=2)

    total_bboxes = sum(len(v) for v in all_bboxes.values())

    print(f"\n{'='*55}")
    print(f"  处理完成!")
    print(f"    成功: {success} 张")
    if failed > 0:
        print(f"    失败: {failed} 张")
    print(f"    提取 BBox 总数: {total_bboxes}")
    print(f"    双通道图像: {dual_dir}")
    print(f"    BBox JSON:   {BBOX_JSON_PATH}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
