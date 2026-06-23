#!/usr/bin/env python3
"""
离线预处理完整 WHU-Mix 测试集 (8402张)
生成 dual_channel_labels 和 bboxes.json
python src/data/preprocess_full_test.py
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def process_full_test(input_images_dir, input_labels_dir, output_dual_dir, output_bbox_json):
    """
    处理完整测试集
    - input_images_dir: 原始图像目录
    - input_labels_dir: 原始标签目录（文件名需与图像一一对应）
    - output_dual_dir: 输出双通道标签目录
    - output_bbox_json: 输出 bbox JSON 文件路径
    """
    # 创建输出目录
    Path(output_dual_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有标签文件（假设为 .png 或 .tif）
    label_files = list(Path(input_labels_dir).glob("*.png")) + list(Path(input_labels_dir).glob("*.tif"))
    if not label_files:
        raise FileNotFoundError(f"在 {input_labels_dir} 中未找到任何标签文件")
    
    bbox_dict = {}
    kernel = np.ones((3,3), np.uint8)
    
    for label_path in tqdm(label_files, desc="处理测试集"):
        # 读取标签
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if label is None:
            print(f"警告: 无法读取 {label_path}，跳过")
            continue
        
        # 二值化（确保是0/255）
        _, label_bin = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
        
        # 提取物理边界：腐蚀后相减
        eroded = cv2.erode(label_bin, kernel)
        boundary = label_bin - eroded
        
        # 生成双通道图像：通道0=原始标签，通道1=边界，通道2=0
        dual = np.zeros((*label.shape, 3), dtype=np.uint8)
        dual[:,:,0] = label_bin
        dual[:,:,1] = boundary
        
        # 保存双通道图像（与标签同名）
        out_path = Path(output_dual_dir) / label_path.name
        cv2.imwrite(str(out_path), dual)
        
        # 提取所有建筑物的 bounding boxes
        contours, _ = cv2.findContours(label_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([x, y, x+w, y+h])
        bbox_dict[label_path.name] = boxes
    
    # 保存 bbox JSON
    with open(output_bbox_json, "w") as f:
        json.dump(bbox_dict, f, indent=2)
    
    print(f"✅ 处理完成！共处理 {len(label_files)} 张图像")
    print(f"   双通道标签保存至: {output_dual_dir}")
    print(f"   BBox JSON 保存至: {output_bbox_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预处理完整 WHU-Mix 测试集")
    parser.add_argument("--images_dir", type=str, 
                        default="./data/data/raw/whu_mix_full_test/test/image",
                        help="原始图像目录")
    parser.add_argument("--labels_dir", type=str,
                        default="./data/data/raw/whu_mix_full_test/test/label",
                        help="原始标签目录")
    parser.add_argument("--output_dual", type=str,
                        default="./data/data/raw/whu_mix_full_test/dual_channel_labels",
                        help="输出双通道标签目录")
    parser.add_argument("--output_bbox", type=str,
                        default="./data/data/raw/whu_mix_full_test/bbox.json",
                        help="输出 bbox JSON 文件路径")
    args = parser.parse_args()
    
    process_full_test(args.images_dir, args.labels_dir, args.output_dual, args.output_bbox)