"""
=============================================================================
dataset.py  ---  遥感少样本数据集加载器 (v2.0 双轨制重构)
=============================================================================
功能说明:
  1. FewShotRSIDDataset: 从指定目录读取影像 + 双通道标签 + bbox JSON
     - 支持 few-shot 采样 (通过 num_shots + seed 控制)
     - 支持全量模式 (num_shots=None)
     - 自动处理影像/标签配对、缩放、归一化
  2. ValDataset: 独立的验证/测试数据集, 与训练集地理绝对隔离
     - 从独立 val 或 test 目录读取, 不受训练集随机划分影响
     - 自动从二值 mask 提取边界 (腐蚀法)

设计原则:
  - 所有路径通过构造函数参数显式传入, 消除硬编码
  - bbox JSON 读取中使用 .get() 防御键值缺失
  - 影像/标签严格配对检查

用法:
  from data.dataset import FewShotRSIDDataset, ValDataset

  train_ds = FewShotRSIDDataset(
      image_dir="data/raw/processed_slim/source_whu/train/images",
      dual_label_dir="data/raw/processed_slim/source_whu/train/dual_channel_labels",
      bbox_json_path="data/raw/processed_slim/source_whu/train/bboxes.json",
      num_shots=5, seed=42,
  )

  val_ds = ValDataset(
      image_dir="data/raw/processed_slim/source_whu/val/images",
      dual_label_dir="data/raw/processed_slim/source_whu/val/dual_channel_labels",
  )
=============================================================================
"""

# ==========================================================================
# 模块 1: 导入与常量定义
# ==========================================================================
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

TARGET_SIZE: int = 512
IMAGE_EXTS: tuple = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

DUMMY_BOX = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)

DUAL_LABEL_TRY_EXTS: tuple = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def _find_dual_label(img_path: Path, dual_label_dir: Path) -> Optional[Path]:
    for ext in DUAL_LABEL_TRY_EXTS:
        candidate = dual_label_dir / f"{img_path.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


# ==========================================================================
# 模块 2: FewShotRSIDDataset --- 训练用少样本数据集
# ==========================================================================
class FewShotRSIDDataset(Dataset):

    def __init__(
        self,
        image_dir: str,
        dual_label_dir: str,
        bbox_json_path: str,
        num_shots: Optional[int] = None,
        seed: int = 42,
        target_size: int = TARGET_SIZE,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.dual_label_dir = Path(dual_label_dir)
        self.bbox_json_path = Path(bbox_json_path)
        self.target_size = target_size

        if not self.bbox_json_path.exists():
            raise FileNotFoundError(f"bbox JSON 文件不存在: {self.bbox_json_path}")

        with open(self.bbox_json_path, "r", encoding="utf-8") as f:
            self.bboxes_all: Dict[str, List[List[int]]] = json.load(f)

        image_candidates = sorted([
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        ])
        all_filenames = []
        for img_path in image_candidates:
            dual_path = _find_dual_label(img_path, self.dual_label_dir)
            if dual_path is not None:
                all_filenames.append(img_path)

        if not all_filenames:
            raise FileNotFoundError(
                f"未找到配对的影像与双通道标签。\n"
                f"  影像目录: {self.image_dir}\n"
                f"  标签目录: {self.dual_label_dir}"
            )

        if num_shots is not None:
            rng = random.Random(seed)
            num_shots = min(num_shots, len(all_filenames))
            self.file_paths = rng.sample(all_filenames, num_shots)
        else:
            self.file_paths = all_filenames

        print(f"FewShotRSIDDataset: {len(self.file_paths)} 张配对样本已就绪 "
              f"(目录: {self.image_dir})")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        img_path = self.file_paths[index]
        stem = img_path.stem
        orig_filename = img_path.name

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取影像: {img_path}")
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.target_size, self.target_size))

        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        dual_path = _find_dual_label(img_path, self.dual_label_dir)
        if dual_path is None:
            raise FileNotFoundError(f"未找到双通道标签: {stem}.* (目录: {self.dual_label_dir})")
        dual_bgr = cv2.imread(str(dual_path), cv2.IMREAD_COLOR)
        if dual_bgr is None:
            raise FileNotFoundError(f"无法读取双通道标签: {dual_path}")
        dual_resized = cv2.resize(dual_bgr, (self.target_size, self.target_size))

        channel_b = dual_resized[:, :, 0].astype(np.float32) / 255.0
        channel_g = dual_resized[:, :, 1].astype(np.float32) / 255.0

        mask_tensor = torch.from_numpy(channel_b).unsqueeze(0)
        boundary_tensor = torch.from_numpy(channel_g).unsqueeze(0)

        bboxes_raw = self.bboxes_all.get(orig_filename, [])
        if not bboxes_raw:
            boxes_tensor = DUMMY_BOX
        else:
            scale_x = self.target_size / orig_w
            scale_y = self.target_size / orig_h
            scaled = []
            for box in bboxes_raw:
                if len(box) < 4:
                    continue
                x1, y1, x2, y2 = box[:4]
                scaled.append([
                    max(0.0, x1 * scale_x),
                    max(0.0, y1 * scale_y),
                    min(float(self.target_size), x2 * scale_x),
                    min(float(self.target_size), y2 * scale_y),
                ])
            if scaled:
                boxes_tensor = torch.tensor(scaled, dtype=torch.float32)
            else:
                boxes_tensor = DUMMY_BOX

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "boundary": boundary_tensor,
            "boxes": boxes_tensor,
            "name": stem,
        }


# ==========================================================================
# 模块 3: ValDataset --- 验证/测试用数据集 (地理绝对隔离)
# ==========================================================================
class ValDataset(Dataset):

    def __init__(
        self,
        image_dir: str,
        dual_label_dir: str,
        target_size: int = TARGET_SIZE,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.dual_label_dir = Path(dual_label_dir)
        self.target_size = target_size

        image_candidates = sorted([
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        ])

        self.file_paths: List[Path] = []
        for img_path in image_candidates:
            dual_path = _find_dual_label(img_path, self.dual_label_dir)
            if dual_path is not None:
                self.file_paths.append(img_path)

        if not self.file_paths:
            raise FileNotFoundError(
                f"未找到配对的影像与双通道标签。\n"
                f"  影像目录: {self.image_dir}\n"
                f"  标签目录: {self.dual_label_dir}"
            )

        print(f"ValDataset: {len(self.file_paths)} 张配对样本已就绪 "
              f"(目录: {self.image_dir})")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        img_path = self.file_paths[index]
        stem = img_path.stem

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取影像: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.target_size, self.target_size))

        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        dual_path = _find_dual_label(img_path, self.dual_label_dir)
        if dual_path is None:
            raise FileNotFoundError(f"未找到双通道标签: {stem}.* (目录: {self.dual_label_dir})")
        dual_bgr = cv2.imread(str(dual_path), cv2.IMREAD_COLOR)
        if dual_bgr is None:
            raise FileNotFoundError(f"无法读取双通道标签: {dual_path}")
        dual_resized = cv2.resize(dual_bgr, (self.target_size, self.target_size))

        channel_b = dual_resized[:, :, 0].astype(np.float32) / 255.0
        channel_g = dual_resized[:, :, 1].astype(np.float32) / 255.0

        mask_tensor = torch.from_numpy(channel_b).unsqueeze(0)
        boundary_tensor = torch.from_numpy(channel_g).unsqueeze(0)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "boundary": boundary_tensor,
            "name": stem,
        }


# ==========================================================================
# 模块 4: 自测入口
# ==========================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.cloud_paths import get_domain_paths

    for domain in ["source", "target"]:
        dp = get_domain_paths(domain, full_test=False)
        print(f"\n=== 测试 domain={domain} ===")

        train_ds = FewShotRSIDDataset(
            image_dir=dp["train_image_dir"],
            dual_label_dir=dp["train_label_dir"],
            bbox_json_path=dp["train_bbox_json"],
            num_shots=4,
            seed=42,
        )
        print(f"  train dataset 长度: {len(train_ds)}")

        val_ds = ValDataset(
            image_dir=dp["val_image_dir"],
            dual_label_dir=dp["val_label_dir"],
        )
        print(f"  val dataset 长度: {len(val_ds)}")

        for i in range(min(2, len(train_ds))):
            sample = train_ds[i]
            print(f"  [train {i}] name={sample['name']}  "
                  f"img={list(sample['image'].shape)}  "
                  f"mask={list(sample['mask'].shape)}  "
                  f"boxes={sample['boxes'].shape[0]} boxes")
