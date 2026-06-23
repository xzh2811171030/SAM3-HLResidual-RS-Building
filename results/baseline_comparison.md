| Model | Trainable Params (M) | mIoU (%) | F1-score (%) | Boundary IoU (d=5) (%) |
| --- | --- | --- | --- | --- |
| UNet (ResNet34) | 24.44 | 55.08 | 64.23 | 48.95 |
| SegFormer (mit-b2) | **27.35** | 66.69 | 75.07 | 56.45 |
| SAM3 Zero-shot | 0.00 | 74.79 | 81.89 | 61.83 |
| SAM3 Decoder-Only | 0.69 | 56.64 | 63.69 | 43.25 |
| SAM3 LoRA | 1.74 | **82.97** | **88.64** | **66.90** |