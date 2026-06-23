# SAM3-HLResidual-RS-Building

基于 **SAM3 图像编码器 + LoRA 参数高效微调 + High--Low Residual 边界细化分支** 的低标注跨域遥感建筑物提取项目。

本仓库整理了 SAM3 Prompt-free 建筑物分割实验的核心代码、配置模板、数据划分、诊断脚本和结果汇总。模型在推理阶段不使用 box、point 或 text prompt，仅输入 RGB 遥感影像即可直接预测建筑物二值掩膜。

## 项目概述

高分辨率遥感建筑物提取在城市制图、地理数据库更新、灾害评估和空间分析中具有重要意义。SAM 类视觉基础模型具有较强的通用视觉表征能力，但其标准推理流程通常依赖 box、point 或 text prompt，这限制了其在大规模自动化制图任务中的直接应用。

本项目关注 **prompt-free building extraction** 场景。模型推理时只输入 RGB 遥感影像，不依赖人工框、点提示、文本提示或外部检测器。整体方法首先使用 SAM3 图像编码器和 LoRA 构建 prompt-free baseline，然后引入 High--Low Residual 分支，对 baseline logits 进行残差修正，从而改善建筑物边界和局部轮廓质量。

## 核心特点

* **Prompt-free 推理**：推理阶段不使用 box、point 或 text prompt。
* **SAM3 图像编码器适配**：使用 LoRA/PEFT 对 SAM3 image encoder 进行低标注目标域适配。
* **High--Low Residual 分支**：融合低分辨率 SAM3 语义特征和高分辨率 RGB 细节特征。
* **残差修正而非替换解码器**：HL-Residual 只修正 baseline logits，不重新替代完整语义解码器。
* **Validation-calibrated 20-shot 协议**：每个 support seed 使用 20 张目标域标注影像训练，并使用独立验证集进行 checkpoint 选择和推理校准。
* **Held-out 测试评估**：主结果报告在排除 pilot500 诊断子集后的 7,902 张 WHU-Mix held-out non-pilot 测试影像上。
* **可复现实验整理**：包含固定 split、配置模板、诊断脚本、消融实验脚本和结果汇总。

## 方法简述

Prompt-free LoRA baseline 使用 SAM3 图像编码器提取低分辨率语义特征，并通过轻量级 full-image decoder 输出建筑物分割 logits。HL-Residual 在此基础上进一步引入高低分辨率融合分支，预测一个残差 logit correction：

```text
Z_final = Z_LoRA + alpha * R_HL(F_SAM, I_RGB)
```

其中：

* `Z_LoRA` 表示 prompt-free LoRA baseline 输出的 logits；
* `F_SAM` 表示 SAM3 图像编码器输出的低分辨率语义特征；
* `I_RGB` 表示输入的高分辨率 RGB 遥感影像；
* `R_HL` 表示 High--Low Residual 分支输出的残差 logits；
* `alpha` 表示可学习的残差缩放系数。

该设计的目的不是重新学习完整分割模型，而是在 SAM3-LoRA 已有语义预测的基础上进行轻量级边界修正。

## 主实验结果

主结果基于 WHU→WHU-Mix 20-shot 跨域建筑物分割设置，报告在 7,902 张 held-out non-pilot WHU-Mix 测试影像上，并对 3 个 support seed 取平均。

| 方法               | mIoU (%) | F1 (%) | Boundary IoU (%) |
| ---------------- | -------: | -----: | ---------------: |
| Prompt-free LoRA |    74.29 |  82.36 |            44.12 |
| HL-Residual      |    75.38 |  83.10 |            46.66 |
| 提升               |    +1.09 |  +0.74 |            +2.55 |

可以看到，HL-Residual 的总体提升幅度较为克制，但 Boundary IoU 的提升更明显，说明该分支主要发挥的是局部边界细化作用，而不是简单扩大建筑物区域覆盖。

## 当前仓库结构

```text
SAM3-HLResidual-RS-Building/
├── config/
│   └── coverage_config_template.json
├── data/
│   └── splits/
│       └── e0_manifest/
├── results/
├── scripts/
│   ├── paired_significance_analysis.py
│   ├── run_exp2_component_ablation.sh
│   ├── run_overnight_unet_baseline.sh
│   ├── run_stageA_shot_pilot_5_10.sh
│   └── run_target20_segformer_baseline.sh
├── src/
├── .gitignore
├── README.md
├── README_CH.md
└── requirements.txt
```

## 目录说明

* `config/`：保存诊断分析和复现实验相关的配置模板。
* `data/splits/e0_manifest/`：保存固定 support、validation、pilot diagnostic 和 held-out evaluation 子集划分文件。
* `results/`：保存轻量级结果汇总和导出表格，不应保存大体积权重文件。
* `scripts/`：保存诊断分析、消融实验、低 shot pilot 实验和 baseline 实验的运行脚本。
* `src/`：保存核心 Python 代码，包括模型结构、数据读取、训练/评估逻辑、评价指标和工具函数。
* `weights/`：仅本地使用，用于存放 SAM3 checkpoint 或训练得到的 checkpoint，不上传 GitHub。

## 当前脚本说明

| 脚本                                           | 用途                                         |
| -------------------------------------------- | ------------------------------------------ |
| `scripts/paired_significance_analysis.py`    | 对 Prompt-free LoRA 和 HL-Residual 进行图像级配对分析 |
| `scripts/run_exp2_component_ablation.sh`     | 运行残差分支组件消融实验                               |
| `scripts/run_overnight_unet_baseline.sh`     | 运行 U-Net 相关 baseline 实验                    |
| `scripts/run_stageA_shot_pilot_5_10.sh`      | 运行 5-shot 和 10-shot pilot 诊断实验             |
| `scripts/run_target20_segformer_baseline.sh` | 运行 20-shot SegFormer baseline 实验           |

运行这些脚本前，需要检查并修改本地数据路径、checkpoint 路径、GPU 编号和输出目录。

## 环境配置

本项目主要实验环境为：

```text
Python 3.12
PyTorch 2.8
CUDA 12.8
SAM3 checkpoint: sam3.1-st-bf16
Task: binary building extraction
```

安装依赖：

```bash
conda create -n sam3_hl python=3.12
conda activate sam3_hl
pip install -r requirements.txt
```

SAM3 官方代码和 checkpoint 请按照 SAM3 官方说明自行安装和下载。本仓库不重新分发 SAM3 官方权重。

## 数据准备

本项目使用 WHU aerial building dataset 和 WHU-Mix building dataset。由于数据许可和仓库存储限制，本仓库不包含原始影像和 mask。

推荐本地数据组织方式：

```text
data_root/
├── whu_aerial/
│   ├── images/
│   └── masks/
└── whu_mix/
    ├── images/
    └── masks/
```

运行实验前，请在配置文件或脚本中修改本地数据路径和 checkpoint 路径。

本仓库中的 `data/` 目录只建议保存轻量级 split manifest 和文件列表，不上传原始遥感影像或 mask。

## 权重说明

本仓库不包含：

* SAM3 官方 checkpoint；
* LoRA 训练 checkpoint；
* HL-Residual 训练 checkpoint；
* `.pth`、`.pt`、`.ckpt`、`.safetensors` 等权重文件；
* 完整预测 mask。

如需运行实验，请在本地创建：

```text
weights/
```

并将 SAM3 checkpoint 或自己训练得到的 checkpoint 放入该目录。`weights/` 已被 `.gitignore` 忽略，不会上传到 GitHub。

如果 `results/` 中生成了 checkpoint 或权重文件，也应只保留在本地，不应上传到 GitHub。

## 协议说明

* 主实验设置为 **validation-calibrated 20-shot target adaptation**，不是严格的“只使用 20 张图像”的协议。
* 目标域验证集用于 checkpoint selection 和 inference calibration，但不参与梯度训练。
* 固定 pilot500 子集只用于诊断分析。
* 主结果报告在 7,902 张 held-out non-pilot WHU-Mix 测试影像上。
* Prompt-free LoRA 和 HL-Residual 在推理阶段均不使用 box、point 或 text prompt。

## 评价指标

本项目主要报告：

* **mIoU**：前景建筑物 IoU，在测试影像层面平均；
* **F1 score**：前景建筑物类别的 F1 分数；
* **Boundary IoU**：固定边界容忍宽度下的边界重叠指标。

验证集校准、阈值选择、最小连通域过滤、形态学处理、孔洞填充和 TTA 等操作均属于 validation calibration protocol 的一部分。

## 引用

如果本项目对你的研究有帮助，请引用相关论文和 SAM3 官方工作。

```bibtex
@article{xia2026sam3hlresidual,
  title   = {Prompt-Free High--Low Residual Refinement of SAM3 Image Encoders for Validation-Calibrated 20-Shot Cross-Domain Building Extraction},
  author  = {Xia, Zihan and Guo, Jingpeng and Xia, Lang},
  journal = {Manuscript in preparation},
  year    = {2026}
}
```

## License

本仓库仅供学术研究和学习使用。使用本项目时，请同时遵守 SAM3、WHU、WHU-Mix 以及其他第三方资源的许可要求。
