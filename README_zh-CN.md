# SAM3-HLResidual-RS-Building

基于 **SAM3 图像编码器 + LoRA 参数高效微调 + High--Low Residual 边界细化分支** 的低标注跨域遥感建筑物提取项目。

本仓库整理了 SAM3 Prompt-free 建筑物分割实验的核心代码、配置模板、数据划分、诊断脚本和结果汇总。模型在推理阶段不使用 box、point 或 text prompt，仅输入 RGB 遥感影像即可直接预测建筑物二值掩膜。

English version: [README.md](README.md).

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
* **部署诊断分析**：仓库包含 prompt drift、image corruption、region-wise、efficiency、qualitative visualization、paired analysis 和 baseline comparison 等诊断脚本。

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
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   ├── utils/
│   ├── generate_mask_predictions.py
│   ├── make_paper_tables.py
│   ├── recompute_excluding_pilot_metrics.py
│   ├── run_calibrated_fulltest.py
│   ├── run_calibrated_inference_pilot.py
│   ├── run_deeplabv3plus_baseline_revised.py
│   ├── run_e7a_prompt_drift.py
│   ├── run_e7b_image_corruption.py
│   ├── run_e8_regionwise_eval.py
│   ├── run_e9_efficiency.py
│   ├── run_foundation_prompt_baselines.py
│   ├── run_hl_residual_component_ablation.py
│   ├── run_qualitative_visualization_v2.py
│   └── run_target20_segformer_baseline.py
├── .gitignore
├── README.md
├── README_zh-CN.md
└── requirements.txt
```

## 代码组织

本仓库是整理后的 research-code release。如果本地环境中将官方 SAM3 代码放在 `src/models/sam3/` 下，该部分应视为外部依赖，不计入本项目自定义实验管线。

```text
SAM3-HLResidual-RS-Building/
├── config/
│   └── coverage_config_template.json
├── data/
│   └── splits/
│       └── e0_manifest/
├── results/
├── scripts/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   ├── utils/
│   └── run_*.py
├── .gitignore
├── README.md
├── README_zh-CN.md
└── requirements.txt
```

### 模块级说明

| 模块                         | 作用                                                                            |
| -------------------------- | ----------------------------------------------------------------------------- |
| `src/models/`              | 自定义模型定义，包括 SAM3 prompt-free segmentation 模块和 High--Low Residual refinement 组件 |
| `src/training/`            | LoRA 适配、全监督 baseline、PEFT baseline 和实验编排相关训练代码                                |
| `src/evaluation/`          | mIoU、F1、Boundary IoU 以及 baseline evaluation 相关评估代码                            |
| `src/data/`                | 数据集读取和 split manifest 加载工具                                                    |
| `src/utils/`               | 路径管理、manifest 审计、mask 保存、混淆矩阵生成和诊断工具                                          |
| `scripts/`                 | 部分实验的 shell wrapper 和高层分析脚本                                                   |
| `data/splits/e0_manifest/` | support、validation、pilot 和 held-out evaluation 使用的固定划分文件                      |
| `results/`                 | 轻量级结果汇总和导出表格；不上传大体积 checkpoint 或预测 mask                                       |

## 实验管线

本仓库自定义代码覆盖以下实验阶段：

1. **Prompt-free SAM3-LoRA 训练**
   训练不使用 prompt 的 SAM3-LoRA baseline，输入 RGB 遥感影像并直接预测建筑物 mask。

2. **HL-Residual 残差细化训练**
   在 LoRA baseline 基础上训练 High--Low residual refinement 分支，包括完整 HL-Residual 以及诊断变体。

3. **Baseline 训练与对比**
   运行 UNetFormer-style ResNet34、SegFormer、DeepLabV3+ 和 SAM3 prompt-dependent diagnostic baselines。

4. **Validation-calibrated 推理**
   使用验证集选择阈值、后处理参数、TTA 和 checkpoint，并进行 full-test 评估。

5. **Held-out non-pilot 重算**
   从官方 8,402 张测试池中排除固定 pilot500 诊断子集，重新计算 7,902 张 held-out non-pilot 主结果。

6. **部署导向诊断**
   包括 prompt drift、image corruption、region-wise behavior、efficiency、qualitative examples 和 coverage-oriented metrics。

7. **论文表格与分析**
   将 JSON/CSV 指标聚合为论文表格，并生成图像级配对分析结果。

## 主要入口脚本

| 阶段                        | 入口脚本                                        | 用途                                             |
| ------------------------- | ------------------------------------------- | ---------------------------------------------- |
| LoRA baseline 训练          | `src/run_e6v2_pilot.py`                     | 20-shot prompt-free LoRA pilot 训练              |
| HL-Residual 训练            | `src/run_hl_residual_refine_pilot.py`       | 从 LoRA checkpoint 出发训练 HL-Residual refinement  |
| 组件消融                      | `src/run_hl_residual_component_ablation.py` | 训练并评估 `rgb_only` 和 `sam_only` 残差变体             |
| 校准 full-test 评估           | `src/run_calibrated_fulltest.py`            | validation-calibrated TTA 推理和指标评估              |
| pilot 校准推理                | `src/run_calibrated_inference_pilot.py`     | 面向模型变体的 pilot-scale calibrated inference       |
| held-out 重算               | `src/recompute_excluding_pilot_metrics.py`  | 排除 pilot500 后重新计算 held-out non-pilot 指标        |
| mask 导出                   | `src/generate_mask_predictions.py`          | 保存预测 mask，用于 coverage 和 confusion 诊断           |
| prompt drift              | `src/run_e7a_prompt_drift.py`               | prompt sensitivity 和 prompt-free robustness 诊断 |
| image corruption          | `src/run_e7b_image_corruption.py`           | 多种图像扰动下的鲁棒性诊断                                  |
| region-wise evaluation    | `src/run_e8_regionwise_eval.py`             | 分区域/城市统计模型表现                                   |
| efficiency analysis       | `src/run_e9_efficiency.py`                  | 参数量、FLOPs 和推理时间分析                              |
| qualitative visualization | `src/run_qualitative_visualization_v2.py`   | 生成定性预测图和 error map                             |
| paper tables              | `src/make_paper_tables.py`                  | 从导出指标生成论文表格                                    |

## Scripts 目录

| 脚本                                           | 用途                                             |
| -------------------------------------------- | ---------------------------------------------- |
| `scripts/paired_significance_analysis.py`    | Prompt-free LoRA 与 HL-Residual 的图像级配对分析        |
| `scripts/run_exp2_component_ablation.sh`     | residual branch 组件消融实验的批处理脚本                   |
| `scripts/run_overnight_unet_baseline.sh`     | UNetFormer-style baseline 实验                   |
| `scripts/run_stageA_shot_pilot_5_10.sh`      | 5-shot 和 10-shot pilot sensitivity diagnostics |
| `scripts/run_target20_segformer_baseline.sh` | target-domain 20-shot SegFormer baseline       |

<details>
<summary>完整自定义脚本索引</summary>

### 核心训练脚本

| 文件                                          | 用途                                      |
| ------------------------------------------- | --------------------------------------- |
| `src/run_e6v2_pilot.py`                     | 20-shot Prompt-free LoRA pilot 训练       |
| `src/run_hl_residual_refine_pilot.py`       | 基于 LoRA checkpoint 训练 HL-Residual 残差精炼器 |
| `src/run_hl_lite_pilot.py`                  | HL-Lite / HL-Lite-Aux 等轻量残差变体训练         |
| `src/run_distillation_pilot.py`             | 基于 SAM3 伪标签的蒸馏 pilot 训练                 |
| `src/run_distillation_pilot_1.py`           | 蒸馏 pilot 的变体或备份版本                       |
| `src/run_hl_residual_component_ablation.py` | RGB-only / SAM-only 残差分支组件消融            |
| `src/run_all_experiments.py`                | 旧版多实验编排脚本                               |
| `src/run_experiments_gbg.py`                | GBG-SAM3 实验运行脚本                         |

### Baseline 训练脚本

| 文件                                          | 用途                                                          |
| ------------------------------------------- | ----------------------------------------------------------- |
| `src/run_overnight_unet_baseline.py`        | UNetFormer-style ResNet34 baseline 训练和校准评估                  |
| `src/run_target20_segformer_baseline.py`    | Target-domain 20-shot SegFormer baseline                    |
| `src/run_deeplabv3plus_baseline_revised.py` | 修订版 DeepLabV3+ baseline                                     |
| `src/run_foundation_prompt_baselines.py`    | SAM3 prompt-dependent foundation-model diagnostic baselines |

### 训练子模块

| 文件                                           | 用途                                                            |
| -------------------------------------------- | ------------------------------------------------------------- |
| `src/training/experiment_runner.py`          | LoRA 训练、checkpoint 保存和 validation monitoring                  |
| `src/training/experiment_runner_gbg.py`      | GBG-SAM3 实验编排，包含 GatedBoundaryAdapter 和 source-weight loading |
| `src/training/train_fully_supervised.py`     | 全监督训练工具，支持多 seed 和完整指标评估                                      |
| `src/training/train_fully_supervised_seg.py` | SegFormer-style 全监督训练                                         |
| `src/training/train_peft_baselines.py`       | PEFT baseline 训练，包括 LoRA-style adaptation                     |

### 校准推理与评估脚本

| 文件                                            | 用途                                         |
| --------------------------------------------- | ------------------------------------------ |
| `src/run_calibrated_fulltest.py`              | TTA、后处理搜索和指标评估的校准 full-test 推理             |
| `src/run_calibrated_inference_pilot.py`       | 面向多模型变体的 pilot-scale calibrated inference  |
| `src/02_run_sam_calibrated_save_confusion.py` | 校准推理并保存 confusion map                      |
| `src/recompute_excluding_pilot_metrics.py`    | 排除 pilot500 后重算 7,902 张 held-out 指标        |
| `src/generate_mask_predictions.py`            | 保存预测 mask，用于 coverage-oriented diagnostics |

### 鲁棒性与部署诊断

| 文件                                     | 用途                                                |
| -------------------------------------- | ------------------------------------------------- |
| `src/run_e7_robustness.py`             | 旧版 E7 图像腐蚀鲁棒性评估                                   |
| `src/run_e7_robustness_revised.py`     | 修订版 E7 robustness evaluation                      |
| `src/run_e7a_prompt_drift.py`          | 比较 box-prompted 与 prompt-free 行为的 prompt-drift 诊断 |
| `src/run_e7a_prompt_drift_residual.py` | HL-Residual 的简化 prompt-drift 诊断                   |
| `src/run_e7b_image_corruption.py`      | 多腐蚀类型、多模型、多 seed 的 image-corruption 诊断            |
| `src/run_e7b_resume.py`                | E7b 未完成腐蚀实验的续跑脚本                                  |
| `src/run_e7b_resume_select.py`         | E7b 续跑选模脚本                                        |
| `src/merge_e7_json_parts.py`           | 合并分批 E7 JSON 结果                                   |
| `src/plot_e7_results.py`               | 绘制 E7 诊断结果                                        |

### E8/E9 分析脚本

| 文件                              | 用途                               |
| ------------------------------- | -------------------------------- |
| `src/run_e8_regionwise_eval.py` | 按城市或区域组进行 region-wise evaluation |
| `src/run_e9_efficiency.py`      | 参数量、FLOPs 和推理时间分析                |

### 分析与可视化脚本

| 文件                                        | 用途                                     |
| ----------------------------------------- | -------------------------------------- |
| `src/exp1_val_budget_end2end.py`          | validation budget sensitivity analysis |
| `src/01_run_three_model_coverage_eval.py` | 三模型 coverage-oriented evaluation       |
| `src/03_run_unetformer_save_confusion.py` | UNetFormer confusion-map 保存            |
| `src/cross_city_variance_analysis.py`     | 跨城市方差分析                                |
| `src/make_paper_tables.py`                | 从导出指标生成论文表格                            |
| `src/run_qualitative_visualization.py`    | 预测结果与原图叠加的定性可视化                        |
| `src/run_qualitative_visualization_v2.py` | 修订版定性可视化                               |

### 工具与数据脚本

| 文件                                             | 用途                                                  |
| ---------------------------------------------- | --------------------------------------------------- |
| `src/evaluation/eval_metrics.py`               | mIoU、F1、Boundary IoU、precision 和 recall 指标计算        |
| `src/evaluation/e5_eval_baselines_full.py`     | E5 full-scale baseline evaluation                   |
| `src/evaluation/eval_segformer.py`             | SegFormer evaluation                                |
| `src/utils/diagnose.py`                        | 数据、模型和路径诊断工具                                        |
| `src/utils/cloud_paths.py`                     | AutoDL 与本地环境的路径适配工具                                 |
| `src/utils/audit_manifest_masks.py`            | Manifest 和 image-label 一致性审计                        |
| `src/utils/mask_saving_and_confusion_utils.py` | Mask 保存和 confusion-map 生成工具函数                       |
| `src/models/model.py`                          | 自定义模型定义，包括 GatedBoundaryAdapter 和 HLLiteDecoder 等变体 |
| `src/__init__.py`                              | Python package 初始化文件                                |

</details>


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

本仓库中的 `data/` 目录只保存轻量级 split manifest 和文件列表，不上传原始遥感影像或 mask。

## 最小复现说明

运行实验前，需要先检查并修改：

* 数据集路径；
* SAM3 checkpoint 路径；
* 输出目录；
* GPU 编号；
* 结果文件路径。

常见命令包括：

```bash
# 图像级配对分析
python scripts/paired_significance_analysis.py

# 组件消融实验
bash scripts/run_exp2_component_ablation.sh

# 5-shot 和 10-shot pilot 诊断
bash scripts/run_stageA_shot_pilot_5_10.sh

# Target 20-shot SegFormer baseline
bash scripts/run_target20_segformer_baseline.sh
```

核心训练和评估代码位于 `src/` 目录。运行大规模实验前，请检查对应入口脚本中的本地路径和输出路径。

## 权重说明

本仓库不包含：

* SAM3 官方 checkpoint；
* LoRA 训练 checkpoint；
* HL-Residual 训练 checkpoint；
* `.pth`、`.pt`、`.ckpt`、`.safetensors` 等权重文件；
* 完整预测 mask；
* 个人日志或临时输出文件。

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
