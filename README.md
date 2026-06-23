# SAM3-HLResidual-RS-Building

**Prompt-free SAM3-LoRA with High--Low Residual refinement for validation-calibrated 20-shot cross-domain building extraction in remote sensing imagery.**

This repository contains the research code, split manifests, diagnostic scripts, and lightweight result summaries for a SAM3-based remote-sensing building extraction project. The model performs full-image building segmentation without box, point, or text prompts during inference.

中文说明见 [README_zh-CN.md](README_zh-CN.md).

## Overview

Building extraction from high-resolution remote sensing imagery is important for urban mapping, geographic database updating, disaster response, and downstream geospatial analysis. Although SAM-style foundation models provide transferable visual representations, their standard prompt-based inference is not ideal for fully automated large-scale mapping.

This project focuses on a **prompt-free** setting. The model receives only RGB remote-sensing images during inference and directly predicts binary building masks. The framework first adapts the SAM3 image encoder with LoRA to build a prompt-free baseline, and then introduces a High--Low Residual branch to refine local building boundaries.

## Key Features

* **Prompt-free inference**: no box, point, or text prompt is used during inference.
* **SAM3 image encoder adaptation**: LoRA is used to adapt the SAM3 image encoder with limited target-domain labels.
* **High--Low Residual refinement**: the residual branch combines low-resolution SAM3 semantic features with high-resolution RGB details.
* **Residual correction**: HL-Residual corrects baseline logits instead of replacing the decoder.
* **Validation-calibrated 20-shot protocol**: each support seed uses 20 labeled target images for training, with a separate validation subset for checkpoint selection and inference calibration.
* **Held-out evaluation**: the primary comparison is reported on the 7,902-image held-out non-pilot WHU-Mix subset.
* **Deployment-oriented diagnostics**: the repository includes scripts for prompt drift, image corruption, region-wise behavior, efficiency analysis, qualitative visualization, paired analysis, and baseline comparison.

## Method Summary

The prompt-free LoRA baseline uses the SAM3 image encoder and a lightweight full-image decoder to output building logits. HL-Residual further predicts a residual logit correction:

```text
Z_final = Z_LoRA + alpha * R_HL(F_SAM, I_RGB)
```

where:

* `Z_LoRA` is the prompt-free LoRA baseline logit;
* `F_SAM` is the low-resolution SAM3 semantic feature;
* `I_RGB` is the high-resolution RGB image;
* `R_HL` is the High--Low Residual branch output;
* `alpha` is a learnable residual scale.

The residual branch is designed to improve local contours and building boundaries while preserving the semantic stability of the prompt-free LoRA baseline.

## Main Results

The primary comparison is conducted on the 7,902-image held-out non-pilot WHU-Mix subset under a validation-calibrated 20-shot target adaptation protocol. Results are averaged over three support seeds.

| Method           | mIoU (%) | F1 (%) | Boundary IoU (%) |
| ---------------- | -------: | -----: | ---------------: |
| Prompt-free LoRA |    74.29 |  82.36 |            44.12 |
| HL-Residual      |    75.38 |  83.10 |            46.66 |
| Gain             |    +1.09 |  +0.74 |            +2.55 |

The improvement is modest but mainly concentrated on Boundary IoU, supporting the interpretation that HL-Residual acts as a lightweight boundary refinement module rather than a universal few-shot segmentation solution.

## Repository Structure

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

## Code Organization

The repository is organized as a research-code release. The official SAM3 codebase, if locally installed or vendored under `src/models/sam3/`, is treated as an external dependency and is not counted as part of the custom experimental pipeline.

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

### Pipeline-level modules

| Module                     | Role                                                                                                                         |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `src/models/`              | Custom model definitions, including SAM3-based prompt-free segmentation modules and High--Low residual refinement components |
| `src/training/`            | Training runners for LoRA adaptation, fully supervised baselines, PEFT baselines, and experiment orchestration               |
| `src/evaluation/`          | Evaluation metrics and baseline evaluation utilities, including mIoU, F1, Boundary IoU, and related segmentation metrics     |
| `src/data/`                | Dataset and split-loading utilities                                                                                          |
| `src/utils/`               | Path handling, manifest auditing, mask saving, confusion-map generation, and diagnostic utilities                            |
| `scripts/`                 | Shell wrappers and high-level analysis scripts for selected experiments                                                      |
| `data/splits/e0_manifest/` | Fixed split manifests used by the support, validation, pilot, and held-out evaluation protocol                               |
| `results/`                 | Lightweight result summaries and exported tables; large checkpoints and masks are intentionally excluded                     |

## Experimental Pipeline

The custom code covers the following experimental stages:

1. **Prompt-free SAM3-LoRA training**
   Train the baseline model that adapts the SAM3 image encoder with LoRA and predicts full-image building masks without prompts.

2. **HL-Residual refinement training**
   Train residual refinement variants on top of the LoRA baseline, including full High--Low residual fusion and diagnostic variants.

3. **Baseline training and comparison**
   Run non-SAM or prompt-dependent diagnostic baselines such as UNetFormer-style ResNet34, SegFormer, DeepLabV3+, and SAM3 prompt-based references.

4. **Validation-calibrated inference**
   Apply validation-selected thresholding, post-processing, TTA, and full-test evaluation.

5. **Held-out non-pilot recomputation**
   Exclude the fixed pilot500 diagnostic subset from the official 8,402-image pool and recompute the 7,902-image held-out aggregate.

6. **Deployment-oriented diagnostics**
   Evaluate prompt drift, image corruption robustness, region-wise behavior, efficiency, qualitative examples, and coverage-oriented metrics.

7. **Paper table generation and analysis**
   Aggregate JSON/CSV metrics into paper-ready tables and paired image-level summaries.

## Main Entry Points

The most relevant entry points for reproducing and auditing the experiments are:

| Stage                           | Entry point                                 | Purpose                                                            |
| ------------------------------- | ------------------------------------------- | ------------------------------------------------------------------ |
| LoRA baseline training          | `src/run_e6v2_pilot.py`                     | 20-shot prompt-free LoRA pilot training                            |
| HL-Residual training            | `src/run_hl_residual_refine_pilot.py`       | Train HL-Residual refinement from a LoRA checkpoint                |
| Component ablation              | `src/run_hl_residual_component_ablation.py` | Train and evaluate `rgb_only` and `sam_only` residual variants     |
| Calibrated full-test evaluation | `src/run_calibrated_fulltest.py`            | Run validation-calibrated TTA inference and metric evaluation      |
| Pilot calibrated inference      | `src/run_calibrated_inference_pilot.py`     | Pilot-scale calibrated inference for model variants                |
| Held-out recomputation          | `src/recompute_excluding_pilot_metrics.py`  | Recompute metrics after excluding pilot500 from the full test pool |
| Mask export                     | `src/generate_mask_predictions.py`          | Save prediction masks for coverage and confusion diagnostics       |
| Prompt drift                    | `src/run_e7a_prompt_drift.py`               | Evaluate prompt sensitivity and prompt-free robustness             |
| Image corruption                | `src/run_e7b_image_corruption.py`           | Evaluate corruption robustness under multiple perturbation types   |
| Region-wise evaluation          | `src/run_e8_regionwise_eval.py`             | Summarize model behavior across region groups                      |
| Efficiency analysis             | `src/run_e9_efficiency.py`                  | Compare parameters, FLOPs, and inference time                      |
| Qualitative visualization       | `src/run_qualitative_visualization_v2.py`   | Generate qualitative prediction and error-map visualizations       |
| Paper tables                    | `src/make_paper_tables.py`                  | Aggregate exported metrics into paper-ready tables                 |

## Available Shell and Analysis Scripts

| Script                                       | Purpose                                                              |
| -------------------------------------------- | -------------------------------------------------------------------- |
| `scripts/paired_significance_analysis.py`    | Paired image-level analysis between Prompt-free LoRA and HL-Residual |
| `scripts/run_exp2_component_ablation.sh`     | Batch wrapper for residual-branch component ablation                 |
| `scripts/run_overnight_unet_baseline.sh`     | Batch wrapper for UNetFormer-style baseline experiments              |
| `scripts/run_stageA_shot_pilot_5_10.sh`      | 5-shot and 10-shot pilot sensitivity diagnostics                     |
| `scripts/run_target20_segformer_baseline.sh` | Target-domain 20-shot SegFormer baseline run                         |

<details>
<summary>Complete custom script index</summary>

### Core training scripts

| File                                        | Purpose                                                        |
| ------------------------------------------- | -------------------------------------------------------------- |
| `src/run_e6v2_pilot.py`                     | 20-shot prompt-free LoRA pilot training                        |
| `src/run_hl_residual_refine_pilot.py`       | HL-Residual refinement training from a LoRA checkpoint         |
| `src/run_hl_lite_pilot.py`                  | Lightweight residual variants such as HL-Lite and HL-Lite-Aux  |
| `src/run_distillation_pilot.py`             | Knowledge distillation pilot using SAM3 pseudo-labels          |
| `src/run_distillation_pilot_1.py`           | Distillation pilot variant or backup version                   |
| `src/run_hl_residual_component_ablation.py` | Component ablation for RGB-only and SAM-only residual variants |
| `src/run_all_experiments.py`                | Legacy multi-experiment orchestration script                   |
| `src/run_experiments_gbg.py`                | GBG-SAM3 experiment runner                                     |

### Baseline training scripts

| File                                        | Purpose                                                               |
| ------------------------------------------- | --------------------------------------------------------------------- |
| `src/run_overnight_unet_baseline.py`        | UNetFormer-style ResNet34 baseline training and calibrated evaluation |
| `src/run_target20_segformer_baseline.py`    | Target-domain 20-shot SegFormer baseline                              |
| `src/run_deeplabv3plus_baseline_revised.py` | Revised DeepLabV3+ baseline                                           |
| `src/run_foundation_prompt_baselines.py`    | SAM3 prompt-dependent foundation-model diagnostic baselines           |

### Training submodules

| File                                         | Purpose                                                                               |
| -------------------------------------------- | ------------------------------------------------------------------------------------- |
| `src/training/experiment_runner.py`          | Core LoRA training, checkpoint saving, and validation monitoring                      |
| `src/training/experiment_runner_gbg.py`      | GBG-SAM3 experiment orchestration with GatedBoundaryAdapter and source-weight loading |
| `src/training/train_fully_supervised.py`     | Fully supervised training utilities with multi-seed and full-metric evaluation        |
| `src/training/train_fully_supervised_seg.py` | SegFormer-style fully supervised training                                             |
| `src/training/train_peft_baselines.py`       | PEFT baseline training, including LoRA-style adaptation                               |

### Calibrated inference and evaluation scripts

| File                                          | Purpose                                                                                |
| --------------------------------------------- | -------------------------------------------------------------------------------------- |
| `src/run_calibrated_fulltest.py`              | Calibrated full-test inference with TTA, post-processing search, and metric evaluation |
| `src/run_calibrated_inference_pilot.py`       | Pilot-scale calibrated inference for multiple model variants                           |
| `src/02_run_sam_calibrated_save_confusion.py` | Calibrated inference with confusion-map saving                                         |
| `src/recompute_excluding_pilot_metrics.py`    | Recompute held-out 7,902-image metrics after excluding pilot500                        |
| `src/generate_mask_predictions.py`            | Save prediction masks for coverage-oriented diagnostics                                |

### Robustness and deployment diagnostics

| File                                   | Purpose                                                                 |
| -------------------------------------- | ----------------------------------------------------------------------- |
| `src/run_e7_robustness.py`             | Legacy E7 image-corruption robustness evaluation                        |
| `src/run_e7_robustness_revised.py`     | Revised E7 robustness evaluation                                        |
| `src/run_e7a_prompt_drift.py`          | Prompt-drift diagnostic comparing box-prompted and prompt-free behavior |
| `src/run_e7a_prompt_drift_residual.py` | Simplified prompt-drift diagnostic for HL-Residual                      |
| `src/run_e7b_image_corruption.py`      | Image-corruption diagnostic across corruption types, models, and seeds  |
| `src/run_e7b_resume.py`                | Resume script for unfinished E7b corruption experiments                 |
| `src/run_e7b_resume_select.py`         | Resume-and-select script for E7b corruption experiments                 |
| `src/merge_e7_json_parts.py`           | Merge split E7 JSON result files                                        |
| `src/plot_e7_results.py`               | Plot E7 diagnostic results                                              |

### E8/E9 analysis scripts

| File                            | Purpose                                                 |
| ------------------------------- | ------------------------------------------------------- |
| `src/run_e8_regionwise_eval.py` | Region-wise evaluation by city or inferred region group |
| `src/run_e9_efficiency.py`      | Parameter count, FLOPs, and inference-time analysis     |

### Analysis and visualization scripts

| File                                      | Purpose                                               |
| ----------------------------------------- | ----------------------------------------------------- |
| `src/exp1_val_budget_end2end.py`          | Validation-budget sensitivity analysis                |
| `src/01_run_three_model_coverage_eval.py` | Three-model coverage-oriented evaluation              |
| `src/03_run_unetformer_save_confusion.py` | UNetFormer confusion-map saving                       |
| `src/cross_city_variance_analysis.py`     | Cross-city variance analysis                          |
| `src/make_paper_tables.py`                | Generate paper-ready tables from exported metrics     |
| `src/run_qualitative_visualization.py`    | Qualitative visualization of predictions and overlays |
| `src/run_qualitative_visualization_v2.py` | Revised qualitative visualization                     |

### Utility and data scripts

| File                                           | Purpose                                                                             |
| ---------------------------------------------- | ----------------------------------------------------------------------------------- |
| `src/evaluation/eval_metrics.py`               | Metric computation for mIoU, F1, Boundary IoU, precision, and recall                |
| `src/evaluation/e5_eval_baselines_full.py`     | Full-scale E5 baseline evaluation                                                   |
| `src/evaluation/eval_segformer.py`             | SegFormer evaluation                                                                |
| `src/utils/diagnose.py`                        | Diagnostic utilities for data, models, and paths                                    |
| `src/utils/cloud_paths.py`                     | Cloud/local path adaptation utilities                                               |
| `src/utils/audit_manifest_masks.py`            | Manifest and image-label consistency auditing                                       |
| `src/utils/mask_saving_and_confusion_utils.py` | Utilities for mask saving and confusion-map generation                              |
| `src/models/model.py`                          | Custom model definitions, including GatedBoundaryAdapter and HLLiteDecoder variants |
| `src/__init__.py`                              | Package initialization file                                                         |

</details>


## Environment

The main experimental environment used in this project:

```text
Python 3.12
PyTorch 2.8
CUDA 12.8
SAM3 checkpoint: sam3.1-st-bf16
Task: binary building extraction
```

Install dependencies:

```bash
conda create -n sam3_hl python=3.12
conda activate sam3_hl
pip install -r requirements.txt
```

Please install the official SAM3 package and prepare the SAM3 checkpoint according to the official SAM3 instructions and license requirements.

This repository does not redistribute SAM3 checkpoints.

## Dataset Preparation

This project uses the WHU aerial building dataset and WHU-Mix building dataset. Due to dataset licenses and storage constraints, raw images and masks are not included in this repository.

A recommended local dataset layout is:

```text
data_root/
├── whu_aerial/
│   ├── images/
│   └── masks/
└── whu_mix/
    ├── images/
    └── masks/
```

The `data/` directory in this repository is intended to store only lightweight split manifests and file lists, not raw images or masks.

## Minimal Reproduction Guide

Before running experiments, update local paths in configuration files and shell scripts, including:

* dataset root;
* SAM3 checkpoint path;
* output directory;
* GPU ID;
* result file path.

Typical commands include:

```bash
# Paired image-level analysis
python scripts/paired_significance_analysis.py

# Component ablation
bash scripts/run_exp2_component_ablation.sh

# 5-shot and 10-shot pilot diagnostics
bash scripts/run_stageA_shot_pilot_5_10.sh

# Target 20-shot SegFormer baseline
bash scripts/run_target20_segformer_baseline.sh
```

Core training and evaluation code is organized under `src/`. For large-scale experiments, please inspect the corresponding entry-point scripts and update machine-specific paths before running.

## Checkpoints and Weights

This repository does not include:

* SAM3 official checkpoints;
* LoRA training checkpoints;
* HL-Residual training checkpoints;
* `.pth`, `.pt`, `.ckpt`, or `.safetensors` files;
* full prediction masks;
* private logs or temporary runtime outputs.

If you need to run the experiments, create a local directory:

```text
weights/
```

and place your SAM3 checkpoint or trained checkpoints there. The `weights/` directory is ignored by Git.

Large checkpoint files inside `results/` should also be kept local and should not be uploaded to GitHub.

## Protocol Clarification

* The main setting is **validation-calibrated 20-shot target adaptation**, not a strictly 20-image-only protocol.
* The target validation subset is used for checkpoint selection and inference calibration, but not for gradient-based training.
* The fixed pilot500 subset is used only for diagnostics.
* The main quantitative comparison is reported on the 7,902-image held-out non-pilot WHU-Mix subset.
* Prompt-free LoRA and HL-Residual use no box, point, or text prompts during inference.

## Evaluation Metrics

The reported metrics are:

* **mIoU**: foreground building IoU averaged over evaluation images;
* **F1**: foreground building F1 score;
* **Boundary IoU**: boundary overlap under a fixed boundary tolerance.

Validation-based thresholding, component filtering, morphological operations, hole filling, and test-time augmentation are handled through the validation calibration protocol.

## Citation

If this repository is useful for your research, please cite the related work:

```bibtex
@article{xia2026sam3hlresidual,
  title   = {Prompt-Free High--Low Residual Refinement of SAM3 Image Encoders for Validation-Calibrated 20-Shot Cross-Domain Building Extraction},
  author  = {Xia, Zihan and Guo, Jingpeng and Xia, Lang},
  journal = {Manuscript in preparation},
  year    = {2026}
}
```

## License

This repository is released for academic and research use. Please also follow the licenses of SAM3, WHU, WHU-Mix, and any third-party resources used in this project.
