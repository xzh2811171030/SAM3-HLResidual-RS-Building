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

### Core modules

* `src/models/`: model components for SAM3-based prompt-free segmentation and High--Low Residual refinement.
* `src/training/`: training runners, LoRA adaptation logic, lightweight decoder training, and optimization utilities.
* `src/evaluation/`: evaluation metrics and utilities for mIoU, F1, Boundary IoU, and baseline evaluation.
* `src/data/`: dataset and split-loading utilities.
* `src/utils/`: shared utilities for paths, experiment management, and analysis.

### Main experiment and diagnostic scripts

| Script                                      | Purpose                                                                     |
| ------------------------------------------- | --------------------------------------------------------------------------- |
| `src/run_calibrated_fulltest.py`            | Validation-calibrated full-test evaluation                                  |
| `src/run_calibrated_inference_pilot.py`     | Calibrated pilot-scale inference                                            |
| `src/generate_mask_predictions.py`          | Export prediction masks for downstream diagnostics                          |
| `src/recompute_excluding_pilot_metrics.py`  | Recompute held-out non-pilot metrics after excluding the fixed pilot subset |
| `src/run_hl_residual_component_ablation.py` | Component ablation for residual-branch variants                             |
| `src/run_e7a_prompt_drift.py`               | Prompt-drift diagnostic                                                     |
| `src/run_e7b_image_corruption.py`           | Image-corruption diagnostic                                                 |
| `src/run_e8_regionwise_eval.py`             | Region-wise evaluation                                                      |
| `src/run_e9_efficiency.py`                  | Parameter and inference-time analysis                                       |
| `src/run_qualitative_visualization_v2.py`   | Qualitative visualization                                                   |
| `src/run_deeplabv3plus_baseline_revised.py` | DeepLabv3+ baseline                                                         |
| `src/run_target20_segformer_baseline.py`    | Target 20-shot SegFormer baseline                                           |
| `src/run_foundation_prompt_baselines.py`    | Prompt-dependent foundation-model diagnostic baselines                      |
| `src/make_paper_tables.py`                  | Generate result tables from exported metrics                                |

### Shell scripts

| Script                                       | Purpose                                                              |
| -------------------------------------------- | -------------------------------------------------------------------- |
| `scripts/paired_significance_analysis.py`    | Paired image-level analysis between Prompt-free LoRA and HL-Residual |
| `scripts/run_exp2_component_ablation.sh`     | Component ablation experiments                                       |
| `scripts/run_overnight_unet_baseline.sh`     | Overnight U-Net baseline experiment                                  |
| `scripts/run_stageA_shot_pilot_5_10.sh`      | 5-shot and 10-shot pilot diagnostics                                 |
| `scripts/run_target20_segformer_baseline.sh` | Target 20-shot SegFormer baseline run                                |

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
