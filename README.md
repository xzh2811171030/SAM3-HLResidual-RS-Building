# SAM3-HLResidual-RS-Building

**Prompt-free SAM3-LoRA with High--Low Residual refinement for validation-calibrated 20-shot cross-domain building extraction in remote sensing imagery.**

This repository contains the core code, configuration templates, split manifests, diagnostic scripts, and result summaries for a SAM3-based remote-sensing building extraction project. The model performs full-image building segmentation without box, point, or text prompts during inference.

## Overview

Building extraction from high-resolution remote sensing imagery is important for urban mapping, geographic database updating, disaster response, and downstream geospatial analysis. Although SAM-style foundation models provide transferable visual representations, their standard prompt-based inference is not ideal for fully automated large-scale mapping.

This project focuses on a **prompt-free** setting. The model receives only RGB remote sensing images during inference and directly predicts binary building masks. The framework first adapts the SAM3 image encoder with LoRA to build a prompt-free baseline, and then introduces a High--Low Residual branch to refine local building boundaries.

## Key Features

* **Prompt-free inference**: no box, point, or text prompt is used during inference.
* **SAM3 image encoder adaptation**: LoRA is used to adapt the SAM3 image encoder with limited target-domain labels.
* **High--Low Residual refinement**: the residual branch combines low-resolution SAM3 semantic features with high-resolution RGB details.
* **Residual correction**: HL-Residual corrects baseline logits instead of replacing the decoder.
* **Validation-calibrated 20-shot protocol**: each support seed uses 20 labeled target images for training, with a separate validation subset for checkpoint selection and inference calibration.
* **Held-out evaluation**: the primary comparison is reported on the 7,902-image held-out non-pilot WHU-Mix subset.
* **Reproducibility utilities**: fixed split manifests, configuration templates, diagnostic scripts, and result summaries are included.

## Method Summary

The framework first builds a prompt-free LoRA baseline using the SAM3 image encoder and a lightweight full-image decoder. On top of this baseline, HL-Residual predicts a residual logit correction:

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
├── .gitignore
├── README.md
├── README_CH.md
└── requirements.txt
```

## Directory Description

* `config/`: configuration templates for diagnostic analysis and reproducibility.
* `data/splits/e0_manifest/`: fixed split manifests for support, validation, pilot diagnostic, and held-out evaluation subsets.
* `results/`: lightweight result summaries and exported tables. Large checkpoints should not be stored here.
* `scripts/`: shell scripts and analysis utilities for diagnostics, ablation studies, low-shot pilot experiments, and baseline runs.
* `src/`: core Python source code, including model components, data handling, training/evaluation logic, metrics, and utilities.
* `weights/`: local-only directory for SAM3 checkpoints or trained checkpoints. This directory is ignored by Git and should not be uploaded.

## Available Scripts

The current public scripts include:

| Script                                       | Purpose                                                              |
| -------------------------------------------- | -------------------------------------------------------------------- |
| `scripts/paired_significance_analysis.py`    | Paired image-level analysis between Prompt-free LoRA and HL-Residual |
| `scripts/run_exp2_component_ablation.sh`     | Component ablation experiments for residual-branch variants          |
| `scripts/run_overnight_unet_baseline.sh`     | Overnight baseline run for U-Net-related experiments                 |
| `scripts/run_stageA_shot_pilot_5_10.sh`      | Pilot-scale 5-shot and 10-shot sensitivity diagnostics               |
| `scripts/run_target20_segformer_baseline.sh` | 20-shot SegFormer baseline experiment                                |

Before running these scripts, please check and update local dataset paths, checkpoint paths, GPU IDs, and output directories.

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

Update local dataset paths and checkpoint paths in your configuration files or scripts before running experiments.

The `data/` directory in this repository is intended to store only lightweight split manifests and file lists, not raw images or masks.

## Checkpoints and Weights

This repository does not include:

* SAM3 official checkpoints;
* LoRA training checkpoints;
* HL-Residual training checkpoints;
* `.pth`, `.pt`, `.ckpt`, or `.safetensors` files;
* full prediction masks.

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
    