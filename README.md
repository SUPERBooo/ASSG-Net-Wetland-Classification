# ASSG-Net: Adaptive Scale and Sparse Graph Network via Gated Fusion for Wetland Vegetation Classification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch 1.13.1](https://img.shields.io/badge/PyTorch-1.13.1-EE4C2C.svg)](https://pytorch.org/)

[cite_start]Official PyTorch implementation of the paper **"ASSG-Net: Adaptive Scale and Sparse Graph Network via Gated Fusion for Wetland Vegetation Classification"**[cite: 1, 2, 14].

## 📖 Overview

[cite_start]Accurate classification of fine-scale wetland vegetation is fundamentally constrained by the mismatch between rigid model structures and the complex spatial heterogeneity of coastal landscapes[cite: 11]. [cite_start]**ASSG-Net** resolves these challenges through three synergistic innovations[cite: 40]:
1. [cite_start]**Physically-Aware Scale Perception Module (ASPM):** Integrates heterogeneity priors to adaptively switch receptive fields, preserving fine-grained features of fragmented vegetation (e.g., *Suaeda salsa*)[cite: 41, 42].
2. [cite_start]**Topology-Reconstruction Graph Module (AGSM):** Employs a learnable sparsity regularization to dynamically prune low-relevance connections, effectively preventing over-smoothing across ecological boundaries[cite: 43, 44].
3. [cite_start]**Gated Fusion Module (GFM):** Performs adaptive cross-modal integration via dual-level gating to optimally weigh Sentinel-1 SAR and Sentinel-2 optical features[cite: 45, 46].

[cite_start]With only **0.15 M parameters**, ASSG-Net achieves state-of-the-art accuracy while maintaining high computational efficiency[cite: 20].

## 📂 Repository Structure

```text
├── configs.py               # Hyperparameters and path configurations
├── dataloader.py            # PyTorch dataset for multi-source remote sensing data
├── model.py                 # Core architecture (ASPM, LightAGSM, GatedFusion)
├── preprocess.py            # Data preprocessing & Bayesian Optimization for SNIC
├── train.py                 # Training script with Focal/Tversky loss and EMA
├── predict.py               # Inference script using sliding window strategy
├── plot_bo_convergence.py   # Visualization for Bayesian Optimization trace
├── requirements.txt         # Environment dependencies
└── data/                    # Folder for sample Sentinel-1/2 images and ground truth
