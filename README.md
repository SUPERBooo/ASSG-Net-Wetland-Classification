# ASSG-Net: Adaptive Scale and Sparse Graph Network via Gated Fusion for Wetland Vegetation Classification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18894512.svg)](https://doi.org/10.5281/zenodo.18894512)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-red.svg)

Official PyTorch implementation of **"ASSG-Net: Adaptive Scale and Sparse Graph Network via Gated Fusion for Wetland Vegetation Classification"**.

## 📖 Overview

Accurate classification of fine-scale wetland vegetation remains challenging because of strong spatial heterogeneity, fragmented vegetation patches, graph over-smoothing across ecological boundaries, and varying reliability of multi-source remote sensing observations.

ASSG-Net is a lightweight dual-branch framework for joint learning from Sentinel-1 SAR and Sentinel-2 multispectral imagery. It contains three main components:

1. **Physically-Aware Scale Perception Module (ASPM)**  
   ASPM introduces Sobel-edge and local-variance information as explicit spatial heterogeneity priors. Two lightweight depthwise convolution branches with different receptive fields are adaptively selected through pixel-wise gating, helping preserve fine-scale spectral-textural characteristics of fragmented wetland vegetation.

2. **Adaptive Graph Sparsity Module (AGSM)**  
   Sentinel-1 SAR pixels are first aggregated into SNIC superpixels, which are used as graph nodes. A mutual k-nearest-neighbor graph is constructed from superpixel centroids. AGSM then applies learnable edge gating, sparsity regularization, and DropEdge to suppress low-relevance graph connections and reduce over-smoothing across ecological boundaries.

3. **Gated Fusion Module (GFM)**  
   GFM integrates SAR-derived graph features and MSI-derived CNN features through global channel gating and local spatial gating. The adaptive fusion mechanism balances complementary spectral and structural information according to their spatially varying discriminative contributions.

ASSG-Net contains approximately **0.142 M trainable parameters** and provides a lightweight solution with competitive wetland vegetation classification performance.

## 📁 Repository Structure

```text
├── configs.py              # Hyperparameters, paths, and experiment settings
├── dataloader.py           # Dataset loading, normalization, augmentation, and class weighting
├── model.py                # ASSG-Net: ASPM, AGSM, GFM, and classifier
├── snic.py                 # SNIC superpixel generation
├── preprocess.py           # Data preprocessing and superpixel preparation
├── train.py                # Model training and evaluation
├── predict.py              # Model inference and classification-map generation
├── analyze_gates.py        # Analysis of ASPM, AGSM, and GFM gating behavior
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── Data.zip                # Example / accompanying dataset files
