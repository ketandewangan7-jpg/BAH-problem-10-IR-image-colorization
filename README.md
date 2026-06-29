# Deep-IR: Multi-Task Physics-Constrained Sat-DLSS Framework

Official proposal repository for the **Bharatiya Antariksh Hackathon (BAH) 2026** organized by **ISRO** and managed by **Hack2Skill**.

---

# 🚀 Problem Statement

## Infrared image colorization and enhancement for improved object interpretation

**Domain:** Geospatial Data Analytics, Remote Sensing, Image-to-Image Translation

**Target Platforms:** Landsat 8/9 Level-2 Science Products (TIRS Band 10 & Visible RGB)

---

# 👥 Team Details

**Team Name:** ORBITALLY INCLINED

**Members:**
1.VISHVESH MISHRA(group leader)
2.KETAN DEWANGAN
3.NITIN BHANDARI
4.KANISHKA TYAGI
---

# 💡 Core Innovation & Solution Architecture

Most standard image colorizers (like CycleGAN) guess textures blindly, leading to spatial hallucinations that destroy tactical ground-truth data. **Deep-IR** introduces a **Multi-Task Neural Network with a Shared Backbone** that optimizes structural sharpness and physical color consistency simultaneously.

## Technical Workflow

1. **Data Ingestion & Scaling**

   Ingest 16-bit GeoTIFF thermal matrices.

   Apply a 2% Cumulative Linear Contrast Stretch to normalize the thermal image into the range **[-1,1]**.

2. **SwinIR Backbone**

   Uses Shifted Window Transformer blocks to reconstruct structural information from thermal imagery.

3. **Dual Prediction Heads**

   - **Head 1 (Physical Anchor Head)** reconstructs high-resolution infrared imagery to preserve thermal physics.
   - **Head 2 (Painter Head)** predicts a three-channel RGB approximation from the learned structural features.

---

# Architecture

```text
               [ Raw 16-Bit Landsat IR ]
                          |
            [ 2% Contrast Stretch Scaling ]
                          |
             [ SwinIR Transformer Backbone ]
                          |
             +-------------+-------------+
             |                           |
     [ Head 1: IR Anchor ]      [ Head 2: RGB Painter ]
      (Optimizes L1 Physics)     (Translates to Color)
```

---

# Composite Loss Function

The proposed model is trained using a composite loss consisting of:

- Pixel-wise **L1 Loss** for infrared reconstruction.
- **Cross Entropy Loss** for semantic consistency.

This combination preserves physical infrared characteristics while maintaining semantic correctness.

---

# Repository Structure

```text
Deep-IR/
│
├── README.md
├── requirements.txt
│
├── models/
│   ├── backbone.py
│   └── dual_heads.py
│
├── losses/
│   └── composite_loss.py
│
├── utils/
│   └── preprocessing.py
│
└── datasets/
```
