# Task 1 — EEG Data Processing, Visualization, and Classical Neural Networks

This repository contains my implementation for **Task 1**:

> **Objective:** Establish a high‑fidelity EEG data processing pipeline and demonstrate an understanding of EEG signal characteristics through **classical deep learning models**.

I reproduced **Table‑5 style** results on three MOABB datasets using **braindecode** models (**EEGNet** and **ShallowFBCSPNet**) and provided required visualizations (TensorBoard curves, t‑SNE, gradients, and a Figure‑3‑style interpretation figure).

---
 training stages


---

## Datasets (MOABB)

We evaluate on:

- **BNCI2015_001** (2 classes)
- **BNCI2014_001** (4 classes)
- **Lee2019_MI** (2 classes)

---

## Preprocessing (MOABB + MNE)

Pipeline (suggested + extra steps):

- Resample to **250 Hz**
- Bandpass/temporal filter: **4–36 Hz**
- Crop epochs to **≤ 3 seconds**
- *(extra)* Optional **notch** 
- *(extra)* **Average reference (CAR)**
- *(extra)* **Trial‑wise z‑score normalization** 

---

## Models (braindecode)

- **EEGNet**
- **ShallowFBCSPNet**

---

## Evaluation protocol

For each dataset, we run:

- **Inter‑subject**: GroupKFold over **subject**
- **Inter‑session**: GroupKFold over **subject_session**

Metrics (reported in results table + TensorBoard):

- **Balanced Accuracy (BalAcc)**
- **Macro‑F1**
- **MCC**
- **Cohen’s Kappa**

---

## Results

| Dataset | Setting | Classes | Model | BalAcc | MacroF1 | MCC | Kappa |
|---|---|---|---|---:|---:|---:|---:|
| BNCI2015_001 | inter-subject | feet, right_hand | EEGNet | 0.598 | 0.522 | 0.321 | 0.195 |
| BNCI2015_001 | inter-subject | feet, right_hand | ShallowFBCSPNet | 0.820 | 0.816 | 0.671 | 0.640 |
| BNCI2015_001 | inter-session | feet, right_hand | EEGNet | 0.520 | 0.380 | 0.128 | 0.040 |
| BNCI2015_001 | inter-session | feet, right_hand | ShallowFBCSPNet | 0.945 | 0.945 | 0.890 | 0.890 |
| BNCI2014_001 | inter-subject | feet, left_hand, right_hand, tongue | EEGNet | 0.248 | 0.109 | -0.010 | -0.002 |
| BNCI2014_001 | inter-subject | feet, left_hand, right_hand, tongue | ShallowFBCSPNet | 0.540 | 0.532 | 0.392 | 0.387 |
| BNCI2014_001 | inter-session | feet, left_hand, right_hand, tongue | EEGNet | 0.486 | 0.435 | 0.363 | 0.315 |
| BNCI2014_001 | inter-session | feet, left_hand, right_hand, tongue | ShallowFBCSPNet | 0.668 | 0.646 | 0.567 | 0.558 |
| Lee2019_MI | inter-subject | left_hand, right_hand | EEGNet | 0.440 | 0.385 | -0.150 | -0.120 |
| Lee2019_MI | inter-subject | left_hand, right_hand | ShallowFBCSPNet | 0.560 | 0.560 | 0.120 | 0.120 |
| Lee2019_MI | inter-session | left_hand, right_hand | EEGNet | 0.530 | 0.501 | 0.068 | 0.060 |
| Lee2019_MI | inter-session | left_hand, right_hand | ShallowFBCSPNet | 0.580 | 0.577 | 0.163 | 0.160 |


---

## TensorBoard

TensorBoard curves include **train/val/test loss** and **balanced performance metrics**.


Launch (Colab):
```bash
%load_ext tensorboard
%tensorboard --logdir /content/runs_task1
```

---

## Visualizations

### 1) t‑SNE (raw / hidden / logits)
- **Raw**: flattened EEG input (time downsampled)
- **Hidden**: penultimate (pre‑logits) representation
- **Logits**: network outputs

Saved to: `out_root/<dataset>/<model>/<mode>/tsne/`

### 2) Gradient diagnostics (early/mid/late)
Gradient norm snapshots at **early / middle / late** epochs.

Saved to: `out_root/<dataset>/<model>/<mode>/grad_*.png`




---


