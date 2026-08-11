# DermaFusion: Multimodal Skin Lesion Classification via Cross-Modal Attention

DermaFusion is a multimodal deep learning system for skin lesion classification that fuses dermoscopic images with structured clinical patient data using cross-modal attention, rather than simple feature concatenation.

## Overview

Skin lesion diagnosis in practice draws on two complementary sources of information: what a lesion *looks like* (dermoscopic imaging) and *who the patient is* (age, lesion history, anatomical site, family history, and other clinical metadata). Most prior multimodal approaches fuse these two modalities by concatenating their feature vectors. DermaFusion instead uses a cross-modal attention mechanism, letting the image features attend over clinical features so the model can learn *which* clinical attributes are relevant to *which* visual patterns, rather than treating both feature sets as a flat, undifferentiated vector.

## Dataset

- **PAD-UFES-20** — a dermoscopic image dataset paired with structured clinical metadata (33 clinical features per sample after preprocessing).
- **Classes (6):** ACK, BCC, MEL, NEV, SCC, SEK
- The dataset is not included in this repository. Download it separately and update the data paths in the notebook to point to your local/Kaggle copy.

## Architecture: AttentionFusionResNet

| Component | Details |
|---|---|
| Image backbone | ResNet50 (ImageNet-pretrained), only `layer4` fine-tuned |
| Clinical encoder | 3-layer MLP, 33-dim clinical input |
| Fusion | Cross-modal attention, 256-dim, 8-head multi-head attention |
| Classifier head | 768 → 256 → 6 |

Two fusion strategies were compared:
- **Concatenation fusion** — image and clinical features simply concatenated before the classifier
- **Cross-modal attention fusion (proposed)** — image features attend over clinical features before classification

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Concatenation Fusion | ~75% | 71.56% |
| Attention Fusion | ~71.88% | 70.64% |
| **ResNet50 + Test-Time Augmentation (best)** | **80.29%** | **74.03%** |

Attention fusion achieved a higher validation F1 (0.7512 vs. 0.7456) than concatenation fusion despite a lower raw test accuracy, and applying test-time augmentation (TTA) gave the largest overall boost in both accuracy and macro F1.

*Note: exact figures can vary slightly between training runs due to random initialization and data shuffling; see the notebook's printed output for the run-specific numbers.*

## Baseline comparison

The primary published baseline for this task is the MiSC framework (Mohamed et al., 2025), which reports 95.7% weighted F1 using concatenation-based fusion on a related setup. DermaFusion's cross-modal attention fusion is presented as the paper's main contribution, evaluated against that baseline and against an internal concatenation-fusion ablation.

## Repository Structure

```
.
├── dermafusion-development.ipynb   # Main notebook: data prep, all model variants, training, evaluation
├── README.md
├── requirements.txt
└── images/                         # Loss curve and confusion matrix plots
    ├── loss_curve_v2.png
    └── confusion_matrix_v2_tta.png
```

## How to Run

This notebook was developed and run on Kaggle (GPU-accelerated environment) against the PAD-UFES-20 dataset mounted at `/kaggle/input/...`. To run it elsewhere:

1. Install dependencies: `pip install -r requirements.txt`
2. Download PAD-UFES-20 and update the dataset path variables near the top of the notebook
3. Run cells sequentially — the notebook is organized in phases:
   - **Phase 1–2:** EDA, stratified train/val/test split
   - **Phase 3:** Clinical-only baseline (MLP)
   - **Phase 4:** Concatenation fusion baseline
   - **Phase 5:** Attention fusion (proposed model) + test-time augmentation
   - **Retrain (R1–R4):** Full-dataset retraining of all four model variants for the final reported results

Training is GPU-intensive; running the full notebook end-to-end (all model variants plus TTA and ensembling) can take several hours depending on hardware.

## Status

This is an active research project. Current work includes an IEEE-formatted conference paper and a React-based web app demo for interactive inference.
