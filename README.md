# 🧠 Medical VQA — Hallucination-Aware Visual Question Answering

A deep learning system for Medical Visual Question Answering (VQA) using the **SLAKE dataset**.
It combines a traditional CNN baseline with a hybrid model that fuses **BiomedCLIP**, 
**Knowledge Graph Embeddings (RotatE)**, and a **Graph Attention Network (GAT)** 
to answer clinical questions about medical images.

---

## 📌 Project Overview

Given a medical image and a natural language question (e.g., *"What organ is shown?"*, 
*"Are there abnormalities?"*), the system predicts the correct clinical answer.

Two models are trained and compared:

| Model | Architecture | Task |
| **Traditional** | ConvNeXt-Tiny | Organ + Modality classification |
| **Hybrid** | BiomedCLIP + RotatE + GAT | Full VQA (Answer + Organ + Modality) |

---

## 📊 Results

| Model | Answer Acc | Organ Acc | Modality Acc |
| Traditional (ConvNeXt) | — | **97.88%** | **100%** |
| Hybrid (BiomedCLIP+GAT) | **40.04%** | **94.95%** | **100%** |

---

## 🗂️ Dataset — SLAKE

- **642 unique medical images** (MRI, CT, X-Ray)
- **7,033 English Q&A pairs**
- Each image folder (`xmlab0` to `xmlab641`) contains:
  - `source.jpg` — the medical image
  - `question.json` — multiple Q&A pairs
- **Split:** 80% Train (~514 images, ~5,627 QA) / 20% Test (~128 images, ~1,406 QA)
- Grouped by image to prevent data leakage

> Dataset is not included in this repo due to size. Download SLAKE from [https://www.med-vqa.com/slake/](https://www.med-vqa.com/slake/) and place in `data/imgs/`

---

## 🏗️ Architecture

### Model 1 — ConvNeXt-Tiny (Traditional Baseline)
- Pretrained ConvNeXt-Tiny backbone (ImageNet)
- Dual classification heads: organ + modality
- Trained for **6 epochs**, AdamW optimizer (`lr=2e-4`)
- Weighted CrossEntropyLoss to handle class imbalance

### Model 2 — BiomedCLIP Hybrid (Main Model)
