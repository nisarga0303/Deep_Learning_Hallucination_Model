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

### Overview
The hybrid model addresses medical VQA by combining three powerful components:
**vision-language understanding**, **medical knowledge graph reasoning**, and 
**attention-based graph context fusion** — all working together to produce a final answer.

---

### 🧩 Components

#### 1. BiomedCLIP — Vision + Language Encoder (Frozen)
- Pretrained model: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- Vision encoder: **ViT-B/16** (Vision Transformer, 16×16 patch size, 224×224 input)
- Text encoder: **PubMedBERT** (trained on biomedical literature)
- Both encoders are **completely frozen** during training — only the projection layers learn
- Outputs normalized 512-dim image and text feature vectors
- Projected down to **256-dim** via learnable Linear + GELU + Dropout layers

#### 2. RotatE — Knowledge Graph Embedding
- Encodes medical entities (organs, modalities, diseases, intents) as complex-number vectors
- Each entity has a **real part** and an **imaginary part** (128-dim each)
- Relations are encoded as **rotation angles** in complex space
- Triple scoring formula: `score(h, r, t) = -‖ h ∘ r - t ‖²`  
  where `∘` = complex rotation
- Trained using **margin-based loss** on positive vs negative triples:
  `kg_loss = ReLU(1 - positive_score + negative_score)`
- Purpose: teaches the model the relationships between medical concepts

#### 3. QuestionGuidedGAT — Graph Attention Network
- Receives a **subgraph** built from the question's intent, predicted organ, and modality
- Node features come from RotatE entity embeddings (128-dim)
- The **query vector** = concatenation of image + text features (512-dim) — this guides attention
- Attention scores computed as:
  `score = softmax( (nodes + query) × nodes_T / √dim )`
- Masked attention: only connected nodes (per adjacency matrix) can attend to each other
- Output: single pooled 128-dim graph context vector

--


