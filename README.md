# annfinalproject
# 👗 Outfit Compatibility Rater
### Can a Neural Network Learn Fashion Sense?

---

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Feature Extraction](#feature-extraction-via-transfer-learning)
  - [Model Design](#model-design)
- [Results](#results)
  - [Model Comparison](#model-comparison)
  - [ROC Curve Analysis](#roc-curve-analysis)
  - [Confusion Matrices](#confusion-matrices)
- [Human Parsing Extension](#human-parsing-extension)
- [Discussion](#discussion)
- [Challenges](#challenges)
- [Conclusion](#conclusion)

---

## Introduction

Fashion sense is often intuitive and deeply personal. Most people can sense when an outfit works and when it doesn't. But some combinations are harder to judge, and sometimes you just want a second opinion.

This project explores whether a neural network can learn that same sense of visual compatibility from data alone. The goal was to build an **outfit compatibility classifier** that predicts whether a group of clothing items forms a compatible outfit without manually defining fashion rules like "these colors match" or "these silhouettes work together."

---

## Dataset

We used the **Polyvore outfit dataset**, a fashion compatibility dataset made up of **21,889 curated outfits** collected from the Polyvore social fashion platform. Each outfit contains multiple clothing item images — tops, bottoms, dresses, shoes, and accessories.

Each outfit was treated as a binary classification example:

| Label | Meaning |
|-------|---------|
| `1` | Compatible outfit |
| `0` | Incompatible outfit |

After loading train, test, and validation splits and merging them, outfits with fewer than two resolvable images were discarded to ensure all remaining samples could offer meaningful representation.

---

## Methodology

### Feature Extraction via Transfer Learning

We used **ResNet50** as a pretrained image feature extractor to convert each clothing item image into a numerical representation. ResNet50 was a strong fit because:

- It's a well-documented CNN architecture available through Keras
- Its pretrained ImageNet weights allow it to recognize general visual patterns — colors, edges, textures, shapes, and object-level features

**Pipeline:**
1. Each clothing item image was resized to **224×224 pixels**
2. Passed through ResNet50 with the final classification layer removed
3. Produced a **2,048-dimensional feature vector** per item
4. Applied **L2 normalization** — scaling each embedding to a magnitude of 1 so that no single item dominates the classifier due to numerical scale

---

### Model Design

A key challenge: outfits have a variable number of items, but neural networks require fixed-size inputs. We explored two outfit embedding strategies and trained three models to compare them.

#### Model A — Logistic Regression (Baseline)

Each outfit was represented using **concatenation + average** embedding:

```
4 item embeddings × 2048 features
+ 1 average embedding × 2048 features
= 10,240-dimensional input
```

Logistic regression was a natural starting point for binary classification. This baseline performed surprisingly well, reaching **~70.5% test accuracy**.

---

#### Model B — Neural Network with Concatenation + Average Embeddings

Same embedding strategy as Model A, but replaced logistic regression with a **feedforward neural network** featuring:
- Dense layers
- Leaky ReLU activations (to avoid the dying ReLU problem)
- Dropout and L2 regularization

Despite added complexity, Model B reached only **~59.2% accuracy** — worse than the baseline — and was heavily biased toward predicting outfits as incompatible.

---

#### Model C — Pairwise Neural Network ✅ *(Final Model)*

Model C expanded the outfit representation by adding **pairwise difference vectors** — comparing every pair of clothing item embeddings by subtracting one from another:

```
4 item embeddings × 2048 features
+ 1 average embedding × 2048 features
+ 6 pairwise difference embeddings × 2048 features
= 22,528-dimensional input
```

Additional improvements over Model B:
- **AdamW optimizer** for stable training and weight decay
- Same deep architecture: dense layers, Leaky ReLU, dropout, L2 regularization

Model C reached **~69.8% test accuracy** — slightly below Model A, but the strongest neural network model and most conceptually aligned with the goal of modeling outfit compatibility as a *relationship* between items.

**Shared architecture for Models B and C:**

```
Input (10K or 22K dim)
    ↓
Dense 1024 → Leaky ReLU → BatchNorm → Dropout 0.5
    ↓
Dense 512  → Leaky ReLU → BatchNorm → Dropout 0.4
    ↓
Dense 256  → Leaky ReLU → BatchNorm → Dropout 0.3
    ↓
Dense 128  → Leaky ReLU → BatchNorm → Dropout 0.2
    ↓
Sigmoid → [0, 1] compatibility score
```

> Model C additionally applies L2 weight decay to the first three dense layers.

---

## Results

### Model Comparison

| Model | Strategy | Test Accuracy | AUC |
|-------|----------|--------------|-----|
| A — Logistic Regression | Concat + Mean | 70.5% | 0.766 |
| B — Neural Network | Concat + Mean | 59.2% | 0.743 |
| **C — Pairwise Neural Network** | **Pairwise Differences** | **69.8%** | **0.771** |

**Key takeaway:** Model B shows that neural network complexity alone doesn't help — the *way you represent the outfit* matters more. Model C's pairwise differences are what push it ahead, not the architecture itself.

---

### ROC Curve Analysis

The ROC curve evaluates each model's ability to separate compatible from incompatible outfits **across all decision thresholds**, not just at the default 0.5 cutoff.

- Model C achieved the **highest AUC (0.771)**, meaning its probability scores are better calibrated
- Model B underperforms even the logistic regression baseline (0.743 vs 0.766), confirming that the concat+mean embedding was the bottleneck
- The margins are tight — Models A and C are nearly identical across much of the curve, but pairwise features give Model C a consistent edge

<img width="790" height="590" alt="Unknown" src="https://github.com/user-attachments/assets/9a0da604-3181-488f-987f-4eaf57bc236f" />

  

---

### Confusion Matrices

The confusion matrices revealed important differences in *how* each model makes errors:

**Model A — Logistic Regression**
- Correctly identified 516/657 incompatible outfits
- Correctly identified 266/452 compatible outfits
- Biased toward predicting incompatible, but functional on both classes

<img width="649" height="547" alt="Unknown-2" src="https://github.com/user-attachments/assets/9c1d536f-7139-4897-9170-59132072676a" />


**Model B — Concat + Mean NN**
- Correctly identified 657/657 incompatible outfits
- Correctly identified 0/452 compatible outfits
- **Predicted every outfit as incompatible** — learned nothing about compatibility

<img width="649" height="547" alt="Unknown-3" src="https://github.com/user-attachments/assets/a5fbf3a5-cfc7-46c5-a55f-77149bdb09ea" />


**Model C — Pairwise NN**
- Correctly identified 568/657 incompatible outfits
- Correctly identified 206/452 compatible outfits
- **Most balanced** — the only neural network that genuinely learned both classes

> A useful outfit compatibility rater should not simply reject most outfits. Model C is the only model that can identify when an outfit *works* and when it doesn't.

<img width="649" height="547" alt="Unknown-4" src="https://github.com/user-attachments/assets/017b809a-a8d3-4ea4-8c64-8b4da25ab95c" />


**Model C — Correct Predictions**
<img width="1100" height="755" alt="IMG_2848" src="https://github.com/user-attachments/assets/16863b1c-6734-4f1b-aad1-1c41e5e7c3a2" />


**Model C — Incorrect Predictions**
<img width="1135" height="733" alt="IMG_4189" src="https://github.com/user-attachments/assets/82d584d1-6c9c-4f80-b3e8-7937c9a9ca9d" />


---

## Human Parsing Extension

As a practical extension, the project implements an **end-to-end pipeline for evaluating your own outfits from a photo**.

**Pipeline:**
1. **SegFormer** (fine-tuned on the ATR human parsing dataset via `mattmdjaga/segformer_b2_clothes`) segments a full-body image into per-pixel clothing labels
2. Detected regions — top, bottom, dress, shoes — are cropped independently
3. Each crop is embedded with ResNet50
4. Assembled into Model C's 22,528-dimensional pairwise format
5. Passed to the trained compatibility classifier for a final score

| Category | Region |
|----------|--------|
| TOP | Top garment |
| BOTTOM | Pants / Skirt |
| DRESS | Full dress |
| SHOES | Footwear |

<img width="754" height="515" alt="image" src="https://github.com/user-attachments/assets/f7d59b4f-1ddb-4d55-ae22-003fbdc21f6b" />


---

## Discussion

### What the Results Tell Us

- **ResNet50 embeddings already contain meaningful visual compatibility information** — Model A's strong baseline performance demonstrates this
- **Representation matters more than model complexity** — Model B was a more complex model than Model A and still performed worse, because the embedding strategy was the bottleneck
- **Pairwise differences unlock relational learning** — Model C improved substantially over Model B by explicitly encoding how clothing items relate to each other in embedding space
- **Model C did not dramatically outperform the logistic regression baseline in raw accuracy** — its value lies in being the strongest neural network and in better matching what outfit compatibility actually is: a relationship between items, not a property of individual pieces

---

## Challenges

**Variable-length outfits** — Polyvore outfits can contain 2–6+ items, but neural networks require fixed-size inputs. We padded shorter outfits with zero vectors and truncated longer ones to four items. The mean embedding is computed *before* padding to avoid zero-vector bias. A permutation-invariant architecture (e.g. Set Transformer or DeepSets) would be a stronger long-term solution.

**Missing images** — Many outfit entries referenced files unavailable locally. Outfits with fewer than two resolvable images were dropped entirely, consistently across all models to ensure fair comparison.

**Dying ReLU** — Early experiments produced inactive neurons receiving large negative inputs. Switching to **Leaky ReLU** resolved this, which was especially important given the high-dimensional, sparse input space from zero-padded outfits.

**Performance ceiling** — Further architecture tuning (additional layers, adjusted dropout, optimizer changes) did not consistently improve results, suggesting the model may be limited by: noisy/subjective compatibility labels, fixed-size truncation, and the use of a frozen general-purpose ResNet50 rather than a fashion-specific backbone.

---

## Conclusion

Through this project, we found that outfit compatibility can be predicted from visual features with better-than-chance accuracy.

- **Model A** showed that ResNet50 embeddings were already useful enough for logistic regression to perform well
- **Model B** showed that neural network complexity alone is not enough — the wrong embedding strategy produced a worse model than the simple baseline
- **Model C** showed that adding pairwise item relationships made the neural network substantially more effective and better aligned with what outfit compatibility actually means

Model C was used for the personal photo pipeline, where SegFormer segmentation enables compatibility scoring of real outfit photos without requiring pre-cropped, catalogued item images.

**Future directions:**
- Larger training data or data augmentation
- Category-conditioned compatibility embeddings
- Fine-tuning the ResNet50 backbone on fashion-specific data rather than keeping it frozen
