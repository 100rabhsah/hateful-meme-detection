# ViBERT-X: Dual-Path Cross-Attentional Framework for Hateful Meme Detection

Multimodal hateful meme classification using **BERT word embeddings** and **ViT patch embeddings** with a novel **dual-path cross-modal attention** mechanism. Sentences are divided into **individual words** and images into **16×16 pixel patches**, then cross-attention is applied between them — each word independently attends to all 196 image patches.

> **Best Result:** F1 = **0.8365 ± 0.0056**, AUC-ROC = **0.9046 ± 0.0036** (5-fold CV, dual-path, fully unfrozen backbones)

---

## Key Novelty: Dual-Path Word-to-Patch Cross-Attention

Hateful memes work through **incongruity** — benign text paired with harmful imagery, or vice versa. Standard cross-attention captures only alignment (where modalities agree). Our dual-path architecture explicitly models **both agreement and conflict** between words and image patches:

```
Sentence → BERT → ["they", "taking", "over"]     (word embeddings, each 768-D)
                         ↓ Strip CLS, Project to 128-D
                   [S_w × 128] word queries
                         ↓
              ┌─────────────────────────────┐
              │      ALIGNMENT BRANCH       │  ← each word attends to all 196 patches
              │   (with residual + LayerNorm)│     captures word-patch AGREEMENT
              └──────────┬──────────────────┘
                         │
              ┌──────────┴──────────────────┐
              │       GATED FUSION          │  ← σ(g)·align + (1-σ(g))·incon
              └──────────┬──────────────────┘
                         │
              ┌──────────┴──────────────────┐
              │     INCONGRUITY BRANCH      │  ← separate Q projection, NO residual
              │  (decorrelated, no residual) │     captures word-patch CONFLICT
              └─────────────────────────────┘
                         ↓
Image → ViT → [patch₁, ..., patch₁₉₆]           (patch embeddings, each 768-D)
               (14×14 grid, 16×16 px each)
                         ↓ Strip CLS, Project to 128-D
                   [196 × 128] patch keys/values
```

### How Each Word Attends to Image Patches

For a meme with text *"they are taking over"* overlaid on an image:

| Word | Attends to Patches Containing | Learns |
|------|------------------------------|--------|
| **"they"** | Faces, demographic cues (skin tone, clothing) | *Who* "they" refers to |
| **"taking"** | Gestures (hands, body posture), objects | *What action* is depicted |
| **"over"** | Spatial context, symbols, backgrounds | *Where/what* is "taken over" |

Each word (128-D query) independently computes attention over **all 196 image patches** (128-D keys/values), producing a `[S_w × 196]` attention matrix with 8 heads. This is **bidirectional** — patches also attend back to words.

---

## Three Attention Modes

ViBERT-X supports three cross-modal fusion modes. All share the same BERT + ViT backbones, projection layers, and classifier — only the attention mechanism differs:

| Mode | CLI Flag | Granularity | Branches | Conflict Detection | Best F1 |
|------|----------|-------------|----------|-------------------|---------|
| **Dual-Path** | `--mode dual-path` | Per-word × per-patch | Alignment + Incongruity | ✅ Explicit (decorrelation loss) | **0.8365** |
| **Word-to-Patch** | `--mode word-patch` | Per-word × per-patch | Single | ❌ Implicit only | 0.8006 |
| **CLS Pooling** | `--mode cls` | Global CLS vectors | Single | ❌ None | 0.7842 |

### Mode Progression
- **CLS → Word-to-Patch**: +1.64% F1 (fine-grained word-patch alignment beats global fusion)
- **Word-to-Patch → Dual-Path**: +0.18% F1 (explicit incongruity branch captures text-image conflict)

---

## Complete Experiment Results

Nine experiments covering three attention modes, three freeze strategies, two data configurations, and 5-fold cross-validation:

| # | Experiment | Mode | Freeze | F1 | Acc | AUC-ROC |
|---|-----------|------|--------|-----|-----|---------|
| **9** | `v3_dual_aug_none_kf5` | dual-path | none | **0.8365±0.006** | **0.8365±0.006** | **0.9046±0.004** |
| 3 | `v3_dual_aug_none` | dual-path | none | 0.8289 | 0.8292 | 0.9055 |
| 1 | `v3_dual_aug_partial` | dual-path | partial | 0.8024 | 0.8024 | 0.8810 |
| 4 | `v3_wp_aug_partial` | word-patch | partial | 0.8006 | 0.8005 | 0.8826 |
| 5 | `v3_cls_aug_partial` | CLS | partial | 0.7842 | 0.7842 | 0.8644 |
| 2 | `v3_dual_aug_full` | dual-path | full | 0.7394 | 0.7397 | 0.8066 |
| 6 | `v3_dual_bal_partial` | dual-path | partial | 0.6880 | 0.6886 | 0.7594 |
| 7 | `v3_wp_bal_partial` | word-patch | partial | 0.6814 | 0.6820 | 0.7566 |

### Key Findings
1. **Dual-path > word-to-patch > CLS** across all comparable settings
2. **Fully unfrozen > partial > full freeze** (+2.65% and +8.95% F1 respectively)
3. **Augmented >> balanced data**: +11–12% F1 improvement across all architectures
4. **5-fold CV confirms robustness**: σ_F1 = 0.0056 across 5 folds

### 5-Fold Cross-Validation Detail (Experiment 9)

| Fold | F1 | AUC-ROC | Accuracy |
|------|----|---------|----------|
| 1 | 0.8304 | 0.9009 | 0.8305 |
| 2 | 0.8367 | 0.9067 | 0.8368 |
| 3 | 0.8459 | 0.9091 | 0.8459 |
| 4 | 0.8382 | 0.9062 | 0.8382 |
| 5 | 0.8313 | 0.8998 | 0.8314 |
| **Mean ± Std** | **0.8365 ± 0.0056** | **0.9046 ± 0.0036** | **0.8365 ± 0.0056** |

---

## Architecture Details

### Dataflow and Tensor Shapes

| Stage | Words | Patches |
|-------|-------|---------|
| Backbone output | `[B, S_T, 768]` | `[B, 197, 768]` |
| After CLS stripping | `[B, S_w, 768]` | `[B, 196, 768]` |
| After 768→128 projection | `[B, S_w, 128]` | `[B, 196, 128]` |
| Cross-attention scope | S_w × 196 pairs, 8 heads, bidirectional | |
| After pooling | `[B, 128]` | `[B, 128]` |
| Concatenated | `[B, 256]` | |
| Classifier output | `[B, 2]` | |

### Design Choices

| Component | Value | Rationale |
|-----------|-------|-----------|
| Attention heads | 8 | Per-head dim: 128/8 = 16 |
| Cross-attention layers | 1 block | Bidirectional (words→patches + patches→words) |
| CLS tokens | Stripped from both BERT and ViT | Attention operates purely on words and patches |
| Residual connections | Alignment branch only | Incongruity branch omits residuals by design |
| Layer normalization | Both branches | After attention |
| Decorrelation loss | Barlow-Twins (λ=0.5) | Enforces branch orthogonality |
| Gated fusion | Learned sigmoid gate | Dynamic per-sample branch weighting |

### Backbone Freeze Strategies

| Strategy | Trainable Params | Test F1 | Description |
|----------|-----------------|---------|-------------|
| **None (fully unfrozen)** | ~196M | **0.8289** | All BERT + ViT params trainable |
| Partial (top-2 layers) | ~29.6M | 0.8024 | Top 2 transformer layers unfrozen |
| Full (heads only) | ~676K | 0.7394 | Only projection + attention heads trainable |

---

## Data Augmentation

Targets the **minority hateful class only** (3,019 → 5,481 samples) to match the non-hateful count:

### Image Augmentation (`torchvision.transforms`)
- Random resized crop (scale 0.8–1.0)
- Random horizontal flip (p=0.5)
- Color jitter: brightness/contrast ±0.2, saturation ±0.1
- Random rotation (±15°)

### Text Augmentation
- **Synonym replacement**: WordNet, 1–2 non-protected words/sentence (p=0.3)
- **Back-translation**: English → German → English via MarianMT

### Label Safety Safeguards
- Slurs, identity terms, and negation markers excluded via protected-word list
- Back-translated samples audited on 10% subset; altered samples discarded

### Impact (Ablation)
| Architecture | Balanced F1 | Augmented F1 | Δ F1 |
|-------------|------------|-------------|------|
| Dual-Path | 0.6880 | 0.8024 | **+0.1144** |
| Word-to-Patch | 0.6814 | 0.8006 | **+0.1192** |

---

## Training Protocol

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| BERT variant | `bert-base-uncased` (110M params, HuggingFace) |
| ViT variant | `google/vit-base-patch16-224` (86M params, HuggingFace) |
| Image resolution | 224 × 224 |
| Patch size | 16 × 16 (196 patches/image) |
| Tokenizer | WordPiece (BERT default) |
| Max text length | 128 tokens |
| Optimizer | AdamW (weight decay = 0.01) |
| Learning rate | 2 × 10⁻⁵ |
| LR scheduler | Cosine annealing + 1-epoch linear warmup (η_min = 10⁻⁷) |
| Batch size | 64 (train), configurable (val/test) |
| Epochs | 10 (early stopping, patience = 3) |
| Label smoothing | 0.1 |
| Dropout | 0.3 |
| Projection dim | 768 → 128 (per-word/per-patch) |
| Classifier | 256 → 64 → 2 |
| Decorrelation λ | 0.5 |

### Evaluation Protocol
- **Split**: 70% train / 10% val / 20% test (stratified)
- **Model selection**: Based on validation F1 only; test set used only for final evaluation
- **Cross-validation**: 5-fold stratified CV; augmented data derived only from training partition of each fold

---

## Quick Start

### Local Training

```bash
pip install -r requirements.txt

# Best configuration: Dual-path + augmented + fully unfrozen + 5-fold CV
python train.py --mode dual-path --augmented --freeze none --kfold 5 --batch-size 32 --lr 1e-5

# Dual-path with partial freeze (default)
python train.py --mode dual-path --augmented

# Word-to-patch baseline
python train.py --mode word-patch --augmented

# CLS-only baseline
python train.py --mode cls --augmented

# Fully unfrozen (single split)
python train.py --mode dual-path --augmented --freeze none

# Custom hyperparameters
python train.py --mode dual-path --lr 3e-5 --epochs 10 --dropout 0.4

# Evaluate a saved checkpoint
python train.py --eval-only checkpoints/best_model.pth
```

### Google Colab (Pro) / Kaggle

Use `run_experiments.py` — a unified script that auto-detects the environment:

**Colab:**
```python
# Cell 1: Clone the repo
!git clone https://github.com/100rabhsah/hateful-meme-detection.git
%cd hateful-meme-detection

# Cell 2: Run (change EXPERIMENT_ID inside the file first)
%run run_experiments.py
```

**Kaggle:**
1. Add dataset + repo as a Kaggle dataset
2. Copy `run_experiments.py` into a notebook cell
3. Change `EXPERIMENT_ID` and run

#### Experiment Presets (`run_experiments.py`)

| ID | Mode | Data | Freeze | K-Fold | Description |
|----|------|------|--------|--------|-------------|
| 1 | dual-path | augmented | partial | — | Dual-path, partial freeze |
| 2 | dual-path | augmented | full | — | Dual-path, full freeze (heads only) |
| 3 | dual-path | augmented | none | — | Dual-path, fully unfrozen |
| 4 | word-patch | augmented | partial | — | Word-to-patch baseline |
| 5 | cls | augmented | partial | — | CLS-only baseline |
| 6 | dual-path | balanced | partial | — | Dual-path on balanced data |
| 7 | dual-path | augmented | partial | 5 | Dual-path + 5-fold CV |
| 8 | word-patch | balanced | partial | — | Word-to-patch on balanced data |
| **9** | **dual-path** | **augmented** | **none** | **5** | **Best: fully unfrozen + 5-fold CV** |

Change `EXPERIMENT_ID = 9` at the top of the file and run.

**Colab-specific features:**
- Auto-mounts Google Drive (checkpoints persist across sessions)
- Auto-installs all dependencies
- Auto-detects project & data paths
- Prints GPU name and memory at startup

---

## CLI Reference

```
python train.py [-h]
    --mode {word-patch,cls,dual-path}   Attention mode (default: word-patch)
    --augmented                         Use augmented dataset
    --kfold N                           K-fold CV folds, 0=disabled (default: 0)
    --freeze {none,full,partial}        Backbone freeze strategy (default: partial)
    --unfreeze-layers N                 Top-N layers to unfreeze for partial (default: 2)

    --lr FLOAT                          Learning rate (default: 2e-5)
    --epochs N                          Number of epochs (default: 5)
    --batch-size N                      Train batch size (default: 32)
    --val-batch-size N                  Val batch size (default: 8)
    --test-batch-size N                 Test batch size (default: 16)
    --dropout FLOAT                     Dropout rate (default: 0.3)
    --embed-dim N                       Projection dim (default: 128)
    --attn-heads N                      Attention heads (default: 8)
    --weight-decay FLOAT                Weight decay (default: 0.01)
    --seed N                            Random seed (default: 42)

    --bert MODEL                        BERT model name (default: bert-base-uncased)
    --vit MODEL                         ViT model name (default: google/vit-base-patch16-224)

    --incon-lambda FLOAT                Incongruity λ init (default: 0.5)
    --incon-loss-weight FLOAT           Decorrelation loss weight (default: 0.1)

    --no-scheduler                      Disable cosine LR scheduler
    --warmup-epochs N                   Warmup epochs before cosine decay (default: 1)

    --class-weights                     Use inverse-frequency class weights
    --run-augmentation                  Run offline augmentation pipeline first
    --name NAME                         Custom experiment name
    --eval-only CHECKPOINT              Skip training; evaluate checkpoint only
```

---

## Project Structure

```
hateful_meme_detection/
├── src/
│   ├── config.py                    # Hyperparameters + environment detection
│   ├── data/
│   │   ├── dataset.py               # HatefulMemesDataset (PyTorch Dataset)
│   │   ├── loader.py                # JSONL/CSV loading, splits, k-fold loaders
│   │   └── augmentation.py          # Offline augmentation pipeline
│   ├── models/
│   │   ├── cross_attention.py       # CrossModalAttention + DualPathCrossAttention
│   │   └── classifier.py            # Main model (word-patch / CLS / dual-path modes)
│   ├── engine/
│   │   ├── metrics.py               # MetricsTracker with AUC-ROC, confusion matrix
│   │   ├── trainer.py               # Training loop with decorrelation loss + fold support
│   │   └── evaluator.py             # Test evaluation + fold-aware checkpoint loading
│   └── utils/
│       └── visualization.py         # Training curves, bar charts, confusion matrix
├── train.py                         # Local CLI entry point
├── run_experiments.py               # Unified Colab / Kaggle / Local experiment runner
├── subgroup_analysis.py             # Seen vs unseen subgroup evaluation
├── requirements.txt
├── paper_revision/
│   ├── revised_sections.tex         # LaTeX snippets for revised manuscript
│   └── reviewer_response.tex        # Point-by-point reviewer response document
├── SUPERVISOR_RESPONSE.tex          # Detailed response to supervisor comments
├── JSON Files/                      # JSONL data (train, dev_seen, dev_unseen, etc.)
├── CSV Files/                       # Augmented CSV data
├── img/                             # Meme images (10,000+)
├── checkpoints/                     # Saved model checkpoints
├── results/                         # Experiment outputs and plots
├── outputs/                         # Training logs and metrics
└── Notebooks/                       # Original Jupyter notebooks (preserved)
```

---

## Dataset

Based on the [Facebook Hateful Memes Challenge](https://ai.facebook.com/tools/hatefulmemes/):
- **~10,000 memes** with binary labels (hateful / not-hateful)
- Includes benign confounders to prevent unimodal shortcuts
- **Balanced**: 8,500 samples (3,019 hateful, 5,481 non-hateful)
- **Augmented**: ~11,000 samples (hateful class upsampled to match non-hateful)

---

## Commit History (Recent)

| Commit | Description |
|--------|-------------|
| `642e39b` | Experiment 9: dual-path + augmented + fully unfrozen + 5-fold CV |
| `261ca62` | Major changes: v3 architecture overhaul — word-to-patch mode renamed, cosine LR, 3 freeze strategies, configurable BERT/ViT, expanded CLI, 8 experiment presets |
| `7c9e9af` | Fix: correct IMAGE_DIR double-nesting (img/img/) |
| `147bec8` | Fix: add GDrive→SSD data sync for Colab in subgroup_analysis |
| `ebf0e76` | Feat: subgroup analysis script for seen vs unseen evaluation |
| `460c2e8` | Feat: partial unfreeze — keep top 2 layers of BERT + ViT trainable |
| `22ec8a3` | Feat: v2 overhaul — freeze backbones, label smoothing, tuned hyperparams |
| `3390011` | Feat: early stopping with patience=3 |
| `d43d75e` | Fix: remove residual from incongruity branch + Barlow-Twins loss |
| `3faec9e` | Feat: dual-path cross-attention (alignment + incongruity) and k-fold CV |

---

## Paper Revision

The `paper_revision/` directory contains LaTeX documents for the manuscript revision:

- **`revised_sections.tex`**: Complete replacement snippets for the revised paper, including:
  - Section 4.4: Three Attention Modes (CLS, Word-to-Patch, Dual-Path)
  - Section 4.5: Dual-Path Cross-Attention Formal Specification
  - Section 4.6: Training Protocol with Algorithm 1
  - Section 5: Design Rationale and Sensitivity Analysis
  - Section 6.6: Comprehensive 9-experiment Ablation Study
  - Revised contributions, hyperparameters, evaluation protocol, and conclusion

- **`reviewer_response.tex`**: Point-by-point responses to all three reviewers (R1: 7 comments, R2: 8 comments, R3: 2 comments)

---

## Citation

```bibtex
@article{vibert-x,
  title={ViBERT-X: Dual-Path Cross-Attentional Framework for Hateful Meme Detection},
  author={Kedia, Upkar Kumar and Pisharody, Uddhav and Kumari, Kirti and Shah, Md. Fahad},
  journal={ICV},
  year={2026}
}
```
