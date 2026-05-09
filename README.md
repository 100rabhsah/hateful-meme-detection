# ViBERT-X: Dual-Path Cross-Attentional Framework for Hateful Meme Detection

Multimodal hateful meme classification using **BERT token embeddings** and **ViT patch tokens** with a novel **dual-path cross-modal attention** mechanism that explicitly models both semantic alignment and incongruity between modalities.

## Key Novelty: Dual-Path Cross-Attention

Hateful memes often work through **incongruity** — benign text paired with harmful imagery, or vice versa. Standard cross-attention only captures alignment (where modalities agree). Our architecture explicitly models **both agreement and conflict**:

```
                     ┌──────────────────────┐
                     │   Alignment Branch   │──→ captures where text & image AGREE
        query ──────▶│  (standard X-attn)   │
                     └──────────┬───────────┘
                                │
                     ┌──────────┴───────────┐
                     │     Gated Fusion      │──→ output = gate·align + (1-gate)·λ·incon
                     └──────────┬───────────┘
                                │
                     ┌──────────┴───────────┐
        query ──────▶│  Incongruity Branch  │──→ captures where text & image CONFLICT
                     │ (decorrelated X-attn) │
                     └──────────────────────┘
```

- **Alignment branch**: standard multi-head cross-attention
- **Incongruity branch**: separate projections + decorrelation loss to learn mismatched regions
- **Learnable λ**: automatically scales incongruity contribution
- **Learned gating**: sigmoid gate fuses both branches adaptively

## Full Architecture

```
Image ──→ ViT ──→ [CLS, patch₁, ..., patch₁₉₆] ──→ Project(768→128) ──┐
                                                                         │
                                                              Dual-Path Cross-Attention
                                                              (Alignment + Incongruity)
                                                              ×2 bidirectional
                                                                         │
Text ───→ BERT ──→ [CLS, tok₁, ..., tokₙ]  ──→ Project(768→128) ───────┘
                                                                         │
                                                              Mean Pool + Concat (256-D)
                                                                         │
                                                                  MLP(256→64→2)
                                                                         │
                                                              Hateful / Not-Hateful
```

## Three Attention Modes

| Mode | Flag | Description |
|------|------|-------------|
| **Dual-Path** | `--mode dual-path` | 🆕 Alignment + Incongruity branches (novel) |
| Sequence | `--mode sequence` | Full BERT tokens × ViT patches cross-attention |
| CLS-only | `--mode cls` | CLS token baseline |

## Two Validation Strategies

| Strategy | Flag | Description |
|----------|------|-------------|
| Normal Split | *(default)* | 70/10/20 train/val/test split |
| **K-Fold CV** | `--kfold 5` | Stratified k-fold cross-validation with mean ± std |

## Quick Start

### Local

```bash
pip install -r requirements.txt

# NOVEL: Dual-path cross-attention (alignment + incongruity)
python train.py --mode dual-path

# Dual-path with augmented data
python train.py --mode dual-path --augmented

# Dual-path with 5-fold cross-validation
python train.py --mode dual-path --kfold 5

# Standard sequence-level cross-attention
python train.py --mode sequence

# CLS-only baseline
python train.py --mode cls

# Custom hyperparameters
python train.py --mode dual-path --lr 3e-5 --epochs 10 --dropout 0.4

# Custom incongruity settings
python train.py --mode dual-path --incon-lambda 0.3 --incon-loss-weight 0.15
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

#### Experiment Presets (in `run_experiments.py`)

| ID | Mode | Data | Validation | Description |
|----|------|------|------------|-------------|
| 1 | Sequence | Balanced | Normal Split | Baseline cross-attention |
| 2 | Sequence | Augmented | Normal Split | Baseline + augmented |
| **3** | **Dual-Path** | Balanced | Normal Split | 🆕 Novel architecture |
| **4** | **Dual-Path** | Augmented | Normal Split | 🆕 Novel + augmented |
| **5** | **Dual-Path** | Balanced | **5-Fold CV** | 🆕 Novel + k-fold |
| **6** | **Dual-Path** | Augmented | **5-Fold CV** | 🆕 Novel + k-fold + aug |
| 7 | CLS-only | Balanced | Normal Split | CLS baseline |
| 8 | CLS-only | Augmented | Normal Split | CLS + augmented |

Just change `EXPERIMENT_ID = 4` at the top of the file and run.

**Colab-specific features:**
- Auto-mounts Google Drive (checkpoints persist across sessions)
- Auto-installs all dependencies
- Auto-detects project & data paths
- Prints GPU name and memory at startup

## Project Structure

```
hateful_meme_detection/
├── src/
│   ├── config.py                # Hyperparameters + environment detection
│   ├── data/
│   │   ├── dataset.py           # HatefulMemesDataset (PyTorch Dataset)
│   │   ├── loader.py            # JSONL/CSV loading, splits, k-fold loaders
│   │   └── augmentation.py      # Offline augmentation pipeline
│   ├── models/
│   │   ├── cross_attention.py   # CrossModalAttention + DualPathCrossAttention
│   │   └── classifier.py        # Main model (sequence / CLS / dual-path modes)
│   ├── engine/
│   │   ├── metrics.py           # MetricsTracker with AUC-ROC, confusion matrix
│   │   ├── trainer.py           # Training loop with incongruity loss + fold support
│   │   └── evaluator.py         # Test evaluation + fold-aware checkpoint loading
│   └── utils/
│       └── visualization.py     # Training curves, bar charts, confusion matrix
├── train.py                     # Local CLI entry point
├── run_experiments.py           # Unified Colab / Kaggle / Local runner
├── requirements.txt
├── JSON Files/                  # JSONL data (train, dev_seen, dev_unseen, etc.)
├── CSV Files/                   # Augmented CSV data
├── img/                         # Meme images
└── Notebooks/                   # Original notebooks (preserved)
```

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 2e-5 | AdamW optimizer |
| Batch Size (Train/Val/Test) | 32 / 8 / 16 | |
| Data Split | 70% / 10% / 20% | Normal mode |
| K-Fold Splits | 5 | Stratified, class-balanced |
| Epochs | 5 | |
| Weight Decay | 0.01 | |
| Dropout | 0.3 | |
| Attention Heads | 8 | Per cross-attention branch |
| Embedding Dimension | 128 | 768 → 128 projection |
| Incongruity λ (init) | 0.5 | Learnable during training |
| Incongruity Loss Weight | 0.1 | Auxiliary decorrelation loss |

## CLI Reference

```
python train.py [-h]
    --mode {sequence,cls,dual-path}     Attention mode (default: sequence)
    --kfold N                           K-fold CV folds, 0=disabled (default: 0)
    --augmented                         Use augmented dataset
    --class-weights                     Use inverse-frequency class weights
    --incon-lambda FLOAT                Incongruity λ init (default: 0.5)
    --incon-loss-weight FLOAT           Incongruity loss weight (default: 0.1)
    --lr FLOAT                          Learning rate (default: 2e-5)
    --epochs N                          Number of epochs (default: 5)
    --batch-size N                      Train batch size (default: 32)
    --dropout FLOAT                     Dropout rate (default: 0.3)
    --embed-dim N                       Projection dim (default: 128)
    --attn-heads N                      Attention heads (default: 8)
    --eval-only CHECKPOINT              Evaluate a checkpoint only
    --name NAME                         Custom experiment name
```

## Dataset

Based on the [Facebook Hateful Memes Challenge](https://ai.facebook.com/tools/hatefulmemes/):
- **~10,000 memes** with binary labels (hateful / not-hateful)
- Includes benign confounders to prevent unimodal shortcuts
- Balanced to 8,500 samples; augmented to 11,000 samples
