# Hateful Meme Detection — BERT + ViT with Cross-Modal Attention

Multimodal hateful meme classification using **BERT token embeddings** and **ViT patch tokens** with bidirectional cross-modal attention fusion.

## Architecture

```
Image ──→ ViT ──→ [CLS, patch₁, patch₂, ..., patch₁₉₆] ──→ Project(768→128)
                                                                    │
                                                          ┌────────▼────────┐
                                                          │  Cross-Modal    │
Text ───→ BERT ──→ [CLS, tok₁, tok₂, ..., tokₙ] ──→ Project(768→128)
                                                          │  Attention (×2) │
                                                          └────────┬────────┘
                                                                   │
                                                          Mean Pool + Concat
                                                                   │
                                                             MLP → Logits
```

**Key Innovation**: Unlike CLS-only approaches, this model uses the **full sequence** of BERT token embeddings and **all 196 ViT patch tokens** for fine-grained cross-modal attention.

## Quick Start

### Local
```bash
pip install -r requirements.txt

# Sequence-level cross-attention (recommended)
python train.py --mode sequence

# CLS-only baseline for comparison
python train.py --mode cls

# With augmented data
python train.py --mode sequence --augmented

# Custom hyperparameters
python train.py --lr 3e-5 --epochs 10 --dropout 0.4
```

### Kaggle / Colab
1. Upload the project to Google Drive or as a Kaggle dataset
2. Copy `run_kaggle.py` contents into a notebook cell and run
3. Modify the `config = ExperimentConfig(...)` block to change experiments

## Project Structure

```
hateful_meme_detection/
├── src/
│   ├── config.py                # All hyperparameters + environment detection
│   ├── data/
│   │   ├── dataset.py           # HatefulMemesDataset (PyTorch Dataset)
│   │   ├── loader.py            # JSONL/CSV loading, splits, DataLoaders
│   │   └── augmentation.py      # Offline augmentation pipeline
│   ├── models/
│   │   ├── cross_attention.py   # CrossModalAttention module
│   │   └── classifier.py        # Main model (sequence + CLS modes)
│   ├── engine/
│   │   ├── metrics.py           # MetricsTracker with AUC-ROC, confusion matrix
│   │   ├── trainer.py           # Training loop with checkpointing
│   │   └── evaluator.py         # Test evaluation + checkpoint loading
│   └── utils/
│       └── visualization.py     # Training curves, bar charts, confusion matrix
├── train.py                     # Local CLI entry point
├── run_kaggle.py                # Kaggle/Colab entry point
├── requirements.txt
├── JSON Files/                  # JSONL data (train, dev_seen, dev_unseen, etc.)
├── CSV Files/                   # Augmented CSV data
├── img/                         # Meme images
└── Notebooks/                   # Original notebooks (preserved)
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 2e-5 |
| Optimizer | AdamW |
| Batch Size (Train/Val/Test) | 32 / 8 / 16 |
| Data Split | 70% / 10% / 20% |
| Epochs | 5 |
| Weight Decay | 0.01 |
| Dropout | 0.3 |
| Attention Heads | 8 |
| Embedding Dimension | 128 |

## Experiments

| Experiment | Mode | Data | Description |
|-----------|------|------|-------------|
| `seq_bal_e5` | Sequence | Balanced (8.5K) | Full token cross-attention on balanced data |
| `cls_bal_e5` | CLS-only | Balanced (8.5K) | CLS-only baseline for comparison |
| `seq_aug_e5` | Sequence | Augmented (11K) | Full token cross-attention on augmented data |
| `cls_aug_e5` | CLS-only | Augmented (11K) | CLS-only baseline on augmented data |
