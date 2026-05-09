"""
Centralized configuration for all hyperparameters, paths, and experiment settings.
Supports both local and Kaggle/Colab environments via environment auto-detection.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional


def detect_environment() -> str:
    """Auto-detect runtime environment: 'colab', 'kaggle', or 'local'."""
    if os.path.exists("/content"):
        return "colab"
    elif os.path.exists("/kaggle"):
        return "kaggle"
    return "local"


@dataclass
class PathConfig:
    """All path-related configuration. Resolved based on environment."""
    environment: str = field(default_factory=detect_environment)

    # These are set in __post_init__ based on environment
    data_root: str = ""
    image_dir: str = ""
    json_dir: str = ""
    csv_dir: str = ""
    output_dir: str = ""
    checkpoint_dir: str = ""

    def __post_init__(self):
        if self.environment == "colab":
            base = "/content/drive/MyDrive/hateful_memes"
            self.data_root = base
            self.image_dir = base  # images referenced as "img/XXXX.png" relative to data_root
            self.json_dir = base
            self.csv_dir = base
            self.output_dir = os.path.join(base, "outputs")
            self.checkpoint_dir = os.path.join(base, "checkpoints")

        elif self.environment == "kaggle":
            # Kaggle: input data is read-only, outputs go to /kaggle/working
            base_input = "/kaggle/input/hateful-memes"
            base_output = "/kaggle/working"
            self.data_root = base_input
            self.image_dir = base_input
            self.json_dir = base_input
            self.csv_dir = base_input
            self.output_dir = os.path.join(base_output, "outputs")
            self.checkpoint_dir = os.path.join(base_output, "checkpoints")

        else:  # local
            # Resolve relative to project root (one level up from src/)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_root = project_root
            self.image_dir = project_root  # images at project_root/img/XXXX.png
            self.json_dir = os.path.join(project_root, "JSON Files")
            self.csv_dir = os.path.join(project_root, "CSV Files")
            self.output_dir = os.path.join(project_root, "outputs")
            self.checkpoint_dir = os.path.join(project_root, "checkpoints")

        # Ensure output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


@dataclass
class ModelConfig:
    """Model architecture hyperparameters."""
    bert_model_name: str = "bert-base-uncased"
    vit_model_name: str = "google/vit-base-patch16-224"
    bert_hidden_size: int = 768       # BERT base output dim
    vit_hidden_size: int = 768        # ViT base output dim
    embed_dim: int = 128              # Projection dimension for cross-attention
    num_attention_heads: int = 8      # Cross-modal attention heads
    num_classes: int = 2              # Binary: hateful / not-hateful
    dropout: float = 0.3
    fusion_hidden_dim: int = 64       # Hidden dim in fusion MLP
    use_sequence_tokens: bool = True  # True = use full token sequences; False = CLS-only

    # ── Dual-path cross-attention (NOVEL) ──────────────────────────────
    use_dual_path: bool = False       # True = dual-path (alignment + incongruity)
    incongruity_lambda: float = 0.5   # Initial λ for incongruity branch weighting
    incongruity_loss_weight: float = 0.1  # Weight of auxiliary incongruity loss


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 5
    train_batch_size: int = 32
    val_batch_size: int = 8
    test_batch_size: int = 16
    train_ratio: float = 0.70
    val_ratio: float = 0.10
    test_ratio: float = 0.20
    random_seed: int = 42
    num_workers: int = 2               # DataLoader workers (0 for Colab)
    pin_memory: bool = True
    gradient_clip_max_norm: float = 1.0
    use_class_weights: bool = False    # Inverse-frequency class weighting
    use_augmented_data: bool = False   # Load augmented CSV instead of raw JSONL

    # ── K-Fold cross-validation ────────────────────────────────────────
    num_kfolds: int = 5               # Number of folds (0 = disabled, use normal split)


@dataclass
class AugmentationConfig:
    """Data augmentation settings (for offline augmentation pipeline)."""
    target_class: int = 1              # Minority class to augment
    num_augmented_samples: int = 2500
    horizontal_flip_p: float = 0.5
    rotate_limit: int = 30
    rotate_p: float = 0.5
    brightness_contrast_p: float = 0.5
    gaussian_blur_p: float = 0.3


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration bundling all sub-configs."""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    experiment_name: str = "bert_vit_cross_attention"
    device: str = field(default="")

    def __post_init__(self):
        if not self.device:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon
            else:
                self.device = "cpu"

    def summary(self) -> str:
        """Print a human-readable summary of the experiment configuration."""
        # Determine attention mode string
        if self.model.use_dual_path:
            attn_mode = "Dual-Path (Alignment + Incongruity)"
        elif self.model.use_sequence_tokens:
            attn_mode = "Full Sequence Cross-Attention"
        else:
            attn_mode = "CLS-Only Baseline"

        kfold_str = f"{self.training.num_kfolds}-Fold CV" if self.training.num_kfolds > 0 else "Normal Split"

        lines = [
            f"{'='*60}",
            f"  Experiment: {self.experiment_name}",
            f"  Environment: {self.paths.environment}",
            f"  Device: {self.device}",
            f"  Attention Mode: {attn_mode}",
            f"  Validation: {kfold_str}",
            f"{'='*60}",
            f"  BERT: {self.model.bert_model_name}",
            f"  ViT:  {self.model.vit_model_name}",
            f"  Embed Dim: {self.model.embed_dim}  |  Attn Heads: {self.model.num_attention_heads}",
            f"  Dropout: {self.model.dropout}  |  Sequence Tokens: {self.model.use_sequence_tokens}",
        ]

        if self.model.use_dual_path:
            lines.extend([
                f"  Incongruity λ: {self.model.incongruity_lambda}  |  "
                f"Incon. Loss Weight: {self.model.incongruity_loss_weight}",
            ])

        lines.extend([
            f"{'─'*60}",
            f"  LR: {self.training.learning_rate}  |  Epochs: {self.training.num_epochs}",
            f"  Batch (Train/Val/Test): {self.training.train_batch_size}/{self.training.val_batch_size}/{self.training.test_batch_size}",
            f"  Split: {self.training.train_ratio}/{self.training.val_ratio}/{self.training.test_ratio}",
            f"  Weight Decay: {self.training.weight_decay}  |  Class Weights: {self.training.use_class_weights}",
            f"  Augmented Data: {self.training.use_augmented_data}",
            f"{'='*60}",
        ])
        return "\n".join(lines)
