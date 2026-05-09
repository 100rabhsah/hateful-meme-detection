"""
Kaggle / Google Colab entry point for Hateful Meme Detection.

╔══════════════════════════════════════════════════════════════════════╗
║  HOW TO USE ON KAGGLE / COLAB:                                      ║
║                                                                      ║
║  1. Upload the entire project as a dataset (Kaggle) or to Drive     ║
║  2. Copy this file's contents into a notebook cell                  ║
║  3. Run the cell — it handles drive mounting, path setup, and       ║
║     the full training pipeline automatically                        ║
╚══════════════════════════════════════════════════════════════════════╝

Alternatively, you can:
    !git clone <your-repo> /content/hateful_meme_detection
    %cd /content/hateful_meme_detection
    !python run_kaggle.py
"""

import os
import sys

# ════════════════════════════════════════════════════════════════════════
#  STEP 0: Environment Setup
# ════════════════════════════════════════════════════════════════════════

def setup_environment():
    """Auto-detect and configure environment (Colab vs Kaggle vs Local)."""

    # if os.path.exists("/content"):
    #     # ── Google Colab ────────────────────────────────────────────────
    #     print("🔵 Detected Google Colab environment")
    #     try:
    #         from google.colab import drive
    #         drive.mount("/content/drive")
    #         print("  ✅ Google Drive mounted")
    #     except Exception:
    #         print("  ⚠️  Drive mount failed — using local files")

    #     # Install dependencies
    #     os.system("pip install -q transformers torch torchvision scikit-learn tqdm matplotlib")

    #     # Set project root (clone or symlink the project here)
    #     project_root = "/kaggle/working/hateful-meme-detection"
    #     if not os.path.exists(project_root):
    #         # If the src/ folder is in Drive
    #         drive_project = "/content/drive/MyDrive/hateful_meme_detection"
    #         if os.path.exists(drive_project):
    #             os.symlink(drive_project, project_root)
    #             print(f"  Symlinked {drive_project} → {project_root}")
    #         else:
    #             print(f"  ⚠️  Project not found at {drive_project}")
    #             print("  Please clone the repo or upload the project")
    #             return None
    #     return project_root

    if os.path.exists("/kaggle"):
        # ── Kaggle ──────────────────────────────────────────────────────
        print("🟠 Detected Kaggle environment")
        os.system("pip install -q transformers")

        project_root = "/kaggle/working/hateful-meme-detection"
        if not os.path.exists(project_root):
            # Try to find it in available datasets
            for d in os.listdir("/kaggle/input/datasets/sourabhsah04/hateful-memes-data"):
                candidate = f"/kaggle/input/datasets/sourabhsah04/hateful-memes-data{d}"
                if os.path.exists(os.path.join(candidate, "src")):
                    project_root = candidate
                    break
        print(f"  Project root: {project_root}")
        return project_root

    else:
        # ── Local ──────────────────────────────────────────────────────
        print("🟢 Detected local environment")
        return os.path.dirname(os.path.abspath(__file__))


# ════════════════════════════════════════════════════════════════════════
#  STEP 1: Setup paths and imports
# ════════════════════════════════════════════════════════════════════════

project_root = setup_environment()
if project_root:
    sys.path.insert(0, project_root)
    os.chdir(project_root)

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import ExperimentConfig, PathConfig, ModelConfig, TrainingConfig
from src.data.loader import build_dataloaders
from src.models.classifier import HatefulMemesClassifier
from src.engine.trainer import Trainer
from src.engine.evaluator import evaluate, load_model_from_checkpoint
from src.utils.visualization import (
    plot_training_curves,
    plot_metrics_comparison,
    plot_confusion_matrix,
)


# ════════════════════════════════════════════════════════════════════════
#  STEP 2: Configuration
# ════════════════════════════════════════════════════════════════════════
#
#  Modify these settings to run different experiments.
#  All hyperparameters from your spec are set as defaults.
#

# 1. Create the default object
custom_paths = PathConfig(environment="kaggle")
# 2. Manually override the attributes (this happens AFTER __post_init__)
custom_paths.data_root = "/kaggle/input/datasets/sourabhsah04/hateful-memes-data/hateful-memes-data"
custom_paths.image_dir = "/kaggle/input/datasets/sourabhsah04/hateful-memes-data/hateful-memes-data/img"
custom_paths.json_dir = "/kaggle/input/datasets/sourabhsah04/hateful-memes-data/hateful-memes-data/JSON Files"
custom_paths.csv_dir = "/kaggle/input/datasets/sourabhsah04/hateful-memes-data/hateful-memes-data/CSV Files"
custom_paths.output_dir = "/kaggle/working/"
custom_paths.checkpoint_dir = "/kaggle/working/hateful-meme-detection"

config = ExperimentConfig(
    paths=custom_paths,   # Auto-detects environment
    model=ModelConfig(
        bert_model_name="bert-base-uncased",
        vit_model_name="google/vit-base-patch16-224",
        embed_dim=128,
        num_attention_heads=8,
        dropout=0.3,
        use_sequence_tokens=True,   # ← True = BERT token seq + ViT patch tokens
                                    #   False = CLS-only baseline
    ),
    training=TrainingConfig(
        learning_rate=2e-5,
        weight_decay=0.01,
        num_epochs=5,
        train_batch_size=32,
        val_batch_size=8,
        test_batch_size=16,
        random_seed=42,
        use_class_weights=False,
        use_augmented_data=False,   # ← True = use augmented CSV (11K samples)
                                    #   False = balanced JSONL (8.5K samples)
    ),
    experiment_name="seq_balanced_e5",   # Change for each experiment
)

print(config.summary())


# ════════════════════════════════════════════════════════════════════════
#  STEP 3: Seed & Data
# ════════════════════════════════════════════════════════════════════════

torch.manual_seed(config.training.random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.training.random_seed)

train_loader, val_loader, test_loader, max_length = build_dataloaders(config)


# ════════════════════════════════════════════════════════════════════════
#  STEP 4: Model, Loss, Optimizer
# ════════════════════════════════════════════════════════════════════════

model = HatefulMemesClassifier(config.model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n🏗  Model built: {total_params:,} params ({trainable_params:,} trainable)")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=config.training.learning_rate,
    weight_decay=config.training.weight_decay,
)


# ════════════════════════════════════════════════════════════════════════
#  STEP 5: Train
# ════════════════════════════════════════════════════════════════════════

trainer = Trainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
)

history = trainer.train()


# ════════════════════════════════════════════════════════════════════════
#  STEP 6: Test
# ════════════════════════════════════════════════════════════════════════

print("\n📋 Evaluating best model on test set...")
best_model = load_model_from_checkpoint(config)
test_metrics = evaluate(best_model, test_loader, criterion, device=config.device)


# ════════════════════════════════════════════════════════════════════════
#  STEP 7: Visualize
# ════════════════════════════════════════════════════════════════════════

plot_training_curves(
    history["train"], history["val"],
    config.paths.output_dir,
    experiment_name=config.experiment_name,
)

if history["train"] and history["val"]:
    plot_metrics_comparison(
        history["train"][-1],
        history["val"][-1],
        test_metrics,
        config.paths.output_dir,
        experiment_name=config.experiment_name,
    )

if test_metrics.confusion_mat is not None:
    plot_confusion_matrix(
        test_metrics.confusion_mat,
        config.paths.output_dir,
        experiment_name=config.experiment_name,
    )

print(f"\n🎉 Done! Results at: {config.paths.output_dir}")