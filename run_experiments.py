#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║         Hateful Meme Detection — Unified Experiment Runner              ║
║                                                                          ║
║   Supports: Google Colab (Pro) • Kaggle • Local                         ║
║   Modes  : sequence | cls | dual-path                                   ║
║   Eval   : normal split | k-fold cross-validation                       ║
╚══════════════════════════════════════════════════════════════════════════╝

USAGE — GOOGLE COLAB (recommended):
    1. Open a new Colab notebook
    2. In the first cell, run:
           !git clone https://github.com/<your-username>/hateful-meme-detection.git
           %cd hateful-meme-detection
    3. In the second cell, paste or run:
           %run run_experiments.py
       OR
           !python run_experiments.py

USAGE — KAGGLE:
    1. Add your dataset + this repo as a Kaggle dataset
    2. Copy this script into a notebook cell
    3. Run the cell

USAGE — LOCAL:
    python run_experiments.py

CONFIGURATION:
    Change EXPERIMENT_ID below to run different experiments.
    Everything else is auto-detected.
"""

import os
import sys
import time

# ════════════════════════════════════════════════════════════════════════════
#  🎛️  EXPERIMENT SELECTOR — CHANGE THIS TO RUN DIFFERENT EXPERIMENTS
# ════════════════════════════════════════════════════════════════════════════
#
#   ID  │ Mode        │ Data       │ Validation    │ Description
#  ─────┼─────────────┼────────────┼───────────────┼──────────────────────────────
#   1   │ sequence    │ balanced   │ normal split  │ Baseline: full token cross-attn
#   2   │ sequence    │ augmented  │ normal split  │ Baseline + augmented data
#   3   │ dual-path   │ balanced   │ normal split  │ NOVEL dual-path on balanced
#   4   │ dual-path   │ augmented  │ normal split  │ NOVEL dual-path on augmented
#   5   │ dual-path   │ balanced   │ 5-fold CV     │ NOVEL dual-path + k-fold
#   6   │ dual-path   │ augmented  │ 5-fold CV     │ NOVEL dual-path + k-fold + aug
#   7   │ cls         │ balanced   │ normal split  │ CLS-only baseline
#   8   │ cls         │ augmented  │ normal split  │ CLS-only baseline + augmented
#

EXPERIMENT_ID = 4   # ← CHANGE THIS (1–8)


# ════════════════════════════════════════════════════════════════════════════
#  STEP 0: Environment Detection & Setup
# ════════════════════════════════════════════════════════════════════════════

def detect_env():
    """Detect runtime: 'colab', 'kaggle', or 'local'."""
    if "google.colab" in sys.modules or os.path.exists("/content"):
        return "colab"
    elif os.path.exists("/kaggle"):
        return "kaggle"
    return "local"


def setup_colab():
    """
    Full Google Colab setup:
        1. Mount Google Drive (for persistent storage)
        2. Install dependencies
        3. Clone or locate the project
        4. Return (project_root, data_root) paths
    """
    print("🔵 Google Colab Pro detected")
    print("─" * 60)

    # ── 1. Mount Google Drive ──────────────────────────────────────────
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    print("  ✅ Google Drive mounted")

    # ── 2. Install dependencies ────────────────────────────────────────
    print("\n📦 Installing dependencies...")
    os.system("pip install -q transformers torch torchvision scikit-learn "
              "tqdm matplotlib Pillow albumentations")
    print("  ✅ Dependencies installed")

    # ── 3. Locate project ──────────────────────────────────────────────
    # Priority: current directory → Drive
    if os.path.exists("src/config.py"):
        project_root = os.getcwd()
        print(f"  📂 Project found in current directory: {project_root}")
    elif os.path.exists("/content/hateful-meme-detection/src/config.py"):
        project_root = "/content/hateful-meme-detection"
        print(f"  📂 Project found at: {project_root}")
    elif os.path.exists("/content/hateful_meme_detection/src/config.py"):
        project_root = "/content/hateful_meme_detection"
        print(f"  📂 Project found at: {project_root}")
    else:
        # Try Drive
        drive_candidates = [
            "/content/drive/MyDrive/hateful_meme_detection",
            "/content/drive/MyDrive/hateful-meme-detection",
            "/content/drive/MyDrive/Colab Notebooks/hateful_meme_detection",
        ]
        project_root = None
        for candidate in drive_candidates:
            if os.path.exists(os.path.join(candidate, "src", "config.py")):
                project_root = candidate
                print(f"  📂 Project found in Drive: {project_root}")
                break

        if project_root is None:
            print("  ⚠️  Project not found. Cloning from git...")
            os.system("git clone https://github.com/100rabhsah/hateful-meme-detection.git "
                      "/content/hateful-meme-detection")
            project_root = "/content/hateful-meme-detection"

    # ── 4. Locate data ─────────────────────────────────────────────────
    # IMPORTANT: Google Drive FUSE I/O is very slow for random image reads.
    # We copy data to the local SSD (/content/) first for ~10-50x faster I/O.

    drive_data_candidates = [
        "/content/drive/MyDrive/hateful-memes-data",
        "/content/drive/MyDrive/hateful_memes",
        "/content/drive/MyDrive/hateful_memes_data",
        "/content/drive/MyDrive/hateful_meme_detection",
        "/content/drive/MyDrive/datasets/hateful_memes",
    ]

    local_data_path = "/content/hateful-memes-data"

    # Check if data is already on local SSD (from a previous copy)
    local_has_data = (
        os.path.exists(os.path.join(local_data_path, "img")) and
        (os.path.exists(os.path.join(local_data_path, "JSON Files", "train.jsonl")) or
         os.path.exists(os.path.join(local_data_path, "train.jsonl")))
    )

    if local_has_data:
        data_root = local_data_path
        print(f"  📂 Data already on local SSD: {data_root} (fast I/O ⚡)")
    else:
        # Find data on Drive and copy to local SSD
        drive_data_root = None
        for candidate in drive_data_candidates:
            has_json = os.path.exists(os.path.join(candidate, "JSON Files", "train.jsonl")) or \
                       os.path.exists(os.path.join(candidate, "train.jsonl"))
            has_img = os.path.exists(os.path.join(candidate, "img"))
            if has_json or has_img:
                drive_data_root = candidate
                break

        if drive_data_root is None:
            # Fallback: use project root
            data_root = project_root
            print(f"  ⚠️  Data not found on Drive, using project root: {data_root}")
        else:
            # Copy Drive data → local SSD for fast I/O
            # Using OS-level cp -r (much faster than Python shutil over Drive FUSE)
            print(f"  📂 Data found on Drive: {drive_data_root}")
            print(f"  ⏳ Copying data to local SSD for fast I/O...")
            os.makedirs(local_data_path, exist_ok=True)

            import subprocess
            for folder in ["img", "JSON Files", "CSV Files"]:
                src = os.path.join(drive_data_root, folder)
                dst = os.path.join(local_data_path, folder)
                if os.path.exists(src) and not os.path.exists(dst):
                    print(f"     📁 Copying {folder}/ ...", end=" ", flush=True)
                    result = subprocess.run(
                        ["cp", "-r", src, dst],
                        capture_output=True, text=True,
                    )
                    if result.returncode == 0:
                        # Count files copied
                        n_files = sum(len(f) for _, _, f in os.walk(dst))
                        print(f"✅ ({n_files} files)")
                    else:
                        print(f"❌ Error: {result.stderr.strip()}")

            data_root = local_data_path
            print(f"  ✅ Data copied to local SSD: {data_root} (fast I/O ⚡)")

    # ── 5. Set output dirs (persistent in Drive) ───────────────────────
    output_base = "/content/drive/MyDrive/hateful_memes_results"
    os.makedirs(output_base, exist_ok=True)
    output_dir = os.path.join(output_base, "outputs")
    checkpoint_dir = os.path.join(output_base, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"  💾 Outputs → {output_dir}")
    print(f"  💾 Checkpoints → {checkpoint_dir}")

    return project_root, data_root, output_dir, checkpoint_dir


def setup_kaggle():
    """Kaggle environment setup."""
    print("🟠 Kaggle detected")
    os.system("pip install -q transformers")

    project_root = "/kaggle/working/hateful-meme-detection"
    if not os.path.exists(project_root):
        base = "/kaggle/input/datasets/sourabhsah04/hateful-memes-data"
        if os.path.exists(base):
            for d in os.listdir(base):
                candidate = os.path.join(base, d)
                if os.path.exists(os.path.join(candidate, "src")):
                    project_root = candidate
                    break

    data_root = "/kaggle/input/datasets/sourabhsah04/hateful-memes-data/hateful-memes-data"
    output_dir = "/kaggle/working/outputs"
    checkpoint_dir = "/kaggle/working/checkpoints"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"  📂 Project: {project_root}")
    print(f"  📂 Data: {data_root}")
    return project_root, data_root, output_dir, checkpoint_dir


def setup_local():
    """Local environment setup."""
    print("🟢 Local environment detected")
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = project_root
    output_dir = os.path.join(project_root, "outputs")
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    return project_root, data_root, output_dir, checkpoint_dir


# ════════════════════════════════════════════════════════════════════════════
#  STEP 1: Detect environment and configure paths
# ════════════════════════════════════════════════════════════════════════════

env = detect_env()
print(f"\n{'═'*60}")
print(f"  🔬 Hateful Meme Detection — Experiment Runner")
print(f"  📍 Environment: {env.upper()}")
print(f"  🧪 Experiment ID: {EXPERIMENT_ID}")
print(f"{'═'*60}\n")

if env == "colab":
    project_root, data_root, output_dir, checkpoint_dir = setup_colab()
elif env == "kaggle":
    project_root, data_root, output_dir, checkpoint_dir = setup_kaggle()
else:
    project_root, data_root, output_dir, checkpoint_dir = setup_local()

# Add project to path
sys.path.insert(0, project_root)
if env != "local":
    os.chdir(project_root)

print(f"\n{'─'*60}")


# ════════════════════════════════════════════════════════════════════════════
#  STEP 2: Imports (after path setup)
# ════════════════════════════════════════════════════════════════════════════

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import ExperimentConfig, PathConfig, ModelConfig, TrainingConfig
from src.data.loader import build_dataloaders, build_kfold_dataloaders
from src.models.classifier import HatefulMemesClassifier
from src.engine.trainer import Trainer
from src.engine.evaluator import evaluate, load_model_from_checkpoint
from src.engine.metrics import EpochMetrics
from src.utils.visualization import (
    plot_training_curves,
    plot_metrics_comparison,
    plot_confusion_matrix,
)


# ════════════════════════════════════════════════════════════════════════════
#  STEP 3: Experiment Presets
# ════════════════════════════════════════════════════════════════════════════

EXPERIMENTS = {
    1: {"name": "seq_bal_e5",       "mode": "sequence",   "augmented": False, "dual_path": False, "kfold": 0},
    2: {"name": "seq_aug_e5",       "mode": "sequence",   "augmented": True,  "dual_path": False, "kfold": 0},
    3: {"name": "dual_bal_e5",      "mode": "dual-path",  "augmented": False, "dual_path": True,  "kfold": 0},
    4: {"name": "dual_aug_e5",      "mode": "dual-path",  "augmented": True,  "dual_path": True,  "kfold": 0},
    5: {"name": "dual_bal_kf5",     "mode": "dual-path",  "augmented": False, "dual_path": True,  "kfold": 5},
    6: {"name": "dual_aug_kf5",     "mode": "dual-path",  "augmented": True,  "dual_path": True,  "kfold": 5},
    7: {"name": "cls_bal_e5",       "mode": "cls",        "augmented": False, "dual_path": False, "kfold": 0},
    8: {"name": "cls_aug_e5",       "mode": "cls",        "augmented": True,  "dual_path": False, "kfold": 0},
}

exp = EXPERIMENTS[EXPERIMENT_ID]
use_sequence = exp["mode"] in ("sequence", "dual-path")


# ════════════════════════════════════════════════════════════════════════════
#  STEP 4: Build Configuration
# ════════════════════════════════════════════════════════════════════════════

# Resolve data directories
if env == "kaggle":
    json_dir = os.path.join(data_root, "JSON Files")
    csv_dir = os.path.join(data_root, "CSV Files")
    image_dir = os.path.join(data_root, "img")
elif env == "colab":
    # Auto-detect directory structure
    if os.path.exists(os.path.join(data_root, "JSON Files")):
        json_dir = os.path.join(data_root, "JSON Files")
    else:
        json_dir = data_root

    if os.path.exists(os.path.join(data_root, "CSV Files")):
        csv_dir = os.path.join(data_root, "CSV Files")
    else:
        csv_dir = data_root

    # image_dir is the root where img/ folder lives
    image_dir = data_root
else:
    json_dir = os.path.join(data_root, "JSON Files")
    csv_dir = os.path.join(data_root, "CSV Files")
    image_dir = data_root

# Build PathConfig manually (skip auto-detection)
custom_paths = PathConfig(environment=env)
custom_paths.data_root = data_root
custom_paths.image_dir = image_dir
custom_paths.json_dir = json_dir
custom_paths.csv_dir = csv_dir
custom_paths.output_dir = output_dir
custom_paths.checkpoint_dir = checkpoint_dir

config = ExperimentConfig(
    paths=custom_paths,
    model=ModelConfig(
        bert_model_name="bert-base-uncased",
        vit_model_name="google/vit-base-patch16-224",
        embed_dim=128,
        num_attention_heads=8,
        dropout=0.3,
        use_sequence_tokens=use_sequence,
        use_dual_path=exp["dual_path"],
        incongruity_lambda=0.5,
        incongruity_loss_weight=0.1,
    ),
    training=TrainingConfig(
        learning_rate=2e-5,
        weight_decay=0.01,
        num_epochs=5,
        train_batch_size=64,      # A100: safe at 64 (use 32 for T4)
        val_batch_size=64,        # A100: larger = faster eval
        test_batch_size=64,       # A100: larger = faster eval
        random_seed=42,
        num_workers=2,            # Colab Pro handles 2 workers fine
        use_class_weights=False,
        use_augmented_data=exp["augmented"],
        num_kfolds=exp["kfold"],
    ),
    experiment_name=exp["name"],
)

print(config.summary())


# ════════════════════════════════════════════════════════════════════════════
#  STEP 5: Seed Everything
# ════════════════════════════════════════════════════════════════════════════

torch.manual_seed(config.training.random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.training.random_seed)
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    try:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   Memory: {gpu_mem:.1f} GB")
    except AttributeError:
        pass  # older/newer torch versions may differ
print(f"📱 Device: {config.device}\n")


# ════════════════════════════════════════════════════════════════════════════
#  STEP 6: Run Experiment
# ════════════════════════════════════════════════════════════════════════════

start_time = time.time()

if exp["kfold"] > 0:
    # ── K-Fold Cross-Validation ────────────────────────────────────────
    k = exp["kfold"]
    print(f"\n{'═'*70}")
    print(f"  🔄 Running {k}-Fold Stratified Cross-Validation")
    print(f"{'═'*70}")

    fold_metrics = []
    for fold_idx, train_loader, val_loader, max_length in build_kfold_dataloaders(config):
        # Fresh model for each fold
        model = HatefulMemesClassifier(config.model)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        trainer = Trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            fold=fold_idx,
        )
        history = trainer.train()

        best_val = max(history["val"], key=lambda m: m.f1)
        fold_metrics.append(best_val)
        print(f"  📊 Fold {fold_idx} Best Val F1: {best_val.f1:.4f}")

    # ── K-Fold Summary ─────────────────────────────────────────────────
    accuracies = [m.accuracy for m in fold_metrics]
    f1_scores = [m.f1 for m in fold_metrics]
    auc_scores = [m.auc_roc for m in fold_metrics if m.auc_roc is not None]

    print(f"\n{'═'*70}")
    print(f"  📊 {k}-Fold Cross-Validation Results")
    print(f"{'═'*70}")
    print(f"  Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"  F1 Score:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    if auc_scores:
        print(f"  AUC-ROC:   {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"\n  Per-Fold F1: {[f'{f:.4f}' for f in f1_scores]}")
    print(f"{'═'*70}")

    # Save summary
    summary_path = os.path.join(output_dir, f"{exp['name']}_kfold_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"{k}-Fold Cross-Validation Summary\n")
        f.write(f"Experiment: {exp['name']}\n{'='*50}\n")
        f.write(f"Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n")
        f.write(f"F1 Score:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n")
        if auc_scores:
            f.write(f"AUC-ROC:   {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}\n")
        f.write(f"\nPer-Fold:\n")
        for i, m in enumerate(fold_metrics, 1):
            auc_str = f"AUC={m.auc_roc:.4f}" if m.auc_roc else "AUC=N/A"
            f.write(f"  Fold {i}: Acc={m.accuracy:.4f} F1={m.f1:.4f} {auc_str}\n")
    print(f"  💾 Summary saved: {summary_path}")

else:
    # ── Normal Split Training ──────────────────────────────────────────
    train_loader, val_loader, test_loader, max_length = build_dataloaders(config)

    model = HatefulMemesClassifier(config.model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🏗  Model: {total_params:,} params ({trainable_params:,} trainable)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Train
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
    )
    history = trainer.train()

    # Test
    print("\n📋 Evaluating best model on test set...")
    best_model = load_model_from_checkpoint(config)
    test_metrics = evaluate(best_model, test_loader, criterion, device=config.device)

    # Plots
    print("\n📊 Generating plots...")
    plot_training_curves(
        history["train"], history["val"],
        output_dir,
        experiment_name=exp["name"],
    )
    if history["train"] and history["val"]:
        plot_metrics_comparison(
            history["train"][-1],
            history["val"][-1],
            test_metrics,
            output_dir,
            experiment_name=exp["name"],
        )
    if test_metrics.confusion_mat is not None:
        plot_confusion_matrix(
            test_metrics.confusion_mat,
            output_dir,
            experiment_name=exp["name"],
        )


# ════════════════════════════════════════════════════════════════════════════
#  STEP 7: Done
# ════════════════════════════════════════════════════════════════════════════

elapsed = time.time() - start_time
print(f"\n{'═'*60}")
print(f"  🎉 Experiment '{exp['name']}' complete!")
print(f"  ⏱  Total time: {elapsed/60:.1f} minutes")
print(f"  💾 Results at: {output_dir}")
if env == "colab":
    print(f"  📂 Checkpoints saved to Google Drive (persistent)")
print(f"{'═'*60}")
