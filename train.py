#!/usr/bin/env python3
"""
Local training entry point for Hateful Meme Detection.

Supports three modes of operation:
    Phase 1 — Normal split training (standard or dual-path)
    Phase 2 — K-Fold cross-validation

Usage:
    # Default: word-to-patch cross-attention on balanced data
    python train.py

    # CLS-only baseline
    python train.py --mode cls

    # NOVEL: Dual-path word-to-patch cross-attention (alignment + incongruity)
    python train.py --mode dual-path

    # K-Fold cross-validation (5 folds)
    python train.py --kfold 5

    # Dual-path + K-Fold
    python train.py --mode dual-path --kfold 5

    # Use augmented data
    python train.py --augmented

    # Custom hyperparameters
    python train.py --lr 3e-5 --epochs 10 --batch-size 16

    # Run data augmentation pipeline first
    python train.py --run-augmentation
"""

import argparse
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import ExperimentConfig, PathConfig, ModelConfig, TrainingConfig
from src.data.loader import build_dataloaders, build_kfold_dataloaders, compute_class_weights
from src.data.augmentation import run_augmentation
from src.models.classifier import HatefulMemesClassifier
from src.engine.trainer import Trainer
from src.engine.evaluator import evaluate, load_model_from_checkpoint
from src.engine.metrics import EpochMetrics
from src.utils.visualization import (
    plot_training_curves,
    plot_metrics_comparison,
    plot_confusion_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hateful Meme Detection — BERT + ViT with Cross-Modal Attention",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Experiment mode
    parser.add_argument(
        "--mode", choices=["word-patch", "cls", "dual-path"], default="word-patch",
        help="'word-patch' = each word attends to all image patches (word-to-patch cross-attention); "
             "'cls' = CLS-only baseline; "
             "'dual-path' = NOVEL dual-path word-to-patch (alignment + incongruity)",
    )
    parser.add_argument(
        "--augmented", action="store_true",
        help="Use augmented dataset (shuffled_training_data.csv) instead of balanced JSONL",
    )
    parser.add_argument(
        "--run-augmentation", action="store_true",
        help="Run the offline augmentation pipeline before training",
    )
    parser.add_argument(
        "--class-weights", action="store_true",
        help="Use inverse-frequency class weights in loss",
    )

    # K-Fold cross-validation
    parser.add_argument(
        "--kfold", type=int, default=0,
        help="Number of folds for cross-validation (0 = disabled, use normal split)",
    )

    # Dual-path specific
    parser.add_argument(
        "--incon-lambda", type=float, default=0.5,
        help="Initial λ for incongruity branch weighting (dual-path mode)",
    )
    parser.add_argument(
        "--incon-loss-weight", type=float, default=0.1,
        help="Weight for auxiliary incongruity decorrelation loss",
    )

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Train batch size")
    parser.add_argument("--val-batch-size", type=int, default=8, help="Val batch size")
    parser.add_argument("--test-batch-size", type=int, default=16, help="Test batch size")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--embed-dim", type=int, default=128, help="Projection embedding dim")
    parser.add_argument("--attn-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model
    parser.add_argument("--bert", default="bert-base-uncased", help="BERT model name")
    parser.add_argument("--vit", default="google/vit-base-patch16-224", help="ViT model name")

    # Backbone freeze strategy
    parser.add_argument(
        "--freeze", choices=["none", "full", "partial"], default="partial",
        help="'none' = all params trainable (~196M); "
             "'full' = freeze all BERT+ViT (~676K trainable); "
             "'partial' = unfreeze top-N layers (~29.6M trainable)",
    )
    parser.add_argument(
        "--unfreeze-layers", type=int, default=2,
        help="Number of top transformer layers to unfreeze (only for --freeze partial)",
    )

    # LR scheduler
    parser.add_argument(
        "--no-scheduler", action="store_true",
        help="Disable cosine LR scheduler (use constant LR)",
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=1,
        help="Number of warmup epochs before cosine decay",
    )

    # Experiment
    parser.add_argument("--name", default=None, help="Experiment name (auto-generated if None)")

    # Eval only
    parser.add_argument(
        "--eval-only", default=None, metavar="CHECKPOINT",
        help="Skip training; evaluate this checkpoint on the test set",
    )

    return parser.parse_args()


def build_config(args) -> ExperimentConfig:
    """Build ExperimentConfig from CLI args."""
    use_word_patch = (args.mode in ("word-patch", "dual-path"))
    use_dual_path = (args.mode == "dual-path")

    # Auto-generate experiment name
    if args.name:
        experiment_name = args.name
    else:
        mode_str = "dual" if use_dual_path else ("wp" if use_word_patch else "cls")
        data_str = "aug" if args.augmented else "bal"
        freeze_str = f"_{args.freeze}" if args.freeze != "partial" else ""
        kfold_str = f"_kf{args.kfold}" if args.kfold > 0 else ""
        experiment_name = f"{mode_str}_{data_str}_e{args.epochs}{freeze_str}{kfold_str}"

    return ExperimentConfig(
        paths=PathConfig(),
        model=ModelConfig(
            bert_model_name=args.bert,
            vit_model_name=args.vit,
            embed_dim=args.embed_dim,
            num_attention_heads=args.attn_heads,
            dropout=args.dropout,
            use_word_patch_tokens=use_word_patch,
            use_dual_path=use_dual_path,
            incongruity_lambda=args.incon_lambda,
            incongruity_loss_weight=args.incon_loss_weight,
        ),
        training=TrainingConfig(
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            num_epochs=args.epochs,
            train_batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            test_batch_size=args.test_batch_size,
            random_seed=args.seed,
            use_class_weights=args.class_weights,
            use_augmented_data=args.augmented,
            num_kfolds=args.kfold,
            freeze_strategy=args.freeze,
            unfreeze_top_n=args.unfreeze_layers,
            use_lr_scheduler=not args.no_scheduler,
            warmup_epochs=args.warmup_epochs,
        ),
        experiment_name=experiment_name,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Backbone Freeze Strategy
# ═══════════════════════════════════════════════════════════════════════════

def freeze_backbones(model, strategy: str = "partial", unfreeze_top_n: int = 2):
    """
    Apply backbone freeze strategy.

    Args:
        model: HatefulMemesClassifier
        strategy: "none" | "full" | "partial"
            - none:    All parameters trainable (~196M). Best if enough data.
            - full:    Freeze all BERT+ViT. Only heads trainable (~676K).
            - partial: Freeze all except top-N transformer layers (~29.6M).
        unfreeze_top_n: Layers to unfreeze (for "partial" mode).
    """
    if strategy == "none":
        # Everything is trainable (default PyTorch behavior)
        print(f"\n🔓 Freeze: NONE — all parameters trainable")

    elif strategy == "full":
        # Freeze all BERT and ViT parameters
        for name, param in model.named_parameters():
            if name.startswith("bert.") or name.startswith("vit."):
                param.requires_grad = False
        print(f"\n🧊 Freeze: FULL — all BERT+ViT frozen, only heads trainable")

    elif strategy == "partial":
        # Step 1: Freeze everything in backbones
        for name, param in model.named_parameters():
            if name.startswith("bert.") or name.startswith("vit."):
                param.requires_grad = False

        # Step 2: Unfreeze top N layers of BERT
        bert_total_layers = 12  # BERT-base
        for layer_idx in range(bert_total_layers - unfreeze_top_n, bert_total_layers):
            for name, param in model.named_parameters():
                if f"bert.encoder.layer.{layer_idx}." in name:
                    param.requires_grad = True

        # Step 3: Unfreeze top N layers of ViT
        vit_total_layers = 12  # ViT-base
        for layer_idx in range(vit_total_layers - unfreeze_top_n, vit_total_layers):
            for name, param in model.named_parameters():
                if f"vit.encoder.layer.{layer_idx}." in name:
                    param.requires_grad = True

        # Also unfreeze the final layernorms
        for name, param in model.named_parameters():
            if "bert.pooler." in name or "vit.layernorm." in name:
                param.requires_grad = True

        print(f"\n🧊 Freeze: PARTIAL — top {unfreeze_top_n} layers of BERT+ViT unfrozen")

    # Report trainable parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"   Total: {total_params:,}  |  Trainable: {trainable_params:,}  |  Frozen: {frozen_params:,}")

    return model


def build_model_and_optimizer(config: ExperimentConfig):
    """Build model, criterion, optimizer, and LR scheduler."""
    use_word_patch = config.model.use_word_patch_tokens
    use_dual_path = config.model.use_dual_path

    mode_str = "Dual-Path Word-to-Patch" if use_dual_path else ("Word-to-Patch" if use_word_patch else "CLS Only")
    print(f"\n🏗  Building model (mode: {mode_str})...")

    model = HatefulMemesClassifier(config.model)

    # Apply freeze strategy
    model = freeze_backbones(
        model,
        strategy=config.training.freeze_strategy,
        unfreeze_top_n=config.training.unfreeze_top_n,
    )

    # Loss function with label smoothing
    if config.training.use_class_weights:
        weights = torch.tensor([1.0, 1.5], device=config.device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
        print(f"   Using class weights: {weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer — only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # LR Scheduler — cosine annealing with linear warmup
    scheduler = None
    if config.training.use_lr_scheduler:
        warmup_epochs = config.training.warmup_epochs
        total_epochs = config.training.num_epochs

        if warmup_epochs > 0 and total_epochs > warmup_epochs:
            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_epochs,
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-7,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
            print(f"   📈 LR Schedule: warmup ({warmup_epochs} ep) → cosine decay → {1e-7}")
        else:
            scheduler = CosineAnnealingLR(
                optimizer, T_max=max(total_epochs, 1), eta_min=1e-7,
            )
            print(f"   📈 LR Schedule: cosine decay → {1e-7}")

    return model, criterion, optimizer, scheduler


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1: Normal Split Training
# ═══════════════════════════════════════════════════════════════════════════

def run_normal_split(config: ExperimentConfig):
    """Run training with normal train/val/test split."""
    print("\n" + "═" * 70)
    print("  PHASE 1: Normal Split Training")
    print("═" * 70)

    # Build data loaders
    train_loader, val_loader, test_loader, max_length = build_dataloaders(config)

    # Build model
    model, criterion, optimizer, scheduler = build_model_and_optimizer(config)

    # Train
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    history = trainer.train()

    # Test
    print("\n📋 Evaluating on test set...")
    best_model = load_model_from_checkpoint(config)
    test_metrics = evaluate(best_model, test_loader, criterion, device=config.device)

    # Visualize
    _generate_plots(history, test_metrics, config)

    return history, test_metrics


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2: K-Fold Cross-Validation
# ═══════════════════════════════════════════════════════════════════════════

def run_kfold_cv(config: ExperimentConfig):
    """
    Run K-Fold cross-validation.

    For each fold:
        1. Build train/val loaders (stratified)
        2. Train a fresh model
        3. Evaluate on the fold's validation set

    Reports mean ± std across all folds for each metric.
    """
    k = config.training.num_kfolds
    print("\n" + "═" * 70)
    print(f"  PHASE 2: {k}-Fold Stratified Cross-Validation")
    print("═" * 70)

    fold_metrics: list[EpochMetrics] = []

    for fold_idx, train_loader, val_loader, max_length in build_kfold_dataloaders(config):
        # Build a FRESH model for each fold
        model, criterion, optimizer, scheduler = build_model_and_optimizer(config)

        # Train
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            fold=fold_idx,
        )
        history = trainer.train()

        # The best validation metrics for this fold
        best_val = max(history["val"], key=lambda m: m.f1)
        fold_metrics.append(best_val)

        print(f"\n📊 Fold {fold_idx} Best Val: "
              f"Acc={best_val.accuracy:.4f}  F1={best_val.f1:.4f}  "
              f"AUC={best_val.auc_roc:.4f}" if best_val.auc_roc else "")

    # ── Aggregate results across folds ──────────────────────────────────
    _print_kfold_summary(fold_metrics, k, config)

    return fold_metrics


def _print_kfold_summary(fold_metrics: list, k: int, config: ExperimentConfig):
    """Print and save k-fold cross-validation summary."""
    accuracies = [m.accuracy for m in fold_metrics]
    precisions = [m.precision for m in fold_metrics]
    recalls = [m.recall for m in fold_metrics]
    f1_scores = [m.f1 for m in fold_metrics]
    auc_scores = [m.auc_roc for m in fold_metrics if m.auc_roc is not None]

    print("\n" + "═" * 70)
    print(f"  📊 {k}-Fold Cross-Validation Summary")
    print("═" * 70)
    print(f"  {'Metric':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'─'*55}")
    print(f"  {'Accuracy':<15} {np.mean(accuracies):>10.4f} {np.std(accuracies):>10.4f} "
          f"{np.min(accuracies):>10.4f} {np.max(accuracies):>10.4f}")
    print(f"  {'Precision':<15} {np.mean(precisions):>10.4f} {np.std(precisions):>10.4f} "
          f"{np.min(precisions):>10.4f} {np.max(precisions):>10.4f}")
    print(f"  {'Recall':<15} {np.mean(recalls):>10.4f} {np.std(recalls):>10.4f} "
          f"{np.min(recalls):>10.4f} {np.max(recalls):>10.4f}")
    print(f"  {'F1 Score':<15} {np.mean(f1_scores):>10.4f} {np.std(f1_scores):>10.4f} "
          f"{np.min(f1_scores):>10.4f} {np.max(f1_scores):>10.4f}")
    if auc_scores:
        print(f"  {'AUC-ROC':<15} {np.mean(auc_scores):>10.4f} {np.std(auc_scores):>10.4f} "
              f"{np.min(auc_scores):>10.4f} {np.max(auc_scores):>10.4f}")
    print("═" * 70)

    # Per-fold breakdown
    print(f"\n  Per-Fold Breakdown:")
    print(f"  {'Fold':<6} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print(f"  {'─'*56}")
    for i, m in enumerate(fold_metrics, 1):
        auc_str = f"{m.auc_roc:>10.4f}" if m.auc_roc is not None else f"{'N/A':>10}"
        print(f"  {i:<6} {m.accuracy:>10.4f} {m.precision:>10.4f} "
              f"{m.recall:>10.4f} {m.f1:>10.4f} {auc_str}")
    print()

    # Save summary to file
    summary_path = os.path.join(config.paths.output_dir, f"{config.experiment_name}_kfold_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"{k}-Fold Cross-Validation Summary\n")
        f.write(f"Experiment: {config.experiment_name}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n")
        f.write(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}\n")
        f.write(f"Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}\n")
        f.write(f"F1 Score:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n")
        if auc_scores:
            f.write(f"AUC-ROC:   {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}\n")
        f.write(f"{'='*60}\n\n")
        f.write("Per-Fold Results:\n")
        for i, m in enumerate(fold_metrics, 1):
            f.write(f"Fold {i}: Acc={m.accuracy:.4f} P={m.precision:.4f} "
                    f"R={m.recall:.4f} F1={m.f1:.4f} "
                    f"AUC={m.auc_roc:.4f if m.auc_roc else 'N/A'}\n")
    print(f"  💾 K-fold summary saved: {summary_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Plot helpers
# ═══════════════════════════════════════════════════════════════════════════

def _generate_plots(history, test_metrics, config):
    """Generate and save all visualization plots."""
    print("\n📊 Generating plots...")
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


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    config = build_config(args)

    print(config.summary())

    # ── Seed everything ─────────────────────────────────────────────────
    torch.manual_seed(config.training.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.random_seed)

    # ── Optional: run augmentation ──────────────────────────────────────
    if args.run_augmentation:
        run_augmentation(config)
        if not args.augmented:
            print("\n⚠️  Augmentation complete. Add --augmented to use augmented data for training.")

    # ── Eval-only mode ──────────────────────────────────────────────────
    if args.eval_only:
        train_loader, val_loader, test_loader, _ = build_dataloaders(config)
        _, criterion, _, _ = build_model_and_optimizer(config)
        model = load_model_from_checkpoint(config, checkpoint_path=args.eval_only)
        test_metrics = evaluate(model, test_loader, criterion, device=config.device)
        if test_metrics.confusion_mat is not None:
            plot_confusion_matrix(
                test_metrics.confusion_mat,
                config.paths.output_dir,
                experiment_name=config.experiment_name,
            )
        return

    # ── Run experiments ─────────────────────────────────────────────────
    if config.training.num_kfolds > 0:
        # Phase 2: K-Fold cross-validation
        run_kfold_cv(config)
    else:
        # Phase 1: Normal split
        run_normal_split(config)

    print(f"\n🎉 All done! Results saved to: {config.paths.output_dir}")


if __name__ == "__main__":
    main()
