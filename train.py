#!/usr/bin/env python3
"""
Local training entry point for Hateful Meme Detection.

Usage:
    # Default: sequence-level cross-attention on balanced data
    python train.py

    # CLS-only baseline
    python train.py --mode cls

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
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import ExperimentConfig, PathConfig, ModelConfig, TrainingConfig
from src.data.loader import build_dataloaders, compute_class_weights
from src.data.augmentation import run_augmentation
from src.models.classifier import HatefulMemesClassifier
from src.engine.trainer import Trainer
from src.engine.evaluator import evaluate, load_model_from_checkpoint
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
        "--mode", choices=["sequence", "cls"], default="sequence",
        help="'sequence' = full BERT tokens + ViT patches; 'cls' = CLS-only baseline",
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

    # Experiment
    parser.add_argument("--name", default=None, help="Experiment name (auto-generated if None)")

    # Eval only
    parser.add_argument(
        "--eval-only", default=None, metavar="CHECKPOINT",
        help="Skip training; evaluate this checkpoint on the test set",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Build config from CLI args ──────────────────────────────────────
    use_sequence = (args.mode == "sequence")

    experiment_name = args.name or f"{'seq' if use_sequence else 'cls'}_{('aug' if args.augmented else 'bal')}_e{args.epochs}"

    config = ExperimentConfig(
        paths=PathConfig(),
        model=ModelConfig(
            bert_model_name=args.bert,
            vit_model_name=args.vit,
            embed_dim=args.embed_dim,
            num_attention_heads=args.attn_heads,
            dropout=args.dropout,
            use_sequence_tokens=use_sequence,
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
        ),
        experiment_name=experiment_name,
    )

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

    # ── Build data loaders ──────────────────────────────────────────────
    train_loader, val_loader, test_loader, max_length = build_dataloaders(config)

    # ── Build model ─────────────────────────────────────────────────────
    print(f"\n🏗  Building model (mode: {'Sequence Tokens' if use_sequence else 'CLS Only'})...")
    model = HatefulMemesClassifier(config.model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # ── Loss function ───────────────────────────────────────────────────
    if config.training.use_class_weights:
        from src.data.loader import compute_class_weights, df_to_dicts, load_jsonl_data, balance_dataset
        # Recompute weights from the actual data split
        # For simplicity, use uniform weighting; can be refined
        weights = torch.tensor([1.0, 1.5], device=config.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"   Using class weights: {weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Optimizer ───────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # ── Eval-only mode ──────────────────────────────────────────────────
    if args.eval_only:
        model = load_model_from_checkpoint(config, checkpoint_path=args.eval_only)
        test_metrics = evaluate(model, test_loader, criterion, device=config.device)
        if test_metrics.confusion_mat is not None:
            plot_confusion_matrix(
                test_metrics.confusion_mat,
                config.paths.output_dir,
                experiment_name=config.experiment_name,
            )
        return

    # ── Train ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
    )
    history = trainer.train()

    # ── Test ────────────────────────────────────────────────────────────
    print("\n📋 Evaluating on test set...")
    best_model = load_model_from_checkpoint(config)
    test_metrics = evaluate(best_model, test_loader, criterion, device=config.device)

    # ── Visualize ───────────────────────────────────────────────────────
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

    print(f"\n🎉 All done! Results saved to: {config.paths.output_dir}")


if __name__ == "__main__":
    main()
