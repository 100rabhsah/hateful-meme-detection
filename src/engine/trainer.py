"""
Training loop with validation, checkpointing, and metrics logging.
Supports both standard and dual-path (incongruity-aware) training.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, List, Dict
import warnings

from src.config import ExperimentConfig
from src.engine.metrics import MetricsTracker, EpochMetrics


class Trainer:
    """
    Manages the full training lifecycle:
        - Training loop with gradient clipping
        - Per-epoch validation
        - Best-model checkpointing (by validation F1)
        - Training history tracking
        - Auxiliary incongruity loss (when dual-path is enabled)
    """

    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[object] = None,
        fold: Optional[int] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = config.device
        self.fold = fold  # None for normal split, int for k-fold

        # Dual-path settings
        self.use_dual_path = config.model.use_dual_path
        self.incon_loss_weight = config.model.incongruity_loss_weight

        # Early stopping (v2)
        self.early_stop_patience = 3
        self._no_improve_count = 0

        # History
        self.train_history: List[EpochMetrics] = []
        self.val_history: List[EpochMetrics] = []
        self.best_val_f1 = 0.0
        self.best_epoch = 0

    def train(self) -> Dict[str, List[EpochMetrics]]:
        """
        Run the full training loop for `num_epochs`.

        Returns:
            Dictionary with 'train' and 'val' metric histories.
        """
        self.model.to(self.device)
        num_epochs = self.config.training.num_epochs

        fold_str = f" [Fold {self.fold}]" if self.fold is not None else ""
        dual_str = " (Dual-Path)" if self.use_dual_path else ""
        print(f"\n🚀 Starting training{fold_str}{dual_str} for {num_epochs} epochs on {self.device}")
        print(f"   Train batches: {len(self.train_loader)}  |  Val batches: {len(self.val_loader)}")
        print("─" * 70)

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # ── Train one epoch ─────────────────────────────────────────
            train_metrics = self._train_one_epoch(epoch, num_epochs)
            self.train_history.append(train_metrics)

            # ── Validate ────────────────────────────────────────────────
            val_metrics = self._validate(epoch, num_epochs)
            self.val_history.append(val_metrics)

            # ── LR scheduler step ───────────────────────────────────────
            if self.scheduler is not None:
                self.scheduler.step()

            # ── Checkpointing ───────────────────────────────────────────
            if val_metrics.f1 > self.best_val_f1:
                self.best_val_f1 = val_metrics.f1
                self.best_epoch = epoch
                self._save_checkpoint(epoch, is_best=True)
                print(f"   ⭐ New best model! Val F1: {val_metrics.f1:.4f}")
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1

            elapsed = time.time() - epoch_start
            print(f"   ⏱  Epoch {epoch} completed in {elapsed:.1f}s")
            print("─" * 70)

            # ── Early stopping ──────────────────────────────────────────
            if self._no_improve_count >= self.early_stop_patience:
                print(f"\n⏹  Early stopping: no improvement for {self.early_stop_patience} epochs")
                break

        # Save final model
        self._save_checkpoint(num_epochs, is_best=False, tag="final")
        print(f"\n✅ Training complete! Best epoch: {self.best_epoch} (Val F1: {self.best_val_f1:.4f})")

        return {"train": self.train_history, "val": self.val_history}

    def _train_one_epoch(self, epoch: int, num_epochs: int) -> EpochMetrics:
        """Train for one epoch."""
        self.model.train()
        tracker = MetricsTracker()

        fold_str = f" F{self.fold}" if self.fold is not None else ""
        loop = tqdm(
            self.train_loader,
            desc=f"  Epoch {epoch}/{num_epochs}{fold_str} [Train]",
            leave=False,
        )
        for images, input_ids, attention_mask, labels in loop:
            images = images.to(self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=images,
            )

            # Handle dict output (new interface)
            logits = output['logits']
            loss = self.criterion(logits, labels)

            # Add auxiliary incongruity loss if dual-path is enabled
            if self.use_dual_path and 'incongruity_loss' in output:
                incon_loss = output['incongruity_loss']
                if incon_loss.requires_grad:
                    loss = loss + self.incon_loss_weight * incon_loss

            # Backward
            loss.backward()
            if self.config.training.gradient_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_max_norm,
                )
            self.optimizer.step()

            # Track
            tracker.update(loss.item(), logits, labels)
            loop.set_postfix(loss=f"{tracker.running_loss / tracker.num_batches:.4f}")

        metrics = tracker.compute()
        print(f"\n   {metrics.summary_line(prefix='Train  │ ')}")
        return metrics

    @torch.no_grad()
    def _validate(self, epoch: int, num_epochs: int) -> EpochMetrics:
        """Run validation."""
        self.model.eval()
        tracker = MetricsTracker()

        fold_str = f" F{self.fold}" if self.fold is not None else ""
        loop = tqdm(
            self.val_loader,
            desc=f"  Epoch {epoch}/{num_epochs}{fold_str} [Val]  ",
            leave=False,
        )
        for images, input_ids, attention_mask, labels in loop:
            images = images.to(self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)

            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=images,
            )

            logits = output['logits']
            loss = self.criterion(logits, labels)
            tracker.update(loss.item(), logits, labels)

        metrics = tracker.compute()
        print(f"   {metrics.summary_line(prefix='Val    │ ')}")
        print(f"   Classification Report:\n{metrics.classification_report_str}")
        return metrics

    def _save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        tag: str = "",
    ):
        """Save model checkpoint with timestamp to avoid overwriting."""
        from datetime import datetime
        ckpt_dir = self.config.paths.checkpoint_dir
        exp_name = self.config.experiment_name
        fold_suffix = f"_fold{self.fold}" if self.fold is not None else ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        if tag:
            filename = f"{exp_name}{fold_suffix}_{tag}_{timestamp}.pth"
        elif is_best:
            filename = f"{exp_name}{fold_suffix}_best_{timestamp}.pth"
        else:
            filename = f"{exp_name}{fold_suffix}_epoch{epoch}_{timestamp}.pth"

        path = os.path.join(ckpt_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_f1": self.best_val_f1,
            "fold": self.fold,
            "config": {
                "experiment_name": self.config.experiment_name,
                "model": vars(self.config.model),
                "training": vars(self.config.training),
            },
        }, path)
        print(f"   💾 Saved checkpoint: {path}")
