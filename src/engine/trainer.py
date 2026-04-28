"""
Training loop with validation, checkpointing, and metrics logging.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, List, Dict

from src.config import ExperimentConfig
from src.engine.metrics import MetricsTracker, EpochMetrics


class Trainer:
    """
    Manages the full training lifecycle:
        - Training loop with gradient clipping
        - Per-epoch validation
        - Best-model checkpointing (by validation F1)
        - Training history tracking
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
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = config.device

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

        print(f"\n🚀 Starting training for {num_epochs} epochs on {self.device}")
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

            elapsed = time.time() - epoch_start
            print(f"   ⏱  Epoch {epoch} completed in {elapsed:.1f}s")
            print("─" * 70)

        # Save final model
        self._save_checkpoint(num_epochs, is_best=False, tag="final")
        print(f"\n✅ Training complete! Best epoch: {self.best_epoch} (Val F1: {self.best_val_f1:.4f})")

        return {"train": self.train_history, "val": self.val_history}

    def _train_one_epoch(self, epoch: int, num_epochs: int) -> EpochMetrics:
        """Train for one epoch."""
        self.model.train()
        tracker = MetricsTracker()

        loop = tqdm(
            self.train_loader,
            desc=f"  Epoch {epoch}/{num_epochs} [Train]",
            leave=False,
        )
        for images, input_ids, attention_mask, labels in loop:
            images = images.to(self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=images,
            )
            loss = self.criterion(logits, labels)

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

        loop = tqdm(
            self.val_loader,
            desc=f"  Epoch {epoch}/{num_epochs} [Val]  ",
            leave=False,
        )
        for images, input_ids, attention_mask, labels in loop:
            images = images.to(self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=images,
            )
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
        """Save model checkpoint."""
        ckpt_dir = self.config.paths.checkpoint_dir
        if tag:
            filename = f"{self.config.experiment_name}_{tag}.pth"
        elif is_best:
            filename = f"{self.config.experiment_name}_best.pth"
        else:
            filename = f"{self.config.experiment_name}_epoch{epoch}.pth"

        path = os.path.join(ckpt_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_f1": self.best_val_f1,
            "config": {
                "experiment_name": self.config.experiment_name,
                "model": vars(self.config.model),
                "training": vars(self.config.training),
            },
        }, path)
        print(f"   💾 Saved checkpoint: {path}")
