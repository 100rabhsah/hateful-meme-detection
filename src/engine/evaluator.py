"""
Evaluation: test set evaluation + model loading from checkpoint.
Supports both standard and dual-path model outputs.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from src.config import ExperimentConfig
from src.engine.metrics import MetricsTracker, EpochMetrics
from src.models.classifier import HatefulMemesClassifier


def load_model_from_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: Optional[str] = None,
    fold: Optional[int] = None,
) -> HatefulMemesClassifier:
    """
    Load a model from a checkpoint file.

    Args:
        config: Experiment configuration.
        checkpoint_path: Path to .pth file. If None, loads the best checkpoint.
        fold: Fold number (for k-fold). Used to construct checkpoint path.
    """
    if checkpoint_path is None:
        fold_suffix = f"_fold{fold}" if fold is not None else ""
        checkpoint_path = os.path.join(
            config.paths.checkpoint_dir,
            f"{config.experiment_name}{fold_suffix}_best.pth",
        )

    print(f"  Loading checkpoint: {checkpoint_path}")
    model = HatefulMemesClassifier(config.model)
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)
    model.eval()
    print(f"  ✅ Model loaded (epoch {checkpoint.get('epoch', '?')}, "
          f"best F1: {checkpoint.get('best_val_f1', '?')})")
    return model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: str = "cuda",
    desc: str = "Testing",
) -> EpochMetrics:
    """
    Evaluate a model on a DataLoader.

    Args:
        model: The model (already on device, in eval mode).
        test_loader: DataLoader for the test set.
        criterion: Loss function.
        device: Device string.
        desc: Description for the progress bar.

    Returns:
        EpochMetrics with test results.
    """
    model.eval()
    tracker = MetricsTracker()

    loop = tqdm(test_loader, desc=desc, leave=True)
    for images, input_ids, attention_mask, labels in loop:
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=images,
        )

        # Handle dict output (new interface)
        logits = output['logits'] if isinstance(output, dict) else output
        loss = criterion(logits, labels)
        tracker.update(loss.item(), logits, labels)

    metrics = tracker.compute()

    print(f"\n{'═'*60}")
    print(f"  {metrics.summary_line(prefix='Test   │ ')}")
    print(f"{'─'*60}")
    print(f"  Classification Report:\n{metrics.classification_report_str}")
    print(f"  Confusion Matrix:\n{metrics.confusion_mat}")
    print(f"{'═'*60}")

    return metrics
