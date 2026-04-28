"""
Metrics computation for training, validation, and testing.
Centralizes all metric calculations — no more duplicated code across notebooks.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class EpochMetrics:
    """Container for one epoch's metrics."""
    loss: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc_roc: Optional[float] = None
    classification_report_str: str = ""
    confusion_mat: Optional[np.ndarray] = None

    def summary_line(self, prefix: str = "") -> str:
        auc_str = f"  AUC-ROC: {self.auc_roc:.4f}" if self.auc_roc is not None else ""
        return (
            f"{prefix}Loss: {self.loss:.4f}  Acc: {self.accuracy:.4f}  "
            f"P: {self.precision:.4f}  R: {self.recall:.4f}  F1: {self.f1:.4f}{auc_str}"
        )


class MetricsTracker:
    """Accumulates predictions across batches and computes epoch-level metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.all_labels: List[int] = []
        self.all_preds: List[int] = []
        self.all_probs: List[float] = []  # probabilities for positive class
        self.running_loss: float = 0.0
        self.num_batches: int = 0

    def update(
        self,
        loss: float,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Update with one batch of results."""
        self.running_loss += loss
        self.num_batches += 1

        _, preds = torch.max(logits, dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(hateful)

        self.all_labels.extend(labels.cpu().numpy().tolist())
        self.all_preds.extend(preds.cpu().numpy().tolist())
        self.all_probs.extend(probs.cpu().detach().numpy().tolist())

    def compute(self) -> EpochMetrics:
        """Compute all metrics from accumulated predictions."""
        labels = np.array(self.all_labels)
        preds = np.array(self.all_preds)
        probs = np.array(self.all_probs)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0,
        )
        accuracy = accuracy_score(labels, preds)

        # AUC-ROC (only if both classes are present)
        auc = None
        if len(np.unique(labels)) > 1:
            try:
                auc = roc_auc_score(labels, probs)
            except ValueError:
                pass

        report = classification_report(
            labels, preds, target_names=["Not Hateful", "Hateful"], zero_division=0,
        )
        cm = confusion_matrix(labels, preds)

        return EpochMetrics(
            loss=self.running_loss / max(self.num_batches, 1),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc_roc=auc,
            classification_report_str=report,
            confusion_mat=cm,
        )
