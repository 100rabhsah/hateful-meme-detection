"""
Visualization utilities for training curves, metrics comparison, and confusion matrices.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Optional

from src.engine.metrics import EpochMetrics

# Use non-interactive backend when running headless
matplotlib.use("Agg")


def plot_training_curves(
    train_history: List[EpochMetrics],
    val_history: List[EpochMetrics],
    save_dir: str,
    experiment_name: str = "experiment",
):
    """
    Plot training and validation loss/accuracy/F1 curves across epochs.
    Saves the figure to `save_dir`.
    """
    epochs = range(1, len(train_history) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training Curves — {experiment_name}", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(epochs, [m.loss for m in train_history], "b-o", label="Train")
    axes[0].plot(epochs, [m.loss for m in val_history], "r-s", label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, [m.accuracy for m in train_history], "b-o", label="Train")
    axes[1].plot(epochs, [m.accuracy for m in val_history], "r-s", label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1 Score
    axes[2].plot(epochs, [m.f1 for m in train_history], "b-o", label="Train")
    axes[2].plot(epochs, [m.f1 for m in val_history], "r-s", label="Val")
    axes[2].set_title("F1 Score (Weighted)")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{experiment_name}_training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📈 Training curves saved: {path}")


def plot_metrics_comparison(
    train_metrics: EpochMetrics,
    val_metrics: EpochMetrics,
    test_metrics: EpochMetrics,
    save_dir: str,
    experiment_name: str = "experiment",
):
    """
    Bar chart comparing final train/val/test metrics side by side.
    """
    metric_names = ["Loss", "Accuracy", "Precision", "Recall", "F1 Score"]

    train_vals = [train_metrics.loss, train_metrics.accuracy, train_metrics.precision,
                  train_metrics.recall, train_metrics.f1]
    val_vals = [val_metrics.loss, val_metrics.accuracy, val_metrics.precision,
                val_metrics.recall, val_metrics.f1]
    test_vals = [test_metrics.loss, test_metrics.accuracy, test_metrics.precision,
                 test_metrics.recall, test_metrics.f1]

    x = np.arange(len(metric_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, train_vals, width, label="Train", color="#4C72B0")
    ax.bar(x, val_vals, width, label="Validation", color="#55A868")
    ax.bar(x + width, test_vals, width, label="Test", color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Value")
    ax.set_title(f"Metrics Comparison — {experiment_name}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(save_dir, f"{experiment_name}_metrics_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Metrics comparison saved: {path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    save_dir: str,
    experiment_name: str = "experiment",
    class_names: List[str] = None,
    normalize: Optional[str] = None,
):
    """
    Plot a confusion matrix heatmap.

    Args:
        cm: Raw confusion matrix (counts).
        save_dir: Directory to save the plot.
        experiment_name: Name for the title and filename.
        class_names: Class labels.
        normalize: Normalization mode:
            - None: raw counts (default, backward-compatible)
            - 'true': row-wise normalization (percentage per true class)
            - 'pred': column-wise normalization (percentage per predicted class)
    """
    if class_names is None:
        class_names = ["0 (Not Hateful)", "1 (Hateful)"]

    # Compute normalized version if requested
    if normalize == "true":
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        file_suffix = "_pct"
        fmt_func = lambda val, raw: f"{val:.2f}"
        colorbar_label = None
    elif normalize == "pred":
        cm_display = cm.astype(float) / cm.sum(axis=0, keepdims=True) * 100
        file_suffix = "_pct_pred"
        fmt_func = lambda val, raw: f"{val:.2f}"
        colorbar_label = None
    else:
        cm_display = cm.astype(float)
        file_suffix = ""
        fmt_func = lambda val, raw: format(int(raw), "d")
        colorbar_label = None

    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(cm_display, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.outline.set_visible(False)
    if colorbar_label:
        cbar.set_label(colorbar_label, fontsize=11)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True",
        xlabel="Predicted",
    )
    
    # Remove plot borders
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)

    # Write values in cells
    thresh = cm_display.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cell_text = fmt_func(cm_display[i, j], cm[i, j])
            ax.text(j, i, cell_text,
                    ha="center", va="center", fontsize=12,
                    color="white" if cm_display[i, j] > thresh else "black")

    plt.tight_layout()
    path = os.path.join(save_dir, f"{experiment_name}_confusion_matrix{file_suffix}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  🔢 Confusion matrix saved: {path}")
    return path
