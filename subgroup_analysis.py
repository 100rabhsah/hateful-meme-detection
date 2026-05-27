"""
Subgroup Analysis: Evaluate best checkpoints on test_seen vs test_unseen separately.
This tests whether the dual-path architecture generalizes better to unseen categories.

Usage on Colab:
    %run subgroup_analysis.py
"""

import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
)

from src.config import ExperimentConfig, PathConfig, ModelConfig, TrainingConfig
from src.data.loader import read_jsonl, df_to_dicts
from src.data.dataset import HatefulMemesDataset, get_image_transform, compute_max_token_length
from src.models.classifier import HatefulMemesClassifier


# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

# Detect environment
if os.path.exists("/content"):
    ENV = "colab"
    CKPT_DIR = "/content/drive/MyDrive/hateful_memes_results/checkpoints"
    OUTPUT_DIR = "/content/drive/MyDrive/hateful_memes_results/outputs"

    # ── Mount Google Drive ──────────────────────────────────────────────
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)

    # ── Sync data from GDrive to local SSD for fast I/O ────────────────
    local_data_path = "/content/hateful-memes-data"
    drive_data_candidates = [
        "/content/drive/MyDrive/hateful-memes-data",
        "/content/drive/MyDrive/hateful_memes",
        "/content/drive/MyDrive/hateful_memes_data",
    ]

    local_has_data = (
        os.path.exists(os.path.join(local_data_path, "img")) and
        os.path.exists(os.path.join(local_data_path, "JSON Files", "train.jsonl"))
    )

    if local_has_data:
        DATA_DIR = local_data_path
        print(f"  📂 Data already on local SSD: {DATA_DIR} (fast I/O ⚡)")
    else:
        drive_data_root = None
        for candidate in drive_data_candidates:
            if os.path.exists(os.path.join(candidate, "JSON Files", "train.jsonl")) or \
               os.path.exists(os.path.join(candidate, "img")):
                drive_data_root = candidate
                break

        if drive_data_root:
            import subprocess
            print(f"  📂 Data found on Drive: {drive_data_root}")
            print(f"  ⏳ Copying data to local SSD for fast I/O...")
            os.makedirs(local_data_path, exist_ok=True)
            for folder in ["img", "JSON Files", "CSV Files"]:
                src = os.path.join(drive_data_root, folder) + "/"
                dst = os.path.join(local_data_path, folder)
                if os.path.exists(os.path.join(drive_data_root, folder)):
                    os.makedirs(dst, exist_ok=True)
                    print(f"     📁 Syncing {folder}/ ...", end=" ", flush=True)
                    result = subprocess.run(
                        ["rsync", "-a", src, dst],
                        capture_output=True, text=True,
                    )
                    if result.returncode == 0:
                        n_files = sum(len(f) for _, _, f in os.walk(dst))
                        print(f"✅ ({n_files} files)")
                    else:
                        print(f"❌ Error: {result.stderr.strip()}")
            DATA_DIR = local_data_path
            print(f"  ✅ Data copied to local SSD: {DATA_DIR} (fast I/O ⚡)")
        else:
            DATA_DIR = local_data_path
            print(f"  ⚠️  Data not found on Drive, assuming: {DATA_DIR}")
else:
    ENV = "local"
    DATA_DIR = "./data"
    CKPT_DIR = "./checkpoints"
    OUTPUT_DIR = "./outputs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
JSON_DIR = os.path.join(DATA_DIR, "JSON Files")
IMAGE_DIR = DATA_DIR  # JSONL paths already include 'img/' prefix

# Models to evaluate: (name, checkpoint_filename, use_dual_path, use_word_patch)
MODELS = [
    ("CLS-only (Augmented)",     "v2_cls_aug_best.pth",   False, False),
    ("Sequence (Augmented)",     "v2_seq_aug_best.pth",   False, True),
    ("Dual-Path (Augmented)",    "v2_dual_aug_best.pth",  True,  True),
    ("CLS-only (Balanced)",      "v2_cls_bal_best.pth",   False, False),
    ("Sequence (Balanced)",      "v2_seq_bal_best.pth",   False, True),
    ("Dual-Path (Balanced)",     "v2_dual_bal_best.pth",  True,  True),
]

# Test splits to evaluate on
TEST_SPLITS = [
    ("test_seen",   "test_seen.jsonl"),
    ("test_unseen", "test_unseen.jsonl"),
    ("dev_seen",    "dev_seen.jsonl"),
    ("dev_unseen",  "dev_unseen.jsonl"),
]


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def build_eval_loader(jsonl_path, tokenizer, max_length, image_root):
    """Build a DataLoader for a single JSONL split."""
    records = read_jsonl(jsonl_path)
    df = pd.DataFrame(records)
    image_paths, texts, labels = df_to_dicts(df)

    dataset = HatefulMemesDataset(
        image_paths=image_paths,
        texts=texts,
        labels=labels,
        image_root=image_root,
        tokenizer=tokenizer,
        max_length=max_length,
        transform=get_image_transform(),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return loader, len(df), dict(df["label"].value_counts())


def load_model(ckpt_path, use_dual_path, use_sequence):
    """Load model from checkpoint."""
    model_config = ModelConfig(
        bert_model_name="bert-base-uncased",
        vit_model_name="google/vit-base-patch16-224",
        embed_dim=128,
        num_attention_heads=8,
        dropout=0.3,
        use_word_patch_tokens=use_sequence,
        use_dual_path=use_dual_path,
        incongruity_lambda=0.5,
        incongruity_loss_weight=0.5,
    )
    model = HatefulMemesClassifier(model_config)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, ckpt.get("epoch", "?"), ckpt.get("best_val_f1", "?")


@torch.no_grad()
def evaluate_split(model, loader, device):
    """Run evaluation and return metrics dict."""
    all_preds = []
    all_labels = []
    all_probs = []

    for images, input_ids, attention_mask, labels in loader:
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        output = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images)
        logits = output["logits"] if isinstance(output, dict) else output
        probs = torch.softmax(logits, dim=1)

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="macro"),
        "precision": precision_score(all_labels, all_preds, average="macro"),
        "recall": recall_score(all_labels, all_preds, average="macro"),
        "auc_roc": roc_auc_score(all_labels, all_probs),
        "report": classification_report(all_labels, all_preds,
                                        target_names=["Not Hateful", "Hateful"]),
        "confusion": confusion_matrix(all_labels, all_preds),
    }


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  🔬 Subgroup Analysis: test_seen vs test_unseen")
print(f"  📍 Environment: {ENV}")
print(f"  📱 Device: {DEVICE}")
print("=" * 70)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Pre-compute max_length from a sample
sample_records = read_jsonl(os.path.join(JSON_DIR, "test_seen.jsonl"))
sample_texts = [r["text"] for r in sample_records]
max_length = compute_max_token_length(sample_texts, tokenizer, cap=128)
print(f"\n  Max token length: {max_length}")

# Build eval loaders for each split
print("\n📦 Loading test splits...")
loaders = {}
for split_name, split_file in TEST_SPLITS:
    path = os.path.join(JSON_DIR, split_file)
    if os.path.exists(path):
        loader, n_samples, label_dist = build_eval_loader(
            path, tokenizer, max_length, IMAGE_DIR
        )
        loaders[split_name] = loader
        print(f"  ✅ {split_name}: {n_samples} samples {label_dist}")
    else:
        print(f"  ⚠️  {split_name}: {path} not found, skipping")

# Results storage
all_results = []

# Evaluate each model on each split
for model_name, ckpt_file, use_dual, use_seq in MODELS:
    ckpt_path = os.path.join(CKPT_DIR, ckpt_file)
    if not os.path.exists(ckpt_path):
        print(f"\n⚠️  Checkpoint not found: {ckpt_path}, skipping {model_name}")
        continue

    print(f"\n{'═' * 70}")
    print(f"  🏗  Model: {model_name}")
    print(f"  📂 Checkpoint: {ckpt_file}")

    model, epoch, best_f1 = load_model(ckpt_path, use_dual, use_seq)
    print(f"  ✅ Loaded (epoch {epoch}, val F1: {best_f1:.4f})" if isinstance(best_f1, float) else f"  ✅ Loaded (epoch {epoch})")

    for split_name, loader in loaders.items():
        print(f"\n  📊 Evaluating on {split_name}...")
        metrics = evaluate_split(model, loader, DEVICE)

        all_results.append({
            "model": model_name,
            "split": split_name,
            **{k: v for k, v in metrics.items() if k not in ("report", "confusion")},
        })

        print(f"     Acc: {metrics['accuracy']:.4f}  "
              f"F1: {metrics['f1']:.4f}  "
              f"AUC: {metrics['auc_roc']:.4f}")
        print(f"     {metrics['report']}")

    del model
    torch.cuda.empty_cache()

# ════════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 90)
print("  📊 SUBGROUP ANALYSIS SUMMARY")
print("=" * 90)

df_results = pd.DataFrame(all_results)

# Pivot: model × split for F1
print("\n── F1 Score (macro) ──")
pivot_f1 = df_results.pivot(index="model", columns="split", values="f1")
print(pivot_f1.to_string(float_format="%.4f"))

print("\n── AUC-ROC ──")
pivot_auc = df_results.pivot(index="model", columns="split", values="auc_roc")
print(pivot_auc.to_string(float_format="%.4f"))

print("\n── Accuracy ──")
pivot_acc = df_results.pivot(index="model", columns="split", values="accuracy")
print(pivot_acc.to_string(float_format="%.4f"))

# Seen vs Unseen gap analysis
print("\n\n── Seen → Unseen Generalization Gap ──")
print(f"{'Model':<30} {'Seen F1':>10} {'Unseen F1':>10} {'Gap':>10} {'Better on Unseen?':>20}")
print("─" * 85)
for model_name in df_results["model"].unique():
    model_data = df_results[df_results["model"] == model_name]
    seen = model_data[model_data["split"] == "test_seen"]
    unseen = model_data[model_data["split"] == "test_unseen"]
    if len(seen) > 0 and len(unseen) > 0:
        seen_f1 = seen["f1"].values[0]
        unseen_f1 = unseen["f1"].values[0]
        gap = unseen_f1 - seen_f1
        better = "✅ YES" if gap > 0 else "❌ no"
        print(f"{model_name:<30} {seen_f1:>10.4f} {unseen_f1:>10.4f} {gap:>+10.4f} {better:>20}")

# Save results
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "subgroup_analysis_results.csv")
df_results.to_csv(output_path, index=False)
print(f"\n💾 Results saved to: {output_path}")

print("\n" + "=" * 70)
print("  ✅ Subgroup analysis complete!")
print("=" * 70)
