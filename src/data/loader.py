"""
Data loading utilities: read JSONL/CSV, build dictionaries, create DataLoaders.
"""

import json
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer
from typing import Tuple, Dict, List, Optional

from src.config import ExperimentConfig
from src.data.dataset import (
    HatefulMemesDataset,
    get_image_transform,
    get_train_augment_transform,
    compute_max_token_length,
)


# ── I/O helpers ─────────────────────────────────────────────────────────────

def read_jsonl(file_path: str) -> List[dict]:
    """Read a JSONL file into a list of dicts."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_jsonl_data(
    json_dir: str,
    files: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load one or more JSONL files and concatenate into a DataFrame.

    Args:
        json_dir: Directory containing JSONL files.
        files: List of filenames to load. If None, loads only 'train.jsonl'.
    """
    if files is None:
        files = ["train.jsonl"]

    dfs = []
    for fname in files:
        path = os.path.join(json_dir, fname)
        if os.path.exists(path):
            records = read_jsonl(path)
            dfs.append(pd.DataFrame(records))
            print(f"  Loaded {path}: {len(records)} samples")
        else:
            print(f"  WARNING: {path} not found, skipping.")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    df = pd.read_csv(csv_path)
    print(f"  Loaded {csv_path}: {len(df)} samples")
    return df


# ── DataFrame → dicts ──────────────────────────────────────────────────────

def df_to_dicts(
    df: pd.DataFrame,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, int]]:
    """
    Convert a DataFrame with columns [id, img, text, label]
    into three dictionaries keyed by string id.
    """
    # Ensure id is string for consistent keying
    df = df.copy()
    df["id"] = df["id"].astype(str)

    image_paths = dict(zip(df["id"], df["img"]))
    texts = dict(zip(df["id"], df["text"]))
    labels = dict(zip(df["id"], df["label"].astype(int)))
    return image_paths, texts, labels


# ── Balance dataset ─────────────────────────────────────────────────────────

def balance_dataset(
    df: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Undersample the majority class so both classes have equal counts.
    """
    counts = df["label"].value_counts()
    minority_count = counts.min()
    majority_label = counts.idxmax()

    majority = df[df["label"] == majority_label]
    minority = df[df["label"] != majority_label]

    majority_downsampled = majority.sample(n=minority_count, random_state=random_state)
    balanced = pd.concat([majority_downsampled, minority], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"  Balanced dataset: {len(balanced)} samples "
          f"({dict(balanced['label'].value_counts())})")
    return balanced


# ── Compute class weights ───────────────────────────────────────────────────

def compute_class_weights(labels: Dict[str, int], device: str = "cpu") -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    label_vals = list(labels.values())
    counts = np.bincount(label_vals)
    weights = 1.0 / counts.astype(np.float32)
    weights = weights / weights.sum() * len(weights)  # normalize
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ── Master loader builder ──────────────────────────────────────────────────

def build_dataloaders(
    config: ExperimentConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    End-to-end pipeline: load data → build dataset → split → return loaders.

    Returns:
        train_loader, val_loader, test_loader, max_token_length
    """
    print("\n📦 Loading data...")
    tc = config.training
    pc = config.paths

    # ── Step 1: Load raw data ───────────────────────────────────────────
    if tc.use_augmented_data:
        csv_path = os.path.join(pc.csv_dir, "shuffled_training_data.csv")
        df = load_csv_data(csv_path)
    else:
        # Load all JSONL splits and concatenate (like original Balanced_Data notebook)
        df = load_jsonl_data(
            pc.json_dir,
            files=["train.jsonl", "dev_seen.jsonl", "dev_unseen.jsonl",
                   "test_seen.jsonl", "test_unseen.jsonl"],
        )
        # Balance by undersampling majority class
        df = balance_dataset(df, random_state=tc.random_seed)

    # ── Step 2: Build dicts ─────────────────────────────────────────────
    image_paths, texts, labels = df_to_dicts(df)

    # ── Step 3: Tokenizer + max length ──────────────────────────────────
    print("\n🔤 Initializing tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config.model.bert_model_name)
    max_length = compute_max_token_length(list(texts.values()), tokenizer, cap=128)
    print(f"  Max token length: {max_length}")

    # ── Step 4: Build dataset ───────────────────────────────────────────
    transform_eval = get_image_transform()
    dataset = HatefulMemesDataset(
        image_paths=image_paths,
        texts=texts,
        labels=labels,
        image_root=pc.image_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        transform=transform_eval,  # will override for train split below
    )

    # ── Step 5: Split ───────────────────────────────────────────────────
    total = len(dataset)
    train_size = int(tc.train_ratio * total)
    val_size = int(tc.val_ratio * total)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(tc.random_seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], generator=generator,
    )
    print(f"\n📊 Split: Train={train_size}  Val={val_size}  Test={test_size}")

    # ── Step 6: DataLoaders ─────────────────────────────────────────────
    # Determine num_workers based on environment
    num_workers = 0 if pc.environment in ("colab", "kaggle") else tc.num_workers

    train_loader = DataLoader(
        train_ds,
        batch_size=tc.train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=tc.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tc.val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=tc.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=tc.test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=tc.pin_memory,
    )

    return train_loader, val_loader, test_loader, max_length
