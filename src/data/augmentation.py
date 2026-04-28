"""
Offline data augmentation pipeline.
Creates augmented images on disk and produces an augmented CSV.
Uses albumentations for image augmentations.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Optional

from src.config import ExperimentConfig

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


def build_augmentation_pipeline(config: ExperimentConfig) -> "A.Compose":
    """Build the albumentations augmentation pipeline from config."""
    if not HAS_ALBUMENTATIONS:
        raise ImportError(
            "albumentations is required for offline augmentation. "
            "Install with: pip install albumentations"
        )
    ac = config.augmentation
    return A.Compose([
        A.HorizontalFlip(p=ac.horizontal_flip_p),
        A.Rotate(limit=ac.rotate_limit, p=ac.rotate_p),
        A.RandomBrightnessContrast(p=ac.brightness_contrast_p),
        A.GaussianBlur(p=ac.gaussian_blur_p),
    ])


def run_augmentation(config: ExperimentConfig) -> pd.DataFrame:
    """
    Augment the minority class in train.jsonl and save augmented images + CSV.

    Steps:
        1. Load train.jsonl
        2. Filter minority class samples
        3. Apply albumentations to each image
        4. Save augmented images to {data_root}/augmented_img/
        5. Create and save combined CSV (original + augmented)

    Returns:
        The combined DataFrame.
    """
    from src.data.loader import load_jsonl_data

    ac = config.augmentation
    pc = config.paths

    print("\n🔄 Running offline data augmentation...")

    # Load original data
    df = load_jsonl_data(pc.json_dir, files=["train.jsonl"])

    # Build augmentation pipeline
    aug_pipeline = build_augmentation_pipeline(config)

    # Create output directory for augmented images
    aug_img_dir = os.path.join(pc.data_root, "augmented_img")
    os.makedirs(aug_img_dir, exist_ok=True)

    augmented_records = []
    augmented_count = 0
    target_class = ac.target_class

    print(f"  Original class distribution:\n{df['label'].value_counts().to_string()}")
    print(f"  Augmenting class {target_class}, target: {ac.num_augmented_samples} samples")

    for _, row in df.iterrows():
        if row["label"] != target_class:
            continue
        if augmented_count >= ac.num_augmented_samples:
            break

        img_path = os.path.join(pc.image_dir, row["img"])
        if not os.path.exists(img_path):
            continue

        img = np.array(Image.open(img_path))
        augmented = aug_pipeline(image=img)["image"]
        augmented_image = Image.fromarray(augmented)

        # Save
        new_id = f"aug_{row['id']}_{augmented_count}"
        new_img_name = f"augmented_img/aug_{row['id']}_{augmented_count}.png"
        augmented_image.save(os.path.join(pc.data_root, new_img_name))

        augmented_records.append({
            "id": new_id,
            "img": new_img_name,
            "label": row["label"],
            "text": row["text"],
        })
        augmented_count += 1

    # Combine and shuffle
    aug_df = pd.DataFrame(augmented_records)
    combined = pd.concat([df, aug_df], ignore_index=True)
    shuffled = combined.sample(frac=1, random_state=config.training.random_seed).reset_index(drop=True)

    # Save CSVs
    combined.to_csv(os.path.join(pc.csv_dir, "updated_training_data.csv"), index=False)
    shuffled.to_csv(os.path.join(pc.csv_dir, "shuffled_training_data.csv"), index=False)

    print(f"  ✅ Augmented {augmented_count} images")
    print(f"  Final class distribution:\n{shuffled['label'].value_counts().to_string()}")
    print(f"  Saved to: {pc.csv_dir}")

    return shuffled
