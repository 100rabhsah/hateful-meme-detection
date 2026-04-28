"""
Dataset class for hateful meme detection.
Handles image loading, BERT tokenization, and label retrieval.
Decoupled from global state — all paths passed explicitly.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
from typing import Dict, List, Optional


# ── Standard image transforms ──────────────────────────────────────────────
def get_image_transform(image_size: int = 224) -> transforms.Compose:
    """ImageNet-normalized transform for ViT input."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_train_augment_transform(image_size: int = 224) -> transforms.Compose:
    """Training-time augmentation (online, on-the-fly)."""
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ── Tokenizer helper ───────────────────────────────────────────────────────
def compute_max_token_length(
    texts: List[str],
    tokenizer: BertTokenizer,
    cap: int = 128,
) -> int:
    """Compute max token length across all texts, capped at `cap`."""
    max_len = 0
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        max_len = max(max_len, len(tokens))
    return min(max_len, cap)


# ── Dataset ─────────────────────────────────────────────────────────────────
class HatefulMemesDataset(Dataset):
    """
    PyTorch Dataset for the Facebook Hateful Memes challenge.

    Each sample returns:
        - pixel_values : Tensor [3, 224, 224]
        - input_ids    : Tensor [max_length]
        - attention_mask: Tensor [max_length]
        - label        : Tensor scalar (0 or 1)
    """

    def __init__(
        self,
        image_paths: Dict[str, str],   # id -> relative image path (e.g. "img/42953.png")
        texts: Dict[str, str],         # id -> meme text
        labels: Dict[str, int],        # id -> label
        image_root: str,               # absolute root to resolve image paths
        tokenizer: BertTokenizer,
        max_length: int,
        transform: Optional[transforms.Compose] = None,
    ):
        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform or get_image_transform()
        self.ids = list(image_paths.keys())

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        sample_id = self.ids[idx]

        # Image
        img_rel = self.image_paths[sample_id]
        img_path = os.path.join(self.image_root, img_rel)
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)

        # Text
        text = self.texts[sample_id]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Label
        label = torch.tensor(self.labels[sample_id], dtype=torch.long)

        return pixel_values, input_ids, attention_mask, label
