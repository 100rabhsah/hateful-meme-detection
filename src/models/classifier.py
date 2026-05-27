"""
Hateful Meme Classifier using BERT word embeddings + ViT patch embeddings
with bidirectional word-to-patch cross-modal attention.

Supports three modes:
    1. CLS-only baseline (use_word_patch_tokens=False, use_dual_path=False)
    2. Word-to-patch cross-attention (use_word_patch_tokens=True, use_dual_path=False)
    3. Dual-path word-to-patch cross-attention (use_word_patch_tokens=True, use_dual_path=True)  [NOVEL]

Architecture (word-to-patch / dual-path mode):
    BERT(text) → [CLS, word1, word2, ..., wordN, SEP, PAD...]
                   ↓ strip CLS → [word1, word2, ..., wordN, SEP, PAD...]
                   ↓ project  → [B, S_w, 128]

    ViT(image) → [CLS, patch1, patch2, ..., patch196]
                   ↓ strip CLS → [patch1, patch2, ..., patch196]
                   ↓ project  → [B, 196, 128]

    Word-to-Patch Cross-Attention (bidirectional):
        Each word (e.g., "they", "taking", "over") attends to all 196 image patches
        → captures which patches (faces, symbols, gestures, objects, demographic cues)
           are relevant to each word.

        Each image patch attends to all words
        → captures which words are relevant to each spatial region.

    Dual-Path mode additionally decomposes each direction into:
        Alignment branch  → captures word-patch agreement
        Incongruity branch → captures word-patch conflict

    Pool attended sequences → Concatenate → MLP → Binary classification

Dimensions:
    Word embedding:   768-dim (BERT) → 128-dim (projected), per WordPiece token
    Patch embedding:  768-dim (ViT)  → 128-dim (projected), per 16×16 pixel patch
    Number of patches: 14×14 = 196 (for 224×224 image with 16×16 patch size)
    Cross-attention:   [S_w × 196] attention matrix with 8 heads (per-head dim = 16)
"""

import torch
import torch.nn as nn
from transformers import BertModel, ViTModel

from src.models.cross_attention import (
    CrossModalAttention,
    DualPathCrossAttention,
    compute_incongruity_loss,
)
from src.config import ModelConfig


class HatefulMemesClassifier(nn.Module):
    """
    Multimodal classifier for hateful meme detection.

    Uses word-to-patch cross-attention where each word in the sentence
    independently attends to all image patches, and vice versa.

    Supports three modes controlled by config:
        use_word_patch_tokens=True, use_dual_path=False → Word-to-patch cross-attention
        use_word_patch_tokens=False                     → CLS-only baseline
        use_dual_path=True                              → Dual-path (alignment + incongruity)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.use_word_patch = config.use_word_patch_tokens
        self.use_dual_path = config.use_dual_path

        # ── Pretrained backbones ────────────────────────────────────────
        self.bert = BertModel.from_pretrained(config.bert_model_name)
        self.vit = ViTModel.from_pretrained(config.vit_model_name)

        # ── Projection heads: backbone_dim (768) → embed_dim (128) ─────
        # Applied per-word and per-patch independently
        self.text_projection = nn.Sequential(
            nn.Linear(config.bert_hidden_size, config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, config.embed_dim),
        )

        self.image_projection = nn.Sequential(
            nn.Linear(config.vit_hidden_size, config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, config.embed_dim),
        )

        # ── Cross-modal attention (bidirectional) ───────────────────────
        if self.use_dual_path:
            # NOVEL: Dual-path word-to-patch cross-attention
            self.text_to_image_attn = DualPathCrossAttention(
                embed_dim=config.embed_dim,
                num_heads=config.num_attention_heads,
                dropout=config.dropout,
                lambda_init=config.incongruity_lambda,
            )
            self.image_to_text_attn = DualPathCrossAttention(
                embed_dim=config.embed_dim,
                num_heads=config.num_attention_heads,
                dropout=config.dropout,
                lambda_init=config.incongruity_lambda,
            )
        else:
            # Standard single-path word-to-patch cross-attention
            self.text_to_image_attn = CrossModalAttention(
                embed_dim=config.embed_dim,
                num_heads=config.num_attention_heads,
                dropout=config.dropout,
            )
            self.image_to_text_attn = CrossModalAttention(
                embed_dim=config.embed_dim,
                num_heads=config.num_attention_heads,
                dropout=config.dropout,
            )

        # ── Classification head ─────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dim, config.num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,      # [B, seq_len]
        attention_mask: torch.Tensor,  # [B, seq_len]
        pixel_values: torch.Tensor,    # [B, 3, 224, 224]
    ) -> dict:
        """
        Forward pass.

        Returns:
            dict with:
                'logits': [B, num_classes]
                'incongruity_loss': scalar (only when use_dual_path=True)
        """
        # ── 1. Extract backbone features ───────────────────────────────
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        vit_out = self.vit(pixel_values=pixel_values)

        if self.use_word_patch:
            if self.use_dual_path:
                return self._forward_dual_path(bert_out, vit_out, attention_mask)
            else:
                return self._forward_word_patch(bert_out, vit_out, attention_mask)
        else:
            return self._forward_cls(bert_out, vit_out)

    def _forward_dual_path(self, bert_out, vit_out, attention_mask):
        """
        Dual-path word-to-patch cross-attention (NOVEL).

        Each direction (words→patches, patches→words) has two branches:
            - Alignment: captures where words and patches agree
            - Incongruity: captures where words and patches conflict

        CLS tokens are stripped from both BERT and ViT outputs so that
        cross-attention operates purely on words ↔ patches.

        Dimensions:
            Word embeddings:  [B, S_w, 768] → project → [B, S_w, 128]
            Patch embeddings: [B, 196, 768] → project → [B, 196, 128]
            S_w = seq_len - 1 (CLS stripped; SEP/PAD still present, handled by mask)

        An auxiliary decorrelation loss enforces branch separation.
        """
        # Strip CLS token (position 0) from both modalities
        # BERT: [CLS, word1, ..., wordN, SEP, PAD...] → [word1, ..., wordN, SEP, PAD...]
        word_hidden = bert_out.last_hidden_state[:, 1:, :]   # [B, S_w, 768]
        word_mask = attention_mask[:, 1:]                      # [B, S_w]

        # ViT: [CLS, patch1, ..., patch196] → [patch1, ..., patch196]
        patch_hidden = vit_out.last_hidden_state[:, 1:, :]    # [B, 196, 768]

        # Project per-word and per-patch: 768 → 128
        word_embeddings = self.text_projection(word_hidden)    # [B, S_w, 128]
        patch_embeddings = self.image_projection(patch_hidden) # [B, 196, 128]

        # Transpose to [S, B, D] for nn.MultiheadAttention
        word_seq = word_embeddings.permute(1, 0, 2)    # [S_w, B, 128]
        patch_seq = patch_embeddings.permute(1, 0, 2)  # [196, B, 128]

        # Key padding mask for words (True = ignore pad tokens)
        word_key_padding_mask = (word_mask == 0)  # [B, S_w]

        # ── Dual-path cross-attention: words → patches ─────────────────
        # Each word attends to all 196 image patches
        words_attended, t2i_align, t2i_incon = self.text_to_image_attn(
            query=word_seq,
            key=patch_seq,
            value=patch_seq,
            key_padding_mask=None,  # No padding in ViT patches
        )

        # ── Dual-path cross-attention: patches → words ─────────────────
        # Each image patch attends to all words
        patches_attended, i2t_align, i2t_incon = self.image_to_text_attn(
            query=patch_seq,
            key=word_seq,
            value=word_seq,
            key_padding_mask=word_key_padding_mask,
        )

        # ── Compute auxiliary incongruity loss ─────────────────────────
        incon_loss_t2i = compute_incongruity_loss(
            t2i_align, t2i_incon, word_seq,
        )
        incon_loss_i2t = compute_incongruity_loss(
            i2t_align, i2t_incon, patch_seq,
        )
        total_incon_loss = (incon_loss_t2i + incon_loss_i2t) / 2.0

        # ── Pool and classify ──────────────────────────────────────────
        words_attended = words_attended.permute(1, 0, 2)     # [B, S_w, 128]
        word_mask_expanded = word_mask.unsqueeze(-1).float()  # [B, S_w, 1]
        word_pooled = (words_attended * word_mask_expanded).sum(dim=1) / word_mask_expanded.sum(dim=1).clamp(min=1)

        patches_attended = patches_attended.permute(1, 0, 2)  # [B, 196, 128]
        patch_pooled = patches_attended.mean(dim=1)            # [B, 128]

        combined = torch.cat([word_pooled, patch_pooled], dim=1)  # [B, 256]
        logits = self.classifier(combined)

        return {
            'logits': logits,
            'incongruity_loss': total_incon_loss,
        }

    def _forward_word_patch(self, bert_out, vit_out, attention_mask):
        """
        Word-to-patch cross-attention.

        Each word in the sentence independently attends to all image patches,
        and each image patch attends to all words. This enables fine-grained
        cross-modal interaction where individual words (e.g., "they", "taking",
        "over") can separately attend to relevant image patches containing
        faces, symbols, gestures, objects, or demographic cues.

        CLS tokens are stripped from both BERT and ViT outputs.

        Dimensions:
            BERT words:  [B, S_w, 768] → project → [B, S_w, 128]
            ViT patches: [B, 196, 768] → project → [B, 196, 128]
              where S_w = seq_len - 1 (CLS removed), 196 = 14×14 patches
              Each patch corresponds to a 16×16 pixel region of the 224×224 image
        """
        # Strip CLS token (position 0) from both modalities
        # BERT: [CLS, word1, ..., wordN, SEP, PAD...] → [word1, ..., wordN, SEP, PAD...]
        word_hidden = bert_out.last_hidden_state[:, 1:, :]   # [B, S_w, 768]
        word_mask = attention_mask[:, 1:]                      # [B, S_w]

        # ViT: [CLS, patch1, ..., patch196] → [patch1, ..., patch196]
        patch_hidden = vit_out.last_hidden_state[:, 1:, :]    # [B, 196, 768]

        # Project per-word and per-patch: 768 → 128
        word_embeddings = self.text_projection(word_hidden)    # [B, S_w, 128]
        patch_embeddings = self.image_projection(patch_hidden) # [B, 196, 128]

        # Transpose to [S, B, D] for nn.MultiheadAttention
        word_seq = word_embeddings.permute(1, 0, 2)    # [S_w, B, 128]
        patch_seq = patch_embeddings.permute(1, 0, 2)  # [196, B, 128]

        # Key padding mask for words (True = ignore pad tokens)
        word_key_padding_mask = (word_mask == 0)  # [B, S_w]

        # Cross-attention: each word attends to all image patches
        words_attended = self.text_to_image_attn(
            query=word_seq,
            key=patch_seq,
            value=patch_seq,
            key_padding_mask=None,  # No padding in ViT patches
        )  # [S_w, B, 128]

        # Cross-attention: each image patch attends to all words
        patches_attended = self.image_to_text_attn(
            query=patch_seq,
            key=word_seq,
            value=word_seq,
            key_padding_mask=word_key_padding_mask,
        )  # [196, B, 128]

        # Mean-pool over words (masked) and patches
        # words_attended: [S_w, B, 128] → [B, 128]
        words_attended = words_attended.permute(1, 0, 2)      # [B, S_w, 128]
        word_mask_expanded = word_mask.unsqueeze(-1).float()   # [B, S_w, 1]
        word_pooled = (words_attended * word_mask_expanded).sum(dim=1) / word_mask_expanded.sum(dim=1).clamp(min=1)

        # patches_attended: [196, B, 128] → [B, 128]
        patches_attended = patches_attended.permute(1, 0, 2)   # [B, 196, 128]
        patch_pooled = patches_attended.mean(dim=1)             # [B, 128]

        # Fuse and classify
        combined = torch.cat([word_pooled, patch_pooled], dim=1)  # [B, 256]
        logits = self.classifier(combined)                         # [B, num_classes]
        return {'logits': logits, 'incongruity_loss': torch.tensor(0.0)}

    def _forward_cls(self, bert_out, vit_out):
        """
        CLS-only baseline (reproduces original notebook behavior).

        Uses only the [CLS] token from BERT and the [CLS] token from ViT.
        Cross-attention is applied on single vectors (unsqueezed to length-1 sequences).
        """
        # CLS tokens
        bert_cls = bert_out.last_hidden_state[:, 0, :]  # [B, 768]
        vit_cls = vit_out.last_hidden_state[:, 0, :]    # [B, 768]

        # Project
        text_emb = self.text_projection(bert_cls)   # [B, D]
        image_emb = self.image_projection(vit_cls)  # [B, D]

        # Unsqueeze for attention: [1, B, D]
        text_emb_t = text_emb.unsqueeze(0)
        image_emb_t = image_emb.unsqueeze(0)

        # Cross-attention
        text_attended = self.text_to_image_attn(
            query=text_emb_t, key=image_emb_t, value=image_emb_t,
        )
        image_attended = self.image_to_text_attn(
            query=image_emb_t, key=text_emb_t, value=text_emb_t,
        )

        # Handle both single-path (tensor) and dual-path (tuple) returns
        if isinstance(text_attended, tuple):
            text_attended = text_attended[0]
        if isinstance(image_attended, tuple):
            image_attended = image_attended[0]

        text_attended = text_attended.squeeze(0)    # [B, D]
        image_attended = image_attended.squeeze(0)  # [B, D]

        # Fuse and classify
        combined = torch.cat([text_attended, image_attended], dim=1)  # [B, 2*D]
        logits = self.classifier(combined)
        return {'logits': logits, 'incongruity_loss': torch.tensor(0.0)}
