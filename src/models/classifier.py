"""
Hateful Meme Classifier using BERT token embeddings + ViT patch tokens
with bidirectional cross-modal attention.

Supports three modes:
    1. CLS-only baseline (use_sequence_tokens=False, use_dual_path=False)
    2. Full sequence cross-attention (use_sequence_tokens=True, use_dual_path=False)
    3. Dual-path cross-attention (use_sequence_tokens=True, use_dual_path=True)  [NOVEL]

Architecture (dual-path mode):
    BERT(text) → [CLS, tok1, tok2, ..., tokN]  → Project to embed_dim
    ViT(image) → [CLS, patch1, patch2, ..., patchM] → Project to embed_dim

    Dual-Path Cross-Attention (per direction):
        Alignment branch  → captures text-image agreement
        Incongruity branch → captures text-image conflict
        Gated fusion combines both branches

    Pool attended sequences → Concatenate → MLP → Binary classification
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

    Supports three modes controlled by config:
        use_sequence_tokens=True, use_dual_path=False → Full token-level cross-attention
        use_sequence_tokens=False                     → CLS-only baseline
        use_dual_path=True                            → Dual-path (alignment + incongruity)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.use_sequence = config.use_sequence_tokens
        self.use_dual_path = config.use_dual_path

        # ── Pretrained backbones ────────────────────────────────────────
        self.bert = BertModel.from_pretrained(config.bert_model_name)
        self.vit = ViTModel.from_pretrained(config.vit_model_name)

        # ── Projection heads: backbone_dim (768) → embed_dim (128) ─────
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
            # NOVEL: Dual-path cross-attention
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
            # Standard single-path cross-attention
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

        if self.use_sequence:
            if self.use_dual_path:
                return self._forward_dual_path(bert_out, vit_out, attention_mask)
            else:
                return self._forward_sequence(bert_out, vit_out, attention_mask)
        else:
            return self._forward_cls(bert_out, vit_out)

    def _forward_dual_path(self, bert_out, vit_out, attention_mask):
        """
        Dual-path cross-attention (NOVEL).

        Each direction (text→image, image→text) has two branches:
            - Alignment: captures where modalities agree
            - Incongruity: captures where modalities conflict

        An auxiliary decorrelation loss is computed to enforce branch separation.
        """
        # Project all tokens
        text_tokens = self.text_projection(bert_out.last_hidden_state)    # [B, S_t, D]
        image_tokens = self.image_projection(vit_out.last_hidden_state)   # [B, S_i, D]

        # Transpose to [S, B, D] for nn.MultiheadAttention
        text_tokens_t = text_tokens.permute(1, 0, 2)    # [S_t, B, D]
        image_tokens_t = image_tokens.permute(1, 0, 2)  # [S_i, B, D]

        # Key padding mask for text
        text_key_padding_mask = (attention_mask == 0)  # [B, S_t]

        # ── Dual-path cross-attention: text → image ────────────────────
        text_attended, t2i_align, t2i_incon = self.text_to_image_attn(
            query=text_tokens_t,
            key=image_tokens_t,
            value=image_tokens_t,
            key_padding_mask=None,
        )

        # ── Dual-path cross-attention: image → text ────────────────────
        image_attended, i2t_align, i2t_incon = self.image_to_text_attn(
            query=image_tokens_t,
            key=text_tokens_t,
            value=text_tokens_t,
            key_padding_mask=text_key_padding_mask,
        )

        # ── Compute auxiliary incongruity loss ─────────────────────────
        incon_loss_t2i = compute_incongruity_loss(
            t2i_align, t2i_incon, text_tokens_t,
        )
        incon_loss_i2t = compute_incongruity_loss(
            i2t_align, i2t_incon, image_tokens_t,
        )
        total_incon_loss = (incon_loss_t2i + incon_loss_i2t) / 2.0

        # ── Pool and classify ──────────────────────────────────────────
        text_attended = text_attended.permute(1, 0, 2)   # [B, S_t, D]
        text_mask = attention_mask.unsqueeze(-1).float()  # [B, S_t, 1]
        text_pooled = (text_attended * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp(min=1)

        image_attended = image_attended.permute(1, 0, 2)  # [B, S_i, D]
        image_pooled = image_attended.mean(dim=1)          # [B, D]

        combined = torch.cat([text_pooled, image_pooled], dim=1)  # [B, 2*D]
        logits = self.classifier(combined)

        return {
            'logits': logits,
            'incongruity_loss': total_incon_loss,
        }

    def _forward_sequence(self, bert_out, vit_out, attention_mask):
        """
        Full token-level cross-attention.

        BERT: last_hidden_state → [B, seq_text, 768] → project → [B, seq_text, 128]
        ViT:  last_hidden_state → [B, num_patches+1, 768] → project → [B, num_patches+1, 128]
        Cross-attend both directions, then mean-pool each attended sequence.
        """
        # Project all tokens
        text_tokens = self.text_projection(bert_out.last_hidden_state)    # [B, S_t, D]
        image_tokens = self.image_projection(vit_out.last_hidden_state)   # [B, S_i, D]

        # Transpose to [S, B, D] for nn.MultiheadAttention
        text_tokens_t = text_tokens.permute(1, 0, 2)    # [S_t, B, D]
        image_tokens_t = image_tokens.permute(1, 0, 2)  # [S_i, B, D]

        # Key padding mask for text (True = ignore pad tokens)
        text_key_padding_mask = (attention_mask == 0)  # [B, S_t]

        # Cross-attention: text attending to image patches
        text_attended = self.text_to_image_attn(
            query=text_tokens_t,
            key=image_tokens_t,
            value=image_tokens_t,
            key_padding_mask=None,  # No padding in ViT patches
        )  # [S_t, B, D]

        # Cross-attention: image patches attending to text tokens
        image_attended = self.image_to_text_attn(
            query=image_tokens_t,
            key=text_tokens_t,
            value=text_tokens_t,
            key_padding_mask=text_key_padding_mask,
        )  # [S_i, B, D]

        # Mean-pool over sequence dimension (masking pad tokens for text)
        # text_attended: [S_t, B, D] → [B, D]
        text_attended = text_attended.permute(1, 0, 2)   # [B, S_t, D]
        text_mask = attention_mask.unsqueeze(-1).float()  # [B, S_t, 1]
        text_pooled = (text_attended * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp(min=1)

        # image_attended: [S_i, B, D] → [B, D]
        image_attended = image_attended.permute(1, 0, 2)  # [B, S_i, D]
        image_pooled = image_attended.mean(dim=1)          # [B, D]

        # Fuse and classify
        combined = torch.cat([text_pooled, image_pooled], dim=1)  # [B, 2*D]
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
