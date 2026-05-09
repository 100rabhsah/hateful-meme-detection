"""
Cross-modal attention modules.

Contains:
    1. CrossModalAttention      — Standard single-path cross-attention (original)
    2. DualPathCrossAttention   — Novel dual-path: alignment + incongruity branches

The dual-path module explicitly models both *agreement* and *conflict*
between modalities. This is motivated by the observation that hateful
memes frequently exploit the **incongruity** between benign text and
harmful imagery (or vice versa). Standard cross-attention only captures
alignment; the incongruity branch is trained to focus on mismatched
regions via a decorrelation auxiliary loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    Multi-head cross-attention between two modalities.

    Given query tokens from one modality and key/value tokens from another,
    this computes cross-attention — allowing each token in the query modality
    to attend to all tokens in the key/value modality.

    Input shapes (batch_first=False convention for nn.MultiheadAttention):
        query: [seq_len_q, batch_size, embed_dim]
        key:   [seq_len_kv, batch_size, embed_dim]
        value: [seq_len_kv, batch_size, embed_dim]

    Output shape: [seq_len_q, batch_size, embed_dim]
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [seq_q, B, D]
            key:   [seq_kv, B, D]
            value: [seq_kv, B, D]
            key_padding_mask: [B, seq_kv] — True for positions to ignore.

        Returns:
            Attended output: [seq_q, B, D]
        """
        attn_output, _ = self.cross_attn(
            query=query, key=key, value=value,
            key_padding_mask=key_padding_mask,
        )
        # Residual connection + LayerNorm
        output = self.layer_norm(query + self.dropout(attn_output))
        return output


# ═══════════════════════════════════════════════════════════════════════════
#  NOVEL: Dual-Path Cross-Attention (Alignment + Incongruity)
# ═══════════════════════════════════════════════════════════════════════════

class DualPathCrossAttention(nn.Module):
    """
    Dual-path cross-attention with explicit alignment and incongruity modelling.

    Architecture:
        ┌─────────────────────┐
        │   Alignment Branch  │ → learns where text and image AGREE
        │  (standard X-attn)  │
        └─────────┬───────────┘
                  │
        query ────┤
                  │
        ┌─────────┴───────────┐
        │  Incongruity Branch │ → learns where text and image CONFLICT
        │  (inverted X-attn)  │
        └─────────┬───────────┘
                  │
        output = align_out + λ · incon_out

    The incongruity branch uses a separate set of projection weights
    and is trained with an auxiliary **decorrelation loss** that minimizes
    cosine similarity between the query and attended output, forcing it
    to focus on mismatched/conflicting regions.

    This is the key architectural novelty: explicitly modelling humor/hate
    as contrast between modalities, not just similarity.

    Args:
        embed_dim:  Dimension of input embeddings.
        num_heads:  Number of attention heads in each branch.
        dropout:    Dropout probability.
        lambda_init: Initial value for the incongruity weighting factor λ.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        lambda_init: float = 0.5,
    ):
        super().__init__()

        # ── Alignment branch ───────────────────────────────────────────
        self.align_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.align_norm = nn.LayerNorm(embed_dim)
        self.align_dropout = nn.Dropout(dropout)

        # ── Incongruity branch ─────────────────────────────────────────
        # Separate projection for keys/values to learn a different subspace
        self.incon_key_proj = nn.Linear(embed_dim, embed_dim)
        self.incon_val_proj = nn.Linear(embed_dim, embed_dim)
        self.incon_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.incon_norm = nn.LayerNorm(embed_dim)
        self.incon_dropout = nn.Dropout(dropout)

        # ── Learnable incongruity weight λ ─────────────────────────────
        # Initialized at lambda_init; learned during training
        self._lambda = nn.Parameter(torch.tensor(lambda_init))

        # ── Gating layer: learns how to fuse the two branches ──────────
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )

    @property
    def lambda_value(self) -> float:
        """Current value of the incongruity weighting factor."""
        return self._lambda.item()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> tuple:
        """
        Dual-path forward pass.

        Args:
            query: [seq_q, B, D]
            key:   [seq_kv, B, D]
            value: [seq_kv, B, D]
            key_padding_mask: [B, seq_kv] — True for positions to ignore.

        Returns:
            output:      [seq_q, B, D]  — Fused alignment + incongruity
            align_out:   [seq_q, B, D]  — Alignment branch output (for loss)
            incon_out:   [seq_q, B, D]  — Incongruity branch output (for loss)
        """
        # ── Alignment branch: standard cross-attention ─────────────────
        align_attn_out, _ = self.align_attn(
            query=query, key=key, value=value,
            key_padding_mask=key_padding_mask,
        )
        align_out = self.align_norm(query + self.align_dropout(align_attn_out))

        # ── Incongruity branch: cross-attention in a different subspace ─
        incon_key = self.incon_key_proj(key)
        incon_val = self.incon_val_proj(value)
        incon_attn_out, _ = self.incon_attn(
            query=query, key=incon_key, value=incon_val,
            key_padding_mask=key_padding_mask,
        )
        incon_out = self.incon_norm(query + self.incon_dropout(incon_attn_out))

        # ── Gated fusion ───────────────────────────────────────────────
        # Instead of simple addition, use a learned gate to combine
        gate_input = torch.cat([align_out, incon_out], dim=-1)  # [S, B, 2D]
        gate_weight = self.gate(gate_input)                      # [S, B, D] ∈ (0,1)

        # gate_weight → how much to weight alignment vs incongruity
        output = gate_weight * align_out + (1 - gate_weight) * (self._lambda * incon_out)

        return output, align_out, incon_out


def compute_incongruity_loss(
    align_out: torch.Tensor,
    incon_out: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    """
    Auxiliary loss to enforce the dual-path separation:

    1. **Alignment similarity**: maximize cosine similarity between
       alignment output and the original query (they should agree).
    2. **Incongruity decorrelation**: minimize cosine similarity between
       incongruity output and the original query (they should disagree).

    The total loss encourages the alignment branch to find agreement
    and the incongruity branch to find conflict.

    Args:
        align_out: [S, B, D] — alignment branch output
        incon_out: [S, B, D] — incongruity branch output
        query:     [S, B, D] — original query input

    Returns:
        Scalar loss value.
    """
    # Mean-pool over sequence dimension → [B, D]
    align_pooled = align_out.mean(dim=0)
    incon_pooled = incon_out.mean(dim=0)
    query_pooled = query.mean(dim=0)

    # Alignment branch should be similar to query
    align_sim = F.cosine_similarity(align_pooled, query_pooled, dim=-1).mean()

    # Incongruity branch should be different from query
    incon_sim = F.cosine_similarity(incon_pooled, query_pooled, dim=-1).mean()

    # Additionally: the two branches should be decorrelated from each other
    branch_sim = F.cosine_similarity(align_pooled, incon_pooled, dim=-1).mean()

    # Loss = -align_similarity + incongruity_similarity + branch_correlation
    # We want: high align_sim, low incon_sim, low branch_sim
    loss = -align_sim + incon_sim + branch_sim

    return loss
