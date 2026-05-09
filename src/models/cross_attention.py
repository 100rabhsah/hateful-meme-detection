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
        # Separate projections for query/key/value to learn a different subspace
        self.incon_query_proj = nn.Linear(embed_dim, embed_dim)
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
        # NOTE: NO residual connection here — forces the branch to learn
        # genuinely different (contrastive) representations instead of
        # collapsing to ≈ query via the skip connection.
        incon_query = self.incon_query_proj(query)
        incon_key = self.incon_key_proj(key)
        incon_val = self.incon_val_proj(value)
        incon_attn_out, _ = self.incon_attn(
            query=incon_query, key=incon_key, value=incon_val,
            key_padding_mask=key_padding_mask,
        )
        incon_out = self.incon_norm(self.incon_dropout(incon_attn_out))

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
    Barlow-Twins-style decorrelation loss to enforce dual-path separation.

    Computes the cross-covariance matrix between the alignment and
    incongruity branch outputs, then penalizes off-diagonal elements
    (redundancy) while encouraging on-diagonal elements to be zero
    (decorrelation). This provides much stronger gradient signal than
    cosine similarity, which was trivially satisfied by residual connections.

    Args:
        align_out: [S, B, D] — alignment branch output
        incon_out: [S, B, D] — incongruity branch output
        query:     [S, B, D] — original query input (unused, kept for API compat)

    Returns:
        Scalar loss value.
    """
    # Mean-pool over sequence dimension → [B, D]
    align_pooled = align_out.mean(dim=0)   # [B, D]
    incon_pooled = incon_out.mean(dim=0)   # [B, D]

    # Normalize each feature dimension across the batch (zero mean, unit variance)
    align_norm = (align_pooled - align_pooled.mean(dim=0, keepdim=True))
    align_std = align_norm.std(dim=0, keepdim=True).clamp(min=1e-6)
    align_norm = align_norm / align_std

    incon_norm = (incon_pooled - incon_pooled.mean(dim=0, keepdim=True))
    incon_std = incon_norm.std(dim=0, keepdim=True).clamp(min=1e-6)
    incon_norm = incon_norm / incon_std

    B = align_norm.shape[0]

    # Cross-correlation matrix [D, D]
    cross_corr = (align_norm.T @ incon_norm) / B

    # Barlow Twins loss: penalize all correlations between branches
    # On-diagonal: should be 0 (decorrelated) → penalize (diag)^2
    # Off-diagonal: should be 0 (no redundancy) → penalize (off-diag)^2
    loss = (cross_corr ** 2).mean()

    return loss
