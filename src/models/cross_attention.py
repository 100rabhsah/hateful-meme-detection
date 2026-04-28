"""
Cross-modal attention module.
Bidirectional attention between text token sequences and image patch sequences.
"""

import torch
import torch.nn as nn


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
