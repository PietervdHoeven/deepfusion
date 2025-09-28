import math
import torch

import torch.nn as nn
import torch.nn.functional as F

class AxialBlock(nn.Module):
    """
    One axial block = Spatial MSA (within direction over S) + Sequence MSA (across directions over Q) + FFN.

    Shapes:
      - Input H: (B, Q, S, embed_dim)
      - Spatial MSA runs on (B*Q, S, embed_dim)
      - Sequence MSA runs on (B*S, Q, embed_dim)
      - FFN is pointwise per token
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_mult: int = 4, attn_dropout=0.1, ffn_dropout=0.1):
        super().__init__()
        self.msa_spatial = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
        self.msa_seq = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_mult * embed_dim),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_mult * embed_dim, embed_dim),
            nn.Dropout(ffn_dropout),
        )

    def forward(
        self,
        H: torch.Tensor,                                # (B, Q, S, embed_dim)
        padding_mask: torch.Tensor | None = None  # (B, Q) bool; True = PAD along sequence axis
    ) -> torch.Tensor:
        B, Q, S, embed_dim = H.shape

        # 1) Spatial self-attention (within each direction), over S (prenorm)
        x = H.view(B * Q, S, embed_dim)
        x_ln = self.ln1(x)
        z, _ = self.msa_spatial(x_ln, x_ln, x_ln, need_weights=False)
        x = x + z
        H = x.view(B, Q, S, embed_dim)

        # 2) Sequence self-attention (across directions), over Q (prenorm)
        x = H.permute(0, 2, 1, 3).contiguous()
        x = x.view(B * S, Q, embed_dim)
        x_ln = self.ln2(x)

        padding_mask = padding_mask[:, None, :].expand(B, S, Q)
        padding_mask = padding_mask.reshape(B * S, Q)

        z, _ = self.msa_seq(x_ln, x_ln, x_ln, key_padding_mask=padding_mask, need_weights=False)
        x = x + z
        H = x.view(B, S, Q, embed_dim).permute(0, 2, 1, 3).contiguous()

        # 3) Pointwise FFN with residual (prenorm)
        y = self.ffn(self.ln3(H))
        H = H + y
        return H


class AttentionPool(nn.Module):
    """Single-query attention pooling over (Q * S) tokens to get a single vector per sample."""
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()

        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.query, std=0.02)

        self.mha = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=0.0, batch_first=True)

    def forward(self, H: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        H: (B, Q, S, embed_dim)
        pad_mask: (B, Q) bool; True = PAD
        returns: (B, embed_dim)
        """
        B, Q, S, embed_dim = H.shape
        L = Q * S
        H = H.reshape(B, L, embed_dim)

        q = self.query.expand(B, 1, embed_dim)

        padding_mask = padding_mask.unsqueeze(-1)
        padding_mask = padding_mask.expand(B, Q, S)
        padding_mask = padding_mask.reshape(B, L)

        out, _ = self.mha(q, H, H, key_padding_mask=padding_mask)
        out = out.squeeze(1)

        return out