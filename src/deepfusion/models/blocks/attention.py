import math
import torch

import torch.nn as nn
import torch.nn.functional as F

class MSA(nn.Module):
    """
    Multi-head self-attention operating on sequences of shape (N, L, d).

    Math:
      - Split model dim d across H heads: d_h = d / H
      - Linear projections (shared, then reshaped into heads):
          Q = X W_Q,  K = X W_K,  V = X W_V
      - Scaled dot-product attention (per head):
          scores = (Q K^T) / sqrt(d_h)                       # (N, H, L, L)
          scores[pad_keys] = -inf                             # apply key padding mask
          A = softmax(scores, dim=-1)                         # attention weights
          O_head = A @ V                                      # (N, H, L, d_h)
      - Merge heads + output projection:
          O = concat_h(O_head) W_O                            # (N, L, d) -> (N, L, d)

    Dropout:
      - Attention dropout on A
      - Projection dropout after output projection

    Args:
      d (int): model dimension
      H (int): number of heads (d must be divisible by H)
      attn_dropout (float): dropout on attention weights
      proj_dropout (float): dropout after output projection
    """
    def __init__(self, d: int, H: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        assert d % H == 0, "d must be divisible by H"
        self.d  = d
        self.H  = H
        self.dh = d // H  # per-head dimension d_h = d / H

        # Q, K, V projections: (N,L,d) -> (N,L,d)
        self.q_proj = nn.Linear(d, d, bias=True)
        self.k_proj = nn.Linear(d, d, bias=True)
        self.v_proj = nn.Linear(d, d, bias=True)

        # Output projection back to d
        self.o_proj = nn.Linear(d, d, bias=True)

        # Dropouts
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(
        self,
        x: torch.Tensor,                       # (N, L, d)
        key_padding_mask: torch.Tensor | None = None  # (N, L) bool; True = PAD (mask out as keys)
    ) -> torch.Tensor:
        N, L, d = x.shape  # N=batch-like axis, L=sequence length, d=model dim

        # --- Linear projections (shared, then split into heads) ---
        # q,k,v: (N, L, d)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to heads: (N, L, d) -> (N, H, L, d_h)
        q = q.view(N, L, self.H, self.dh).transpose(1, 2)  # (N, H, L, d_h)
        k = k.view(N, L, self.H, self.dh).transpose(1, 2)  # (N, H, L, d_h)
        v = v.view(N, L, self.H, self.dh).transpose(1, 2)  # (N, H, L, d_h)

        # --- Scaled dot-product attention (per head) ---
        # scores = (q @ k^T) / sqrt(d_h): (N,H,L,d_h) x (N,H,d_h,L) -> (N,H,L,L)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)  # (N,H,L,L)

        # Apply key padding mask on KEYS (mask == True means PAD -> disallow attention to that key)
        if key_padding_mask is not None:
            # key_padding_mask: (N, L) -> (N, 1, 1, L) to broadcast over heads & queries
            mask = key_padding_mask[:, None, None, :]  # True where PAD
            scores = scores.masked_fill(mask, float("-inf"))

        # Softmax over last dim (keys), then attention dropout
        attn = F.softmax(scores, dim=-1)  # (N,H,L,L)
        attn = self.attn_drop(attn)

        # Weighted sum over values: (N,H,L,L) @ (N,H,L,d_h) -> (N,H,L,d_h)
        out = torch.matmul(attn, v)  # (N,H,L,d_h)

        # Merge heads: (N,H,L,d_h) -> (N,L,H*d_h=d)
        out = out.transpose(1, 2).contiguous().view(N, L, d)  # (N,L,d)

        # Output projection + projection dropout: (N,L,d) -> (N,L,d)
        out = self.o_proj(out)
        out = self.proj_drop(out)
        return out
    

class AxialBlock(nn.Module):
    """
    One axial block = Spatial MSA (within direction over S) + Sequence MSA (across directions over Q) + FFN.

    Shapes:
      - Input H: (B, Q, S, d)
      - Spatial MSA runs on (B*Q, S, d)      # Section 3(a) in math
      - Sequence MSA runs on (B*S, Q, d)     # Section 3(b) in math
      - FFN is pointwise per token           # Section 3(c) in math

    Dropout locations:
      - Inside MSA: attention dropout + projection dropout
      - Inside FFN: dropout between linear layers (and optionally after second linear)
    """
    def __init__(self, d: int, H: int, ffn_mult: int = 4, attn_dropout=0.1, proj_dropout=0.1, ffn_dropout=0.1):
        super().__init__()
        self.msa_spatial = MSA(d, H, attn_dropout=attn_dropout, proj_dropout=proj_dropout)  # over S
        self.msa_seq     = MSA(d, H, attn_dropout=attn_dropout, proj_dropout=proj_dropout)  # over Q

        self.ln1 = nn.LayerNorm(d)  # after spatial residual
        self.ln2 = nn.LayerNorm(d)  # after sequence residual
        self.ln3 = nn.LayerNorm(d)  # after FFN residual

        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_mult * d),
            nn.GELU(),
            nn.Dropout(ffn_dropout),          # FFN dropout between layers
            nn.Linear(ffn_mult * d, d),
            nn.Dropout(ffn_dropout),          # (optional) after second linear
        )

    def forward(
        self,
        H: torch.Tensor,                                # (B, Q, S, d)
        seq_key_padding_mask: torch.Tensor | None = None  # (B, Q) bool; True = PAD along sequence axis
    ) -> torch.Tensor:
        B, Q, S, d = H.shape

        #  1) Spatial self-attention (within each direction), over S 
        x = H.view(B * Q, S, d)                 # (B*Q, S, d)
        z = self.msa_spatial(x)                 # (B*Q, S, d)
        x = self.ln1(x + z)                     # residual + LN
        H = x.view(B, Q, S, d)                  # (B, Q, S, d)

        #  2) Sequence self-attention (across directions), over Q 
        x = H.permute(0, 2, 1, 3).contiguous()  # (B, S, Q, d)  swap (Q,S)
        x = x.view(B * S, Q, d)                 # (B*S, Q, d)

        # Repeat the sequence padding mask over S (so it matches (B*S, Q))
        if seq_key_padding_mask is not None:
            kpm = seq_key_padding_mask[:, None, :].expand(B, S, Q)  # (B,S,Q)   Insert S dim, repeat mask over S
            kpm = kpm.reshape(B * S, Q)                             # (B*S, Q)
        else:
            kpm = None

        z = self.msa_seq(x, key_padding_mask=kpm)   # (B*S, Q, d)
        x = self.ln2(x + z)                         # residual + LN
        H = x.view(B, S, Q, d).permute(0, 2, 1, 3).contiguous()  # (B, Q, S, d) swap back

        #  3) Pointwise FFN with residual 
        y = self.ffn(H)                             # (B, Q, S, d)
        H = self.ln3(H + y)                         # (B, Q, S, d)
        return H


class AttentionPool(nn.Module):
    """Single-query attention pooling over (Q * S) tokens to get a single vector per sample."""
    def __init__(self, d: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, d))  # (1, 1, d)
        nn.init.normal_(self.query, std=0.02)

        self.mha = nn.MultiheadAttention(d, num_heads=8, dropout=0.0, batch_first=True)  # 8 heads, no dropout

    def forward(self, H: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        H: (B, Q, S, d)
        pad_mask: (B, Q) bool; True = PAD
        returns: (B, d)
        """
        B, Q, S, d = H.shape
        L = Q * S
        H = H.reshape(B, L, d)  # (B, L, d)

        # Repeat query for each batch item: (1,1,d) -> (B,1,d)
        q = self.query.expand(B, 1, d)  # (B, 1, d)

        # Prepare key padding mask for (B,L): repeat pad_mask over S
        pad_mask = pad_mask.unsqueeze(-1)       # (B,Q,1)   insert S dim
        pad_mask = pad_mask.expand(B, Q, S)     # (B,Q,S)   repeat mask over S
        pad_mask = pad_mask.reshape(B, L)       # (B,L)     flatten Q and S

        # MHA with single query: (B,1,d) x (B,L,d) -> (B,1,d)
        out, _ = self.mha(q, H, H, key_padding_mask=pad_mask)  # (B,1,d)
        out = out.squeeze(1)                      # (B,d)

        return out