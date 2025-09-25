import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from deepfusion.utils.losses import masked_recon_loss

# ============================================================
# Multi-Head Self-Attention (MSA) with dropout and padding mask
# ============================================================
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


# ============================================================
# Axial (Double) Attention Block: Spatial MSA + Sequence MSA + FFN
# ============================================================
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

        # ---- 1) Spatial self-attention (within each direction), over S ----
        x = H.view(B * Q, S, d)                 # (B*Q, S, d)
        z = self.msa_spatial(x)                 # (B*Q, S, d)
        x = self.ln1(x + z)                     # residual + LN
        H = x.view(B, Q, S, d)                  # (B, Q, S, d)

        # ---- 2) Sequence self-attention (across directions), over Q ----
        x = H.permute(0, 2, 1, 3).contiguous()  # (B, S, Q, d)  swap (Q,S)
        x = x.view(B * S, Q, d)                 # (B*S, Q, d)

        # Repeat the sequence padding mask over S (so it matches (B*S, Q))
        if seq_key_padding_mask is not None:
            kpm = seq_key_padding_mask[:, None, :].expand(B, S, Q)  # (B,S,Q)
            kpm = kpm.reshape(B * S, Q)                             # (B*S, Q)
        else:
            kpm = None

        z = self.msa_seq(x, key_padding_mask=kpm)   # (B*S, Q, d)
        x = self.ln2(x + z)                         # residual + LN
        H = x.view(B, S, Q, d).permute(0, 2, 1, 3).contiguous()  # (B, Q, S, d)

        # ---- 3) Pointwise FFN with residual ----
        y = self.ffn(H)                             # (B, Q, S, d)
        H = self.ln3(H + y)                         # (B, Q, S, d)
        return H


# ============================================================
# Minimal End-to-End Axial Model (with dropout, no bias)
# ============================================================
class AxialMaskedLatentModel(nn.Module):
    """
    End-to-end model for masked latent reconstruction.

    Pipeline:
      1) Input projection: X ∈ R^{B,Q,S,C} -> H ∈ R^{B,Q,S,d} (Section 2a)
      2) Add spatial positional embeddings P_s ∈ R^{S,d} (Section 2b)
      3) Add gradient embeddings E_g(g_q) ∈ R^{B,Q,d} broadcast over S (Section 2c)
      4) Replace masked directions with learned [MASK] token (optional)
      5) Stack of axial blocks (spatial MSA → sequence MSA → FFN)^N
      6) Decode back to channel space: H -> X_hat ∈ R^{B,Q,S,C} (Section 4)

    Dropout is already handled inside MSA and FFN.
    """
    def __init__(self, C: int = 512, d: int = 256, H: int = 8, S: int = 36, N: int = 6,
                 attn_dropout=0.1, proj_dropout=0.1, ffn_dropout=0.1):
        super().__init__()
        self.S = S
        self.d = d

        # 2(a) Input projection C -> d
        self.proj_in  = nn.Linear(C, d)  # W_in
        # 4) Output projection d -> C
        self.proj_out = nn.Linear(d, C)  # W_dec

        # 2(b) Learned spatial positional embeddings P_s for s=1..S
        self.spatial_pe = nn.Parameter(torch.zeros(S, d))
        nn.init.normal_(self.spatial_pe, std=0.02)

        # 2(c) Gradient embedding E_g: g ∈ R^4 -> R^d
        self.grad_mlp = nn.Sequential(
            nn.Linear(4, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        # Learned [MASK] token (replaces entire directions after embeddings)
        self.mask_token = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.mask_token, std=0.02)

        # Axial blocks
        self.blocks = nn.ModuleList([
            AxialBlock(d=d, H=H, ffn_mult=4, attn_dropout=attn_dropout,
                       proj_dropout=proj_dropout, ffn_dropout=ffn_dropout)
            for _ in range(N)
        ])

    def forward(
        self,
        X: torch.Tensor,                         # (B, Q, S, C)
        g: torch.Tensor,                         # (B, Q, 4)   [bval, bvec]
        dir_mask: torch.Tensor | None = None,    # (B, Q) bool; True = masked directions
        seq_key_padding_mask: torch.Tensor | None = None  # (B, Q) bool; True = PAD
    ) -> torch.Tensor:
        B, Q, S, C = X.shape
        assert S == self.S, f"expected S={self.S}, got {S}"

        # ---- 2(a) Project channels C -> d ----
        H = self.proj_in(X)                                # (B, Q, S, d)

        # ---- 2(b) Add spatial positional embeddings P_s ----
        H = H + self.spatial_pe.view(1, 1, S, self.d)      # (B, Q, S, d)

        # ---- 2(c) Gradient embedding E_g(g_q) ----
        G = self.grad_mlp(g)                                # (B, Q, d)
        H = H + G.unsqueeze(2)                               # broadcast over S -> (B, Q, S, d)

        # ---- Optional masking: replace whole directions with [MASK] ----
        if dir_mask is not None:
            # dir_mask (B,Q) -> (B,Q,S,d)
            m = dir_mask.view(B, Q, 1, 1).expand(B, Q, S, 1)
            mask_tok = self.mask_token.view(1, 1, 1, self.d).expand(B, Q, S, self.d)
            H = torch.where(m, mask_tok, H)                 # (B, Q, S, d)

        # ---- Axial blocks (Spatial MSA -> Sequence MSA -> FFN) ----
        for blk in self.blocks:
            H = blk(H, seq_key_padding_mask=seq_key_padding_mask)  # (B, Q, S, d)

        # ---- Decode d -> C per token ----
        X_hat = self.proj_out(H)                             # (B, Q, S, C)
        return X_hat



# ===== Dummy forward pass =====
B, Q, S, C = 2, 5, 36, 512

# Random latent maps: (B, Q, S, C)
X = torch.randn(B, Q, S, C)

# Gradient info per direction: (B, Q, 4)   (e.g., [bval, bx, by, bz])
g = torch.randn(B, Q, 4)

# Mask two random directions for reconstruction loss
dir_mask = torch.zeros(B, Q, dtype=torch.bool)
dir_mask[0, 2] = True   # mask direction 2 in sample 0
dir_mask[1, 4] = True   # mask direction 4 in sample 1

# Key padding mask: pretend sample 1 only has Q=3 real directions, rest is PAD
seq_key_padding_mask = torch.zeros(B, Q, dtype=torch.bool)
seq_key_padding_mask[1, 3:] = True   # mark last 2 as PAD for sample 1

# Instantiate the model
model = AxialMaskedLatentModel(
    C=512, d=128, H=4, S=S, N=2,   # smaller d for demo
    attn_dropout=0.1, proj_dropout=0.1, ffn_dropout=0.1
)

# Forward pass
X_hat = model(
    X, g,
    dir_mask=dir_mask,
    seq_key_padding_mask=seq_key_padding_mask
)

print("Input X shape:    ", X.shape)      # (2, 5, 36, 512)
print("Grad g shape:     ", g.shape)      # (2, 5, 4)
print("Dir mask shape:   ", dir_mask.shape)  # (2, 5)
print("Pad mask shape:   ", seq_key_padding_mask.shape)  # (2, 5)
print("Output X_hat shape:", X_hat.shape)  # (2, 5, 36, 512)

# Compute masked reconstruction loss
loss = masked_recon_loss(X_hat, X, dir_mask)
print("Masked recon loss:", loss.item())
