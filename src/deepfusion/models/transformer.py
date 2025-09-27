from torch import nn
import torch

from deepfusion.models.blocks.attention import MSA, AxialBlock, AttentionPool

class AxialMaskedModellingTransformer(nn.Module):
    """
    End-to-end model for masked latent reconstruction.

    Args:
        C (int): Number of input channels per token.
        d (int): Embedding dimension.
        H (int): Number of attention heads.
        S (int): Number of spatial positions.
        N (int): Number of axial blocks.
        attn_dropout (float): Dropout rate for attention layers.
        proj_dropout (float): Dropout rate for projection layers.
        ffn_dropout (float): Dropout rate for feed-forward layers.

    Pipeline:
      1) Input projection: X ∈ ℝ^{B,Q,S,C} → H ∈ ℝ^{B,Q,S,d}
      2) Add spatial positional embeddings: P_s ∈ ℝ^{S,d}
      3) Add gradient embeddings: E_g(g_q) ∈ ℝ^{B,Q,d}, broadcast over S
      4) Replace masked directions with learned [MASK] token (optional)
      5) Stack of axial blocks: (spatial MSA → sequence MSA → FFN)^N
      6) Decode back to channel space: H → X_hat ∈ ℝ^{B,Q,S,C}

    Dropout is handled inside MSA and FFN.

    Maths:
        - Input: X ∈ ℝ^{B,Q,S,C}
        - Project: H = W_in X ∈ ℝ^{B,Q,S,d}
        - Add positional embeddings: H = H + P_s
        - Add gradient embeddings: H = H + E_g(g_q)
        - Masking: H[q] = [MASK] if Q_mask[q] is True
        - Axial blocks: H' = AxialBlock(H)
        - Output: X_hat = W_dec H' ∈ ℝ^{B,Q,S,C}
    """
    def __init__(self, C: int = 384, d: int = 256, H: int = 8, S: int = 36, N: int = 6,
                 attn_dropout=0.1, proj_dropout=0.1, ffn_dropout=0.1):
        super().__init__()
        self.S = S
        self.d = d

        # 2(a) Input projection C -> d
        self.proj_in  = nn.Linear(C, d)  # W_in

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

        # 3) Axial blocks
        self.blocks = nn.ModuleList([
            AxialBlock(d=d, H=H, ffn_mult=4, attn_dropout=attn_dropout,
                       proj_dropout=proj_dropout, ffn_dropout=ffn_dropout)
            for _ in range(N)
        ])

        # 4) Output projection d -> C
        self.proj_out = nn.Linear(d, C)  # W_dec

    def forward(
        self,
        X: torch.Tensor,                                # (B, Q, S, C)
        g: torch.Tensor,                                # (B, Q, 4)   [bval, bvec]
        Q_mask: torch.Tensor | None = None,             # (B, Q) bool; True = masked q-space direction
        padding_mask: torch.Tensor | None = None        # (B, Q) bool; True = PAD
    ) -> torch.Tensor:
        B, Q, S, C = X.shape
        assert S == self.S, f"expected S={self.S}, got {S}"

        #  1(a) Project channels C -> d 
        H = self.proj_in(X)                                # (B, Q, S, d)

        #  1(b) Add spatial positional embeddings P_s 
        H = H + self.spatial_pe.view(1, 1, S, self.d)      # (B, Q, S, d)

        #  1(c) Gradient embedding E_g(g_q) 
        G = self.grad_mlp(g)                                # (B, Q, d)
        H = H + G.unsqueeze(2)                               # broadcast over S -> (B, Q, S, d)

        #  Optional masking: replace whole directions with [MASK] 
        if Q_mask is not None:
            mask = Q_mask.view(B, Q, 1, 1).expand(B, Q, S, 1)                                # Mask is fixed for qs but broadcast over S and d
            mask_tok = self.mask_token.view(1, 1, 1, self.d).expand(B, Q, S, self.d)         # Expand mask token for all positions
            H = torch.where(mask, mask_tok, H)                 # (B, Q, S, d)                  Replace H with [MASK] where masked

        #  Axial blocks (Spatial MSA -> Sequence MSA -> FFN) 
        for blk in self.blocks:
            H = blk(H, seq_key_padding_mask=padding_mask)  # (B, Q, S, d)

        #  Decode d -> C per token 
        X_hat = self.proj_out(H)                             # (B, Q, S, C)
        return X_hat, H
    

class AxialPredictingTransformer(nn.Module):
    """
    Wraps a pretrained transformer backbone to produce pooled features.

    Args:
        backbone (nn.Module): Transformer backbone module that outputs hidden states H of shape (B, Q, S, d).
        d (int): Embedding dimension of the backbone output.
        heads (int, optional): Number of attention heads for pooling. Default is 8.

    Input:
        x (tuple): Tuple from the datamodule, either (X, G, Q_mask, pad_mask) or (X, G, pad_mask).
            - X (torch.Tensor): Input tensor of shape (B, Q, S, C).
            - G (torch.Tensor): Gradient tensor of shape (B, Q, 4).
            - Q_mask (torch.Tensor, optional): Mask tensor of shape (B, Q), True for masked directions.
            - pad_mask (torch.Tensor): Padding mask of shape (B, Q), True for padded tokens.

    Output:
        z (torch.Tensor): Pooled feature tensor of shape (B, d).

    Maths:
        - Backbone: H = backbone(X, G, Q_mask=None, padding_mask=pad_mask) ∈ ℝ^{B,Q,S,d}
        - Attention pooling: z = AttentionPool(H, pad_mask) ∈ ℝ^{B,d}
    """
    def __init__(self, heads: int = 8):
        super().__init__()
        self.transformer = AxialMaskedModellingTransformer()    # should output H: (B, Q, S, d)
        self.d = self.transformer.d
        self.pool = AttentionPool(self.d)

    def forward(
            self,
            X: torch.Tensor,                                # (B, Q, S, C)
            g: torch.Tensor,                                # (B, Q, 4)   [bval, bvec]
            Q_mask: torch.Tensor | None = None,             # (B, Q) bool; True = masked q-space direction
            padding_mask: torch.Tensor | None = None        # (B, Q) bool; True = PAD
    ) -> torch.Tensor:
        _, H = self.transformer(X, g, Q_mask, padding_mask)        # (B, Q, S, d)
        z = self.pool(H, padding_mask)                          # (B, d)
        return z
