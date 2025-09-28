from torch import nn
import torch

from deepfusion.models.blocks.attention import AxialBlock, AttentionPool

class AxialMaskedModellingTransformer(nn.Module):
    """
    End-to-end model for masked latent reconstruction.

    Args:
        in_channels (int): Number of input channels per token.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        num_spatials (int): Number of spatial positions.
        num_layers (int): Number of axial blocks.
        attn_dropout (float): Dropout rate for attention layers.
        proj_dropout (float): Dropout rate for projection layers.
        ffn_dropout (float): Dropout rate for feed-forward layers.
    """
    def __init__(
        self,
        in_channels: int = 384,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_spatials: int = 36,
        num_layers: int = 6,
        attn_dropout: float = 0.02,
        ffn_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_spatials = num_spatials
        self.embed_dim = embed_dim

        # Input projection: in_channels -> embed_dim
        self.proj_in = nn.Linear(in_channels, embed_dim)

        # Learned spatial positional embeddings
        self.spatial_pe = nn.Parameter(torch.zeros(num_spatials, embed_dim))
        nn.init.normal_(self.spatial_pe, std=0.02)

        # Gradient embedding: g ∈ R^4 -> R^embed_dim
        self.mlp_grad = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Learned [MASK] token
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Axial blocks
        self.blocks = nn.ModuleList([
            AxialBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_mult=4,
                attn_dropout=attn_dropout,
                ffn_dropout=ffn_dropout,
            )
            for _ in range(num_layers)
        ])

        # Output projection: embed_dim -> in_channels
        self.ln_out = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Linear(embed_dim, in_channels)

    def forward(
        self,
        X: torch.Tensor,                                # (B, Q, S, in_channels)
        g: torch.Tensor,                                # (B, Q, 4)
        Q_mask: torch.Tensor | None = None,             # (B, Q) bool
        padding_mask: torch.Tensor | None = None        # (B, Q) bool
    ) -> torch.Tensor:
        B, Q, S, C = X.shape
        assert S == self.num_spatials, f"expected S={self.num_spatials}, got {S}"

        # Input projection
        H = self.proj_in(X)

        # Masking
        Q_mask = Q_mask.view(B, Q, 1, 1)
        M = self.mask_token.view(1, 1, 1, self.embed_dim)
        H = torch.where(Q_mask, M, H)

        # Positional embedding
        P = self.spatial_pe.view(1, 1, S, self.embed_dim)

        # Gradient embedding
        G = self.mlp_grad(g).unsqueeze(2)

        # Add embeddings
        H = H + P + G

        # Axial blocks
        for block in self.blocks:
            H = block(H, padding_mask)

        # Output projection
        X_hat = self.proj_out(self.ln_out(H))

        return X_hat, H


class AxialPredictingTransformer(nn.Module):
    """
    Wraps a pretrained transformer backbone to produce pooled features.

    Args:
        backbone (nn.Module): Transformer backbone module that outputs hidden states H of shape (B, Q, S, embed_dim).
        embed_dim (int): Embedding dimension of the backbone output.
        num_heads (int, optional): Number of attention heads for pooling. Default is 8.
    """
    def __init__(
        self,
        in_channels: int = 384,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_spatials: int = 36,
        num_layers: int = 6,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
    ):
        super().__init__()
        self.transformer = AxialMaskedModellingTransformer(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_spatials=num_spatials,
            num_layers=num_layers,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
        )
        self.embed_dim = embed_dim
        self.pool = AttentionPool(embed_dim=embed_dim, num_heads=num_heads)

    def forward(
        self,
        X: torch.Tensor,                                # (B, Q, S, in_channels)
        g: torch.Tensor,                                # (B, Q, 4)
        Q_mask: torch.Tensor | None = None,             # (B, Q) bool
        padding_mask: torch.Tensor | None = None        # (B, Q) bool
    ) -> torch.Tensor:
        _, H = self.transformer(X, g, Q_mask, padding_mask)
        z = self.pool(H, padding_mask)
        return z
