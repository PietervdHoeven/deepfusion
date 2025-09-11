import pytorch_lightning as pl
from deepfusion.models.encoders import ResEncoder, BasicEncoder
from deepfusion.models.decoder import Decoder
from deepfusion.models.blocks import Embedder
import torch.nn as nn
import torch
import torch.nn.functional as F


class DeepFusion(pl.LightningModule):
    def __init__(
            self,
            in_ch: int = 1,
            base_ch: int = 16,
            dim_enc: int = 256,
            dim_model: int = 384,
            num_head: int = 6,
            dim_feedforward: int = 1536,
            dropout: float = 0.1,
            activation: str = 'gelu',
            num_layers: int = 6,
            # training flags
            mdm: bool = True,               # masked diffusion modeling (randomly mask input tokens during training)
            mask_prob: float = 0.5,         # probability of masking each token when mdm=True
            lr: float = 1e-4,
            weight_decay: float = 1e-5,
    ):
        super().__init__()
        # Model backbone
        self.encoder = BasicEncoder(in_ch=in_ch, base=base_ch, out_dim=dim_enc, downsample="max")
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_model))  # [1, 1, d_model]
        self.embedder = Embedder(d_enc=dim_enc, d_model=dim_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model,
                nhead=num_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                norm_first=True,
                batch_first=True,
            ),
            num_layers=num_layers
        )
        self.unembedder = nn.Linear(dim_model, dim_enc)
        self.decoder = Decoder(d_enc=dim_enc, base=base_ch, out_ch=in_ch)

        # Training parameters
        self.mdm = mdm
        self.mask_prob = mask_prob
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x, g, attn_mask):
        """
        x: (B, L, D, H, W)
        g: (B, L, 4)
        attn_mask: (B, L)
        """
        B, L, D, H, W = x.shape
        x = x.unsqueeze(2)  # (B, L, 1, D, H, W) add channel dim

        # 1) sample a masked diffusion modelling mask (mdm_mask) and a visible to the model mask (vis_mask)
        mdm_mask = (torch.rand(B, L, device=x.device) < self.mask_prob)     # [B, L] bool: True = mask, False = keep
        vis_mask = attn_mask & (~mdm_mask)                                  # [B, L] bool: True = visible, False = masked or padding

        # 2) pack volumes that are visible (not masked, not padding) into (N_vis, 1, D, H, W) -> Encode -> Reshape back to (B, L, d_enc)
        flat_vis_mask = vis_mask.view(B * L)                                            # [B*L]
        vis_idx = flat_vis_mask.nonzero(as_tuple=False).squeeze(1)                      # [N_vis]   indices of visible tokens in flattened batch
        x_flat = x.view(B * L, 1, D, H, W)                                              # [B*L, 1, D, H, W]
        x_vis = x_flat.index_select(0, vis_idx)                                         # [N_vis, 1, D, H, W]   visible volumes only
        z_vis = self.encoder(x_vis)                                                     # [N_vis, d_enc]    encoded visible volumes
        z_flat = torch.zeros(B * L, z_vis.shape[1], device=x.device, dtype=z_vis.dtype) # [B*L, d_enc]    zero initialize all latents
        z_flat.index_copy_(0, vis_idx, z_vis)                                           # copy z_flat[vis_idx] = z_vis
        z = z_flat.view(B, L, -1)                                                       # [B, L, d_enc]   reshape back to (B, L, d_enc)

        # 3) Embed latents and gradients + prepend [CLS] token
        t = self.embedder(z, g, mdm_mask if self.mdm else None)         # [B, L, d_model]   Combine latent or [mask] and gradient embeddings
        cls = self.cls_token.expand(B, 1, -1)                           # [B, 1, d_model]   CLS token, same for all in batch
        t = torch.cat([cls, t], dim=1)                                  # [B, L+1, d_model]  prepend CLS token
        cls_pad = torch.ones(B, 1, dtype=torch.bool, device=x.device)   # [B, 1] True = [CLS] padding
        attn_mask = torch.cat([cls_pad, attn_mask], dim=1)              # [B, L+1] True = pay attention, False = ignore (we pay attention to [CLS])

        # 4) Transformer Encoder
        h = self.transformer(t, src_key_padding_mask=~attn_mask)  # [B, L+1, d_model]   CLS attends to all visible tokens, visible tokens attend to CLS and other visible tokens
        h = h[:, 1:, :]                                           # [B, L, d_model]   discard CLS token for pretraining

        # 5) Predict latents for masked tokens only, then decode to volumes
        attn_mdm_mask = (attn_mask[:, 1:] & mdm_mask)    # [B, L] True = attended and masked, False = visible or padding
        if attn_mdm_mask.any():
            h_mask = h[attn_mdm_mask]      # [N_mdm, d_model]   only the masked tokens (not visible, not padding, not CLS)
            z_pred = self.unembedder(h_mask)  # [N_mdm, d_enc]   predict latents for masked tokens
            x_pred = self.decoder(z_pred)  # [N_mdm, 1, 128,128,128]   decode predicted latents to volumes
            x_true = x[attn_mdm_mask]  # [N_mdm, 1, 128,128,128]   target true volumes for masked tokens
            loss = F.l1_loss(x_pred, x_true)  # L1 loss on masked tokens only
        else:
            loss = (h.sum() * 0.0)
            x_pred = x_true = None

        return loss, x_pred, x_true
    
# ---- 1) init model ----
model = DeepFusion(
    in_ch=1,
    base_ch=16,
    dim_enc=256,
    dim_model=384,
    num_head=6,
    dim_feedforward=1536,
    num_layers=2,        # keep small for quick test
    mask_prob=0.5,
)

# ---- 2) dummy batch ----
B, L, D, H, W = 1, 65, 128, 128, 128   # 2 patients, 70 volumes each
x = torch.randn(B, L, D, H, W)        # diffusion volumes
g = torch.randn(B, L, 4)              # gradient embeddings (bval+bvec)

# attend_mask: 1 = attend, 0 = pad
attend_mask = torch.ones(B, L, dtype=torch.bool)  # all valid here

# ---- 3) forward pass ----
for _ in range(10):
    loss, x_pred, x_true = model(x, g, attend_mask)

    print("loss:", loss.item())
    print("x_pred:", None if x_pred is None else x_pred.shape)
    print("x_true:", None if x_true is None else x_true.shape)
