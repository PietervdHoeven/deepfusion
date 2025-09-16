from deepfusion.models.autoencoders import ConvBlock3D, Downsample3D, Upsample3D
from deepfusion.models.blocks import Embedder
import torch.nn as nn
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

class ZMapEncoder(nn.Module):
    def __init__(self, in_ch=256, grid=(4,4,4), out_dim=512):
        super().__init__()
        self.conv1 = ConvBlock3D(in_ch, 384, kernel_size=3, stride=1, padding=1)
        self.down1 = Downsample3D(384, 384, "learned")                     # 4->2
        self.conv2 = ConvBlock3D(384, 512, kernel_size=3, stride=1, padding=1)    # 2->1
        self.down2 = Downsample3D(512, 512, "learned")                     # 2->1
        self.proj  = nn.Linear(512, out_dim)
    
    def forward(self, x):
        x = self.conv1(x)                   # [B,384,4,4,4]
        x = self.down1(x)                   # [B,384,2,2,2]
        x = self.conv2(x)                   # [B,512,2,2,2]
        x = self.down2(x)                   # [B,512,1,1,1]
        x = x.flatten(1)                    # [B,512]
        token = self.proj(x)                # [B,out_dim]
        return token


class ZMapDecoder(nn.Module):
    def __init__(self, in_dim=512, out_ch=256, grid=(4,4,4)):
        super().__init__()
        self.proj = nn.Linear(in_dim, 512)
        self.conv1 = ConvBlock3D(512, 384, kernel_size=3, stride=1, padding=1)
        self.up1   = Upsample3D(384, 384, "learned")                       # 1->2
        self.conv2 = ConvBlock3D(384, 256, kernel_size=3, stride=1, padding=1)  # 2->2
        self.up2   = Upsample3D(256, out_ch, "learned")                    # 2->4

    def forward(self, x):
        x = self.proj(x)                    # [B,512]
        x = x[:, :, None, None, None]       # [B,512,1,1,1]
        x = self.conv1(x)                   # [B,384,1,1,1]
        x = self.up1(x)                     # [B,384,2,2,2]
        x = self.conv2(x)                   # [B,256,2,2,2]
        x = self.up2(x)                     # [B,256,4,4,4]
        return x
    

class Embedder(nn.Module):
    def __init__(self, d_enc=256, d_model=384):
        super().__init__()
        self.proj_z = nn.Linear(d_enc, d_model, bias=False)
        self.proj_g = nn.Linear(4,     d_model, bias=True)
        self.z_mask = nn.Parameter(torch.zeros(d_model))  # [d_model]
        self.norm   = nn.LayerNorm(d_model)

    def forward(self, z, g, mdm_mask=None):
        """
        z: [B, L, d_enc]
        g: [B, L, 4]
        mdm_mask: [B, L] bool or None
        """
        z_emb = self.proj_z(z)        # [B, L, d_model]
        g_emb = self.proj_g(g)        # [B, L, d_model]

        if mdm_mask is not None:
            mask = mdm_mask.unsqueeze(-1)         # [B, L, 1]
            z_emb = torch.where(
                mask,                             # condition
                self.z_mask.view(1, 1, -1),       # replacement [1,1,d_model] -> broadcast
                z_emb                             # original
            )                                    # [B, L, d_model]

        token = z_emb + g_emb                     # [B, L, d_model]
        return self.norm(token)
    

class DeepFusion(pl.LightningModule):
    def __init__(
            self,
            # model params
            in_ch: int = 256,
            dim_model: int = 384,
            num_head: int = 6,
            dim_feedforward: int = 1536,
            dropout: float = 0.1,
            activation: str = 'gelu',
            num_layers: int = 6,
            # training params
            mask_prob: float = 0.5,
            lr: float = 1e-4,
            weight_decay: float = 0.05,
            betas: tuple = (0.9, 0.95)
            ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = ZMapEncoder(in_ch=in_ch, out_dim=dim_model)
        self.embedder = Embedder(d_enc=dim_model, d_model=dim_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_model))
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
        self.unembedder = nn.Linear(dim_model, dim_model)
        self.decoder = ZMapDecoder(in_dim=dim_model, out_ch=in_ch)


    def forward(self, x, g, attn_mask, mdm_mask=None):
        device = x.device
        B, L, C, D, H, W = x.shape
        if mdm_mask is None:
            mdm_mask = torch.zeros(B, L, dtype=torch.bool, device=device)  # no masking

        # 1) pack volumes that are visible (not masked, not padding) into (N_vis, C, D, H, W) -> Encode -> Reshape back to (B, L, d_enc)
        vis_mask = attn_mask.to(device) & (~mdm_mask)
        flat_vis_mask = vis_mask.view(B * L)                            # [B*L] bool   
        vis_idx = flat_vis_mask.nonzero(as_tuple=False).squeeze(1)      # [N_vis] indices of visible tokens in masked sequence
        x_flat = x.view(B * L, C, D, H, W)                              # [B*L, C, D, H, W] Stack all volumes in batch
        x_vis = x_flat.index_select(0, vis_idx)                         # [N_vis, C, D, H, W] Select only visible volumes
        z_vis = self.encoder(x_vis)                                     # [N_vis, d_enc] Encode visible volumes
        z_flat = torch.zeros(B * L, z_vis.shape[1], device=device, dtype=z_vis.dtype)   # [B*L, d_enc] Prepare flat latent tensor as destination for index_copy
        z_flat = torch.index_copy(z_flat, 0, vis_idx, z_vis)            # out-of-place index_copy keeps autograd graph
        z = z_flat.view(B, L, -1)                                       # [B, L, d_enc] Reshape back to (B,L,d_enc)

        # 2) Embed latents and gradients + prepend [CLS] token
        g = g.to(device)
        t = self.embedder(z, g, mdm_mask)
        cls = self.cls_token.expand(B, 1, -1).to(device)
        t = torch.cat([cls, t], dim=1)
        cls_pad = torch.ones(B, 1, dtype=torch.bool, device=device)
        attn_mask = torch.cat([cls_pad, attn_mask.to(device)], dim=1)

        # 3) Transformer Encoder
        h_cls = self.transformer(t, src_key_padding_mask=~attn_mask)
        h = h_cls[:, 1:, :]

        # 4) Predict latents for masked tokens only, then decode to volumes
        attn_mdm_mask = (attn_mask[:, 1:] & mdm_mask)
        if attn_mdm_mask.any():
            h_mask = h[attn_mdm_mask]
            z_pred = self.unembedder(h_mask)
            x_pred = self.decoder(z_pred)
            x_true = x[attn_mdm_mask]
        else:
            x_pred = x_true = None

        return x_pred, x_true
    
    def training_step(self, batch, batch_idx):
        x, g, attn_mask = batch # x: [B,L,C,D,H,W], g: [B,L,4], attn_mask: [B,L] bool
        mdm_mask = (torch.rand_like(attn_mask, dtype=torch.float) < self.hparams.mask_prob) & attn_mask   # Sample a mask based on mask_prob but only for valid tokens where attn_mask == 1

        x_pred, x_true = self(x, g, attn_mask, mdm_mask)

        loss = self._ssl_loss(x_pred, x_true)

        self.log(f"train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, g, attn_mask = batch
        gen = torch.Generator(device=attn_mask.device).manual_seed(0)  # fixed seed
        mdm_mask = (torch.rand(attn_mask.shape, generator=gen, device=attn_mask.device) < self.hparams.mask_prob) & attn_mask

        x_pred, x_true = self(x, g, attn_mask, mdm_mask)
        loss = self._ssl_loss(x_pred, x_true)
        self.log(f"val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, g, attn_mask = batch
        gen = torch.Generator(device=attn_mask.device).manual_seed(0)  # fixed seed
        mdm_mask = (torch.rand(attn_mask.shape, generator=gen, device=attn_mask.device) < self.hparams.mask_prob) & attn_mask

        x_pred, x_true = self(x, g, attn_mask, mdm_mask)
        loss = self._ssl_loss(x_pred, x_true)
        self.log(f"test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def _ssl_loss(self, x_pred, x_true):
        if x_pred is not None and x_true is not None:
            return F.l1_loss(x_pred, x_true)
        else:
            return torch.tensor(0.0, device=x_pred.device, dtype=x_pred.dtype)
        
    def configure_optimizers(self):
        # split params into decay / no-decay (biases & norms)
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay,    "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
            betas=(0.9, 0.95),
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = max(1, int(0.05 * total_steps))
        decay_steps  = max(1, total_steps - warmup_steps)

        warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=self.hparams.lr * 0.01)

        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step"
            }
        }



# # ---- 1) init model ----
# model = DeepFusion(
#     in_ch=256,
#     dim_model=384,
#     num_head=6,
#     dim_feedforward=1536,
#     num_layers=6,        # keep small for quick test
#     mask_prob=0.5,
# ).cuda()

# # ---- 2) dummy batch ----
# B, L, C, D, H, W = 20, 100, 256, 4, 4, 4   # 2 patients, 70 volumes each
# x = torch.randn(B, L, C, D, H, W).cuda()        # diffusion volumes
# g = torch.randn(B, L, 4).cuda()              # gradient embeddings (bval+bvec)

# # attend_mask: 1 = attend, 0 = pad
# attend_mask = torch.ones(B, L, dtype=torch.bool)  # all valid here

# # ---- 3) forward pass ----
# for _ in range(100):
#     loss, x_pred, x_true = model(x, g, attend_mask)

#     print("loss:", loss.item())
#     print("x_pred:", None if x_pred is None else x_pred.shape)
#     print("x_true:", None if x_true is None else x_true.shape)