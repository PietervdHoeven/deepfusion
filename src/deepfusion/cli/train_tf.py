import torch
from torch.utils.data import DataLoader

from deepfusion.datasets.deepfusion_dataset import DeepFusionDataset
from deepfusion.utils.collate import collate_deepfusion_pretrain
from deepfusion.models.transformer import AxialMaskedLatentModel
from deepfusion.utils.losses import masked_recon_loss
# from your module: AxialMaskedLatentModel, masked_recon_loss

train_loader = DataLoader(
    dataset=DeepFusionDataset(stage="train", task="pretraining"),
    batch_size=4,              # tune for VRAM; 4–8 is typical on 8GB with d~256, N~4–6
    shuffle=True,
    num_workers=12,
    pin_memory=True,
    collate_fn=lambda b: collate_deepfusion_pretrain(b, mask_ratio=0.35),
)

val_loader = DataLoader(
    dataset=DeepFusionDataset(stage="val", task="pretraining"),
    batch_size=6,
    shuffle=False,
    num_workers=12,
    pin_memory=True,
    collate_fn=lambda b: collate_deepfusion_pretrain(b, mask_ratio=0.35),
)

# Instantiate model for your shapes (C=384, S=36)
model = AxialMaskedLatentModel(
    C=384, d=256, H=8, S=36, N=6,     # safe defaults for 8GB; reduce d/N if tight
    attn_dropout=0.1, proj_dropout=0.1, ffn_dropout=0.1
).cuda()

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-2)

# one dummy iteration
model.train()
for X, G, dir_mask, pad_mask in train_loader:
    X = X.cuda(non_blocking=True)           # (B,Q,S,384)
    G = G.cuda(non_blocking=True)           # (B,Q,4)
    dir_mask = dir_mask.cuda(non_blocking=True)         # (B,Q)
    pad_mask = pad_mask.cuda(non_blocking=True)         # (B,Q)

    X_hat = model(X, G, dir_mask=dir_mask, seq_key_padding_mask=pad_mask)  # (B,Q,S,384)
    loss = masked_recon_loss(X_hat, X, dir_mask, lam_cos=0.1)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    print("loss:", float(loss))

model.eval()
with torch.no_grad():
    for X, G, dir_mask, pad_mask in val_loader:
        X = X.cuda(non_blocking=True)           # (B,Q,S,384)
        G = G.cuda(non_blocking=True)           # (B,Q,4)
        dir_mask = dir_mask.cuda(non_blocking=True)         # (B,Q)
        pad_mask = pad_mask.cuda(non_blocking=True)         # (B,Q)

        X_hat = model(X, G, dir_mask=dir_mask, seq_key_padding_mask=pad_mask)  # (B,Q,S,384)
        loss = masked_recon_loss(X_hat, X, dir_mask, lam_cos=0.1)
        print("val loss:", float(loss))

    