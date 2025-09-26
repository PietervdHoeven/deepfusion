from typing import List, Tuple, Union
import torch

# def collate_selfsupervised(batch):
#     """
#     batch: list of (X_i, G_i)
#       - X_i: [L_i, C, D, H, W]  (latent maps per volume)
#       - G_i: [L_i, 4]           (side info per volume)

#     Returns:
#       X: [B, L_max, C, D, H, W]
#       G: [B, L_max, 4]
#       attn_mask: [B, L_max] (bool)
#     """

#     Ls = [x.shape[0] for x, _ in batch] # list of sequence lengths
#     L_max = max(Ls) # max sequence length in the batch

#     # Extract shapes
#     B = len(batch)
#     C, D, H, W = batch[0][0].shape[1:]

#     # Prepare padded tensors
#     X = torch.zeros(B, L_max, C, D, H, W, dtype=batch[0][0].dtype)
#     G = torch.zeros(B, L_max, 4, dtype=batch[0][1].dtype)
#     attn_mask = torch.zeros(B, L_max, dtype=torch.bool)  # False = ignore, True = pay attention

#     for b,(x,g) in enumerate(batch):
#         L = x.shape[0]
#         X[b, :L] = x    # broadcast (D,H,W) over L
#         G[b, :L] = g    # (4) over L
#         attn_mask[b, :L] = True # True = pay attention, not padding

#     return X, G, attn_mask

# def collate_finetune(batch):
#     """
#     batch: list of (X_i, G_i, y_i)
#       - X_i: [L_i, C, D, H, W]  (latent maps per volume)
#       - G_i: [L_i, 4]           (side info per volume)
#       - y_i: int or float       (label per volume)

#     Returns:
#       X: [B, L_max, C, D, H, W]
#       G: [B, L_max, 4]
#       attn_mask: [B, L_max] (bool)
#       y: [B] (int or float)
#     """

#     Ls = [x.shape[0] for x, _, _ in batch] # list of sequence lengths
#     L_max = max(Ls) # max sequence length in the batch

#     # Extract shapes
#     B = len(batch)
#     C, D, H, W = batch[0][0].shape[1:]

#     # Prepare padded tensors
#     X = torch.zeros(B, L_max, C, D, H, W, dtype=batch[0][0].dtype)
#     G = torch.zeros(B, L_max, 4, dtype=batch[0][1].dtype)
#     attn_mask = torch.zeros(B, L_max, dtype=torch.bool)  # False = ignore, True = pay attention
#     y_dtype = torch.long if isinstance(batch[0][2], int) else torch.float
#     y = torch.zeros(B, dtype=y_dtype)

#     for b,(x,g,label) in enumerate(batch):
#         L = x.shape[0]
#         X[b, :L] = x    # broadcast (D,H,W) over L
#         G[b, :L] = g    # (4) over L
#         attn_mask[b, :L] = True # True = pay attention, not padding
#         y[b] = label

#     return X, G, attn_mask, y


def collate_transformer(
    batch: List[Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    mask_ratio: float = 0.3,
):
    """
    Batch items:
      x: (Q_i, C, 3, 4, 3)
      g: (Q_i, 4)
      [y: ...]  # optional for downstream task, ignored here

    Returns:
      X:  (B, Q_max, S=36, C)  float32
      G:  (B, Q_max, 4)        float32
      dir_mask:            (B, Q_max) bool, True = masked directions for MLM loss
      seq_key_padding_mask:(B, Q_max) bool, True = PAD positions
      [y_batch if present]
    """
    B = len(batch)
    has_labels = (len(batch[0]) == 3)

    # 1) find maximum Q in this batch
    Qs = [item[0].shape[0] for item in batch]
    Q_max = max(Qs)

    # 2) shapes
    _, C, D, H, W = batch[0][0].shape
    assert (D, H, W) == (3, 4, 3), "Expected (3,4,3) spatial shape"
    S = D * H * W  # 36

    # 3) allocate padded tensors
    X = torch.zeros(B, Q_max, S, C, dtype=torch.float32)   # (B,Q,S,C)
    G = torch.zeros(B, Q_max, 4,   dtype=torch.float32)    # (B,Q,4)
    seq_key_padding_mask = torch.ones(B, Q_max, dtype=torch.bool)  # True = PAD
    dir_mask = torch.zeros(B, Q_max, dtype=torch.bool)             # True = MLM mask
    Ys = []  # optional labels

    for b, item in enumerate(batch):
        if has_labels:
            x, g, y = item    # x = (Q_i, C, 3,4,3), g = (Q_i, 4), y = scalar
            Ys.append(y)
        else:
            x, g = item

        Qi = x.shape[0]

        # ---- reshape x: (Q_i, C, 3,4,3) -> (Q_i, S=36, C) ----
        xi = x.view(Qi, C, S).permute(0, 2, 1).contiguous()

        # copy into padded batch
        X[b, :Qi] = xi
        G[b, :Qi] = g
        seq_key_padding_mask[b, :Qi] = False  # False = real token, True = PAD

        # build direction mask (random MLM on real tokens only)
        if mask_ratio > 0:
            num_mask = max(1, int(mask_ratio * Qi))
            mask_idx = torch.randperm(Qi)[:num_mask]
            dir_mask[b, mask_idx] = True

    if has_labels:
        Y = torch.stack(Ys, dim=0)
        return X, G, dir_mask, seq_key_padding_mask, Y
    else:
        return X, G, dir_mask, seq_key_padding_mask
