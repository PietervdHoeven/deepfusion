import torch

def collate_selfsupervised(batch):
    """
    batch: list of (X_i, G_i)
      - X_i: [L_i, C, D, H, W]  (latent maps per volume)
      - G_i: [L_i, 4]           (side info per volume)

    Returns:
      X: [B, L_max, C, D, H, W]
      G: [B, L_max, 4]
      attn_mask: [B, L_max] (bool)
    """

    Ls = [x.shape[0] for x, _ in batch] # list of sequence lengths
    L_max = max(Ls) # max sequence length in the batch

    # Extract shapes
    B = len(batch)
    C, D, H, W = batch[0][0].shape[1:]

    # Prepare padded tensors
    X = torch.zeros(B, L_max, C, D, H, W, dtype=batch[0][0].dtype)
    G = torch.zeros(B, L_max, 4, dtype=batch[0][1].dtype)
    attn_mask = torch.zeros(B, L_max, dtype=torch.bool)  # False = ignore, True = pay attention

    for b,(x,g) in enumerate(batch):
        L = x.shape[0]
        X[b, :L] = x    # broadcast (D,H,W) over L
        G[b, :L] = g    # (4) over L
        attn_mask[b, :L] = True # True = pay attention, not padding

    return X, G, attn_mask

def collate_finetune(batch):
    """
    batch: list of (X_i, G_i, y_i)
      - X_i: [L_i, C, D, H, W]  (latent maps per volume)
      - G_i: [L_i, 4]           (side info per volume)
      - y_i: int or float       (label per volume)

    Returns:
      X: [B, L_max, C, D, H, W]
      G: [B, L_max, 4]
      attn_mask: [B, L_max] (bool)
      y: [B] (int or float)
    """

    Ls = [x.shape[0] for x, _, _ in batch] # list of sequence lengths
    L_max = max(Ls) # max sequence length in the batch

    # Extract shapes
    B = len(batch)
    C, D, H, W = batch[0][0].shape[1:]

    # Prepare padded tensors
    X = torch.zeros(B, L_max, C, D, H, W, dtype=batch[0][0].dtype)
    G = torch.zeros(B, L_max, 4, dtype=batch[0][1].dtype)
    attn_mask = torch.zeros(B, L_max, dtype=torch.bool)  # False = ignore, True = pay attention
    y_dtype = torch.long if isinstance(batch[0][2], int) else torch.float
    y = torch.zeros(B, dtype=y_dtype)

    for b,(x,g,label) in enumerate(batch):
        L = x.shape[0]
        X[b, :L] = x    # broadcast (D,H,W) over L
        G[b, :L] = g    # (4) over L
        attn_mask[b, :L] = True # True = pay attention, not padding
        y[b] = label

    return X, G, attn_mask, y
