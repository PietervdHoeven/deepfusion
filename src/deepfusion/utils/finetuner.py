# src/models/build_finetuner.py
import torch
from deepfusion.models.deepfusion import DeepFusion, DeepFusionFinetuner         # your pretrained SSL class

def build_finetuner_from_pretrained(
    ssl_ckpt_path: str,
    task: str = "tri_cdr",
    **overrides,  # lr, weight_decay, dropout, etc. (Hydra can override at CLI)
):
    # 1) load SSL checkpoint (Lightning .ckpt)
    ssl = DeepFusion.load_from_checkpoint(ssl_ckpt_path, strict=False)

    # 2) init finetuner with SAME backbone hparams (others overridable)
    finetuner = DeepFusionFinetuner(
        task=task,
        in_ch=ssl.hparams.in_ch,
        dim_model=ssl.hparams.dim_model,
        num_head=ssl.hparams.num_head,
        dim_feedforward=ssl.hparams.dim_feedforward,
        activation=ssl.hparams.activation,
        num_layers=ssl.hparams.num_layers,
        **overrides,
    )

    # 3) transfer weights; new head stays randomly init
    missing, unexpected = finetuner.load_state_dict(ssl.state_dict(), strict=False)
    print("[finetuner] missing (expected: fc.*):", missing)
    print("[finetuner] unexpected (expected: unembedder.*, decoder.*):", unexpected)
    return finetuner
