# run_transformer.py
import argparse, os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from deepfusion.datamodules import TransformerDataModule
from deepfusion.models.wrappers import TransformerPretrainer


def build_args():
    p = argparse.ArgumentParser("Transformer pretraining (masked modeling)")
    p.add_argument("--train", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--test", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ckpt_path", type=str, default="")
    return p.parse_args()


def main():
    args = build_args()
    pl.seed_everything(42)

    if (not args.train) and args.test and not args.ckpt_path:
        raise SystemExit("Provide --ckpt_path when running with --no-train and --test.")

    # Data (fixed 0.3 mask ratio for train/val/test)
    datamodule = TransformerDataModule(
        data_dir="data",
        task="pretraining",
        batch_size=4,
        num_workers=14,
        pin_memory=True,
        prefetch_factor=4,
        mask_ratio_train=0.3,
        mask_ratio_val=0.3,
        mask_ratio_test=0.3,
    )

    # Model (hardcoded hparams)
    model = TransformerPretrainer(
        C=384, d=256, H=8, S=36, N=6,
        attn_dropout=0.1, proj_dropout=0.1, ffn_dropout=0.1,
        lr=1e-4, weight_decay=5e-2
    )

    # Logging/checkpoints
    exp_name = "transformer_pretraining"
    logger = TensorBoardLogger(save_dir="logs", name=exp_name)
    ckpt_dir = os.path.join("checkpoints", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch-{epoch:02d}_val_loss-{val_loss:.4f}",
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        mode="min",
        auto_insert_metric_name=False,
    )
    earlystop_cb = EarlyStopping(monitor="val_loss", mode="min", patience=15, min_delta=0.0005)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision="bf16-mixed",
        logger=logger,
        callbacks=[checkpoint_cb, earlystop_cb],
        gradient_clip_val=1.0,
        accumulate_grad_batches=8,
        log_every_n_steps=10,
        max_epochs=150,
    )

    # Run
    if args.train:
        trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path or None)
        print(f"[fit] best: {checkpoint_cb.best_model_path}")

    if args.test:
        test_ckpt = args.ckpt_path or checkpoint_cb.best_model_path
        if not test_ckpt or not os.path.exists(test_ckpt):
            raise SystemExit(f"No checkpoint available for testing: {test_ckpt}")
        print(f"[test] using: {test_ckpt}")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=test_ckpt)


if __name__ == "__main__":
    main()
