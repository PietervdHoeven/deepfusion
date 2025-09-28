# run_transformer_finetune.py
import argparse, os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from deepfusion.datamodules import TransformerDataModule
from deepfusion.models.wrappers import TransformerFinetuner


TASKS = ["bin_cdr", "tri_cdr", "handedness", "age", "gender"]


def build_args():
    p = argparse.ArgumentParser("Transformer fine-tuning (downstream)")
    p.add_argument("--task", type=str, required=True, choices=TASKS)
    p.add_argument("--pretrain_ckpt", type=str, required=True, help="Path to pretraining .ckpt")
    p.add_argument("--train", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--test", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ckpt_path", type=str, default="")
    return p.parse_args()


def main():
    args = build_args()
    pl.seed_everything(42)

    if (not args.train) and args.test and not args.ckpt_path:
        raise SystemExit("Provide --ckpt_path when running with --no-train and --test.")

    # Data (no masking for downstream)
    datamodule = TransformerDataModule(
        data_dir="data",
        task=args.task,
        batch_size=8,
        num_workers=14,
        pin_memory=True,
        prefetch_factor=4,
        mask_ratio_train=0.0,
        mask_ratio_val=0.0,
        mask_ratio_test=0.0,
    )

    # Model: Finetuner builds backbone+pool internally and loads pretrain ckpt
    model = TransformerFinetuner(
        embed_dim=384,
        num_heads=6,
        task=args.task,
        lr=1e-5,
        weight_decay=1e-4,
        pretrain_ckpt=args.pretrain_ckpt,
        attn_pool=True
    )

    # Logging/checkpoints
    exp_name = f"transformer_finetuning_{args.task}"
    logger = TensorBoardLogger(save_dir="logs", name=exp_name)
    ckpt_dir = os.path.join("checkpoints", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Monitor metric depends on task
    if args.task == "age":
        monitor_metric, monitor_mode = "val_mae", "min"
        ckpt_filename = "epoch-{epoch:02d}_val_mae-{val_mae:.4f}"
    else:
        monitor_metric, monitor_mode = "val_auroc", "max"
        ckpt_filename = "epoch-{epoch:02d}_val_auroc-{val_auroc:.4f}"

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=ckpt_filename,
        save_top_k=1,
        save_last=True,
        monitor=monitor_metric,
        mode=monitor_mode,
        auto_insert_metric_name=False,
    )
    earlystop_cb = EarlyStopping(monitor=monitor_metric, mode=monitor_mode, patience=15, min_delta=1e-3)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision="32",
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
