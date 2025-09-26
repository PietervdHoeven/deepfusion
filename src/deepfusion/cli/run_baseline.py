# run.py
import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# DataModule: keep this hardcoded; adjust kwargs below.
from deepfusion.datamodules import BaselineDataModule

# Models: register the ones you want to toggle between.
from deepfusion.models.baselines import EncoderOnly, ResNet10, ResNet18

MODEL_REGISTRY = {
    "encoder": EncoderOnly,
    "resnet10": ResNet10,
    "resnet18": ResNet18,
}

def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="encoder", choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--task", type=str, default="tri_cdr", choices=["bin_cdr", "tri_cdr", "gender", "handedness", "age"])

    # toggles
    p.add_argument("--train", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--test", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ckpt_path", type=str, default="")  # used for resume/test

    return p.parse_args()

def main():
    args = build_args()
    pl.seed_everything(42)

    if not args.train and args.test and not args.ckpt_path:
        raise SystemExit("Provide --ckpt_path when running with --no-train and --test.")

    # Hardcoded DataModule
    datamodule = BaselineDataModule(
        data_dir="data",
        task=args.task,
        use_sampler=True,
        batch_size=2,
        num_workers=14,
        pin_memory=True,
        prefetch_factor=4
    )

    # Hardcoded model params (adjust if you like)
    ModelClass = MODEL_REGISTRY[args.model]
    model = ModelClass(
        task=args.task,
        in_ch=4,
        lr=1e-4,
        weight_decay=1e-4,
    )

    # Logger / Checkpoints / Early stop (hardcoded)
    exp_name = f"{args.model}_{args.task}"
    logger = TensorBoardLogger(save_dir="logs", name=exp_name)

    ckpt_dir = os.path.join("checkpoints", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Choose metric to monitor based on task
    monitor_metric = "val_auroc" if args.task != "age" else "val_mae"
    monitor_mode = "max" if args.task != "age" else "min"

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:02d}-{" + monitor_metric + ":.4f}",
        save_top_k=1,
        save_last=True,
        monitor=monitor_metric,
        mode=monitor_mode,
        auto_insert_metric_name=False,
    )
    earlystop_cb = EarlyStopping(
        monitor=monitor_metric,
        mode=monitor_mode,
        patience=20,
        min_delta=0.0001,
    )

    # Trainer (hardcoded)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        logger=logger,
        callbacks=[checkpoint_cb, earlystop_cb],
        max_epochs=200,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        log_every_n_steps=10
    )

    # Run
    if args.train:
        trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path or None)
        print(f"[fit] best:  {checkpoint_cb.best_model_path}")

    if args.test:
        test_ckpt = args.ckpt_path or checkpoint_cb.best_model_path
        if not test_ckpt or not os.path.exists(test_ckpt):
            raise SystemExit(f"No checkpoint available for testing: {test_ckpt}")
        print(f"[test] using: {test_ckpt}")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=test_ckpt)

if __name__ == "__main__":
    main()
