"""Trains a DimeNet model on a dataset."""

import json
import os
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from model import GeoGNN, GeoGNNDatamodule  # type: ignore

os.environ["SLURM_JOB_NAME"] = "bash"
torch.set_float32_matmul_precision("high")


def main(
    lit_json_path: Path,
    seed: int | None = None,
    folder_name: Path = Path("models"),
) -> None:
    """This script trains a DimeNet model on a dataset."""
    L.seed_everything(seed, workers=True)

    # Extract the split number from json_path
    split_str = lit_json_path.stem.split("_")[-1]
    split = f"{int(split_str):03d}"

    # Create a folder for the split
    model_folder = folder_name / f"split_{split}/seed_{seed}"
    if any(model_folder.rglob("args.json")):
        print(f"Split {split} already trained.")
        return

    exp_json_path = Path("data/experimental/jsons/experimental.json")

    datamodule = GeoGNNDatamodule(
        literature_json=lit_json_path,
        experiment_json=exp_json_path,
        sample_dir=Path("data/xyz"),
        topo_dir=None,
        batch_size=8,
        num_workers=16,
    )

    model = GeoGNN(learning_rate=1e-3)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{val_loss:.3f}",
        monitor="val_loss",
        save_top_k=3,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    # early_stopping = EarlyStopping(monitor="val_loss", patience=10)

    # Logger
    tb_logger = TensorBoardLogger(model_folder)
    csv_logger = CSVLogger(model_folder, version=tb_logger.version)

    trainer = L.Trainer(
        deterministic=False,
        max_epochs=25,
        logger=[tb_logger, csv_logger],
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=0.25,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=5,
        fast_dev_run=False,
    )

    trainer.fit(model, datamodule=datamodule)

    results = trainer.test(model, datamodule=datamodule, ckpt_path="last")

    # Save inputs for reproducibility and predictions
    args_dict = {
        "seed": seed,
        "json_path": str(lit_json_path),
        "best_ckpt": str(checkpoint_callback.best_model_path),
        "train_loss": results[0]["test_loss/dataloader_idx_0"],
        "val_loss": results[1]["test_loss/dataloader_idx_1"],
        "test_loss": results[2]["test_loss/dataloader_idx_2"],
    }

    args_json = model_folder / f"lightning_logs/version_{tb_logger.version}/args.json"

    with open(args_json, "w") as f:
        json.dump(args_dict, f, indent=4)


if __name__ == "__main__":
    print("Starting training...")
    from sys import argv

    seed = argv[1] if len(argv) > 1 else 0
    seed = int(seed)
    start = argv[2] if len(argv) > 2 else 0
    start = int(start)
    end = argv[3] if len(argv) > 3 else 156
    end = int(end)

    for i in range(start, end):
        try:
            json_path = Path(f"data/jsons/split_{i}.json")
            main(lit_json_path=json_path, seed=seed)
        except Exception as e:
            print(f"Error training split {i:03d} with seed {seed}: {e}")
    print("Training complete.")
