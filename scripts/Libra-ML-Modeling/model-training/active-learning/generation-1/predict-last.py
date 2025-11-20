"""Trains a DimeNet model on a dataset."""

from pathlib import Path

import typer


def main(training_logs: Path, start: int, end: int) -> None:
    """Predicts the output of a DimeNet model."""
    import json

    import lightning as L
    import numpy as np
    import torch
    from lightning.pytorch.accelerators import find_usable_cuda_devices
    from model import GeoGNN, GeoGNNDatamodule

    torch.set_float32_matmul_precision("high")

    arg_jsons = sorted(list([x for x in training_logs.rglob("args.json")]))

    print(f"Found {len(arg_jsons)} args.json files.")

    for arg_json in arg_jsons:
        if (arg_json.parent / "exp_results-last.json").exists():
            print(f"Skipping {arg_json} as exp_results.json already exists.")
            continue
        else:
            print(f"Predicting for {arg_json.parent.name}.")

        with open(arg_json) as f:
            args = json.load(f)

        ckpt_path = Path(args["best_ckpt"]).parent / "last.ckpt"
        lit_json_path = Path(args["json_path"])
        exp_json_path = Path("data/jsons/experimental.json")

        datamodule = GeoGNNDatamodule(
            literature_json=lit_json_path,
            experiment_json=exp_json_path,
            sample_dir=Path("data/xyz"),
            topo_dir=None,
            batch_size=32,
            num_workers=16,
        )

        model = GeoGNN.load_from_checkpoint(ckpt_path)

        trainer = L.Trainer(
            logger=False,
            enable_checkpointing=False,
            devices=find_usable_cuda_devices(1),
        )

        preds = trainer.predict(model=model, datamodule=datamodule)

        if preds:
            test_pred, exp_pred = preds

        # Save predictions
        test_pred = [pred.unsqueeze(0) if pred.dim() == 0 else pred for pred in test_pred]
        exp_pred = [pred.unsqueeze(0) if pred.dim() == 0 else pred for pred in exp_pred]

        test_pred = torch.cat(test_pred).numpy().astype(np.float64)
        exp_pred = torch.cat(exp_pred).numpy().astype(np.float64)

        with open(lit_json_path) as f:
            split = json.load(f)

        split["test"]["y_pred"] = list(test_pred)

        with open(arg_json.parent / "results-last.json", "w") as f:
            json.dump(split, f, indent=4)

        with open(exp_json_path) as f:
            exp_json = json.load(f)

        exp_json["y_pred"] = list(exp_pred)

        with open(arg_json.parent / "exp_results-last.json", "w") as f:
            json.dump(exp_json, f, indent=4)


if __name__ == "__main__":
    typer.run(main)
