import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch_geometric

torch.multiprocessing.set_sharing_strategy("file_system")
script_dir = os.path.dirname(os.path.abspath(__file__))


def csd_model():
    base_dir = Path(f"{script_dir}/../TM-foundation-model/models/DimeNet/")

    model_ckpt = (
        base_dir
        / "logs"
        / "dimenet3a_csd_new-nonaug_lr1e-4_dropout0.2"
        / "version_0"
        / "checkpoints"
        / "epoch=23-val_loss=0.340.ckpt"
    )
    data_dir = Path(f"{script_dir}/../TM-foundation-model/data/pretraining")
    sys.path.append(str(base_dir))

    from dimenet.pl_wrapper import GeoGNN, HDF5DataModule

    print("CSD model imported successfully.")

    datamodule = HDF5DataModule(
        h5_file=data_dir / "processed_geometries_augmented.h5",
        input_json=data_dir / "csd_original_dataset.json",
        batch_size=1,
        num_workers=16,
        radial_cutoff=True,
    )
    datamodule.setup()

    model = GeoGNN.load_from_checkpoint(model_ckpt)
    model.eval()

    train = datamodule.train_dataloader()
    test = datamodule.test_dataloader()

    return model, datamodule, train, test


def dimenet_model(model_type="dimenet-10conf", checkpoint_index=0, seed=0):
    """
    Load DimeNet model with specified type and checkpoint index.

    Args:
        model_type: One of "dimenet-10conf", "dimenet-100conf", "dimenet-noaug"
        checkpoint_index: Index of the checkpoint to load (0-based)
    """
    current_file = Path(__file__).resolve()
    base_dir = (
        current_file.parent.parent
        / "Libra-ML-Modeling"
        / "model-training"
        / "butene"
        / "dimenet-10conf"
    )
    sys.path.append(str(base_dir))
    from model import GeoGNN, GeoGNNDatamodule

    # Find the project root directory
    data_dir = (
        current_file.parent.parent
        / "Libra-ML-Modeling"
        / "model-training"
        / "butene"
        / model_type
    )

    torch.set_float32_matmul_precision("high")

    models_path = data_dir / "models"
    subfolders = sorted([f.path for f in os.scandir(models_path) if f.is_dir()])

    arg_json = (
        f"{subfolders[checkpoint_index]}/seed_{seed}/lightning_logs/version_0/args.json"
    )
    print(f"Using args.json: {arg_json}")

    with open(arg_json) as f:
        args = json.load(f)

    ckpt_path = data_dir / args["best_ckpt"]
    lit_json_path = data_dir / args["json_path"]
    print(f"Checkpoint path: {ckpt_path}")
    print(f"Literature JSON path: {lit_json_path}")

    # Configure datamodule based on model type
    if model_type == "dimenet-100conf":
        datamodule = GeoGNNDatamodule(
            literature_json=lit_json_path,
            experiment_json=data_dir / "data/jsons/experimental.json",
            sample_dir=data_dir / "data/full_conf/xyz",
            topo_dir=None,
            batch_size=1,
            num_workers=0,  # Reduce workers to avoid memory issues
        )
    elif model_type == "dimenet-10conf":
        datamodule = GeoGNNDatamodule(
            literature_json=lit_json_path,
            experiment_json=data_dir / "data/experimental/jsons/experimental.json",
            sample_dir=data_dir / "data/xyz",
            topo_dir=None,
            batch_size=1,
            num_workers=0,  # Reduce workers to avoid memory issues
        )
    elif model_type == "dimenet-noaug":
        datamodule = GeoGNNDatamodule(
            literature_json=lit_json_path,
            experiment_json=data_dir / "data/experimental/jsons/experimental.json",
            sample_dir=data_dir / "data/xyz",
            topo_dir=None,
            batch_size=1,
            num_workers=0,  # Reduce workers to avoid memory issues
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    datamodule.setup()
    train_loader, _, test_loader = datamodule.test_dataloader()
    _, exp_loader = datamodule.predict_dataloader()

    # Load model with error handling and retry logic
    import time

    max_retries = 3
    retry_delay = 2.0

    for attempt in range(max_retries):
        try:
            print(
                f"Loading checkpoint: {ckpt_path} (attempt {attempt + 1}/{max_retries})"
            )
            # Load model on CPU first to avoid GPU memory issues
            model = GeoGNN.load_from_checkpoint(ckpt_path, map_location="cpu")
            model.eval()
            print(f"✓ Successfully loaded checkpoint")
            break
        except (OSError, IOError) as e:
            if attempt < max_retries - 1:
                print(
                    f"Warning: I/O error loading checkpoint (attempt {attempt + 1}/{max_retries}): {e}"
                )
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                print(
                    f"Error: Failed to load checkpoint after {max_retries} attempts: {e}"
                )
                print(f"Checkpoint file exists: {os.path.exists(ckpt_path)}")
                print(
                    f"Checkpoint file size: {os.path.getsize(ckpt_path) if os.path.exists(ckpt_path) else 'N/A'}"
                )
                raise
        except Exception as e:
            print(f"Error: Unexpected error loading checkpoint: {e}")
            raise

    cd = torch.utils.data.ConcatDataset(
        [datamodule.train_dataset, datamodule.test_dataset, datamodule.exp_dataset]
    )
    data = torch_geometric.loader.DataLoader(
        cd,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        num_workers=0,  # Reduce workers to avoid memory issues
    )

    # Load SMILES data if available
    try:
        smiles_data = load_smiles_multi_model(lit_json_path, data_dir, model_type)
    except Exception as e:
        print(f"Warning: Could not load SMILES data: {e}")
        smiles_data = None

    return model, data, smiles_data


def dimenet100conf(checkpoint_index=0):
    """Load dimenet-100conf model with specified checkpoint index"""
    return dimenet_model("dimenet-100conf", checkpoint_index)


def dimenet_noaug(checkpoint_index=0):
    """Load dimenet-noaug model with specified checkpoint index"""
    return dimenet_model("dimenet-noaug", checkpoint_index)


def dimenet10conf():
    """Load dimenet-10conf model with default checkpoint index 0 (backward compatibility)"""
    return dimenet_model("dimenet-10conf", 0)


def load_smiles_multi_model(lit_json_path, data_dir, model_type):
    """Load SMILES data for different model types"""
    # Find the project root directory
    current_file = Path(__file__).resolve()

    dd = current_file.parent.parent / "Libra-ML-Modeling"

    try:
        if model_type == "dimenet-10conf":
            desc = pd.read_csv(f"{dd}/data/HF-Literature.tsv", sep="\t")
            jp = f"{dd}/model-training/butene/{model_type}/data/jsons/split_0.json"

            with open(jp) as f:
                data = json.load(f)
            train = pd.DataFrame(data["train"])
            val = pd.DataFrame(data["val"])
            test = pd.DataFrame(data["test"])

            df = pd.concat([train, test, val])
            df = df.merge(desc, left_on="rxn_id", right_on="rxn_id", how="left")
            return df

        elif model_type == "dimenet-100conf":
            # For dimenet-100conf, try different possible data file locations
            project_root = (
                current_file.parent.parent
                / "Libra-ML-Modeling"
                / "model-training"
                / "butene"
            )
            possible_lit_paths = [
                Path(
                    "/global/cfs/projectdirs/m636/nhadler/projects/Libra-ML-Modeling/data/dft-descriptors/HF-Literature.tsv"
                ),
                project_root / "dimenet-100conf/data/HF-Literature-QM.tsv",
                project_root / "dimenet-100conf/data/literature/HF-Literature-QM.tsv",
                project_root
                / "dimenet-10conf/data/HF-Literature-QM.tsv",  # Fallback to 10conf data
            ]
            possible_exp_paths = [
                Path(
                    "/global/cfs/projectdirs/m636/nhadler/projects/Libra-ML-Modeling/data/dft-descriptors/HF-Experimental.tsv"
                ),
                project_root / "dimenet-100conf/data/HF-Experimental-QM.tsv",
                project_root
                / "dimenet-100conf/data/experimental/HF-Experimental-QM.tsv",
                project_root
                / "dimenet-10conf/data/HF-Experimental-QM.tsv",  # Fallback to 10conf data
            ]

            # Try to find the literature dataset
            dataset = None
            for path in possible_lit_paths:
                if path.exists():
                    dataset = pd.read_csv(path, sep="\t")
                    print(f"Found literature dataset at: {path}")
                    break

            if dataset is None:
                print("Warning: Could not find literature dataset for dimenet-100conf")
                return None, None, None

            # Try to find the experimental dataset
            experimental_dataset = None
            for path in possible_exp_paths:
                if path.exists():
                    experimental_dataset = pd.read_csv(path, sep="\t")
                    print(f"Found experimental dataset at: {path}")
                    break

            if experimental_dataset is None:
                print(
                    "Warning: Could not find experimental dataset for dimenet-100conf"
                )
                return None, None, None

            with open(lit_json_path) as f:
                literature = json.load(f)
            with open(data_dir / "data/jsons/experimental.json") as f:
                experimental = json.load(f)

            train_cas = [x.split("_")[0] for x in literature["train"]["catalyst_key"]]
            test_cas = [x.split("_")[0] for x in literature["test"]["catalyst_key"]]
            experimental_cas = [x.split("_")[0] for x in experimental["catalyst_key"]]

            smiles = dataset[
                ["ligand_1_cas", "ligand_1_smiles", "bite_angle", "C1_C2", "pbv_3.5A"]
            ]
            smiles.index = smiles["ligand_1_cas"]
            smiles = smiles.drop_duplicates()

            experimental_smiles = experimental_dataset[
                ["ligand_1_cas", "ligand_1_smiles", "bite_angle", "C1_C2", "pbv_3.5A"]
            ]
            experimental_smiles.index = experimental_smiles["ligand_1_cas"]
            experimental_smiles_list = experimental_smiles.drop_duplicates()

            train_smiles = smiles.loc[train_cas]
            test_smiles = smiles.loc[test_cas]
            experimental_smiles = experimental_smiles_list.loc[experimental_cas]

            return train_smiles, test_smiles, experimental_smiles

        elif model_type == "dimenet-noaug":
            desc = pd.read_csv(f"{dd}/data/HF-Literature.tsv", sep="\t")
            jp = f"{dd}/model-training/butene/{model_type}/data/jsons/split_0.json"

            with open(jp) as f:
                data = json.load(f)
            train = pd.DataFrame(data["train"])
            val = pd.DataFrame(data["val"])
            test = pd.DataFrame(data["test"])

            df = pd.concat([train, test, val])
            df = df.merge(desc, left_on="rxn_id", right_on="rxn_id", how="left")
            return df

        else:
            print(
                f"Warning: Unknown model type {model_type}, returning None for SMILES data"
            )
            return None, None, None

    except Exception as e:
        print(f"Warning: Could not load SMILES data for {model_type}: {e}")
        return None, None, None


def load_smiles(lit_json_path, data_dir):
    dataset = pd.read_csv(
        Path(
            "/global/cfs/projectdirs/m636/nhadler/projects/Libra-ML-Modeling/data/dft-descriptors/HF-Literature.tsv"
        ),
        sep="\t",
    )

    experimental_dataset = pd.read_csv(
        Path(
            "/global/cfs/projectdirs/m636/nhadler/projects/Libra-ML-Modeling/data/dft-descriptors/HF-Experimental.tsv"
        ),
        sep="\t",
    )

    with open(lit_json_path) as f:
        literature = json.load(f)
    with open(f"{data_dir}/data/experimental/jsons/experimental.json") as f:
        experimental = json.load(f)

    train_cas = [x.split("_")[0] for x in literature["train"]["catalyst_key"]]
    test_cas = [x.split("_")[0] for x in literature["test"]["catalyst_key"]]
    experimental_cas = [x.split("_")[0] for x in experimental["catalyst_key"]]

    smiles = dataset[
        ["ligand_1_cas", "ligand_1_smiles", "bite_angle", "C1_C2", "pbv_3.5A"]
    ]
    smiles.index = smiles["ligand_1_cas"]
    smiles = smiles.drop_duplicates()

    experimental_smiles = experimental_dataset[
        ["ligand_1_cas", "ligand_1_smiles", "bite_angle", "C1_C2", "pbv_3.5A"]
    ]
    experimental_smiles.index = experimental_smiles["ligand_1_cas"]
    experimental_smiles_list = experimental_smiles.drop_duplicates()

    # These are the smiles strings for the train, test, and experimental datasets
    train_smiles = smiles.loc[train_cas]
    test_smiles = smiles.loc[test_cas]
    experimental_smiles = experimental_smiles_list.loc[experimental_cas]

    return train_smiles, test_smiles, experimental_smiles
