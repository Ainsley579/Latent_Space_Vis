"""
Script for training DimeNet on geometric GNN datasets for use in Libra-ML.

This includes dataset loading, model architecture, and training utilities. Portions of this code
where adapted from the Pytorch Geometric implementation of DimeNet.

Written by: Nicholas Hadler (nhadler@berkeley.edu)
"""

import itertools
import json
import os
from functools import partial
from math import pi as PI
from math import sqrt
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.nn import Embedding, Linear
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import pool, radius_graph
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, SparseTensor
from torch_geometric.utils import scatter, to_dense_batch
from torchmetrics import R2Score

ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
}


class GeoGNNDataset(Dataset):
    """This constructs a dataset for loading in XYZ files and optionally their topology files.

    This expects a data_dict with the following keys:
        - "catalyst_key": List of catalyst keys (str). These should match the filenames of the xyz and topology json files.
        - "reaction_conditions": List of reaction conditions (list of floats) corresponding to each datapoint.
        - "ddG": List of ddG values (float) corresponding to each catalyst key.

    The XYZ files should be in the standard turbomol format (with .xyz file format), and the topology files should be in JSON format
    with a "blist" key containing the bond list. These should generated using GFNFF-xTB:
        ```
            > xtb <xyz_file> --gfnff --wrtopo blist
        ```
    """

    def __init__(
        self,
        data_dict: dict[str, list],
        sample_dir: Path,
        topo_dir: None | Path,
        transform: None | list[torch.nn.Module] = None,
    ) -> None:
        """Initializes the GeoGNNDataset.

        Args:
            data_dict (dict[str, list]): Dictionary containing the data keys and labels with keys:
                - "catalyst_key": List of catalyst identifiers.
                - "reaction_conditions": List of reaction condition values.
                - "ddG": List of energy values.
            sample_dir (Path): Path to the directory containing the xyz files.
            topo_dir (None | Path): Path to the directory containing the topology files. Defaults to None.
            transform (None | list[torch.nn.Module]): Transformation to apply to the data. Defaults to None.
        """
        super().__init__(transform=transform)

        self.keys = data_dict["catalyst_key"]
        self.ddG = data_dict["ddG"]
        self.sample_dir = sample_dir
        self.topo_dir = topo_dir
        self.rxn_conds = data_dict["reaction_conditions"]

    @staticmethod
    def edge_index(topo_json) -> torch.Tensor:
        """Returns the edge index for the dataset."""
        import time
        
        # Retry logic for I/O operations
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                with open(topo_json) as f:
                    blist = json.load(f)["blist"]

                blist = np.array(blist)
                blist = (blist[:, :-1] - 1).T

                return torch.tensor(blist, dtype=torch.long)
            except (OSError, IOError) as e:
                if attempt < max_retries - 1:
                    print(f"Warning: I/O error reading {topo_json} (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Error: Failed to read {topo_json} after {max_retries} attempts: {e}")
                    print(f"File exists: {os.path.exists(topo_json)}")
                    print(f"File size: {os.path.getsize(topo_json) if os.path.exists(topo_json) else 'N/A'}")
                    # Return a default empty edge index
                    return torch.tensor([[], []], dtype=torch.long)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: JSON error reading {topo_json}: {e}")
                # Return a default empty edge index
                return torch.tensor([[], []], dtype=torch.long)

    def len(self) -> int:
        """Returns the length of the dataset."""
        return len(self.keys)
    
    def _generate_basic_edge_index(self, atom_list, xyz_coord, cutoff=3.0):
        """Generate basic edge indices based on distance cutoff when topology data is unavailable."""
        import torch
        from torch_geometric.nn import radius_graph
        
        # Generate edges based on distance cutoff
        edge_index = radius_graph(xyz_coord, r=cutoff, loop=False)
        return edge_index

    def get(self, idx: int) -> Data:
        """Returns the sample at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Data: Sample data.
        """
        try:
            xyz = self.sample_dir / f"{self.keys[idx]}.xyz"
            atom_list, xyz_coord = self.read_xyz(xyz)

            ddG = torch.tensor([self.ddG[idx]])
            rxn_cond = torch.tensor(self.rxn_conds[idx]).unsqueeze(0)

            if self.topo_dir:
                try:
                    edge_index = self.edge_index(self.topo_dir / f"{self.keys[idx]}.json")
                    return Data(
                        z=atom_list,
                        pos=xyz_coord,
                        edge_index=edge_index,
                        y=ddG,
                        rxn_conds=rxn_cond,
                    )
                except (OSError, IOError) as e:
                    print(f"Warning: Skipping topology data for {self.keys[idx]} due to I/O error: {e}")
                    # Generate basic edge indices based on distance
                    edge_index = self._generate_basic_edge_index(atom_list, xyz_coord)
                    return Data(
                        z=atom_list,
                        pos=xyz_coord,
                        edge_index=edge_index,
                        y=ddG,
                        rxn_conds=rxn_cond,
                    )
            else:
                return Data(
                    z=atom_list,
                    pos=xyz_coord,
                    edge_index=None,
                    y=ddG,
                    rxn_conds=rxn_cond,
                )
        except Exception as e:
            print(f"Warning: Error processing sample {idx} ({self.keys[idx]}): {e}")
            # Return a minimal valid Data object
            return Data(
                z=torch.tensor([1]),  # Default atom type
                pos=torch.tensor([[0.0, 0.0, 0.0]]),  # Default position
                edge_index=torch.tensor([[], []]),  # Empty edge index
                y=torch.tensor([0.0]),  # Default ddG
                rxn_conds=torch.tensor([[0.0, 0.0, 0.0]]),  # Default reaction conditions
            )

    @staticmethod
    def read_xyz(xyz_file: Path | str | bytes) -> tuple[torch.Tensor, torch.Tensor]:
        """Reads an xyz file and returns the coordinates and atom list.

        Args:
            xyz_file (Path | str | bytes): Path to the xyz file.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing the atom list and coordinates.
        """
        # Read in the xyz file
        with open(xyz_file) as f:
            xyz_temp = f.read().splitlines()

        # Converts the xyz file into a list of lists
        xyz = []
        for line in xyz_temp:
            xyz += [line.split()]

        # Converts the xyz file into a numpy array of floats
        xyz_coord = np.array(xyz[2:]).T[1:].T.astype("float32")

        # Converts the xyz file into a list of atoms
        atom_list = np.array(xyz[2:]).T[:1].T.tolist()
        atom_list = [str(x) for x in itertools.chain.from_iterable(atom_list)]
        atom_array = np.array([int(ATOMIC_NUMBERS[x]) for x in atom_list])

        return torch.from_numpy(atom_array), torch.from_numpy(xyz_coord)


class GeoGNNDatamodule(L.LightningDataModule):
    """Datamodule for loading in geometric GNN datasets.

    This expects a literature JSON file with the following structure:
        {
            "train": [list of catalyst keys for training],
            "val": [list of catalyst keys for validation],
            "test": [list of catalyst keys for testing]
        }

    And an experiment JSON file with the same structure.
    """

    def __init__(
        self,
        literature_json: Path,
        experiment_json: Path,
        sample_dir: Path,
        topo_dir: None | Path = None,
        batch_size: int = 32,
        num_workers: int = 8,
        train_transform: None | list[torch.nn.Module] = None,
        val_transform: None | list[torch.nn.Module] = None,
        test_transform: None | list[torch.nn.Module] = None,
    ) -> None:
        """Initializes the GeoGNNDatamodule.

        Args:
            literature_json (Path): Path to the JSON file containing the dataset split.
            experiment_json (Path): Path to the JSON file containing the dataset split.
            sample_dir (Path): Path to the directory containing the xyz files.
            topo_dir (None | Path, optional): Path to the directory containing the topology files. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of concurrent workers for data_loaders. Defaults to 8.
            train_transform (None | list[torch.nn.Module], optional): Transformations for training data.
                Defaults to None.
            val_transform (None | list[torch.nn.Module], optional): Transformations for validation data.
                Defaults to None.
            test_transform (None | list[torch.nn.Module], optional): Transformations for test data.
                Defaults to None.
        """
        super().__init__()

        self.sample_dir = sample_dir
        self.topo_dir = topo_dir

        with open(literature_json) as f:
            lit_data = json.load(f)

        self.train_data = lit_data["train"]
        self.val_data = lit_data["val"]
        self.test_data = lit_data["test"]

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        with open(experiment_json) as f:
            self.exp_data = json.load(f)

    def setup(self, stage: None | str = None) -> None:
        """Prepares the data for training, validation, and testing."""
        self.train_dataset = GeoGNNDataset(
            self.train_data,
            self.sample_dir,
            self.topo_dir,
            self.train_transform,
        )
        self.validation_dataset = GeoGNNDataset(
            self.val_data,
            self.sample_dir,
            self.topo_dir,
            self.val_transform,
        )
        self.test_dataset = GeoGNNDataset(
            self.test_data,
            self.sample_dir,
            self.topo_dir,
            self.test_transform,
        )
        self.exp_dataset = GeoGNNDataset(
            self.exp_data,
            self.sample_dir,
            self.topo_dir,
            self.test_transform,
        )

    def train_dataloader(self) -> DataLoader:
        """Returns the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the validation dataloader."""
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Returns the train, validation, and test dataloader."""
        return (
            DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            ),
            DataLoader(
                self.validation_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            ),
            DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            ),
        )

    def predict_dataloader(self) -> tuple[DataLoader, DataLoader]:
        """Returns the test and experimental dataloader."""
        return (
            DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            ),
            DataLoader(
                self.exp_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            ),
        )


class Envelope(torch.nn.Module):
    """Envelope function that ensures a smooth cutoff.

    Args:
        exponent (int): Exponent value for envelope calculation.
    """

    def __init__(self, exponent: int):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for envelope calculation.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Envelope value.
        """
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return (1.0 / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2) * (x < 1.0).to(
            x.dtype
        )


class BesselBasisLayer(torch.nn.Module):
    """Bessel basis expansion layer.

    Args:
        num_radial (int): Number of radial basis functions.
        cutoff (float, optional): Cutoff distance. Defaults to 5.0.
        envelope_exponent (int, optional): Exponent for envelope function. Defaults to 5.
    """

    def __init__(
        self, num_radial: int, cutoff: float = 5.0, envelope_exponent: int = 5
    ):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.empty(num_radial))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the learnable parameters."""
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)
        self.freq.requires_grad_()

    def forward(self, dist: Tensor) -> Tensor:
        """Forward pass.

        Args:
            dist (Tensor): Distance tensor.

        Returns:
            Tensor: Bessel basis expansion.
        """
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class SphericalBasisLayer(torch.nn.Module):
    """Spherical basis expansion layer.

    Args:
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff (float, optional): Cutoff distance. Defaults to 5.0.
        envelope_exponent (int, optional): Exponent for envelope function. Defaults to 5.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
    ):
        super().__init__()
        import sympy as sym  # type: ignore
        from dimenet_scripts.dimenet_utils import (  # type: ignore
            bessel_basis,
            real_sph_harm,
        )

        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols("x theta")
        modules = {"sin": torch.sin, "cos": torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(partial(self._sph_to_tensor, sph1))
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    @staticmethod
    def _sph_to_tensor(sph: float, x: Tensor) -> Tensor:
        """Convert spherical harmonic value to tensor.

        Args:
            sph (float): Spherical harmonic value.
            x (Tensor): Input tensor.

        Returns:
            Tensor: Spherical harmonic as tensor with same shape as x.
        """
        return torch.zeros_like(x) + sph

    def forward(self, dist: Tensor, angle: Tensor, idx_kj: Tensor) -> Tensor:
        """Forward pass.

        Args:
            dist (Tensor): Distance tensor.
            angle (Tensor): Angle tensor.
            idx_kj (Tensor): Indices for the k->j edges.

        Returns:
            Tensor: Combined spherical basis functions.
        """
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


class EmbeddingBlock(torch.nn.Module):
    """Block for embedding initial node and edge features.

    Args:
        num_radial (int): Number of radial basis functions.
        hidden_channels (int): Hidden channels dimension.
        act (Callable | str): Activation function.
    """

    def __init__(self, num_radial: int, hidden_channels: int, act: Callable | str):
        super().__init__()
        self.act = act

        self.emb = Embedding(95, hidden_channels)
        self.lin_rbf = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the learnable parameters."""
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor, j: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Node features.
            rbf (Tensor): Radial basis features.
            i (Tensor): Source node indices.
            j (Tensor): Target node indices.

        Returns:
            Tensor: Embedded node features.
        """
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))


class ResidualLayer(torch.nn.Module):
    """Residual layer for message passing.

    Args:
        hidden_channels (int): Number of hidden channels.
        act (Callable): Activation function.
    """

    def __init__(self, hidden_channels: int, act: Callable):
        super().__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the learnable parameters."""
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection.

        Args:
            x (Tensor): Input features.

        Returns:
            Tensor: Output features with residual connection.
        """
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class InteractionBlock(torch.nn.Module):
    """Interaction block for DimeNet.

    Args:
        hidden_channels (int): Hidden channels dimension.
        num_bilinear (int): Size of bilinear layer.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        num_before_skip (int): Number of residual layers before skip connection.
        num_after_skip (int): Number of residual layers after skip connection.
        act (Callable): Activation function.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
        act: Callable,
    ):
        super().__init__()
        self.act = act

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)
        self.lin_sbf = Linear(num_spherical * num_radial, num_bilinear, bias=False)

        # Dense transformations of input messages.
        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)

        self.W = torch.nn.Parameter(
            torch.empty(hidden_channels, num_bilinear, hidden_channels)
        )

        self.layers_before_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)]
        )
        self.lin = Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)]
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the learnable parameters."""
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)
        self.W.data.normal_(mean=0, std=2 / self.W.size(0))
        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(
        self, x: Tensor, rbf: Tensor, sbf: Tensor, idx_kj: Tensor, idx_ji: Tensor
    ) -> Tensor:
        """Forward pass for interaction block.

        Args:
            x (Tensor): Input node features.
            rbf (Tensor): Radial basis features.
            sbf (Tensor): Spherical basis features.
            idx_kj (Tensor): Indices for the k->j edges.
            idx_ji (Tensor): Indices for the j->i edges.

        Returns:
            Tensor: Updated node features.
        """
        rbf = self.lin_rbf(rbf)
        sbf = self.lin_sbf(sbf)

        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))
        x_kj = x_kj * rbf
        x_kj = torch.einsum("wj,wl,ijl->wi", sbf, x_kj[idx_kj], self.W)
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0), reduce="sum")

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class InteractionPPBlock(torch.nn.Module):
    """Interaction block for DimeNet++.

    Args:
        hidden_channels (int): Hidden channels dimension.
        int_emb_size (int): Size of embedding in the interaction block.
        basis_emb_size (int): Size of basis embedding in the interaction block.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        num_before_skip (int): Number of residual layers before skip connection.
        num_after_skip (int): Number of residual layers after skip connection.
        act (Callable): Activation function.
    """

    def __init__(
        self,
        hidden_channels: int,
        int_emb_size: int,
        basis_emb_size: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
        act: Callable,
    ):
        super().__init__()
        self.act = act

        # Transformation of Bessel and spherical basis representations:
        self.lin_rbf1 = Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = Linear(basis_emb_size, hidden_channels, bias=False)

        self.lin_sbf1 = Linear(num_spherical * num_radial, basis_emb_size, bias=False)
        self.lin_sbf2 = Linear(basis_emb_size, int_emb_size, bias=False)

        # Hidden transformation of input message:
        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)

        # Embedding projections for interaction triplets:
        self.lin_down = Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = Linear(int_emb_size, hidden_channels, bias=False)

        # Residual layers before and after skip connection:
        self.layers_before_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)]
        )
        self.lin = Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)]
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the learnable parameters."""
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(
        self,
        x: Tensor,
        rbf: Tensor,
        sbf: Tensor,
        idx_kj: Tensor,
        idx_ji: Tensor,
    ) -> Tensor:
        """Forward pass for interaction++ block.

        Args:
            x (Tensor): Input node features.
            rbf (Tensor): Radial basis features.
            sbf (Tensor): Spherical basis features.
            idx_kj (Tensor): Indices for the k->j edges.
            idx_ji (Tensor): Indices for the j->i edges.

        Returns:
            Tensor: Updated node features.
        """
        # Initial transformation:
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        # Transformation via Bessel basis:
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down project embedding and generating triple-interactions:
        x_kj = self.act(self.lin_down(x_kj))

        # Transform via 2D spherical basis:
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        # Aggregate interactions and up-project embeddings:
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0), reduce="sum")
        x_kj = self.act(self.lin_up(x_kj))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class OutputBlock(torch.nn.Module):
    """Output block for DimeNet.

    Args:
        num_radial (int): Number of radial basis functions.
        hidden_channels (int): Hidden channels dimension.
        out_channels (int): Output channels dimension.
        num_layers (int): Number of linear layers.
        act (Callable | str): Activation function.
        output_initializer (str, optional): Initialization method for output layer.
            Must be one of {"zeros", "glorot_orthogonal"}. Defaults to "zeros".
    """

    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        act: Callable | str,
        output_initializer: str = "zeros",
    ):
        assert output_initializer in {"zeros", "glorot_orthogonal"}

        super().__init__()

        self.act = act
        self.output_initializer = output_initializer

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lin = Linear(hidden_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the learnable parameters."""
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_initializer == "zeros":
            self.lin.weight.data.fill_(0)
        elif self.output_initializer == "glorot_orthogonal":
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(
        self, x: Tensor, rbf: Tensor, i: Tensor, num_nodes: Optional[int] = None
    ) -> Tensor:
        """Forward pass for output block.

        Args:
            x (Tensor): Input node features.
            rbf (Tensor): Radial basis features.
            i (Tensor): Source node indices.
            num_nodes (Optional[int], optional): Number of nodes. Defaults to None.

        Returns:
            Tensor: Output features.
        """
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes, reduce="sum")
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


class OutputPPBlock(torch.nn.Module):
    """Output block for DimeNet++."""

    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_emb_channels: int,
        out_channels: int,
        num_layers: int,
        act: Callable,
        output_initializer: str = "zeros",
    ):
        """Initialize the output block.

        Args:
            num_radial (int): Number of radial basis functions.
            hidden_channels (int): Number of hidden channels.
            out_emb_channels (int): Number of output embedding channels.
            out_channels (int): Number of output channels.
            num_layers (int): Number of layers in the model.
            act (Callable): Activation function.
            output_initializer (str, optional): Method to initialize the output layer.
                Must be one of {"zeros", "glorot_orthogonal"}. Defaults to "zeros".

        Raises:
            AssertionError: If `output_initializer` is not one of {"zeros", "glorot_orthogonal"}.
        """
        assert output_initializer in {"zeros", "glorot_orthogonal"}

        super().__init__()

        self.act = act
        self.output_initializer = output_initializer

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)

        # The up-projection layer:
        self.lin_up = Linear(hidden_channels, out_emb_channels, bias=False)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(out_emb_channels, out_emb_channels))
        self.lin = Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the learnable parameters."""
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_initializer == "zeros":
            self.lin.weight.data.fill_(0)
        elif self.output_initializer == "glorot_orthogonal":
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(
        self, x: Tensor, rbf: Tensor, i: Tensor, num_nodes: Optional[int] = None
    ) -> Tensor:
        """Forward pass for output block.

        Args:
            x (Tensor): Input node features.
            rbf (Tensor): Radial basis features.
            i (Tensor): Source node indices.
            num_nodes (Optional[int], optional): Number of nodes. Defaults to None.

        Returns:
            Tensor: Output features.
        """
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes, reduce="sum")
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


def triplets(
    edge_index: Tensor, num_nodes: int
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Generates triplets of node indices and edge indices for a given graph.

    Args:
        edge_index (Tensor): The edge indices of the graph, where edge_index[0] contains the source nodes
                             and edge_index[1] contains the target nodes.
        num_nodes (int): The number of nodes in the graph.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
            - col (Tensor): The target nodes of the edges.
            - row (Tensor): The source nodes of the edges.
            - idx_i (Tensor): The target nodes of the triplets.
            - idx_j (Tensor): The intermediate nodes of the triplets.
            - idx_k (Tensor): The source nodes of the triplets.
            - idx_kj (Tensor): The edge indices from source to intermediate nodes in the triplets.
            - idx_ji (Tensor): The edge indices from intermediate to target nodes in the triplets.
    """
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(
        row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
    )
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji


class DimeNet(torch.nn.Module):
    """DimeNet model copied from the implementation in Pytorch Geometric.

    The directional message passing neural network (DimeNet) from the
    `"Directional Message Passing for Molecular Graphs"
    <https://arxiv.org/abs/2003.03123>`_ paper.
    DimeNet transforms messages based on the angle between them in a
    rotation-equivariant fashion.

    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        num_bilinear (int): Size of the bilinear layer tensor.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act (str or Callable, optional): The activation function.
            (default: :obj:`"swish"`)
        output_initializer (str, optional): The initialization method for the
            output layer (:obj:`"zeros"`, :obj:`"glorot_orthogonal"`).
            (default: :obj:`"zeros"`)
        return_embeddings (bool, optional): If set to :obj:`True`, the model
            will return the embeddings of the last layer instead of the output.
        data_aug_interaction_layers (bool, optional): If set to :obj:`True`, the model
            will use data augmentation in the interaction layers.
    """

    url = "https://github.com/klicperajo/dimenet/raw/master/pretrained/dimenet"

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: Union[str, Callable] = "swish",
        output_initializer: str = "zeros",
        return_embeddings: bool = False,
        data_aug_interaction_layers: bool = False,
    ):
        super().__init__()

        if num_spherical < 2:
            raise ValueError("'num_spherical' should be greater than 1")

        act = activation_resolver(act)

        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_blocks = num_blocks
        self.return_embeddings = return_embeddings
        self.data_aug_interaction_layers = data_aug_interaction_layers

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(
            num_spherical, num_radial, cutoff, envelope_exponent
        )

        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)

        self.output_blocks = torch.nn.ModuleList(
            [
                OutputBlock(
                    num_radial,
                    hidden_channels,
                    out_channels,
                    num_output_layers,
                    act,
                    output_initializer,
                )
                for _ in range(num_blocks + 1)
            ]
        )

        self.interaction_blocks = torch.nn.ModuleList(
            [
                InteractionBlock(
                    hidden_channels,
                    num_bilinear,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    act,
                )
                for _ in range(num_blocks)
            ]
        )

    def reset_parameters(self) -> None:
        """Reset the learnable parameters."""
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        edge_index: Optional[Tensor] = None,
        batch: OptTensor = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            z (torch.Tensor): Atomic number of each atom with shape
                :obj:`[num_atoms]`.
            pos (torch.Tensor): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            edge_index (torch.Tensor, optional): Graph connectivity in COO
                format with shape :obj:`[2, num_edges]`. If not provided,
                a radius graph will be constructed based on the cutoff
                distance. (default: :obj:`None`)
            batch (torch.Tensor, optional): Batch indices assigning each atom
                to a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`)
        """
        if edge_index is None:
            edge_index = radius_graph(
                pos,
                r=self.cutoff,
                batch=batch,
                max_num_neighbors=self.max_num_neighbors,
            )

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=z.size(0)
        )

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        if isinstance(self, DimeNetPlusPlus):
            pos_jk, pos_ij = pos[idx_j] - pos[idx_k], pos[idx_i] - pos[idx_j]
            a = (pos_ij * pos_jk).sum(dim=-1)
            b = torch.linalg.cross(pos_ij, pos_jk).norm(dim=-1)
        elif isinstance(self, DimeNet):
            pos_ji, pos_ki = pos[idx_j] - pos[idx_i], pos[idx_k] - pos[idx_i]
            a = (pos_ji * pos_ki).sum(dim=-1)
            b = torch.linalg.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(
            self.interaction_blocks, self.output_blocks[1:]
        ):
            if self.data_aug_interaction_layers:
                gaussian_noise = torch.normal(
                    mean=0.0, std=0.025, size=x.shape, device=x.device
                )
                gaussian_noise = torch.clamp(gaussian_noise, min=-0.1, max=0.1)
                x = x + gaussian_noise
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i, num_nodes=pos.size(0))

        if self.return_embeddings:
            return P

        if batch is None:
            return P.sum(dim=0)
        else:
            return scatter(P, batch, dim=0, reduce="sum")


class DimeNetPlusPlus(DimeNet):
    """DimeNet++ model copied from the implementation in Pytorch Geometric.

    The DimeNet++ from the `"Fast and Uncertainty-Aware
    Directional Message Passing for Non-Equilibrium Molecules"
    <https://arxiv.org/abs/2011.14115>`_ paper.

    :class:`DimeNetPlusPlus` is an upgrade to the :class:`DimeNet` model with
    8x faster and 10% more accurate than :class:`DimeNet`.

    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of interaction blocks.
        int_emb_size (int): Size of embedding in the interaction block.
        basis_emb_size (int): Size of basis embedding in the interaction block.
        out_emb_channels (int): Size of embedding in the output block.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (str or Callable, optional): The activation funtion.
            (default: :obj:`"swish"`)
        output_initializer (str, optional): The initialization method for the
            output layer (:obj:`"zeros"`, :obj:`"glorot_orthogonal"`).
            (default: :obj:`"zeros"`)
        return_embeddings (bool, optional): If set to :obj:`True`, the model
            will return the embeddings of the last layer instead of the output.
        data_aug_interaction_layers (bool, optional): If set to :obj:`True`, the model
            will use data augmentation in the interaction layers.
    """

    url = "https://raw.githubusercontent.com/gasteigerjo/dimenet/master/pretrained/dimenet_pp"

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        int_emb_size: int,
        basis_emb_size: int,
        out_emb_channels: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: Union[str, Callable] = "swish",
        output_initializer: str = "zeros",
        return_embeddings: bool = False,
        data_aug_interaction_layers: bool = False,
    ) -> None:
        """Initialize the DimeNet++ model.

        Args:
            hidden_channels (int): Hidden embedding size.
            out_channels (int): Size of each output sample.
            num_blocks (int): Number of interaction blocks.
            int_emb_size (int): Size of embedding in the interaction block.
            basis_emb_size (int): Size of basis embedding in the interaction block.
            out_emb_channels (int): Size of embedding in the output block.
            num_spherical (int): Number of spherical harmonics.
            num_radial (int): Number of radial basis functions.
            cutoff (float, optional): Cutoff distance for interatomic
                interactions. Defaults to 5.0.
            max_num_neighbors (int, optional): The maximum number of neighbors to
                collect for each node within the cutoff distance. Defaults to 32.
            envelope_exponent (int, optional): Shape of the smooth cutoff.
                Defaults to 5.
            num_before_skip (int, optional): Number of residual layers in the
                interaction blocks before the skip connection. Defaults to 1.
            num_after_skip (int, optional): Number of residual layers in the
                interaction blocks after the skip connection. Defaults to 2.
            num_output_layers (int, optional): Number of linear layers for the
                output blocks. Defaults to 3.
            act (Union[str, Callable], optional): The activation function.
                Defaults to "swish".
            output_initializer (str, optional): The initialization method for the
                output layer ("zeros", "glorot_orthogonal").
                Defaults to "zeros".
            return_embeddings (bool, optional): If set to True, the model
                will return the embeddings of the last layer instead of the output.
                Defaults to False.
            data_aug_interaction_layers (bool, optional): If set to True, the model
                will use data augmentation in the interaction layers.
                Defaults to False.
        """
        act = activation_resolver(act)

        super().__init__(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            num_bilinear=1,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            act=act,
            output_initializer=output_initializer,
            return_embeddings=return_embeddings,
            data_aug_interaction_layers=data_aug_interaction_layers,
        )

        # We are re-using the RBF, SBF and embedding layers of `DimeNet` and
        # redefine output_block and interaction_block in DimeNet++.
        # Hence, it is to be noted that in the above initialization, the
        # variable `num_bilinear` does not have any purpose as it is used
        # solely in the `OutputBlock` of DimeNet:
        self.output_blocks = torch.nn.ModuleList(
            [
                OutputPPBlock(
                    num_radial,
                    hidden_channels,
                    out_emb_channels,
                    out_channels,
                    num_output_layers,
                    act,
                    output_initializer,
                )
                for _ in range(num_blocks + 1)
            ]
        )

        self.interaction_blocks = torch.nn.ModuleList(
            [
                InteractionPPBlock(
                    hidden_channels,
                    int_emb_size,
                    basis_emb_size,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    act,
                )
                for _ in range(num_blocks)
            ]
        )

        self.reset_parameters()


class MLP(torch.nn.Module):
    """Multi-layer perceptron used as the final output layer of the model."""

    def __init__(
        self,
        input_dim: int = 260,
        output_dim: int = 1,
        dropout: float = 0.2,
        act: str = "swish",
    ) -> None:
        """Initializes the MLP.

        Args:
            input_dim (int): The dimension of the input features. Default is 260.
            output_dim (int): The dimension of the output features. Default is 1.
            dropout (float): The dropout rate. Default is 0.5.
            act (str): The activation function to use. Default is "swish".
        """
        super().__init__()

        self.fc1 = torch.nn.Linear(input_dim, input_dim // 2)
        self.act_1 = activation_resolver(act)
        self.dropout_1 = torch.nn.Dropout(dropout)

        self.fc2 = torch.nn.Linear(input_dim // 2, input_dim // 2)
        self.act_2 = activation_resolver(act)
        self.dropout_2 = torch.nn.Dropout(dropout)

        self.fc3 = torch.nn.Linear(input_dim // 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        embeddings = []

        x = self.fc1(x)
        x = self.act_1(x)
        x = self.dropout_1(x)

        embeddings.append(x)

        x = self.fc2(x)
        x = self.act_2(x)
        x = self.dropout_2(x)

        embeddings.append(x)

        x = self.fc3(x)
        return x, embeddings


class GeoGNN(L.LightningModule):
    """LightningModule for Geometric Graph Neural Networks."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        scheduler_patience: int = 5,
        readout: str = "sum",
        mlp_input_dim: int = 260,
        mlp_output_dim: int = 1,
        mlp_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        """Initializes the GeoGNN module with DimeNet++ and an MLP.

        Args:
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
            scheduler_patience (int, optional): Patience for the learning rate scheduler. Defaults to 5.
            readout (str, optional): Graph readout method. Must be one of
                "sum", "mean", "max", "SAG", or "ASA". Defaults to "sum".
            mlp_input_dim (int, optional): Input dimension for the MLP. This should match
                the dimension of graph embeddings + reaction conditions. Defaults to 260.
            mlp_output_dim (int, optional): Output dimension for the MLP. Defaults to 1.
            mlp_dropout (float, optional): Dropout rate for the MLP. Defaults to 0.0.
            **kwargs: Additional arguments to pass to DimeNet++.
        """
        super().__init__()

        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience

        # Initialize the DimeNet++ model
        self.gnn = DimeNetPlusPlus(
            hidden_channels=128,
            out_channels=256,
            num_blocks=4,
            int_emb_size=64,
            basis_emb_size=8,
            out_emb_channels=256,
            num_spherical=7,
            num_radial=6,
            act="swish",
            return_embeddings=True,
            **kwargs,
        )

        # Set the readout method
        if readout == "SAG":
            self.readout = pool.SAGPooling(mlp_input_dim, nonlinearity="swish")
        if readout == "ASA":
            self.readout = pool.ASAPooling(mlp_input_dim)
        if readout == "sum":
            self.readout = "sum"
        if readout == "mean":
            self.readout = "mean"
        if readout == "max":
            self.readout = "max"

        # Initialize the MLP
        self.mlp = MLP(
            input_dim=mlp_input_dim, output_dim=mlp_output_dim, dropout=mlp_dropout
        )

        self.loss = torch.nn.MSELoss()
        self.mae = torch.nn.L1Loss()
        self.r2 = R2Score()

    def forward(
        self, data: Data, mode: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            data (Data): PyTorch Geometric Data object containing the graph data.
            mode (str | None, optional): The mode of operation ('train', 'val', 'test', 'predict').
                This can be used to modify behavior during different phases. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - logits: The predicted values
                - y: The true target values
        """
        z, pos, y, batch, edge_index, rxn_conds = (
            data.z,
            data.pos,
            data.y,
            data.batch,
            data.edge_index,
            data.rxn_conds,
        )

        # Get graph embeddings
        P = self.gnn(z=z, pos=pos, edge_index=edge_index, batch=batch)
        P_dense, _ = to_dense_batch(P, batch, 0, 300)  # 300 is the max number of atoms)

        if self.readout == "sum":
            graph_embeddings = pool.global_add_pool(P, batch)

        if self.readout == "mean":
            graph_embeddings = pool.global_mean_pool(P, batch)

        if self.readout == "max":
            graph_embeddings = pool.global_max_pool(P, batch)

        if isinstance(self.readout, pool.SAGPooling):
            P, _, _, new_batch, _, _ = self.readout(
                x=P, edge_index=edge_index, batch=batch
            )
            graph_embeddings = pool.global_add_pool(P, new_batch)

        if isinstance(self.readout, pool.ASAPooling):
            P, _, _, new_batch, _ = self.readout(
                x=P, edge_index=edge_index, batch=batch
            )
            graph_embeddings = pool.global_add_pool(P, new_batch)

        # Pass the graph embeddings through the MLP
        embedding = torch.cat([graph_embeddings, rxn_conds], dim=-1)

        logits, mlp_embeddings = self.mlp(embedding)

        return logits.squeeze(), y, [embedding] + mlp_embeddings

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch (Data): Batch of data for training.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The calculated loss.
        """
        logits, y = self.forward(batch, mode="train")
        loss = self.loss(logits, y)
        self.log("train_loss", loss, batch_size=len(batch), sync_dist=True)
        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> None:
        """Validation step.

        Args:
            batch (Data): Batch of data for validation.
            batch_idx (int): Index of the batch.
        """
        logits, y = self.forward(batch, mode="val")
        loss = self.loss(logits, y)
        self.log("val_loss", loss, prog_bar=True, batch_size=len(batch), sync_dist=True)

    def modified_forward(
        self, batch: Data, batch_idx: int, dataloader_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the logits, and true values.

        Args:
            batch (Data): Batch of data.
            batch_idx (int): Index of the batch.
            dataloader_idx (int): Index of the dataloader.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - logits: The predicted values
                - y: The true target values
        """
        logits, y = self.forward(batch, mode="test")
        return logits, y

    def test_step(
        self, batch: Data, batch_idx: int, dataloader_idx: int
    ) -> torch.Tensor:
        """Test step. Calculates and returns MSE loss.

        Args:
            batch (Data): Batch of data for testing.
            batch_idx (int): Index of the batch.
            dataloader_idx (int): Index of the dataloader.

        Returns:
            torch.Tensor: The calculated MSE loss.
        """
        logits, y, _ = self.forward(batch, mode="test")
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        loss = self.loss(logits, y)
        self.log("test_loss", loss, batch_size=len(batch), sync_dist=True)
        return loss

    def test_step_mae(
        self, batch: Data, batch_idx: int, dataloader_idx: int
    ) -> torch.Tensor:
        """Test step. Calculates and returns MAE loss.

        Args:
            batch (Data): Batch of data for testing.
            batch_idx (int): Index of the batch.
            dataloader_idx (int): Index of the dataloader.

        Returns:
            torch.Tensor: The calculated MAE loss.
        """
        logits, y = self.forward(batch, mode="test")
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        loss = self.mae(logits, y)
        return loss

    def test_step_r2(
        self, batch: Data, batch_idx: int, dataloader_idx: int
    ) -> torch.Tensor:
        """Test step. Calculates and returns R² score.

        Args:
            batch (Data): Batch of data for testing.
            batch_idx (int): Index of the batch.
            dataloader_idx (int): Index of the dataloader.

        Returns:
            torch.Tensor: The calculated R² score.
        """
        logits, y = self.forward(batch, mode="test")
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        loss = self.r2(logits, y)
        return loss

    def predict_step(
        self, data: Data, batch_idx: int, dataloader_idx: int
    ) -> torch.Tensor:
        """Prediction step. Returns the model predictions.

        Args:
            data (Data): Batch of data for prediction.
            batch_idx (int): Index of the batch.
            dataloader_idx (int): Index of the dataloader.

        Returns:
            torch.Tensor: The model predictions.
        """
        logits, _ = self(data, mode="predict")
        return logits

    def configure_optimizers(self) -> dict:
        """Configures the optimizer and learning rate scheduler for training.

        Returns:
            dict: A dictionary containing the optimizer, scheduler and monitored metric.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.75, patience=self.scheduler_patience
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
