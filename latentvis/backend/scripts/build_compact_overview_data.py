#!/usr/bin/env python3
"""
Build compact overview datasets from existing generated case outputs.

The script does not modify current pipeline outputs. It reads the existing
generated data for the three supported cases and writes a new compact format
under a separate output root.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "data_compact"
PROJECTION_METHODS = ("default", "pca", "mds", "umap", "tsne")
SUPPORTED_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitSource:
    split: str
    scatter_dir: Optional[Path]
    cluster_csv: Optional[Path]
    representatives_json: Optional[Path]
    sankey_json: Optional[Path]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    metadata_json: Path
    split_sources: Tuple[SplitSource, ...]
    source_variant: str


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def sort_epoch_keys(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        ordered.append(text)
        seen.add(text)

    def key_fn(item: str) -> Tuple[int, Any]:
        return (0, int(item)) if item.isdigit() else (1, item)

    return sorted(ordered, key=key_fn)


def infer_layers_from_csv(fieldnames: Sequence[str]) -> List[str]:
    layers: List[str] = []
    for name in fieldnames:
        if name.endswith("_cluster"):
            layers.append(name[: -len("_cluster")])
    return layers


def parse_cluster_value(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def normalize_real_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    text = str(value).strip()
    if re.fullmatch(r"-?\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def load_metadata_rows(path: Path) -> List[Dict[str, Any]]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Metadata JSON must be a list: {path}")
    return [row for row in rows if isinstance(row, dict)]


def canonical_names_and_splits(metadata_rows: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, List[int]], Dict[str, int]]:
    molecule_names: List[str] = []
    molecule_name_to_id: Dict[str, int] = {}
    split_ids: Dict[str, List[int]] = {split: [] for split in SUPPORTED_SPLITS}

    for row in metadata_rows:
        name = normalize_real_id(row.get("real_id") or row.get("id"))
        if not name:
            continue
        if name in molecule_name_to_id:
            raise ValueError(f"Duplicate molecule name found in metadata: {name}")
        molecule_id = len(molecule_names)
        molecule_names.append(name)
        molecule_name_to_id[name] = molecule_id
        split_name = str(row.get("split") or "").strip().lower()
        if split_name in split_ids:
            split_ids[split_name].append(molecule_id)

    return molecule_names, split_ids, molecule_name_to_id


def load_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def load_ordering(split_sources: Sequence[SplitSource]) -> Tuple[List[str], List[str]]:
    for source in split_sources:
        if source.sankey_json and source.sankey_json.is_file():
            payload = read_json(source.sankey_json)
            if not isinstance(payload, dict):
                continue
            epoch_block = payload.get("epoch") or {}
            layer_block = payload.get("layer") or {}
            if isinstance(epoch_block, dict) and isinstance(layer_block, dict) and epoch_block and layer_block:
                return [str(key) for key in epoch_block.keys()], [str(key) for key in layer_block.keys()]

    epoch_values: List[str] = []
    layer_values: List[str] = []
    for source in split_sources:
        if not source.cluster_csv or not source.cluster_csv.is_file():
            continue
        fieldnames, rows = load_csv_rows(source.cluster_csv)
        layer_values.extend(infer_layers_from_csv(fieldnames))
        epoch_values.extend(str(row.get("epoch") or row.get("model_type") or "").strip() for row in rows)
    epochs = [value for value in sort_epoch_keys(epoch_values) if value]
    layers = infer_layers_from_values([value for value in layer_values if value])
    return epochs, layers


def infer_layers_from_values(values: Sequence[str]) -> List[str]:
    unique_layers: List[str] = []
    seen_layers = set()
    for value in values:
        if value in seen_layers:
            continue
        unique_layers.append(value)
        seen_layers.add(value)
    return unique_layers


def build_scatter_path(scatter_dir: Path, epoch: str, layer: str, method: str) -> Path:
    suffix = "" if method == "default" else f"_{method}"
    return scatter_dir / f"{epoch}_{layer}{suffix}.json"


def empty_numeric_array(size: int) -> List[Optional[float]]:
    return [None] * size


def empty_int_array(size: int) -> List[Optional[int]]:
    return [None] * size


def load_dr_payload(
    scatter_dir: Optional[Path],
    epoch: str,
    layer: str,
    method: str,
    molecule_name_to_id: Dict[str, int],
    n_molecules: int,
) -> Optional[Dict[str, List[Optional[float]]]]:
    if scatter_dir is None:
        return None
    path = build_scatter_path(scatter_dir, epoch, layer, method)
    if not path.is_file():
        return None
    rows = read_json(path)
    if not isinstance(rows, list):
        return None
    x_values = empty_numeric_array(n_molecules)
    y_values = empty_numeric_array(n_molecules)
    found_any = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        molecule_name = normalize_real_id(row.get("real_id"))
        if molecule_name not in molecule_name_to_id:
            continue
        molecule_id = molecule_name_to_id[molecule_name]
        try:
            x_values[molecule_id] = float(row.get("x"))
            y_values[molecule_id] = float(row.get("y"))
        except (TypeError, ValueError):
            continue
        found_any = True
    if not found_any:
        return None
    return {"x": x_values, "y": y_values}


def build_split_cell_payloads(
    source: SplitSource,
    epochs: Sequence[str],
    layers: Sequence[str],
    molecule_name_to_id: Dict[str, int],
    n_molecules: int,
) -> Tuple[Dict[str, Dict[str, List[Optional[int]]]], Dict[str, Dict[int, set]], Dict[str, Dict[str, Dict[str, Dict[str, List[Optional[float]]]]]]]:
    labels_by_cell: Dict[str, Dict[str, List[Optional[int]]]] = {
        epoch: {layer: empty_int_array(n_molecules) for layer in layers}
        for epoch in epochs
    }
    memberships_by_cell: Dict[str, Dict[str, set[int]]] = {
        epoch: {layer: set() for layer in layers}
        for epoch in epochs
    }
    dr_by_cell: Dict[str, Dict[str, Dict[str, Dict[str, List[Optional[float]]]]]] = {
        epoch: {layer: {} for layer in layers}
        for epoch in epochs
    }

    if source.cluster_csv and source.cluster_csv.is_file():
        _fieldnames, rows = load_csv_rows(source.cluster_csv)
        for row in rows:
            epoch = str(row.get("epoch") or row.get("model_type") or "").strip()
            if epoch not in labels_by_cell:
                continue
            molecule_name = normalize_real_id(row.get("real_id"))
            if molecule_name not in molecule_name_to_id:
                continue
            molecule_id = molecule_name_to_id[molecule_name]
            for layer in layers:
                cluster_id = parse_cluster_value(row.get(f"{layer}_cluster"))
                if cluster_id is None:
                    continue
                labels_by_cell[epoch][layer][molecule_id] = cluster_id
                memberships_by_cell[epoch][layer].add(molecule_id)

    for epoch in epochs:
        for layer in layers:
            for method in PROJECTION_METHODS:
                payload = load_dr_payload(
                    scatter_dir=source.scatter_dir,
                    epoch=epoch,
                    layer=layer,
                    method=method,
                    molecule_name_to_id=molecule_name_to_id,
                    n_molecules=n_molecules,
                )
                if payload is not None:
                    dr_by_cell[epoch][layer][method] = payload

    return labels_by_cell, memberships_by_cell, dr_by_cell


def build_representative_payloads(
    source: SplitSource,
    epochs: Sequence[str],
    layers: Sequence[str],
    molecule_name_to_id: Dict[str, int],
    labels_by_cell: Dict[str, Dict[str, List[Optional[int]]]],
) -> Dict[str, Dict[str, List[Optional[int]]]]:
    representatives: Dict[str, Dict[str, List[Optional[int]]]] = {
        epoch: {layer: [] for layer in layers}
        for epoch in epochs
    }
    if not source.representatives_json or not source.representatives_json.is_file():
        return representatives

    rows = read_json(source.representatives_json)
    if not isinstance(rows, list):
        return representatives

    max_cluster_by_cell: Dict[Tuple[str, str], int] = {}
    for epoch in epochs:
        for layer in layers:
            max_cluster = max((value for value in labels_by_cell[epoch][layer] if value is not None), default=-1)
            max_cluster_by_cell[(epoch, layer)] = max_cluster

    temp: Dict[Tuple[str, str], Dict[int, int]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        epoch = str(row.get("model_type") or row.get("epoch") or "").strip()
        layer = str(row.get("layer_name") or row.get("layer") or "").strip()
        cluster_id = parse_cluster_value(row.get("cluster_id"))
        candidates = row.get("top_k_closest_real_ids")
        if epoch not in representatives or layer not in representatives[epoch] or cluster_id is None:
            continue
        if not isinstance(candidates, list) or not candidates:
            continue
        representative_name = normalize_real_id(candidates[0])
        molecule_id = molecule_name_to_id.get(representative_name)
        if molecule_id is None:
            continue
        temp.setdefault((epoch, layer), {})[cluster_id] = molecule_id

    for epoch in epochs:
        for layer in layers:
            max_cluster = max_cluster_by_cell[(epoch, layer)]
            if max_cluster < 0:
                representatives[epoch][layer] = []
                continue
            payload = [None] * (max_cluster + 1)
            for cluster_id, molecule_id in temp.get((epoch, layer), {}).items():
                if 0 <= cluster_id < len(payload):
                    payload[cluster_id] = molecule_id
            representatives[epoch][layer] = payload

    return representatives


def build_connectivity_matrices(
    epochs: Sequence[str],
    layers: Sequence[str],
    memberships_by_split: Dict[str, Dict[str, Dict[int, set]]],
) -> Tuple[List[int], List[str], List[List[int]], Dict[str, List[List[int]]], Dict[str, Dict[str, int]], List[Dict[str, Any]]]:
    total_layers_per_epoch = len(layers)
    total_nodes = len(epochs) * len(layers)
    node_keys = list(range(total_nodes))
    names = ["" for _ in range(total_nodes)]
    node_lookup: Dict[str, Dict[str, int]] = {}
    nodes: List[Dict[str, Any]] = []

    for epoch_idx, epoch in enumerate(epochs):
        node_lookup[epoch] = {}
        for layer_idx, layer in enumerate(layers):
            global_index = layer_idx + (epoch_idx * total_layers_per_epoch)
            name = f"epoch_{epoch}__layer_{layer}"
            names[global_index] = name
            node_lookup[epoch][layer] = global_index
            nodes.append(
                {
                    "node_key": global_index,
                    "name": name,
                    "epoch": epoch,
                    "layer": layer,
                    "epoch_idx": epoch_idx,
                    "layer_idx": layer_idx,
                    "global_node_index": global_index,
                }
            )

    split_matrices: Dict[str, List[List[int]]] = {
        split: [[0 for _ in range(total_nodes)] for _ in range(total_nodes)]
        for split in SUPPORTED_SPLITS
    }

    for split, memberships in memberships_by_split.items():
        matrix = split_matrices[split]
        for epoch_idx, epoch in enumerate(epochs):
            for layer_idx in range(len(layers) - 1):
                source_layer = layers[layer_idx]
                target_layer = layers[layer_idx + 1]
                source_ids = memberships[epoch][source_layer]
                target_ids = memberships[epoch][target_layer]
                count = len(source_ids & target_ids)
                if count <= 0:
                    continue
                source_index = layer_idx + (epoch_idx * total_layers_per_epoch)
                target_index = (layer_idx + 1) + (epoch_idx * total_layers_per_epoch)
                matrix[source_index][target_index] = count

        for epoch_idx in range(len(epochs) - 1):
            source_epoch = epochs[epoch_idx]
            target_epoch = epochs[epoch_idx + 1]
            for layer_idx, layer in enumerate(layers):
                source_ids = memberships[source_epoch][layer]
                target_ids = memberships[target_epoch][layer]
                count = len(source_ids & target_ids)
                if count <= 0:
                    continue
                source_index = layer_idx + (epoch_idx * total_layers_per_epoch)
                target_index = layer_idx + ((epoch_idx + 1) * total_layers_per_epoch)
                matrix[source_index][target_index] = count

    combined_matrix = [[0 for _ in range(total_nodes)] for _ in range(total_nodes)]
    for split in SUPPORTED_SPLITS:
        matrix = split_matrices[split]
        for source_index in range(total_nodes):
            for target_index in range(total_nodes):
                combined_matrix[source_index][target_index] += matrix[source_index][target_index]

    return node_keys, names, combined_matrix, split_matrices, node_lookup, nodes


def source_paths_payload(split_sources: Sequence[SplitSource]) -> Dict[str, Dict[str, Optional[str]]]:
    payload: Dict[str, Dict[str, Optional[str]]] = {}
    for source in split_sources:
        payload[source.split] = {
            "scatter_dir": str(source.scatter_dir) if source.scatter_dir else None,
            "cluster_csv": str(source.cluster_csv) if source.cluster_csv else None,
            "representatives_json": str(source.representatives_json) if source.representatives_json else None,
            "sankey_json": str(source.sankey_json) if source.sankey_json else None,
        }
    return payload


def validate_overview(overview: Dict[str, Any]) -> None:
    molecule_names = overview["molecule_names"]
    n_molecules = len(molecule_names)
    expected_ids = list(range(n_molecules))
    if overview["molecule_ids"] != expected_ids:
        raise ValueError("Molecule IDs are not sequential integers.")

    split_ids = overview["splits"]
    for split in SUPPORTED_SPLITS:
        ids = split_ids[split]
        if any(not isinstance(value, int) for value in ids):
            raise ValueError(f"Split {split} contains a non-integer ID.")
        if any(value < 0 or value >= n_molecules for value in ids):
            raise ValueError(f"Split {split} contains an out-of-range ID.")

    sankey = overview["sankey"]
    epochs = overview["epochs"]
    layers = overview["layers"]
    expected_node_count = len(epochs) * len(layers)
    if sankey["node_keys"] != list(range(expected_node_count)):
        raise ValueError("Sankey node keys are not sequential integers.")
    if len(sankey["names"]) != expected_node_count:
        raise ValueError("Sankey names length does not match epoch/layer grid size.")
    if len(sankey["connectivity_matrix"]) != expected_node_count:
        raise ValueError("Sankey matrix row count is invalid.")
    for row in sankey["connectivity_matrix"]:
        if len(row) != expected_node_count:
            raise ValueError("Sankey matrix column count is invalid.")

    for node in overview["cells"]:
        if node["node_key"] != node["global_node_index"]:
            raise ValueError("Cell node_key must equal global_node_index.")
        epoch_idx = node["epoch_idx"]
        layer_idx = node["layer_idx"]
        expected_index = layer_idx + (epoch_idx * len(layers))
        if node["global_node_index"] != expected_index:
            raise ValueError("Global node index formula mismatch.")
        for split in SUPPORTED_SPLITS:
            split_payload = node["split_data"][split]
            labels = split_payload["cluster_labels"]
            if len(labels) != n_molecules:
                raise ValueError("Cluster label array length mismatch.")
            representatives = split_payload["cluster_representative_ids"]
            if any(value is not None and not isinstance(value, int) for value in representatives):
                raise ValueError("Representative ID array contains an invalid value.")
            for method, coords in split_payload["dr"].items():
                if method not in PROJECTION_METHODS:
                    raise ValueError(f"Unexpected DR method: {method}")
                if len(coords["x"]) != n_molecules or len(coords["y"]) != n_molecules:
                    raise ValueError("DR coordinate array length mismatch.")


def build_overview(config: DatasetConfig) -> Dict[str, Any]:
    metadata_rows = load_metadata_rows(config.metadata_json)
    molecule_names, split_ids, molecule_name_to_id = canonical_names_and_splits(metadata_rows)
    epochs, layers = load_ordering(config.split_sources)
    layers = infer_layers_from_values(layers)
    if not epochs or not layers:
        raise ValueError(f"Unable to infer epochs/layers for dataset {config.name}")

    n_molecules = len(molecule_names)
    labels_by_split: Dict[str, Dict[str, Dict[str, List[Optional[int]]]]] = {}
    dr_by_split: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, List[Optional[float]]]]]]] = {}
    representatives_by_split: Dict[str, Dict[str, Dict[str, List[Optional[int]]]]] = {}
    memberships_by_split: Dict[str, Dict[str, Dict[int, set]]] = {}

    for source in config.split_sources:
        labels_by_cell, memberships_by_cell, dr_by_cell = build_split_cell_payloads(
            source=source,
            epochs=epochs,
            layers=layers,
            molecule_name_to_id=molecule_name_to_id,
            n_molecules=n_molecules,
        )
        representatives = build_representative_payloads(
            source=source,
            epochs=epochs,
            layers=layers,
            molecule_name_to_id=molecule_name_to_id,
            labels_by_cell=labels_by_cell,
        )
        labels_by_split[source.split] = labels_by_cell
        dr_by_split[source.split] = dr_by_cell
        memberships_by_split[source.split] = memberships_by_cell
        representatives_by_split[source.split] = representatives

    for split in SUPPORTED_SPLITS:
        if split in labels_by_split:
            continue
        labels_by_split[split] = {epoch: {layer: empty_int_array(n_molecules) for layer in layers} for epoch in epochs}
        dr_by_split[split] = {epoch: {layer: {} for layer in layers} for epoch in epochs}
        memberships_by_split[split] = {epoch: {layer: set() for layer in layers} for epoch in epochs}
        representatives_by_split[split] = {epoch: {layer: [] for layer in layers} for epoch in epochs}

    node_keys, names, connectivity_matrix, split_matrices, node_lookup, nodes = build_connectivity_matrices(
        epochs=epochs,
        layers=layers,
        memberships_by_split=memberships_by_split,
    )

    cell_payloads: List[Dict[str, Any]] = []
    for node in sorted(nodes, key=lambda item: item["global_node_index"]):
        epoch = node["epoch"]
        layer = node["layer"]
        split_data: Dict[str, Any] = {}
        for split in SUPPORTED_SPLITS:
            split_data[split] = {
                "cluster_labels": labels_by_split[split][epoch][layer],
                "cluster_representative_ids": representatives_by_split[split][epoch][layer],
                "dr": dr_by_split[split][epoch][layer],
            }
        cell_payload = dict(node)
        cell_payload["split_data"] = split_data
        cell_payloads.append(cell_payload)

    overview = {
        "api_version": "compact_overview_v1",
        "dataset": config.name,
        "source_variant": config.source_variant,
        "source_paths": source_paths_payload(config.split_sources),
        "molecule_ids": list(range(n_molecules)),
        "molecule_names": molecule_names,
        "splits": split_ids,
        "epochs": list(epochs),
        "layers": list(layers),
        "cells": cell_payloads,
        "sankey": {
            "node_keys": node_keys,
            "names": names,
            "connectivity_matrix": connectivity_matrix,
            "split_connectivity_matrices": split_matrices,
            "bookkeeping": {
                "epochs": list(epochs),
                "layers": list(layers),
                "total_layers_per_epoch": len(layers),
                "global_node_index_formula": "layer_idx + (epoch_idx * total_layers_per_epoch)",
                "node_lookup": node_lookup,
                "nodes": nodes,
            },
        },
    }

    validate_overview(overview)
    return overview


def dataset_configs() -> Tuple[DatasetConfig, ...]:
    chemprop_root = ROOT / "data" / "chemprop"
    chemeleon_root = ROOT / "data" / "chemeleon"
    dimenet_root = ROOT / "data" / "dimenet"

    return (
        DatasetConfig(
            name="chemprop",
            metadata_json=chemprop_root / "id_smiles_y_true_y_pred.json",
            source_variant="knn",
            split_sources=(
                SplitSource(
                    split="train",
                    scatter_dir=chemprop_root / "train" / "knn" / "scatterplot",
                    cluster_csv=chemprop_root / "train" / "knn" / "clustering_metrics" / "cluster_assignments_with_real_id.csv",
                    representatives_json=chemprop_root / "train" / "knn" / "topk" / "cluster_representatives.json",
                    sankey_json=chemprop_root / "train" / "knn" / "sankey" / "all_sankey.json",
                ),
                SplitSource(
                    split="test",
                    scatter_dir=chemprop_root / "test" / "knn" / "scatterplot",
                    cluster_csv=chemprop_root / "test" / "knn" / "clustering_metrics" / "cluster_assignments_with_real_id.csv",
                    representatives_json=chemprop_root / "test" / "knn" / "topk" / "cluster_representatives.json",
                    sankey_json=chemprop_root / "test" / "knn" / "sankey" / "all_sankey.json",
                ),
            ),
        ),
        DatasetConfig(
            name="chemeleon",
            metadata_json=chemeleon_root / "id_smiles_y_true_y_pred.json",
            source_variant="knn",
            split_sources=(
                SplitSource(
                    split="train",
                    scatter_dir=chemeleon_root / "train" / "knn" / "scatterplot",
                    cluster_csv=chemeleon_root / "train" / "knn" / "clustering_metrics" / "cluster_assignments_with_real_id.csv",
                    representatives_json=chemeleon_root / "train" / "knn" / "topk" / "cluster_representatives.json",
                    sankey_json=chemeleon_root / "train" / "knn" / "sankey" / "all_sankey.json",
                ),
                SplitSource(
                    split="test",
                    scatter_dir=chemeleon_root / "test" / "knn" / "scatterplot",
                    cluster_csv=chemeleon_root / "test" / "knn" / "clustering_metrics" / "cluster_assignments_with_real_id.csv",
                    representatives_json=chemeleon_root / "test" / "knn" / "topk" / "cluster_representatives.json",
                    sankey_json=chemeleon_root / "test" / "knn" / "sankey" / "all_sankey.json",
                ),
            ),
        ),
        DatasetConfig(
            name="dimenet",
            metadata_json=dimenet_root / "id_smiles_y_true_y_pred.json",
            source_variant="default",
            split_sources=(
                SplitSource(
                    split="test",
                    scatter_dir=dimenet_root / "test" / "scatterplot",
                    cluster_csv=dimenet_root / "test" / "clustering_metrics" / "cluster_assignments_with_real_id.csv",
                    representatives_json=dimenet_root / "test" / "topk" / "cluster_representatives.json",
                    sankey_json=dimenet_root / "test" / "sankey" / "all_sankey.json",
                ),
            ),
        ),
    )


def build_all(output_root: Path) -> Dict[str, str]:
    manifest: Dict[str, str] = {}
    for config in dataset_configs():
        overview = build_overview(config)
        output_path = output_root / config.name / "overview.json"
        write_json(output_path, overview)
        manifest[config.name] = str(output_path)
    write_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build compact overview datasets from existing generated cases.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output directory for the compact overview datasets.",
    )
    args = parser.parse_args()
    manifest = build_all(args.output_root.resolve())
    print(json.dumps({"ok": True, "output_root": str(args.output_root.resolve()), "manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
