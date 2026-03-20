#!/usr/bin/env python3
"""
Build compact overview datasets from existing generated case outputs.

The compact format is intentionally sparse:
- DR coordinates are rounded and stored once per node across all molecule IDs.
- Cluster labels are stored once per node across all splits.
- Connectivity is stored as sparse adjacency lists rather than dense matrices.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "data_compact"
DEFAULT_COORD_DECIMALS = 2
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


def infer_layers_from_values(values: Sequence[str]) -> List[str]:
    unique_layers: List[str] = []
    seen_layers = set()
    for value in values:
        if value in seen_layers:
            continue
        unique_layers.append(value)
        seen_layers.add(value)
    return unique_layers


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


def round_coordinate(value: Any, coord_decimals: int) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), coord_decimals)
    except (TypeError, ValueError):
        return None


def load_metadata_rows(path: Path) -> List[Dict[str, Any]]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Metadata JSON must be a list: {path}")
    return [row for row in rows if isinstance(row, dict)]


def canonical_metadata(
    metadata_rows: List[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, List[int]], Dict[str, int]]:
    molecule_labels: List[str] = []
    molecule_label_to_id: Dict[str, int] = {}
    split_ids: Dict[str, List[int]] = {split: [] for split in SUPPORTED_SPLITS}

    for row in metadata_rows:
        label = normalize_real_id(row.get("real_id") or row.get("id"))
        if not label:
            continue
        if label in molecule_label_to_id:
            raise ValueError(f"Duplicate molecule label found in metadata: {label}")
        molecule_id = len(molecule_labels)
        molecule_labels.append(label)
        molecule_label_to_id[label] = molecule_id
        split_name = str(row.get("split") or "").strip().lower()
        if split_name in split_ids:
            split_ids[split_name].append(molecule_id)

    return molecule_labels, split_ids, molecule_label_to_id


def maybe_molecule_labels_payload(molecule_labels: Sequence[str]) -> Optional[List[str]]:
    if all(label == str(idx) for idx, label in enumerate(molecule_labels)):
        return None
    return list(molecule_labels)


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


def build_scatter_path(scatter_dir: Path, epoch: str, layer: str, method: str) -> Path:
    suffix = "" if method == "default" else f"_{method}"
    return scatter_dir / f"{epoch}_{layer}{suffix}.json"


def load_dr_payload(
    scatter_dir: Optional[Path],
    epoch: str,
    layer: str,
    method: str,
    molecule_label_to_id: Dict[str, int],
    coord_decimals: int,
) -> Optional[Dict[str, List[float]]]:
    if scatter_dir is None:
        return None
    path = build_scatter_path(scatter_dir, epoch, layer, method)
    if not path.is_file():
        return None
    rows = read_json(path)
    if not isinstance(rows, list):
        return None

    coords: Dict[str, List[float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        molecule_label = normalize_real_id(row.get("real_id"))
        molecule_id = molecule_label_to_id.get(molecule_label)
        if molecule_id is None:
            continue
        x_value = round_coordinate(row.get("x"), coord_decimals)
        y_value = round_coordinate(row.get("y"), coord_decimals)
        if x_value is None or y_value is None:
            continue
        coords[str(molecule_id)] = [x_value, y_value]

    if not coords:
        return None
    return coords


def build_split_payloads(
    source: SplitSource,
    epochs: Sequence[str],
    layers: Sequence[str],
    molecule_label_to_id: Dict[str, int],
    coord_decimals: int,
) -> Tuple[
    Dict[str, Dict[str, Dict[int, int]]],
    Dict[str, Dict[str, Set[int]]],
    Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
]:
    labels_by_cell: Dict[str, Dict[str, Dict[int, int]]] = {
        epoch: {layer: {} for layer in layers}
        for epoch in epochs
    }
    memberships_by_cell: Dict[str, Dict[str, Set[int]]] = {
        epoch: {layer: set() for layer in layers}
        for epoch in epochs
    }
    dr_by_cell: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = {
        epoch: {layer: {} for layer in layers}
        for epoch in epochs
    }

    if source.cluster_csv and source.cluster_csv.is_file():
        _fieldnames, rows = load_csv_rows(source.cluster_csv)
        for row in rows:
            epoch = str(row.get("epoch") or row.get("model_type") or "").strip()
            if epoch not in labels_by_cell:
                continue
            molecule_label = normalize_real_id(row.get("real_id"))
            molecule_id = molecule_label_to_id.get(molecule_label)
            if molecule_id is None:
                continue
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
                    molecule_label_to_id=molecule_label_to_id,
                    coord_decimals=coord_decimals,
                )
                if payload is not None:
                    dr_by_cell[epoch][layer][method] = payload

    return labels_by_cell, memberships_by_cell, dr_by_cell


def build_representative_payloads(
    source: SplitSource,
    epochs: Sequence[str],
    layers: Sequence[str],
    molecule_label_to_id: Dict[str, int],
) -> Dict[str, Dict[str, Dict[str, int]]]:
    representatives: Dict[str, Dict[str, Dict[str, int]]] = {
        epoch: {layer: {} for layer in layers}
        for epoch in epochs
    }
    if not source.representatives_json or not source.representatives_json.is_file():
        return representatives

    rows = read_json(source.representatives_json)
    if not isinstance(rows, list):
        return representatives

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
        representative_label = normalize_real_id(candidates[0])
        molecule_id = molecule_label_to_id.get(representative_label)
        if molecule_id is None:
            continue
        representatives[epoch][layer][str(cluster_id)] = molecule_id

    return representatives


def build_sparse_connectivity(
    epochs: Sequence[str],
    layers: Sequence[str],
    memberships_by_split: Dict[str, Dict[str, Dict[str, Set[int]]]],
) -> Tuple[Dict[str, List[List[int]]], Dict[str, Dict[str, List[List[int]]]], Dict[str, Dict[str, int]]]:
    total_layers_per_epoch = len(layers)
    node_lookup: Dict[str, Dict[str, int]] = {}
    for epoch_idx, epoch in enumerate(epochs):
        node_lookup[epoch] = {}
        for layer_idx, layer in enumerate(layers):
            node_lookup[epoch][layer] = layer_idx + (epoch_idx * total_layers_per_epoch)

    split_edges: Dict[str, Dict[str, List[List[int]]]] = {split: {} for split in SUPPORTED_SPLITS}

    for split, memberships in memberships_by_split.items():
        adjacency: Dict[str, List[List[int]]] = {}
        for epoch_idx, epoch in enumerate(epochs):
            for layer_idx in range(len(layers) - 1):
                source_layer = layers[layer_idx]
                target_layer = layers[layer_idx + 1]
                count = len(memberships[epoch][source_layer] & memberships[epoch][target_layer])
                if count <= 0:
                    continue
                source_index = node_lookup[epoch][source_layer]
                target_index = node_lookup[epoch][target_layer]
                adjacency.setdefault(str(source_index), []).append([target_index, count])

        for epoch_idx in range(len(epochs) - 1):
            source_epoch = epochs[epoch_idx]
            target_epoch = epochs[epoch_idx + 1]
            for layer in layers:
                count = len(memberships[source_epoch][layer] & memberships[target_epoch][layer])
                if count <= 0:
                    continue
                source_index = node_lookup[source_epoch][layer]
                target_index = node_lookup[target_epoch][layer]
                adjacency.setdefault(str(source_index), []).append([target_index, count])

        for source_index, edges in adjacency.items():
            edges.sort(key=lambda item: item[0])
        split_edges[split] = adjacency

    combined: Dict[str, Dict[int, int]] = {}
    for split in SUPPORTED_SPLITS:
        for source_index, edges in split_edges[split].items():
            bucket = combined.setdefault(source_index, {})
            for target_index, count in edges:
                bucket[target_index] = bucket.get(target_index, 0) + count

    connectivity = {
        source_index: [[target_index, count] for target_index, count in sorted(target_counts.items())]
        for source_index, target_counts in sorted(combined.items(), key=lambda item: int(item[0]))
    }

    return connectivity, split_edges, node_lookup


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


def validate_sparse_edges(edges: Dict[str, List[List[int]]], n_nodes: int) -> None:
    for source_key, targets in edges.items():
        source_index = int(source_key)
        if source_index < 0 or source_index >= n_nodes:
            raise ValueError("Connectivity source index out of range.")
        if not isinstance(targets, list):
            raise ValueError("Connectivity payload must contain adjacency lists.")
        for item in targets:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("Connectivity edge must be a two-item list.")
            target_index, count = item
            if not isinstance(target_index, int) or target_index < 0 or target_index >= n_nodes:
                raise ValueError("Connectivity target index out of range.")
            if not isinstance(count, int) or count <= 0:
                raise ValueError("Connectivity counts must be positive integers.")


def validate_overview(overview: Dict[str, Any]) -> None:
    molecule_ids = overview["molecule_ids"]
    n_molecules = len(molecule_ids)
    if molecule_ids != list(range(n_molecules)):
        raise ValueError("Molecule IDs are not sequential integers.")

    molecule_labels = overview.get("molecule_labels")
    if molecule_labels is not None and len(molecule_labels) != n_molecules:
        raise ValueError("Molecule labels length mismatch.")

    for split in SUPPORTED_SPLITS:
        ids = overview["splits"][split]
        if any(not isinstance(value, int) for value in ids):
            raise ValueError(f"Split {split} contains a non-integer ID.")
        if any(value < 0 or value >= n_molecules for value in ids):
            raise ValueError(f"Split {split} contains an out-of-range ID.")

    epochs = overview["epochs"]
    layers = overview["layers"]
    expected_node_count = len(epochs) * len(layers)
    if len(overview["cells"]) != expected_node_count:
        raise ValueError("Cell count does not match epoch/layer grid size.")

    for cell in overview["cells"]:
        epoch_idx = cell["epoch_idx"]
        layer_idx = cell["layer_idx"]
        expected_index = layer_idx + (epoch_idx * len(layers))
        if cell["global_node_index"] != expected_index or cell["node_key"] != expected_index:
            raise ValueError("Cell node indexing mismatch.")

        for molecule_id, cluster_id in cell["cluster_labels"].items():
            molecule_index = int(molecule_id)
            if molecule_index < 0 or molecule_index >= n_molecules or not isinstance(cluster_id, int):
                raise ValueError("Cluster labels contain invalid entries.")

        for split in SUPPORTED_SPLITS:
            representatives = cell["cluster_representatives"][split]
            if not isinstance(representatives, dict):
                raise ValueError("Cluster representatives must be stored as a sparse dict.")
            for cluster_id, molecule_id in representatives.items():
                int(cluster_id)
                if not isinstance(molecule_id, int) or molecule_id < 0 or molecule_id >= n_molecules:
                    raise ValueError("Representative ID is invalid.")

        for method, coords in cell["dr"].items():
            if method not in PROJECTION_METHODS:
                raise ValueError(f"Unexpected DR method: {method}")
            if not isinstance(coords, dict):
                raise ValueError("DR payload must be stored as a sparse dict.")
            for molecule_id, pair in coords.items():
                molecule_index = int(molecule_id)
                if molecule_index < 0 or molecule_index >= n_molecules:
                    raise ValueError("DR payload contains an invalid molecule ID.")
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError("DR coordinates must be stored as [x, y].")
                if not all(isinstance(value, (int, float)) for value in pair):
                    raise ValueError("DR coordinates must be numeric.")

    sankey = overview["sankey"]
    validate_sparse_edges(sankey["connectivity"], expected_node_count)
    for split in SUPPORTED_SPLITS:
        validate_sparse_edges(sankey["split_connectivity"][split], expected_node_count)


def build_overview(config: DatasetConfig, coord_decimals: int) -> Dict[str, Any]:
    metadata_rows = load_metadata_rows(config.metadata_json)
    molecule_labels, split_ids, molecule_label_to_id = canonical_metadata(metadata_rows)
    epochs, layers = load_ordering(config.split_sources)
    layers = infer_layers_from_values(layers)
    if not epochs or not layers:
        raise ValueError(f"Unable to infer epochs/layers for dataset {config.name}")

    labels_by_split: Dict[str, Dict[str, Dict[str, Dict[int, int]]]] = {}
    dr_by_split: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]] = {}
    representatives_by_split: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = {}
    memberships_by_split: Dict[str, Dict[str, Dict[str, Set[int]]]] = {}

    for source in config.split_sources:
        labels_by_cell, memberships_by_cell, dr_by_cell = build_split_payloads(
            source=source,
            epochs=epochs,
            layers=layers,
            molecule_label_to_id=molecule_label_to_id,
            coord_decimals=coord_decimals,
        )
        representatives = build_representative_payloads(
            source=source,
            epochs=epochs,
            layers=layers,
            molecule_label_to_id=molecule_label_to_id,
        )
        labels_by_split[source.split] = labels_by_cell
        dr_by_split[source.split] = dr_by_cell
        representatives_by_split[source.split] = representatives
        memberships_by_split[source.split] = memberships_by_cell

    for split in SUPPORTED_SPLITS:
        if split in labels_by_split:
            continue
        labels_by_split[split] = {epoch: {layer: {} for layer in layers} for epoch in epochs}
        dr_by_split[split] = {epoch: {layer: {} for layer in layers} for epoch in epochs}
        representatives_by_split[split] = {epoch: {layer: {} for layer in layers} for epoch in epochs}
        memberships_by_split[split] = {epoch: {layer: set() for layer in layers} for epoch in epochs}

    connectivity, split_connectivity, node_lookup = build_sparse_connectivity(
        epochs=epochs,
        layers=layers,
        memberships_by_split=memberships_by_split,
    )

    cell_payloads: List[Dict[str, Any]] = []
    for epoch_idx, epoch in enumerate(epochs):
        for layer_idx, layer in enumerate(layers):
            node_key = layer_idx + (epoch_idx * len(layers))
            cluster_labels: Dict[str, int] = {}
            for split in SUPPORTED_SPLITS:
                for molecule_id, cluster_id in labels_by_split[split][epoch][layer].items():
                    cluster_labels[str(molecule_id)] = cluster_id

            dr_payload: Dict[str, Dict[str, List[float]]] = {}
            for split in SUPPORTED_SPLITS:
                for method, coords in dr_by_split[split][epoch][layer].items():
                    dr_payload.setdefault(method, {}).update(coords)

            cell_payloads.append(
                {
                    "node_key": node_key,
                    "epoch": epoch,
                    "layer": layer,
                    "epoch_idx": epoch_idx,
                    "layer_idx": layer_idx,
                    "global_node_index": node_key,
                    "cluster_labels": cluster_labels,
                    "cluster_representatives": {
                        split: representatives_by_split[split][epoch][layer]
                        for split in SUPPORTED_SPLITS
                    },
                    "dr": dr_payload,
                }
            )

    overview: Dict[str, Any] = {
        "api_version": "compact_overview_v2",
        "dataset": config.name,
        "source_variant": config.source_variant,
        "dr_coord_decimals": coord_decimals,
        "source_paths": source_paths_payload(config.split_sources),
        "molecule_ids": list(range(len(molecule_labels))),
        "splits": split_ids,
        "epochs": list(epochs),
        "layers": list(layers),
        "cells": cell_payloads,
        "sankey": {
            "connectivity": connectivity,
            "split_connectivity": split_connectivity,
            "node_lookup": node_lookup,
        },
    }

    molecule_labels_payload = maybe_molecule_labels_payload(molecule_labels)
    if molecule_labels_payload is not None:
        overview["molecule_labels"] = molecule_labels_payload

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
                    split="val",
                    scatter_dir=chemprop_root / "val" / "knn" / "scatterplot",
                    cluster_csv=chemprop_root / "val" / "knn" / "clustering_metrics" / "cluster_assignments_with_real_id.csv",
                    representatives_json=chemprop_root / "val" / "knn" / "topk" / "cluster_representatives.json",
                    sankey_json=chemprop_root / "val" / "knn" / "sankey" / "all_sankey.json",
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
                    split="val",
                    scatter_dir=chemeleon_root / "val" / "knn" / "scatterplot",
                    cluster_csv=chemeleon_root / "val" / "knn" / "clustering_metrics" / "cluster_assignments_with_real_id.csv",
                    representatives_json=chemeleon_root / "val" / "knn" / "topk" / "cluster_representatives.json",
                    sankey_json=chemeleon_root / "val" / "knn" / "sankey" / "all_sankey.json",
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


def build_all(output_root: Path, coord_decimals: int) -> Dict[str, str]:
    manifest: Dict[str, str] = {}
    for config in dataset_configs():
        overview = build_overview(config, coord_decimals=coord_decimals)
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
    parser.add_argument(
        "--coord-decimals",
        type=int,
        default=DEFAULT_COORD_DECIMALS,
        help="Number of decimal places to keep for compact DR coordinates.",
    )
    args = parser.parse_args()
    decimals = max(0, int(args.coord_decimals))
    manifest = build_all(args.output_root.resolve(), coord_decimals=decimals)
    print(
        json.dumps(
            {
                "ok": True,
                "output_root": str(args.output_root.resolve()),
                "coord_decimals": decimals,
                "manifest": manifest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
