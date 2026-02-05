#!/usr/bin/env python3
"""
Build all_sankey.json from two epoch CSVs of cluster assignments.

Each CSV row is a data point, with columns like:
  real_id, <layer>_cluster, <layer>_prob_max, <layer>_entropy, ...

Outputs JSON structure similar to sankey_test/data/all_sankey.json:
{
  "epoch": { ... },   # per-epoch spatial transitions (nodes grouped by layer)
  "layer": { ... }    # per-layer temporal transitions (nodes grouped by epoch)
}
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _safe_float(val: Any) -> float:
    try:
        return float(val)
    except Exception:
        return 0.0


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def _infer_layers(fieldnames: Iterable[str]) -> List[str]:
    layers: List[str] = []
    for name in fieldnames:
        if name.endswith("_cluster"):
            layers.append(name[: -len("_cluster")])
    return layers


def _label(layer: str, cluster: Any) -> str:
    return f"{layer}_C{cluster}"


def _assign_node_ids(edges: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    label_to_id: Dict[str, int] = {}
    next_id = 0
    for e in edges:
        for key in ("source_label", "target_label"):
            label = e.get(key)
            if label is None:
                continue
            if label not in label_to_id:
                label_to_id[label] = next_id
                next_id += 1

    enriched = []
    for e in edges:
        s_label = e.get("source_label")
        t_label = e.get("target_label")
        enriched.append(
            {
                "source_node_id": label_to_id.get(s_label),
                "target_node_id": label_to_id.get(t_label),
                "source_label": s_label,
                "target_label": t_label,
                "value": e.get("value", 0),
            }
        )

    return enriched, label_to_id


def _group_from_label(label: str) -> str:
    if not label:
        return "unknown"
    if "_C" in label:
        return label.rsplit("_C", 1)[0]
    return label


def _cluster_from_label(label: str) -> str:
    if not label or "_C" not in label:
        return ""
    return label.rsplit("_C", 1)[1]

def _cluster_num_from_cluster(cluster: str) -> int | None:
    if cluster is None:
        return None
    raw = str(cluster)
    if "_" in raw:
        raw = raw.split("_", 1)[0]
    if raw.startswith("C"):
        raw = raw[1:]
    try:
        return int(raw)
    except Exception:
        return None


def _compute_nodes_grouped_by_layer(
    edges: List[Dict[str, Any]],
    forced_layer: str | None = None,
):
    """
    Returns:
      - flat_nodes: list of {id,label,value}
      - nodes_by_layer: OrderedDict[layer -> list of {id,cluster,label,value,previous}]
    """
    node_label: Dict[int, str] = {}
    node_in: Dict[int, float] = {}
    node_out: Dict[int, float] = {}

    for e in edges:
        sid = e.get("source_node_id")
        tid = e.get("target_node_id")
        slabel = e.get("source_label")
        tlabel = e.get("target_label")
        val = _safe_float(e.get("value", 0))

        if sid is not None and slabel is not None:
            node_label[sid] = slabel
            node_out[sid] = node_out.get(sid, 0.0) + val
        if tid is not None and tlabel is not None:
            node_label[tid] = tlabel
            node_in[tid] = node_in.get(tid, 0.0) + val

    keys = sorted(node_label.keys())
    flat_nodes = []
    for k in keys:
        val = node_in.get(k, None)
        if val is None:
            val = node_out.get(k, 0.0)
        flat_nodes.append({"id": k, "label": node_label[k], "value": round(val, 4)})

    nodes_by_layer: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
    cluster_prefix_by_id: Dict[int, str] = {}

    for k in keys:
        label = node_label[k]
        if forced_layer is not None:
            layer_name = forced_layer
        else:
            layer_name = _group_from_label(str(label))

        cluster = _cluster_from_label(str(label))

        cluster_prefix = ""
        if cluster:
            cluster_prefix = str(cluster).split("_", 1)[0]
        cluster_prefix_by_id[k] = cluster_prefix

        val = node_in.get(k, None)
        if val is None:
            val = node_out.get(k, 0.0)

        nodes_by_layer.setdefault(layer_name, []).append(
            {
                "id": k,
                "cluster": cluster,
                "label": label,
                "value": round(val, 4),
                "previous": 0.0,
            }
        )

    # assign previous within each layer using numeric cluster id order (C0, C1, C2...)
    for _layer_name, node_list in nodes_by_layer.items():
        node_list.sort(
            key=lambda n: (
                _cluster_num_from_cluster(n.get("cluster")) is None,
                _cluster_num_from_cluster(n.get("cluster")) if _cluster_num_from_cluster(n.get("cluster")) is not None else 0,
                str(n.get("label", "")),
                n.get("id"),
            )
        )

        cumulative = 0.0
        for node in node_list:
            node["previous"] = round(cumulative, 4)
            cumulative += _safe_float(node.get("value", 0.0))

    return flat_nodes, nodes_by_layer


def _extract_model_from_label(label: str) -> str | None:
    if not label:
        return None
    if "_C" in label:
        return label.rsplit("_C", 1)[0]
    return None


def _compute_edge_percentages(entries: Dict[str, Any]) -> None:
    for entry in entries.values():
        edges = entry.get("edges", []) if entry else []
        totals: Dict[str, float] = {}
        for e in edges:
            src = e.get("source_label")
            val = _safe_float(e.get("value", 0))
            if src is None:
                continue
            totals[src] = totals.get(src, 0.0) + val
        for e in edges:
            src = e.get("source_label")
            val = _safe_float(e.get("value", 0))
            total = totals.get(src, 0.0)
            if total > 0:
                e["pct_of_source"] = round((val / total) * 100.0, 4)
            else:
                e["pct_of_source"] = None


def _build_spatial_entry(
    rows: List[Dict[str, Any]],
    layers: List[str],
    epoch_name: str,
    source_path: Path,
) -> Dict[str, Any]:
    edge_counts: Dict[Tuple[str, str], int] = {}
    for row in rows:
        for i in range(len(layers) - 1):
            l1 = layers[i]
            l2 = layers[i + 1]
            c1 = row.get(f"{l1}_cluster")
            c2 = row.get(f"{l2}_cluster")
            if c1 is None or c2 is None or c1 == "" or c2 == "":
                continue
            s_label = _label(l1, c1)
            t_label = _label(l2, c2)
            edge_counts[(s_label, t_label)] = edge_counts.get((s_label, t_label), 0) + 1

    edges_raw = [
        {"source_label": s, "target_label": t, "value": v}
        for (s, t), v in edge_counts.items()
    ]
    edges, _ = _assign_node_ids(edges_raw)
    flat_nodes, nodes_by_layer = _compute_nodes_grouped_by_layer(edges)

    return {
        "filename": source_path.name,
        "path": str(source_path.resolve()),
        "type": "spatial_epoch",
        "epoch": epoch_name,
        "edges": edges,
        "nodes": nodes_by_layer,
        "node_count": len(flat_nodes),
    }


def _build_temporal_entry(
    layer: str,
    model0_name: str,
    model1_name: str,
    rows0_by_id: Dict[str, Dict[str, Any]],
    rows1_by_id: Dict[str, Dict[str, Any]],
    model_order: List[str],
) -> Dict[str, Any]:
    edge_counts: Dict[Tuple[str, str], int] = {}
    for rid, row0 in rows0_by_id.items():
        row1 = rows1_by_id.get(rid)
        if row1 is None:
            continue
        c0 = row0.get(f"{layer}_cluster")
        c1 = row1.get(f"{layer}_cluster")
        if c0 is None or c1 is None or c0 == "" or c1 == "":
            continue
        s_label = _label(model0_name, c0)
        t_label = _label(model1_name, c1)
        edge_counts[(s_label, t_label)] = edge_counts.get((s_label, t_label), 0) + 1

    edges_raw = [
        {"source_label": s, "target_label": t, "value": v}
        for (s, t), v in edge_counts.items()
    ]
    edges, _ = _assign_node_ids(edges_raw)
    flat_nodes, nodes_by_layer = _compute_nodes_grouped_by_layer(edges, forced_layer=layer)

    # group nodes by model (from label)
    nodes_by_model: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
    for layer_nodes in nodes_by_layer.values():
        for node in layer_nodes:
            label = node.get("label", "")
            model = _extract_model_from_label(label) or "unknown"
            nodes_by_model.setdefault(model, []).append(node)

    def _model_key(k: str):
        if k in model_order:
            return (0, model_order.index(k))
        if k == "unknown":
            return (2, k)
        return (1, k)

    nodes_by_model_sorted = OrderedDict(
        (k, v) for k, v in sorted(nodes_by_model.items(), key=lambda it: _model_key(it[0]))
    )

    # recompute previous per model using numeric cluster id order (C0, C1, C2...)
    for _model_name, node_list in nodes_by_model_sorted.items():
        node_list.sort(
            key=lambda n: (
                _cluster_num_from_cluster(n.get("cluster")) is None,
                _cluster_num_from_cluster(n.get("cluster")) if _cluster_num_from_cluster(n.get("cluster")) is not None else 0,
                str(n.get("label", "")),
                n.get("id"),
            )
        )

        cumulative = 0.0
        for node in node_list:
            node["previous"] = round(cumulative, 4)
            cumulative += _safe_float(node.get("value", 0.0))

    return {
        "filename": None,
        "path": None,
        "type": "temporal",
        "layername": layer,
        "edges": edges,
        "nodes": nodes_by_model_sorted,
        "node_count": len(flat_nodes),
    }


def build_all(
    epoch0_path: Path,
    epoch1_path: Path,
    model0_name: str,
    model1_name: str,
    out_path: Path,
) -> None:
    rows0 = _read_rows(epoch0_path)
    rows1 = _read_rows(epoch1_path)

    if not rows0 or not rows1:
        raise RuntimeError("One or both CSV files are empty.")

    layers0 = _infer_layers(rows0[0].keys())
    layers1 = _infer_layers(rows1[0].keys())

    if layers0 != layers1:
        common = [l for l in layers0 if l in layers1]
        if not common:
            raise RuntimeError("No shared *_cluster columns found between the two CSVs.")
        layers = common
    else:
        layers = layers0

    epoch_results: Dict[str, Any] = OrderedDict()
    epoch_results[model0_name] = _build_spatial_entry(rows0, layers, model0_name, epoch0_path)
    epoch_results[model1_name] = _build_spatial_entry(rows1, layers, model1_name, epoch1_path)

    rows0_by_id = {str(r.get("real_id")): r for r in rows0 if r.get("real_id") is not None}
    rows1_by_id = {str(r.get("real_id")): r for r in rows1 if r.get("real_id") is not None}

    layer_results: Dict[str, Any] = OrderedDict()
    model_order = [model0_name, model1_name]
    for layer in layers:
        layer_results[layer] = _build_temporal_entry(
            layer,
            model0_name,
            model1_name,
            rows0_by_id,
            rows1_by_id,
            model_order,
        )

    _compute_edge_percentages(epoch_results)
    _compute_edge_percentages(layer_results)

    out = {"epoch": epoch_results, "layer": layer_results}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all_sankey.json from two epoch CSVs")
    parser.add_argument(
        "--epoch0",
        type=Path,
        default=Path(
            "latentvis/backend/data/clustering_metrics/cluster_assignments_with_real_id_dimenet_noaug.csv"
        ),
        help="Path to epoch0 CSV",
    )
    parser.add_argument(
        "--epoch1",
        type=Path,
        default=Path(
            "latentvis/backend/data/clustering_metrics/cluster_assignments_with_real_id.csv"
        ),
        help="Path to epoch1 CSV",
    )
    parser.add_argument("--epoch0-name", default="dimenet10conf", help="Model name for first CSV")
    parser.add_argument("--epoch1-name", default="dimenet_noaug", help="Model name for second CSV")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("latentvis/frontend/public/data/sankey/all_sankey.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    build_all(args.epoch0, args.epoch1, args.epoch0_name, args.epoch1_name, args.out)


if __name__ == "__main__":
    main()
