#!/usr/bin/env python3
"""
parse_sankey.py

Batch parse:
 - data/sankey_spatial_epoch_{epochnames}.csv
 - data/sankey_temporal_{layernames}.csv

Output:
 - out/all_sankey.json
 - out/sankey_spatial_epoch_{epochnames}.json (one per file)
 - out/sankey_temporal_{layernames}.json (one per file)
"""
import argparse
import csv
import json
from pathlib import Path
import re
from typing import List, Dict, Any
from collections import OrderedDict

SPATIAL_RE = re.compile(r"sankey_spatial_epoch_(?P<epoch>.+?)\.csv$", re.IGNORECASE)
TEMPORAL_RE = re.compile(r"sankey_temporal_(?P<layername>.+?)\.csv$", re.IGNORECASE)

def parse_csv_edges(path: Path) -> List[Dict[str, Any]]:
    """
    Parse CSV file and return list of edges.
    Expected row format:
    source_node_id,target_node_id,source_label,target_label,value
    """
    edges = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if any(cell.strip() for cell in r)]  # skip blank lines
        if not rows:
            return edges

        # Detect if first row is header (contains 'source' or 'source_node_id')
        first = [c.strip().lower() for c in rows[0]]
        has_header = any(h in first for h in ("source", "source_node_id", "source_label", "target_label", "value"))
        start_idx = 1 if has_header else 0

        for i, row in enumerate(rows[start_idx:], start=start_idx):
            # Defensive: row may have comma-separated cells with extra spaces
            if len(row) < 5:
                # Skip incomplete rows
                print("Warning: skipping incomplete row %d in %s: %s" % (i + 1, path, row))
                continue
            # Use first 5 columns only
            s_node, t_node, s_label, t_label, val = [c.strip() for c in row[:5]]
            # Convert ids to int when possible, else keep string
            try:
                s_node_i = int(s_node)
            except:
                s_node_i = s_node
            try:
                t_node_i = int(t_node)
            except:
                t_node_i = t_node
            # Convert value to int or float
            try:
                if "." in val:
                    v = float(val)
                else:
                    v = int(val)
            except Exception:
                v = val  # keep string on parse failure
            edges.append({
                "source_node_id": s_node_i,
                "target_node_id": t_node_i,
                "source_label": s_label,
                "target_label": t_label,
                "value": v
            })
    return edges

def compute_nodes(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Given a list of edges, derive a list of unique nodes with their ids, labels,
    and node value.

    This implementation computes both incoming and outgoing totals per node:
      - node_incoming: sum of values where the node is the target
      - node_outgoing: sum of values where the node is the source

    The returned `value` defaults to the incoming total when present (so sinks
    show the amount they receive). If a node has no incoming edges, the
    outgoing total will be used (so pure sources still have a non-zero value).
    """
    node_label: Dict[Any, str] = {}
    node_incoming: Dict[Any, float] = {}
    node_outgoing: Dict[Any, float] = {}

    for e in edges:
        sid = e.get("source_node_id")
        slabel = e.get("source_label")
        tid = e.get("target_node_id")
        tlabel = e.get("target_label")
        val = e.get("value", 0)

        # register source node and accumulate outgoing value
        if sid is not None and slabel is not None:
            node_label[sid] = slabel
            try:
                node_outgoing[sid] = node_outgoing.get(sid, 0.0) + float(val)
            except Exception:
                node_outgoing[sid] = node_outgoing.get(sid, 0.0)

        # register target node and accumulate incoming value
        if tid is not None and tlabel is not None:
            node_label[tid] = tlabel
            try:
                node_incoming[tid] = node_incoming.get(tid, 0.0) + float(val)
            except Exception:
                node_incoming[tid] = node_incoming.get(tid, 0.0)

    # sort ids: ints first then others
    try:
        keys = sorted(node_label.keys(), key=lambda x: (not isinstance(x, int), x))
    except Exception:
        keys = list(node_label.keys())

    nodes = []
    for k in keys:
        # Prefer incoming value when available, otherwise fall back to outgoing
        val = node_incoming.get(k, None)
        if val is None:
            val = node_outgoing.get(k, 0.0)
        nodes.append({
            "id": k,
            "label": node_label[k],
            "value": round(val, 4)
        })
    return nodes

import collections
from typing import Any


def _extract_layer_and_cluster_from_label(label: str):
    """
    From labels like `layer1_C0_3` or `E000_C0_3` extract:
      - layer_name: if label starts with 'layer', return 'layerX'; else None (caller may supply).
      - cluster_id: the part after the first '_', e.g. 'C0_3'
    """
    if not label:
        return (None, None)
    parts = label.split('_', 1)
    if len(parts) == 1:
        return (None, None)
    prefix, rest = parts[0], parts[1]
    layer = prefix if prefix.startswith('layer') else None
    cluster = rest
    return (layer, cluster)


def compute_nodes_grouped_by_layer(edges: List[Dict[str, Any]], forced_layer: str = None):
    """
    Return two structures:
      - flat_nodes: flat list like compute_nodes
      - nodes_by_layer: Ordered dict: layer_name -> [ {id, cluster, label, value, previous}, ... ]

    Implementation notes:
      - compute both incoming and outgoing per-node totals, then choose the
        displayed node.value as incoming if present, otherwise outgoing.
      - this prevents sinks (nodes that only appear as targets) from having
        value == 0 while avoiding double-counting.
    """
    node_label = {}
    node_in = {}
    node_out = {}
    node_layers = {}

    for e in edges:
        sid = e.get("source_node_id")
        slabel = e.get("source_label")
        tid = e.get("target_node_id")
        tlabel = e.get("target_label")
        val = e.get("value", 0)

        # source: record label, accumulate outgoing
        if sid is not None and slabel is not None:
            node_label[sid] = slabel
            try:
                node_out[sid] = node_out.get(sid, 0.0) + float(val)
            except Exception:
                node_out[sid] = node_out.get(sid, 0.0)
            layer, cluster = _extract_layer_and_cluster_from_label(slabel)
            if layer:
                node_layers[sid] = layer

        # target: record label, accumulate incoming
        if tid is not None and tlabel is not None:
            node_label[tid] = tlabel
            try:
                node_in[tid] = node_in.get(tid, 0.0) + float(val)
            except Exception:
                node_in[tid] = node_in.get(tid, 0.0)
            layer, cluster = _extract_layer_and_cluster_from_label(tlabel)
            if layer:
                node_layers[tid] = layer

    # build flat nodes (sorted ids: ints first)
    try:
        keys = sorted(node_label.keys(), key=lambda x: (not isinstance(x, int), x))
    except Exception:
        keys = list(node_label.keys())

    flat_nodes = []
    for k in keys:
        # prefer incoming when available
        val = node_in.get(k, None)
        if val is None:
            val = node_out.get(k, 0.0)
        flat_nodes.append({
            "id": k,
            "label": node_label[k],
            "value": round(val, 4)
        })

    # group into layers
    nodes_by_layer = collections.OrderedDict()
    node_cluster_prefix = {}

    for k in keys:
        label = node_label[k]
        deduced_layer = node_layers.get(k)
        layer_name = forced_layer if forced_layer is not None else deduced_layer
        if layer_name is None:
            p = str(label).split('_', 1)[0]
            layer_name = p

        # cluster is everything after first underscore, e.g. 'C0_3'
        cluster = ''
        if '_' in str(label):
            cluster = str(label).split('_', 1)[1]

        cluster_prefix = ''
        if cluster:
            cluster_prefix = str(cluster).split('_', 1)[0]

        node_cluster_prefix[k] = cluster_prefix

        # use incoming when available, otherwise outgoing
        val = node_in.get(k, None)
        if val is None:
            val = node_out.get(k, 0.0)

        nodes_by_layer.setdefault(layer_name, []).append({
            "id": k,
            "cluster": cluster,
            "label": label,
            "value": round(val, 4),
            "previous": 0.0
        })

    # compute per-cluster totals and cumulative previous
    import re as _re

    for layer_name, node_list in nodes_by_layer.items():
        # Build mapping cluster_prefix -> list of nodes (preserve stable order)
        cluster_to_nodes = {}
        for node in node_list:
            cid = node.get("id")
            prefix = node_cluster_prefix.get(cid, '')  # '' for no cluster
            cluster_to_nodes.setdefault(prefix, []).append(node)

        # Compute cluster totals (sum of values) for ordering if needed
        cluster_totals = {}
        for prefix, nlist in cluster_to_nodes.items():
            total = 0.0
            for node in nlist:
                try:
                    total += float(node.get("value", 0.0))
                except Exception:
                    pass
            cluster_totals[prefix] = total

        # Sorting key for cluster prefixes: numeric C<digits> first, then others, empty last
        def _cluster_sort_key(pref):
            if pref is None or pref == '':
                return (2, pref)
            m = _re.match(r'^C(\d+)$', str(pref))
            if m:
                return (0, int(m.group(1)))
            return (1, str(pref))

        sorted_prefixes = sorted(cluster_to_nodes.keys(), key=_cluster_sort_key)

        # Now assign previous per node, cumulatively within this layer
        cumulative = 0.0
        for pref in sorted_prefixes:
            # sort nodes inside a cluster deterministically:
            # prefer label (string), fallback to id
            nodes_in_cluster = cluster_to_nodes.get(pref, [])
            nodes_in_cluster_sorted = sorted(
                nodes_in_cluster,
                key=lambda n: (str(n.get("label", "")), n.get("id"))
            )
            for node in nodes_in_cluster_sorted:
                # assign previous as current cumulative
                node["previous"] = round(cumulative, 4)
                # increment cumulative by this node's value
                try:
                    cumulative += float(node.get("value", 0.0))
                except Exception:
                    # skip increment on bad value
                    pass

    return flat_nodes, nodes_by_layer

def build_spatial_entry(path: Path, epoch: str) -> Dict[str, Any]:
    edges = parse_csv_edges(path)
    flat_nodes, nodes_by_layer = compute_nodes_grouped_by_layer(edges, forced_layer=None)
    return {
        "filename": path.name,
        "path": str(path.resolve()),
        "type": "spatial_epoch",
        "epoch": epoch,
        "edges": edges,
        "nodes": nodes_by_layer,
        "node_count": len(flat_nodes)
    }

def _extract_epoch_from_label(label: str):
    """
    Try to extract epoch string from label, e.g.:
      - 'E000_C0_3' -> '000'
      - '000_C0_3'  -> '000'
      - 'layer1_E001_C0' -> '001'
    Returns None if extraction fails.
    """
    if not label:
        return None
    # Common form E000 or E000_... / ..._E000_...
    m = re.search(r'E(?P<epoch>\d{1,})', str(label))
    if m:
        return m.group('epoch')
    # Direct numeric prefix 000_xxx
    m2 = re.match(r'^(?P<epoch>\d{1,})[_\-]', str(label))
    if m2:
        return m2.group('epoch')
    # Numeric suffix e.g. ..._000
    m3 = re.search(r'[_\-](?P<epoch>\d{1,})$', str(label))
    if m3:
        return m3.group('epoch')
    return None

def build_temporal_entry(path: Path, layername: str) -> Dict[str, Any]:
    """
    For temporal CSV (one per layer), build an entry whose `nodes` are grouped by epoch names
    (e.g. "000", "001", ...). If epoch can't be extracted, node goes into "unknown".

    Important: recompute `previous` inside each epoch group so that for each epoch
    the cumulative offset starts at 0 and accumulates only nodes that belong to that epoch.
    """
    edges = parse_csv_edges(path)
    flat_nodes, nodes_by_layer = compute_nodes_grouped_by_layer(edges, forced_layer=layername)

    # Aggregate nodes_by_layer into nodes grouped by epoch (e.g. '000', '001', ...)
    nodes_by_epoch: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for layer_nodes in nodes_by_layer.values():
        for node in layer_nodes:
            label = node.get("label", "")
            epoch = _extract_epoch_from_label(label) or "unknown"
            nodes_by_epoch.setdefault(epoch, []).append(node)

    # Sort epoch keys numerically where possible, put 'unknown' last
    def _epoch_key(k):
        if k == "unknown":
            return (1, k)
        try:
            return (0, int(k))
        except Exception:
            return (0, k)

    sorted_items = sorted(nodes_by_epoch.items(), key=lambda it: _epoch_key(it[0]))
    nodes_by_epoch_sorted = OrderedDict((k, v) for k, v in sorted_items)

    # Recompute `previous` per-epoch so each epoch's nodes accumulate from 0
    import re as _re
    for epoch_name, node_list in nodes_by_epoch_sorted.items():
        # Build cluster -> list of nodes for this epoch
        cluster_to_nodes = OrderedDict()
        node_cluster_prefix = {}
        for node in node_list:
            label = node.get("label", "")
            cluster = ''
            if '_' in str(label):
                cluster = str(label).split('_', 1)[1]
            prefix = ''
            if cluster:
                prefix = str(cluster).split('_', 1)[0]
            node_cluster_prefix[node.get('id')] = prefix
            cluster_to_nodes.setdefault(prefix, []).append(node)

        # Sorting key for cluster prefixes: numeric C<digits> first, then others, empty last
        def _cluster_sort_key(pref):
            if pref is None or pref == '':
                return (2, pref)
            m = _re.match(r'^C(\d+)$', str(pref))
            if m:
                return (0, int(m.group(1)))
            return (1, str(pref))

        sorted_prefixes = sorted(cluster_to_nodes.keys(), key=_cluster_sort_key)

        # Now assign previous per node, cumulatively within this epoch only
        cumulative = 0.0
        for pref in sorted_prefixes:
            # preserve a deterministic order inside a cluster: label then id
            nodes_in_cluster = cluster_to_nodes.get(pref, [])
            nodes_in_cluster_sorted = sorted(
                nodes_in_cluster,
                key=lambda n: (str(n.get("label", "")), n.get("id"))
            )
            for node in nodes_in_cluster_sorted:
                node["previous"] = round(cumulative, 4)
                try:
                    cumulative += float(node.get("value", 0.0))
                except Exception:
                    pass

    return {
        "filename": path.name,
        "path": str(path.resolve()),
        "type": "temporal",
        "layername": layername,
        "edges": edges,
        "nodes": nodes_by_epoch_sorted,
        "node_count": len(flat_nodes)
    }

def compute_edge_percentages(results_dict: Dict[str, Any]):
    """
    Given a dict mapping key->entry (each entry has an 'edges' list), compute for each edge
    the percentage of its value relative to the total outgoing value of its source_label.
    Adds a new field `pct_of_source` (float percent) to each edge, or None if total is zero.
    """
    for key, entry in results_dict.items():
        edges = entry.get('edges', []) if entry else []
        # sum totals per source label
        totals = {}
        for e in edges:
            src = e.get('source_label')
            val = e.get('value', 0)
            try:
                num = float(val)
            except Exception:
                num = 0.0
            if src is None:
                continue
            totals[src] = totals.get(src, 0.0) + num
        # compute percent for each edge
        for e in edges:
            src = e.get('source_label')
            val = e.get('value', 0)
            try:
                num = float(val)
            except Exception:
                num = 0.0
            total = totals.get(src, 0.0)
            if total and total > 0:
                pct = (num / total) * 100.0
                e['pct_of_source'] = round(pct, 4)
            else:
                e['pct_of_source'] = None

def main(data_dir: Path, out_dir: Path, write_individual_files: bool = True):
    data_dir = data_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    spatial_results = {}
    temporal_results = {}

    for csv_path in sorted(data_dir.glob("*.csv")):
        name = csv_path.name
        m_sp = SPATIAL_RE.search(name)
        m_temp = TEMPORAL_RE.search(name)
        if m_sp:
            epoch = m_sp.group("epoch")
            entry = build_spatial_entry(csv_path, epoch)
            spatial_results[epoch] = entry
            if write_individual_files:
                out_path = out_dir / f"{csv_path.stem}.json"
                out_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
        elif m_temp:
            layername = m_temp.group("layername")
            entry = build_temporal_entry(csv_path, layername)
            temporal_results[layername] = entry
            if write_individual_files:
                out_path = out_dir / f"{csv_path.stem}.json"
                out_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            print("Skipping non-matching file: %s" % name)

    # --- compute edge percentages (pct_of_source) for spatial and temporal results ---
    compute_edge_percentages(spatial_results)
    compute_edge_percentages(temporal_results)

    # if individual per-file JSONs were written earlier, rewrite them to include the new pct fields
    if write_individual_files:
        for epoch, entry in spatial_results.items():
            out_path = out_dir / f"sankey_spatial_epoch_{epoch}.json"
            out_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
        for layername, entry in temporal_results.items():
            out_path = out_dir / f"sankey_temporal_{layername}.json"
            out_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")

    all_data = {
        "epoch": spatial_results,
        "layer": temporal_results
    }

    all_out = out_dir / "all_sankey.json"
    all_out.write_text(json.dumps(all_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print summary
    print("Parse complete. %d spatial_epoch files, %d temporal files." % (len(spatial_results), len(temporal_results)))
    print("Output dir: %s" % out_dir)
    print("Merged file: %s" % all_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse sankey CSVs and write structured JSON.")
    parser.add_argument("--data-dir", "-d", default="./", help="Directory containing sankey CSVs (default: .)")
    parser.add_argument("--out-dir", "-o", default="./", help="Directory for JSON output (default: .)")
    parser.add_argument("--no-individual", action="store_true", help="Do not write per-CSV JSON files (only all_sankey.json)")
    args = parser.parse_args()
    main(Path(args.data_dir), Path(args.out_dir), write_individual_files=(not args.no_individual))