#!/usr/bin/env python3
"""
parse_sankey.py

批量解析:
 - data/sankey_spatial_epoch_{epochnames}.csv
 - data/sankey_temporal_{layernames}.csv

输出:
 - out/all_sankey.json
 - out/sankey_spatial_epoch_{epochnames}.json (每个文件一份)
 - out/sankey_temporal_{layernames}.json (每个文件一份)
"""
import argparse
import csv
import json
from pathlib import Path
import re
from typing import List, Dict, Any

SPATIAL_RE = re.compile(r"sankey_spatial_epoch_(?P<epoch>.+?)\.csv$", re.IGNORECASE)
TEMPORAL_RE = re.compile(r"sankey_temporal_(?P<layername>.+?)\.csv$", re.IGNORECASE)

def parse_csv_edges(path: Path) -> List[Dict[str, Any]]:
    """
    解析 CSV 文件，返回 edges 列表。
    期望每行格式:
    source_node_id,target_node_id,source_label,target_label,value
    """
    edges = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if any(cell.strip() for cell in r)]  # 去掉空行
        if not rows:
            return edges

        # 检测是否第一行是 header（包含 'source' 或 'source_node_id' 文本）
        first = [c.strip().lower() for c in rows[0]]
        has_header = any(h in first for h in ("source", "source_node_id", "source_label", "target_label", "value"))
        start_idx = 1 if has_header else 0

        for i, row in enumerate(rows[start_idx:], start=start_idx):
            # 保护性处理：行可能含有逗号分隔但多余空格
            if len(row) < 5:
                # 尝试用 join 切分（若 source_label, target_label 含逗号会出问题，但假设不含）
                # 跳过不完整行
                print(f"警告: 跳过不完整行 {i+1} in {path}: {row}")
                continue
            # 只取前5列
            s_node, t_node, s_label, t_label, val = [c.strip() for c in row[:5]]
            # 将 id 转为 int（若不能转则保持原字符串）
            try:
                s_node_i = int(s_node)
            except:
                s_node_i = s_node
            try:
                t_node_i = int(t_node)
            except:
                t_node_i = t_node
            # 将 value 转为 int 或 float
            try:
                if "." in val:
                    v = float(val)
                else:
                    v = int(val)
            except:
                # 若解析失败保留字符串
                v = val
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
    从像 `layer1_C0_3` 或 `E000_C0_3` 的 label 中提取:
      - layer_name: 如果 label 以 'layer' 开头，返回 'layerX'；否则 return None (caller may supply).
      - cluster_id: 整个第一个 '_' 后面的部分，例如 'C0_3'
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
        cluster_totals = {}
        for node in node_list:
            cid = node.get("id")
            prefix = node_cluster_prefix.get(cid, '')
            cluster_totals.setdefault(prefix, 0.0)
            try:
                cluster_totals[prefix] += float(node.get("value", 0.0))
            except Exception:
                pass

        def _cluster_sort_key(pref):
            if not pref:
                return (1e9, pref)
            m = _re.match(r'^C(\d+)$', str(pref))
            if m:
                return (int(m.group(1)), pref)
            return (1e6, pref)

        sorted_prefixes = sorted(cluster_totals.keys(), key=_cluster_sort_key)

        cumulative = 0.0
        for pref in sorted_prefixes:
            prev_for_cluster = cumulative
            for node in node_list:
                cid = node.get("id")
                if node_cluster_prefix.get(cid, '') == pref:
                    node["previous"] = round(prev_for_cluster, 4)
            try:
                cumulative += float(cluster_totals.get(pref, 0.0))
            except Exception:
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

def build_temporal_entry(path: Path, layername: str) -> Dict[str, Any]:
    edges = parse_csv_edges(path)
    flat_nodes, nodes_by_layer = compute_nodes_grouped_by_layer(edges, forced_layer=layername)
    return {
        "filename": path.name,
        "path": str(path.resolve()),
        "type": "temporal",
        "layername": layername,
        "edges": edges,
        "nodes": nodes_by_layer,
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
            print(f"跳过非匹配文件: {name}")

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

    # 打印摘要
    print(f"解析完成。{len(spatial_results)} 个 spatial_epoch 文件，{len(temporal_results)} 个 temporal 文件。")
    print(f"输出目录: {out_dir}")
    print(f"主合并文件: {all_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse sankey CSVs and write structured JSON.")
    parser.add_argument("--data-dir", "-d", default="./", help="存放 sankey CSV 的目录（默认: data）")
    parser.add_argument("--out-dir", "-o", default="./", help="JSON 输出目录（默认: out）")
    parser.add_argument("--no-individual", action="store_true", help="不为每个 CSV 生成单独的 JSON 文件（只生成 all_sankey.json）")
    args = parser.parse_args()
    main(Path(args.data_dir), Path(args.out_dir), write_individual_files=(not args.no_individual))