#!/usr/bin/env python3
"""
Compute layer-to-layer transition statistics for frontend: how many points
change cluster between consecutive layers (by_epoch) or between consecutive
epochs (by_layer). Writes layer_transition_stats.json next to each
cluster_assignments_with_real_id.csv for each case and clustering method.
"""
import csv
import json
import sys
from pathlib import Path

# Backend on path
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

PROJECT_ROOT = _BACKEND.resolve().parents[1]  # repo root (parent of latentvis)
DATA_ROOT = PROJECT_ROOT / "latentvis" / "frontend" / "public" / "data"
CASES = ("chemprop", "chemeleon")


def _find_assignment_csvs(case_dir: Path):
    """Yield (method_name, csv_path) for each cluster_assignments_with_real_id.csv."""
    # Root: clustering_metrics/... -> gmm
    root_ca = case_dir / "clustering_metrics" / "cluster_assignments_with_real_id.csv"
    if root_ca.is_file():
        yield "gmm", root_ca
    # clustering/<method>/clustering_metrics/...
    clustering = case_dir / "clustering"
    if clustering.is_dir():
        for sub in clustering.iterdir():
            if sub.is_dir():
                ca = sub / "clustering_metrics" / "cluster_assignments_with_real_id.csv"
                if ca.is_file():
                    yield sub.name, ca


def _layer_columns(headers):
    """Return list of column names that end with _cluster (layer columns), in header order (model layer order)."""
    return [h for h in headers if h.endswith("_cluster") and h != "epoch"]


def _normalize_cluster_value(val):
    """Normalize cluster ID for comparison so that 0 and '0' and ' 0 ' are equal. Returns a comparable value."""
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return s


def _cluster_eq(a, b):
    """True if both are None or both are the same cluster ID (after normalization)."""
    na, nb = _normalize_cluster_value(a), _normalize_cluster_value(b)
    return na == nb


def _epoch_column(headers):
    if "epoch" in headers:
        return "epoch"
    return None


def _reorder_layer_columns(layer_cols, preferred_layer_order):
    """Return layer_cols reordered so preferred_layer_order (layer names without _cluster) come first.
    Must match all_sankey layer key order so transition stats (cell-to-cell %) match.
    """
    if not preferred_layer_order:
        return layer_cols
    layer_names = [c.replace("_cluster", "") for c in layer_cols]
    seen = set()
    reordered = []
    for p in preferred_layer_order:
        col = p + "_cluster"
        if col in layer_cols and p not in seen:
            reordered.append(col)
            seen.add(p)
    for name in layer_names:
        if name not in seen:
            reordered.append(name + "_cluster")
    return reordered


def _transition_matrix_and_outflow(rows, from_col, to_col):
    """
    Build (source_cluster, target_cluster) -> count from rows; then compute Sankey-consistent
    'changed' count: for each source cluster, outflow = src_tot - max(target count).
    Returns (n_changed, n_total) so pct_changed matches Sankey outflow.
    """
    counts = {}
    for r in rows:
        c0 = _normalize_cluster_value(r.get(from_col))
        c1 = _normalize_cluster_value(r.get(to_col))
        if c0 is None or c1 is None:
            continue
        key = (c0, c1)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return 0, 0
    n_total = sum(counts.values())
    n_changed = 0
    src_totals = {}
    for (c0, c1), val in counts.items():
        src_totals[c0] = src_totals.get(c0, 0) + val
    for c0, src_tot in src_totals.items():
        max_to = max((counts.get((c0, c1), 0) for c1 in set(c1 for (c0_, c1) in counts if c0_ == c0)), default=0)
        n_changed += src_tot - max_to
    return n_changed, n_total


def compute_stats(csv_path, preferred_layer_order=None):
    """Read CSV and return by_epoch and by_layer transition stats.

    Percentages are computed from the same flow logic as Sankey: for each source cluster,
    'changed' = outflow = (total from source) - (max flow to any single target). So if
    most points from a cluster go to one target (no split), change is small; if they
    split across many targets, change is large. This matches the Sankey visualization.
    Layer order must match all_sankey.json.
    """
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = list(reader.fieldnames or [])
    if not rows:
        return {"by_epoch": [], "by_layer": []}
    layer_cols = _layer_columns(headers)
    layer_cols = _reorder_layer_columns(layer_cols, preferred_layer_order)
    epoch_col = _epoch_column(headers)
    if not layer_cols:
        return {"by_epoch": [], "by_layer": []}

    # by_epoch: same logic as Sankey edges (layer A -> layer B). Changed = outflow from each source cluster.
    by_epoch = []
    for i in range(len(layer_cols) - 1):
        from_l = layer_cols[i]
        to_l = layer_cols[i + 1]
        from_key = from_l.replace("_cluster", "")
        to_key = to_l.replace("_cluster", "")
        if epoch_col:
            epochs_seen = set()
            for r in rows:
                ep = r.get(epoch_col)
                if ep is None:
                    continue
                epochs_seen.add(ep)
            for ep in sorted(epochs_seen, key=lambda x: (int(x) if str(x).isdigit() else x)):
                subset = [r for r in rows if r.get(epoch_col) == ep]
                n_changed, n_total = _transition_matrix_and_outflow(subset, from_l, to_l)
                pct = (100.0 * n_changed / n_total) if n_total else 0.0
                by_epoch.append({
                    "epoch": ep,
                    "from_layer": from_key,
                    "to_layer": to_key,
                    "n_changed": n_changed,
                    "n_total": n_total,
                    "pct_changed": round(pct, 2),
                })
        else:
            n_changed, n_total = _transition_matrix_and_outflow(rows, from_l, to_l)
            pct = (100.0 * n_changed / n_total) if n_total else 0.0
            by_epoch.append({
                "epoch": None,
                "from_layer": from_key,
                "to_layer": to_key,
                "n_changed": n_changed,
                "n_total": n_total,
                "pct_changed": round(pct, 2),
            })

    # by_layer: same logic (epoch A -> epoch B). Changed = outflow from each source cluster.
    by_layer = []
    if epoch_col:
        real_ids = sorted(set(r.get("real_id") for r in rows if r.get("real_id") is not None))
        epochs_sorted = sorted(set(r.get(epoch_col) for r in rows if r.get(epoch_col) is not None),
                               key=lambda x: (int(x) if str(x).isdigit() else x))
        for layer_col in layer_cols:
            layer_key = layer_col.replace("_cluster", "")
            for j in range(len(epochs_sorted) - 1):
                e_from = epochs_sorted[j]
                e_to = epochs_sorted[j + 1]
                from_vals = {r["real_id"]: r.get(layer_col) for r in rows if r.get(epoch_col) == e_from}
                to_vals = {r["real_id"]: r.get(layer_col) for r in rows if r.get(epoch_col) == e_to}
                # Build rows for this (e_from, e_to) pair: list of {from_col, to_col} for common real_ids
                pair_rows = []
                for rid in real_ids:
                    if rid not in from_vals or rid not in to_vals:
                        continue
                    pair_rows.append({layer_col + "_from": from_vals[rid], layer_col + "_to": to_vals[rid]})
                n_changed, n_total = _transition_matrix_and_outflow(
                    pair_rows, layer_col + "_from", layer_col + "_to"
                )
                pct = (100.0 * n_changed / n_total) if n_total else 0.0
                by_layer.append({
                    "layer": layer_key,
                    "from_epoch": e_from,
                    "to_epoch": e_to,
                    "n_changed": n_changed,
                    "n_total": n_total,
                    "pct_changed": round(pct, 2),
                })

    return {"by_epoch": by_epoch, "by_layer": by_layer}


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Compute layer/epoch transition stats. Layer order must match all_sankey.json."
    )
    ap.add_argument(
        "--layer-order",
        type=str,
        default="",
        help="Comma-separated layer order (model order), e.g. fingerprint,gnn_layer_1,gnn_layer_2,gnn_layer_3,ffn_layer_1,ffn_layer_2. Use same order as all_sankey. If omitted, CSV header order is used (correct when CSV from current pipeline).",
    )
    ap.add_argument(
        "--layer-order-chemeleon",
        type=str,
        default="",
        help="Override layer order for chemeleon only (9 layers). If set, used instead of --layer-order for chemeleon.",
    )
    args = ap.parse_args()
    preferred_by_case = {}
    if args.layer_order:
        preferred_by_case["chemprop"] = [s.strip() for s in args.layer_order.split(",") if s.strip()]
    if args.layer_order_chemeleon:
        preferred_by_case["chemeleon"] = [s.strip() for s in args.layer_order_chemeleon.split(",") if s.strip()]
    if not preferred_by_case:
        preferred_by_case = {"chemprop": None, "chemeleon": None}
    else:
        # If only --layer-order given, use it for chemeleon too (extra layers appended by _reorder_layer_columns).
        if "chemeleon" not in preferred_by_case and "chemprop" in preferred_by_case:
            preferred_by_case["chemeleon"] = preferred_by_case["chemprop"]

    for case in CASES:
        for split in ("test", "train", ""):
            case_dir = (DATA_ROOT / case / split) if split else (DATA_ROOT / case)
            if not case_dir.is_dir():
                continue
            preferred = preferred_by_case.get(case)
            for method, csv_path in _find_assignment_csvs(case_dir):
                stats = compute_stats(csv_path, preferred_layer_order=preferred)
                out_dir = csv_path.parent
                (out_dir / "layer_transition_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
                (out_dir / "layer_transition_by_epoch.json").write_text(
                    json.dumps({"by_epoch": stats["by_epoch"]}, indent=2),
                    encoding="utf-8",
                )
                (out_dir / "layer_transition_by_layer.json").write_text(
                    json.dumps({"by_layer": stats["by_layer"]}, indent=2),
                    encoding="utf-8",
                )
                print(
                    "Wrote %s (method=%s): by_epoch=%s by_layer=%s"
                    % (out_dir, method, len(stats["by_epoch"]), len(stats["by_layer"]))
                )


if __name__ == "__main__":
    main()
