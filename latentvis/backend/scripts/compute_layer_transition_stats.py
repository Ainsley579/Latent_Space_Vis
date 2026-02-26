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


def compute_stats(csv_path: Path):
    """Read CSV and return by_epoch and by_layer transition stats.

    - by_epoch: within same epoch, for each consecutive layer pair (model order), count how many
      real_ids have a different cluster in the next layer. Same real_id = same row; compare
      from_layer column vs to_layer column (normalized so 0 and '0' are equal).
    - by_layer: within same layer, for each consecutive epoch pair, count how many real_ids
      have a different cluster in the next epoch. Same real_id compared across two epoch rows.
    """
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = list(reader.fieldnames or [])
    if not rows:
        return {"by_epoch": [], "by_layer": []}
    layer_cols = _layer_columns(headers)
    epoch_col = _epoch_column(headers)
    if not layer_cols:
        return {"by_epoch": [], "by_layer": []}

    # by_epoch: within same epoch, same real_id; compare cluster in consecutive layers (model order).
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
                n_total = len(subset)
                # Changed = same real_id has different cluster in to_layer vs from_layer (normalized comparison).
                n_changed = sum(1 for r in subset if not _cluster_eq(r.get(from_l), r.get(to_l)))
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
            n_total = len(rows)
            n_changed = sum(1 for r in rows if not _cluster_eq(r.get(from_l), r.get(to_l)))
            pct = (100.0 * n_changed / n_total) if n_total else 0.0
            by_epoch.append({
                "epoch": None,
                "from_layer": from_key,
                "to_layer": to_key,
                "n_changed": n_changed,
                "n_total": n_total,
                "pct_changed": round(pct, 2),
            })

    # by_layer: within same layer, same real_id; compare cluster in consecutive epochs (epoch order).
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
                common_ids = [rid for rid in real_ids if rid in from_vals and rid in to_vals]
                n_total = len(common_ids)
                # Changed = same real_id has different cluster in to_epoch vs from_epoch (normalized comparison).
                n_changed = sum(
                    1 for rid in common_ids
                    if not _cluster_eq(from_vals.get(rid), to_vals.get(rid))
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
    for case in CASES:
        for split in ("test", "train", ""):
            case_dir = (DATA_ROOT / case / split) if split else (DATA_ROOT / case)
            if not case_dir.is_dir():
                continue
            for method, csv_path in _find_assignment_csvs(case_dir):
                stats = compute_stats(csv_path)
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
