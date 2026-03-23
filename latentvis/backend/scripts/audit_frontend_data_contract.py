#!/usr/bin/env python3
"""
Audit frontend-facing data integrity for chemprop, chemeleon, and dimenet.

Checks:
- backend/data baseline outputs
- backend/data sweeps completeness
- backend/data_compact outputs
- dimenet frontend/public mirror files
- known static metadata contract issues
"""

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DATA = ROOT / "data"
COMPACT_ROOT = ROOT / "data_compact"
FRONTEND_PUBLIC = ROOT.parent / "frontend" / "public"

ESOL_CASES = ("chemprop", "chemeleon")
ESOL_SPLITS = ("train", "val", "test")
DIMENET_SPLITS = ("test",)
METHODS = ("pgmeans", "agglomerative", "knn", "hdbscan")
EXPECTED_SWEEP_COUNTS = {
    "pgmeans": 45,
    "agglomerative": 27,
    "knn": 9,
    "hdbscan": 1,
}
REQUIRED_METHOD_PATHS = (
    "scatterplot",
    "sankey/all_sankey.json",
    "sankey/sankey_samples.json",
    "topk/cluster_representatives.json",
    "clustering_metrics/cluster_assignments_with_real_id.csv",
)


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"__error__": str(exc)}


def count_csv_rows(path):
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return max(sum(1 for _ in csv.reader(handle)) - 1, 0)
    except Exception:
        return None


def collect_split_counts(rows):
    counts = Counter()
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            split = str(row.get("split", "")).strip().lower()
            if split:
                counts[split] += 1
    return counts


def audit_case(case, splits, problems):
    case_root = BACKEND_DATA / case
    metadata_path = case_root / "id_smiles_y_true_y_pred.json"
    metadata_rows = load_json(metadata_path) if metadata_path.is_file() else []
    split_counts = collect_split_counts(metadata_rows)

    stale_resume_dirs = sorted(case_root.rglob(".pipeline_resume"))
    for resume_dir in stale_resume_dirs:
        problems.append("stale resume cache %s" % resume_dir)

    for rel in ("real_id_to_smiles.json", "id_smiles_y_true_y_pred.json", "xy_all.json"):
        if not (case_root / rel).exists():
            problems.append("missing %s" % (case_root / rel))

    if case in ESOL_CASES and not (case_root / "scatterplot_bundles").is_dir():
        problems.append("missing %s" % (case_root / "scatterplot_bundles"))

    for split in splits:
        split_root = case_root / split
        if not split_root.is_dir():
            problems.append("missing %s" % split_root)
            continue

        for method in METHODS:
            method_root = split_root / method
            for rel in REQUIRED_METHOD_PATHS:
                if not (method_root / rel).exists():
                    problems.append("missing %s" % (method_root / rel))
            csv_path = method_root / "clustering_metrics" / "cluster_assignments_with_real_id.csv"
            expected_rows = split_counts.get(split)
            actual_rows = count_csv_rows(csv_path) if csv_path.exists() else None
            if actual_rows is not None and expected_rows and actual_rows != expected_rows:
                problems.append(
                    "rowcount mismatch %s got=%s expected=%s" % (method_root, actual_rows, expected_rows)
                )

        for method, expected_count in EXPECTED_SWEEP_COUNTS.items():
            sweep_root = split_root / "sweeps" / method
            if not sweep_root.is_dir():
                problems.append("missing %s" % sweep_root)
                continue
            run_dirs = sorted(path for path in sweep_root.iterdir() if path.is_dir())
            expected_tags = expected_sweep_tags(method)
            actual_tags = {path.name for path in run_dirs}
            if len(run_dirs) != expected_count:
                problems.append(
                    "sweep count mismatch %s got=%s expected=%s" % (sweep_root, len(run_dirs), expected_count)
                )
            missing_tags = sorted(expected_tags - actual_tags)
            unexpected_tags = sorted(actual_tags - expected_tags)
            if missing_tags:
                problems.append("missing sweep tags %s %s" % (sweep_root, ",".join(missing_tags)))
            if unexpected_tags:
                problems.append("unexpected sweep tags %s %s" % (sweep_root, ",".join(unexpected_tags)))
            manifest_path = sweep_root / "manifest.json"
            if not manifest_path.exists():
                problems.append("missing %s" % manifest_path)
            for run_dir in run_dirs:
                for rel in REQUIRED_METHOD_PATHS:
                    if not (run_dir / rel).exists():
                        problems.append("missing %s" % (run_dir / rel))
                if not (run_dir / "run_meta.json").exists():
                    problems.append("missing %s" % (run_dir / "run_meta.json"))
                if (run_dir / ".pipeline_resume").exists():
                    problems.append("stale resume cache %s" % (run_dir / ".pipeline_resume"))


def expected_sweep_tags(method):
    if method == "pgmeans":
        return {
            "kmin_%s_kmax_%s" % (k_min, k_max)
            for k_min in range(2, 11)
            for k_max in range(k_min, 11)
        }
    if method == "agglomerative":
        return {
            "k_%s_linkage_%s" % (k, linkage)
            for linkage in ("ward", "complete", "average")
            for k in range(2, 11)
        }
    if method == "knn":
        return {"k_%s" % k for k in range(2, 11)}
    if method == "hdbscan":
        return {"default"}
    raise ValueError("Unsupported method: %s" % method)


def audit_frontend_static(problems):
    dimenet_front = FRONTEND_PUBLIC / "data" / "dimenet" / "test"
    for rel in REQUIRED_METHOD_PATHS:
        if not (dimenet_front / rel).exists():
            problems.append("missing %s" % (dimenet_front / rel))

    merged_meta = FRONTEND_PUBLIC / "id_smiles_y_true_y_pred.json"
    dimenet_meta = FRONTEND_PUBLIC / "dimnet" / "id_smiles_y_true_y_pred.json"
    for path in (merged_meta, dimenet_meta):
        if not path.exists():
            problems.append("missing %s" % path)

    if merged_meta.exists():
        payload = load_json(merged_meta)
        if isinstance(payload, dict) and "__error__" in payload:
            problems.append("broken %s" % merged_meta)


def main():
    problems = []
    for case in ESOL_CASES:
        audit_case(case, ESOL_SPLITS, problems)
    audit_case("dimenet", DIMENET_SPLITS, problems)

    for case in ("chemprop", "chemeleon", "dimenet"):
        overview = COMPACT_ROOT / case / "overview.json"
        if not overview.exists():
            problems.append("missing %s" % overview)
    if not (COMPACT_ROOT / "manifest.json").exists():
        problems.append("missing %s" % (COMPACT_ROOT / "manifest.json"))

    audit_frontend_static(problems)

    if problems:
        for item in problems:
            print(item)
        print("audit_status=fail")
        print("problem_count=%s" % len(problems))
        raise SystemExit(1)

    print("audit_status=ok")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
