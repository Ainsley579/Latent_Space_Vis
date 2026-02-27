#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "frontend" / "public" / "data"
    scatter_files = list(data_root.glob("**/scatterplot/*.json"))

    converted = 0
    skipped = 0
    for path in scatter_files:
        stem = path.stem
        if stem.endswith("_xy") or stem.endswith("_cluster"):
            skipped += 1
            continue

        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue

        if not isinstance(rows, list):
            skipped += 1
            continue

        xy_rows = []
        cluster_rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            rid = row.get("real_id")
            if rid is None:
                continue
            xy_rows.append(
                {
                    "real_id": rid,
                    "x": row.get("x"),
                    "y": row.get("y"),
                }
            )
            cluster_rows.append(
                {
                    "real_id": rid,
                    "cluster_id": row.get("cluster_id"),
                }
            )

        xy_path = path.with_name(f"{stem}_xy.json")
        cluster_path = path.with_name(f"{stem}_cluster.json")
        xy_path.write_text(json.dumps(xy_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        cluster_path.write_text(
            json.dumps(cluster_rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        converted += 1

    print(
        f"[split-scatterplot-files] converted={converted}, skipped={skipped}, "
        f"total={len(scatter_files)}"
    )


if __name__ == "__main__":
    main()

