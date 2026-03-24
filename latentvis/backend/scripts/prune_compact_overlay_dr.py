#!/usr/bin/env python3
"""
Prune redundant DR payload from compact overlay JSON files.

Targets:
  - data_compact/<dataset>/clusterings/*.json
  - data_compact/<dataset>/sweeps/**/*.json

Keeps only overlay fields required by backend merge:
  - api_version, dataset, source_variant
  - cells: node_key, cluster_labels, cluster_representatives
  - nodes, connectivity, split_connectivity, node_lookup

Safety:
  - If overlay cell node_keys are not a subset of base.json node_keys, skip file.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPACT_ROOT = ROOT / "data_compact"


@dataclass
class FileResult:
    path: Path
    changed: bool
    skipped: bool
    reason: str
    old_bytes: int
    new_bytes: int


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(path)


def _compact_overlay_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    overlay: Dict[str, Any] = {}
    for k in ("api_version", "dataset", "source_variant"):
        if k in payload:
            overlay[k] = payload[k]
    if "split_names" in payload:
        overlay["split_names"] = payload["split_names"]

    out_cells: List[Dict[str, Any]] = []
    cells = payload.get("cells")
    if isinstance(cells, list):
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            out: Dict[str, Any] = {}
            if "node_key" in cell:
                out["node_key"] = cell["node_key"]
            if "cluster_labels" in cell:
                out["cluster_labels"] = cell["cluster_labels"]
            if "cluster_representatives" in cell:
                out["cluster_representatives"] = cell["cluster_representatives"]
            if out:
                out_cells.append(out)
    overlay["cells"] = out_cells

    for k in ("nodes", "connectivity", "split_connectivity", "node_lookup"):
        if k in payload:
            overlay[k] = payload[k]
    return overlay


def _node_keys_from_cells(cells: Any) -> Set[str]:
    out: Set[str] = set()
    if not isinstance(cells, list):
        return out
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        if "node_key" in cell:
            out.add(str(cell["node_key"]))
    return out


def _iter_target_files(dataset_dir: Path) -> Iterable[Path]:
    clusterings = dataset_dir / "clusterings"
    sweeps = dataset_dir / "sweeps"
    if clusterings.is_dir():
        for p in sorted(clusterings.glob("*.json")):
            yield p
    if sweeps.is_dir():
        for p in sorted(sweeps.rglob("*.json")):
            yield p


def process_dataset(dataset_dir: Path, write: bool) -> List[FileResult]:
    results: List[FileResult] = []
    base_path = dataset_dir / "base.json"
    if not base_path.is_file():
        return results
    base = _read_json(base_path)
    base_keys = _node_keys_from_cells(base.get("cells") if isinstance(base, dict) else None)
    if not base_keys:
        return results

    for path in _iter_target_files(dataset_dir):
        try:
            payload = _read_json(path)
        except Exception:
            results.append(
                FileResult(path=path, changed=False, skipped=True, reason="invalid_json", old_bytes=0, new_bytes=0)
            )
            continue
        if not isinstance(payload, dict):
            results.append(
                FileResult(path=path, changed=False, skipped=True, reason="not_object", old_bytes=0, new_bytes=0)
            )
            continue
        cells = payload.get("cells")
        overlay_keys = _node_keys_from_cells(cells)
        if overlay_keys and not overlay_keys.issubset(base_keys):
            results.append(
                FileResult(
                    path=path,
                    changed=False,
                    skipped=True,
                    reason="node_key_mismatch_vs_base",
                    old_bytes=path.stat().st_size,
                    new_bytes=path.stat().st_size,
                )
            )
            continue

        compact = _compact_overlay_from_payload(payload)
        old_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        new_text = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        old_bytes = len(old_text.encode("utf-8"))
        new_bytes = len(new_text.encode("utf-8"))
        changed = old_text != new_text
        if changed and write:
            _write_json(path, compact)

        results.append(
            FileResult(
                path=path,
                changed=changed,
                skipped=False,
                reason="ok",
                old_bytes=old_bytes,
                new_bytes=new_bytes,
            )
        )

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Prune redundant DR from compact overlay files.")
    ap.add_argument(
        "--compact-root",
        type=Path,
        default=DEFAULT_COMPACT_ROOT,
        help="Path to backend/data_compact",
    )
    ap.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset name under compact root (repeatable). Default: all datasets with base.json",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="Actually rewrite files. Default is dry-run.",
    )
    args = ap.parse_args()

    root = args.compact_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"compact root not found: {root}")

    if args.dataset:
        dataset_dirs = [root / name for name in args.dataset]
    else:
        dataset_dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "base.json").is_file()]

    all_results: List[FileResult] = []
    for ds in dataset_dirs:
        if not ds.is_dir():
            continue
        all_results.extend(process_dataset(ds, write=args.write))

    changed = [r for r in all_results if r.changed and not r.skipped]
    skipped = [r for r in all_results if r.skipped]
    old_total = sum(r.old_bytes for r in changed)
    new_total = sum(r.new_bytes for r in changed)
    saved = old_total - new_total

    print(
        json.dumps(
            {
                "ok": True,
                "mode": "write" if args.write else "dry-run",
                "files_scanned": len(all_results),
                "files_changed": len(changed),
                "files_skipped": len(skipped),
                "bytes_before_changed_files": old_total,
                "bytes_after_changed_files": new_total,
                "bytes_saved": saved,
            },
            indent=2,
        )
    )
    if skipped:
        for item in skipped[:20]:
            print(f"skip: {item.path} ({item.reason})")
        if len(skipped) > 20:
            print(f"... {len(skipped) - 20} more skipped files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
