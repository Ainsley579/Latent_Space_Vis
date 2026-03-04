#!/usr/bin/env python3
"""
Write id/smiles/y_true/y_pred rows (with split) from ESOL-format embedding .pt files.

Usage:
  python scripts/write_id_smiles_y_with_split.py --both-cases

  python scripts/write_id_smiles_y_with_split.py \
    --input /path/to/chemprop-embeddings.pt \
    --output ../frontend/public/data/chemprop/id_smiles_y_true_y_pred.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required to load .pt files") from exc


_BACKEND = Path(__file__).resolve().parent.parent
FRONTEND_DATA = _BACKEND.resolve().parents[1] / "latentvis" / "frontend" / "public" / "data"
DEFAULT_CHEMPROP_PT = Path("/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemprop-embeddings.pt")
DEFAULT_CHEMELEON_PT = Path("/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemeleon-embeddings.pt")


def _to_plain_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


def _load_raw_pt(pt_path: Path) -> Any:
    try:
        return torch.load(str(pt_path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(pt_path), map_location="cpu")


def load_rows(pt_path: Path, include_splits: List[str]) -> List[Dict[str, Any]]:
    raw = _load_raw_pt(pt_path)
    if not isinstance(raw, dict):
        return []

    rows: List[Dict[str, Any]] = []
    for split_name in include_splits:
        ids = smiles = y_true = y_pred = None
        for _, epoch_data in raw.items():
            if not isinstance(epoch_data, dict):
                continue
            split_data = epoch_data.get(split_name)
            if not isinstance(split_data, dict):
                continue
            ids_local = split_data.get("id")
            smiles_local = split_data.get("smiles")
            y_local = split_data.get("y")
            y_pred_local = split_data.get("y_pred")
            if not all(isinstance(x, (list, tuple)) for x in [ids_local, smiles_local, y_local, y_pred_local]):
                continue
            n = min(len(ids_local), len(smiles_local), len(y_local), len(y_pred_local))
            if n <= 0:
                continue
            ids = list(ids_local[:n])
            smiles = list(smiles_local[:n])
            y_true = list(y_local[:n])
            y_pred = list(y_pred_local[:n])
            break

        if ids is None or smiles is None or y_true is None or y_pred is None:
            continue

        for i in range(len(ids)):
            rid = str(ids[i])
            rows.append(
                {
                    "id": rid,
                    "real_id": rid,
                    "smiles": str(smiles[i]),
                    "y_true": _to_plain_float(y_true[i]),
                    "y_pred": _to_plain_float(y_pred[i]),
                    "split": split_name,
                }
            )
    return rows


def write_rows(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _write_case(name: str, pt_path: Path, out_path: Path, include_splits: List[str]) -> int:
    if not pt_path.exists():
        print(f"Skip {name}: input not found: {pt_path}")
        return 1
    rows = load_rows(pt_path, include_splits=include_splits)
    if not rows:
        print(f"Skip {name}: no rows loaded from {pt_path}")
        return 1
    write_rows(rows, out_path)
    by_split = {}
    for s in include_splits:
        by_split[s] = sum(1 for r in rows if r.get("split") == s)
    print(f"Wrote {out_path} ({len(rows)} rows; split counts: {by_split})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Write id/smiles/y_true/y_pred JSON for frontend.")
    ap.add_argument("--input", type=Path, help="Input ESOL-format .pt")
    ap.add_argument("--output", type=Path, help="Output JSON path")
    ap.add_argument(
        "--splits",
        type=str,
        default="test,train,val",
        help="Comma-separated splits to include; default: test,train,val",
    )
    ap.add_argument(
        "--both-cases",
        action="store_true",
        help="Write both chemprop and chemeleon to frontend/public/data/<case>/id_smiles_y_true_y_pred.json",
    )
    args = ap.parse_args()

    include_splits = [s.strip().lower() for s in args.splits.split(",") if s.strip()]
    if not include_splits:
        print("No valid splits requested.")
        return 1

    if args.both_cases:
        if args.input or args.output:
            print("Do not combine --both-cases with --input/--output.")
            return 1
        cases = [
            ("chemprop", DEFAULT_CHEMPROP_PT, FRONTEND_DATA / "chemprop" / "id_smiles_y_true_y_pred.json"),
            ("chemeleon", DEFAULT_CHEMELEON_PT, FRONTEND_DATA / "chemeleon" / "id_smiles_y_true_y_pred.json"),
        ]
        failures = 0
        for name, pt_path, out_path in cases:
            failures += _write_case(name, pt_path, out_path, include_splits=include_splits)
        return 1 if failures else 0

    if not args.input or not args.output:
        print("Provide --input and --output, or use --both-cases.")
        return 1
    return _write_case("single", args.input, args.output, include_splits=include_splits)


if __name__ == "__main__":
    sys.exit(main())
