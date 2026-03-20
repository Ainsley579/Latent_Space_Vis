#!/usr/bin/env python3
"""
Build id_smiles_y_true_y_pred.json for the current 152-sample DimeNet CAS case.

Outputs:
  1. backend/data/dimenet/id_smiles_y_true_y_pred.json
  2. merged frontend/public/id_smiles_y_true_y_pred.json

The current DimeNet case uses the 152-row CAS-deduplicated ordering from
latent-spaces/dimenet_noaug_embeddings_cas_152 and
latent-spaces/dimenet10conf_embeddings/*_unique_cas.pt.

For noaug predictions, the available source tensor is the older 156-row CAS
file. The first 152 rows are verified to match the current 152-row embedding
order exactly, so we trim to that prefix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch


DEFAULT_SPLIT_JSON = Path(
    "/global/cfs/cdirs/m636/nhadler/projects/Libra-ML-Modeling/model-training/butene/dimenet-noaug/data/jsons/split_0.json"
)
DEFAULT_LITERATURE_TSV = Path(
    "/global/cfs/cdirs/m636/nhadler/projects/Libra-ML-Modeling/data/HF-Literature.tsv"
)
DEFAULT_NOAUG_EMB_152 = Path(
    "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet_noaug_embeddings_cas_152/emb_unique_cas.pt"
)
DEFAULT_NOAUG_EMB_156 = Path(
    "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet_noaug_embeddings_cas/emb_unique_cas.pt"
)
DEFAULT_NOAUG_PREDS_156 = Path(
    "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet_noaug_embeddings_cas/u_preds_cas.pt"
)
DEFAULT_10CONF_PREDS_152 = Path(
    "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/dimenet10conf_embeddings/u_preds_cas.pt"
)
DEFAULT_BACKEND_OUT = Path(
    "/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/data/dimenet/id_smiles_y_true_y_pred.json"
)
DEFAULT_FRONTEND_OUT = Path(
    "/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/frontend/public/id_smiles_y_true_y_pred.json"
)
EXPECTED_COUNT = 152


def _load_tensor(path: Path) -> torch.Tensor:
    return torch.load(str(path), map_location="cpu")


def _to_float_list(tensor: Any) -> List[float]:
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu()
    if hasattr(tensor, "reshape"):
        tensor = tensor.reshape(-1)
    return [float(x) for x in tensor.tolist()]


def build_case_rows(split_json: Path, literature_tsv: Path) -> pd.DataFrame:
    split_data = json.loads(split_json.read_text(encoding="utf-8"))
    train_df = pd.DataFrame(split_data.get("train", {}))
    test_df = pd.DataFrame(split_data.get("test", {}))
    val_df = pd.DataFrame(split_data.get("val", {}))
    ordered_df = pd.concat([train_df, test_df, val_df], ignore_index=True)

    literature_df = pd.read_csv(literature_tsv, sep="\t")
    merged = ordered_df.merge(literature_df, on="rxn_id", how="left")
    need = {"ligand_1_cas", "ligand_1_smiles", "ddG"}
    missing = [c for c in need if c not in merged.columns]
    if missing:
        raise ValueError(f"Missing required columns after merge: {missing}")

    dedup = merged.drop_duplicates(subset=["ligand_1_cas"], keep="first").reset_index(drop=True)
    if len(dedup) < EXPECTED_COUNT:
        raise ValueError(f"Deduplicated metadata has only {len(dedup)} rows, expected at least {EXPECTED_COUNT}")
    return dedup.iloc[:EXPECTED_COUNT].copy().reset_index(drop=True)


def verify_noaug_alignment(emb_152: Path, emb_156: Path) -> None:
    a = _load_tensor(emb_152)
    b = _load_tensor(emb_156)
    if tuple(a.shape[:1]) != (EXPECTED_COUNT,):
        raise ValueError(f"Unexpected 152 embedding shape: {tuple(a.shape)}")
    if b.shape[0] < EXPECTED_COUNT:
        raise ValueError(f"Unexpected 156 embedding shape: {tuple(b.shape)}")
    if not torch.equal(a, b[:EXPECTED_COUNT]):
        raise ValueError("Current 152-row noaug embedding order does not match first 152 rows of CAS-156 source")


def build_rows(
    split_json: Path,
    literature_tsv: Path,
    emb_152: Path,
    emb_156: Path,
    noaug_preds_156: Path,
    conf_preds_152: Path,
) -> List[Dict[str, Any]]:
    verify_noaug_alignment(emb_152, emb_156)
    rows_df = build_case_rows(split_json, literature_tsv)

    noaug = _to_float_list(_load_tensor(noaug_preds_156))[:EXPECTED_COUNT]
    conf = _to_float_list(_load_tensor(conf_preds_152))
    if len(noaug) != EXPECTED_COUNT:
        raise ValueError(f"Unexpected noaug prediction count: {len(noaug)}")
    if len(conf) != EXPECTED_COUNT:
        raise ValueError(f"Unexpected 10conf prediction count: {len(conf)}")

    out: List[Dict[str, Any]] = []
    for idx, row in rows_df.iterrows():
        real_id = str(idx)
        item = {
            "dataset": "dimenet",
            "id": real_id,
            "real_id": real_id,
            "smiles": str(row["ligand_1_smiles"]),
            "y_true": float(row["ddG"]),
            "y_pred_by_epoch": {
                "dimenet_noaug": noaug[idx],
                "dimenet10conf": conf[idx],
            },
            # Keep compatibility with the frontend's current scalar y_pred handling.
            "y_pred": conf[idx],
            "y_pred_noaug": noaug[idx],
            "y_pred_dimenet10conf": conf[idx],
            "split": "test",
        }
        out.append(item)
    return out


def write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def merge_frontend_json(path: Path, dimenet_rows: List[Dict[str, Any]]) -> None:
    existing: List[Dict[str, Any]] = []
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                existing = [row for row in raw if isinstance(row, dict)]
        except Exception:
            existing = []
    kept = [row for row in existing if str(row.get("dataset", "")).strip().lower() != "dimenet"]
    merged = kept + dimenet_rows
    write_json(path, merged)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build DimeNet id/smiles/y_true/y_pred JSON.")
    ap.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    ap.add_argument("--literature-tsv", type=Path, default=DEFAULT_LITERATURE_TSV)
    ap.add_argument("--noaug-emb-152", type=Path, default=DEFAULT_NOAUG_EMB_152)
    ap.add_argument("--noaug-emb-156", type=Path, default=DEFAULT_NOAUG_EMB_156)
    ap.add_argument("--noaug-preds-156", type=Path, default=DEFAULT_NOAUG_PREDS_156)
    ap.add_argument("--conf-preds-152", type=Path, default=DEFAULT_10CONF_PREDS_152)
    ap.add_argument("--backend-output", type=Path, default=DEFAULT_BACKEND_OUT)
    ap.add_argument("--frontend-merged-output", type=Path, default=DEFAULT_FRONTEND_OUT)
    args = ap.parse_args()

    rows = build_rows(
        split_json=args.split_json,
        literature_tsv=args.literature_tsv,
        emb_152=args.noaug_emb_152,
        emb_156=args.noaug_emb_156,
        noaug_preds_156=args.noaug_preds_156,
        conf_preds_152=args.conf_preds_152,
    )
    write_json(args.backend_output, rows)
    merge_frontend_json(args.frontend_merged_output, rows)
    print(f"Wrote {len(rows)} rows to {args.backend_output}")
    print(f"Merged DimeNet rows into {args.frontend_merged_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
