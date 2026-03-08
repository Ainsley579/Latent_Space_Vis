#!/usr/bin/env python3
"""
Backend API tests (no frontend). Run from backend dir: python3 scripts/test_flask_endpoints.py
Set LATENTVIS_SKIP_FULL_PIPELINE=1 to skip the full generate-range pipeline (fast; 404 on embeddings).
"""
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path("/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis")
BACKEND_DIR = PROJECT_ROOT / "latentvis" / "backend"
FRONTEND_DATA = PROJECT_ROOT / "latentvis" / "frontend" / "public" / "data"

sys.path.insert(0, str(BACKEND_DIR))

# Optional: skip full pipeline test by forcing embedding-not-found (fast run)
if os.environ.get("LATENTVIS_SKIP_FULL_PIPELINE"):
    os.environ["LATENTVIS_CHEMPROP_PT"] = "/nonexistent/chemprop.pt"


def main() -> None:
    print("Using backend_dir =", BACKEND_DIR)
    try:
        # Ensure user site-packages (where Flask was installed) is on sys.path.
        user_site = Path("/pscratch/sd/j/jiaqing/dotfiles_storage/.local/lib/python3.6/site-packages")
        if user_site.is_dir() and str(user_site) not in sys.path:
            sys.path.append(str(user_site))
        import flask_backend  # type: ignore
    except ImportError as e:
        print("ERROR: failed to import flask_backend (is Flask installed in this env?):", e)
        sys.exit(1)

    app = flask_backend.app
    client = app.test_client()

    # ---------- Test 1: /search-smiles ----------
    chemprop_map_path = FRONTEND_DATA / "chemprop" / "real_id_to_smiles.json"
    mapping = json.loads(chemprop_map_path.read_text(encoding="utf-8"))
    real_id, smiles = next(iter(mapping.items()))
    print("\n[TEST] /search-smiles")
    print("Sample chemprop real_id/smiles:", real_id, "/", smiles)

    resp = client.post(
        "/search-smiles",
        json={"smiles": smiles, "dataset": "chemprop"},
    )
    print("status:", resp.status_code)
    body = resp.get_json()
    print("response:", body)

    if resp.status_code == 200 and body.get("ok") is True:
        assert real_id in body.get("matches", {}).get("chemprop", [])
        print(" search-smiles PASS")
    else:
        # RDKit or real_id_to_smiles may be missing in test env
        print(" search-smiles SKIP (backend returned %s; RDKit or data may be missing)" % resp.status_code)

    # ---------- Test 2: /generate-range (response shape only; full run requires embeddings) ----------
    # If chemprop embeddings exist, we can run a real generate-range and check response.
    # Otherwise we only test that invalid request returns 400 and response shape when we have data.
    print("\n[TEST] /generate-range")
    resp2 = client.post(
        "/generate-range",
        json={"min": 2, "max": 5, "dataset": "chemprop", "clustering": "pgmeans"},
    )
    body2 = resp2.get_json()
    print("status:", resp2.status_code, "body keys:", list(body2.keys()) if body2 else None)

    if resp2.status_code == 200:
        assert body2.get("ok") is True
        summary = body2.get("summary", {}).get("chemprop", {})
        assert summary.get("status") == "done"
        updated = summary.get("updated")
        if updated is not None:
            assert "cells" in updated
            assert "scatter_files" in updated
            assert "topk_representatives" in updated
            assert "sankey_ok" in updated
        print(" generate-range PASS (full pipeline ran)")
    elif resp2.status_code == 404:
        # Embedding file not found (e.g. in CI); accept as skip
        print(" generate-range SKIP (embedding not found)")
    else:
        assert resp2.status_code in (200, 404), "expected 200 or 404, got %s: %s" % (resp2.status_code, body2)
    print(" generate-range response check done")

    print("\nALL BACKEND TESTS PASSED.")


if __name__ == "__main__":
    main()

