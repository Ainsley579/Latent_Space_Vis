#!/usr/bin/env python3
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path("/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis")
BACKEND_DIR = PROJECT_ROOT / "latentvis" / "backend"
FRONTEND_DATA = PROJECT_ROOT / "latentvis" / "frontend" / "public" / "data"

sys.path.insert(0, str(BACKEND_DIR))


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

    assert resp.status_code == 200 and body.get("ok") is True
    assert real_id in body.get("matches", {}).get("chemprop", [])
    print("✅ search-smiles PASS")

    # ---------- Test 2: /generate-range ----------
    sankey_dir = FRONTEND_DATA / "chemprop" / "sankey"
    all_sankey_path = sankey_dir / "all_sankey.json"
    samples_path = sankey_dir / "sankey_samples.json"

    all_data = json.loads(all_sankey_path.read_text(encoding="utf-8"))
    epoch_dict = all_data.get("epoch") or {}
    epoch_ints = []
    for k in epoch_dict.keys():
        try:
            epoch_ints.append(int(k))
        except (ValueError, TypeError):
            continue
    if not epoch_ints:
        raise RuntimeError("No numeric epochs in all_sankey.json for chemprop")
    min_epoch, max_epoch = min(epoch_ints), max(epoch_ints)
    print("\n[TEST] /generate-range")
    print("Chemprop epochs range:", min_epoch, "..", max_epoch)

    orig_rows = len(json.loads(samples_path.read_text(encoding="utf-8")))
    print("Original sankey_samples rows:", orig_rows)

    resp2 = client.post(
        "/generate-range",
        json={"min": min_epoch, "max": max_epoch, "dataset": "chemprop"},
    )
    print("status:", resp2.status_code)
    body2 = resp2.get_json()
    print("response:", body2)

    assert resp2.status_code == 200 and body2.get("ok") is True
    kept = body2.get("summary", {}).get("chemprop", {}).get("kept_epochs")
    if kept is not None:
        assert kept == len(set(epoch_ints))

    new_rows = len(json.loads(samples_path.read_text(encoding="utf-8")))
    print("New sankey_samples rows:", new_rows)
    assert new_rows <= orig_rows
    print("✅ generate-range PASS")

    print("\nALL BACKEND TESTS PASSED.")


if __name__ == "__main__":
    main()

