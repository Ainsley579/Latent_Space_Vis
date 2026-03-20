#!/usr/bin/env python3
"""
Test that run_pipeline produces all expected output files (no frontend).
Uses a small synthetic .pt so tests run without real data.
Run: python -m unittest tests.test_pipeline_outputs
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Backend on path
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _make_synthetic_pt(path: Path, n_epochs: int = 2, n_layers: int = 2, n_samples: int = 30, dim: int = 6) -> None:
    """Write a minimal .pt compatible with embedding_loader (nested epoch -> layer -> array)."""
    try:
        import torch
    except ImportError:
        raise unittest.SkipTest("PyTorch required for synthetic .pt")
    data = {}
    for ei in range(n_epochs):
        data[f"epoch_{ei}"] = {}
        for li, layer_name in enumerate(["emb", "layer1"][:n_layers]):
            data[f"epoch_{ei}"][layer_name] = np.random.randn(n_samples, dim).astype(np.float32) * (1 + ei * 0.1 + li * 0.05)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


class TestPipelineOutputs(unittest.TestCase):
    """Run pipeline with synthetic .pt and assert all expected outputs exist and are valid."""

    def test_pipeline_writes_all_outputs(self):
        from run_pipeline_from_embeddings import run_pipeline

        with tempfile.TemporaryDirectory(prefix="latentvis_pipeline_test_") as tmp:
            tmp_path = Path(tmp)
            pt_path = tmp_path / "synthetic.pt"
            _make_synthetic_pt(pt_path, n_epochs=2, n_layers=2, n_samples=25, dim=6)

            ok = run_pipeline(
                input_path=str(pt_path),
                output_dir=str(tmp_path),
                k_min=2,
                k_max=5,
                clustering_method="pgmeans",
                skip_tsne=True,
            )
            self.assertTrue(ok)

            base = tmp_path
            scatter_dir = base / "scatterplot"
            self.assertTrue(scatter_dir.is_dir(), "scatterplot dir missing")
            pca_files = list(scatter_dir.glob("*_pca.json"))
            self.assertGreaterEqual(len(pca_files), 2, "expected at least 2 cell scatter files (epoch_layer)")
            for f in pca_files:
                data = json.loads(f.read_text(encoding="utf-8"))
                self.assertIsInstance(data, list)
                self.assertGreater(len(data), 0)
                for p in data:
                    self.assertIn("real_id", p)
                    self.assertIn("x", p)
                    self.assertIn("y", p)
                    self.assertIn("cluster_id", p)

            topk_dir = base / "topk"
            self.assertTrue((topk_dir / "cluster_representatives.json").is_file())
            reps = json.loads((topk_dir / "cluster_representatives.json").read_text(encoding="utf-8"))
            self.assertIsInstance(reps, list)
            self.assertTrue((topk_dir / "cluster_sorted_members.json").is_file())
            members = json.loads((topk_dir / "cluster_sorted_members.json").read_text(encoding="utf-8"))
            self.assertIsInstance(members, list)

            self.assertTrue((base / "sankey" / "all_sankey.json").is_file())
            sankey = json.loads((base / "sankey" / "all_sankey.json").read_text(encoding="utf-8"))
            self.assertTrue("epoch" in sankey or "layer" in sankey)

            self.assertTrue((base / "clustering_metrics" / "cluster_assignments_with_real_id.csv").is_file())

            sub_dir = base / "substructures"
            if (sub_dir / "cluster_substructures.json").is_file():
                subs = json.loads((sub_dir / "cluster_substructures.json").read_text(encoding="utf-8"))
                self.assertIsInstance(subs, list)


class TestCollectUpdatedSummary(unittest.TestCase):
    """Unit test for _collect_updated_results_summary: mock dir yields expected keys and counts."""

    def test_collect_updated_results_summary(self):
        from flask_backend import _collect_updated_results_summary

        with tempfile.TemporaryDirectory(prefix="latentvis_updated_summary_") as tmp:
            root = Path(tmp)
            (root / "scatterplot").mkdir()
            (root / "scatterplot" / "epoch_0_layer1_pca.json").write_text("[]")
            (root / "scatterplot" / "epoch_0_layer1_mds.json").write_text("[]")
            (root / "topk").mkdir()
            (root / "topk" / "cluster_representatives.json").write_text("[{}, {}]")
            (root / "topk" / "cluster_sorted_members.json").write_text("[{}, {}, {}]")
            (root / "substructures").mkdir()
            (root / "substructures" / "cluster_substructures.json").write_text("[{}]")
            (root / "sankey").mkdir()
            (root / "sankey" / "all_sankey.json").write_text("{}")
            (root / "clustering_metrics").mkdir()
            (root / "clustering_metrics" / "cluster_assignments_with_real_id.csv").write_text("real_id,epoch\n")

            summary = _collect_updated_results_summary(root)
            self.assertEqual(summary["cells"], 1)
            self.assertEqual(summary["scatter_files"], 2)
            self.assertEqual(summary["topk_representatives"], 2)
            self.assertEqual(summary["topk_sorted_members"], 3)
            self.assertEqual(summary["substructure_entries"], 1)
            self.assertTrue(summary["sankey_ok"])
            self.assertTrue(summary["cluster_assignments_ok"])

        empty = _collect_updated_results_summary(Path("/nonexistent"))
        self.assertEqual(empty["cells"], 0)
        self.assertEqual(empty["scatter_files"], 0)
        self.assertFalse(empty["sankey_ok"])


class TestSplitSupport(unittest.TestCase):
    def test_load_real_id_smiles_rows_includes_val(self):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch required for ESOL split test")

        from embedding_loader import load_real_id_smiles_rows

        with tempfile.TemporaryDirectory(prefix="latentvis_split_rows_") as tmp:
            pt_path = Path(tmp) / "esol_like.pt"
            payload = {
                "0": {
                    "train": {
                        "id": ["train_a"],
                        "smiles": ["CC"],
                        "embeddings": {"fingerprint": np.zeros((1, 2), dtype=np.float32)},
                    },
                    "val": {
                        "id": ["val_a"],
                        "smiles": ["CO"],
                        "embeddings": {"fingerprint": np.zeros((1, 2), dtype=np.float32)},
                    },
                    "test": {
                        "id": ["test_a"],
                        "smiles": ["CN"],
                        "embeddings": {"fingerprint": np.zeros((1, 2), dtype=np.float32)},
                    },
                }
            }
            torch.save(payload, pt_path)

            rows = load_real_id_smiles_rows(pt_path)
            self.assertEqual([row["split"] for row in rows], ["test", "val", "train"])
            self.assertEqual({row["real_id"] for row in rows}, {"train_a", "val_a", "test_a"})

    def test_compact_dataset_config_includes_val_sources(self):
        source = (
            _BACKEND / "scripts" / "build_compact_overview_data.py"
        ).read_text(encoding="utf-8")
        self.assertIn("DEFAULT_COORD_DECIMALS = 2", source)
        self.assertIn('api_version": "compact_overview_v2"', source)
        self.assertIn('split="train"', source)
        self.assertIn('split="val"', source)
        self.assertIn('split="test"', source)
        self.assertIn('chemprop_root / "val" / "knn" / "scatterplot"', source)
        self.assertIn('chemeleon_root / "val" / "knn" / "scatterplot"', source)
        self.assertIn('"cluster_representatives"', source)
        self.assertIn('"split_connectivity"', source)


if __name__ == "__main__":
    unittest.main()
