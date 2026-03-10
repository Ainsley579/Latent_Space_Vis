#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path


_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from scripts.materialize_sankey_compat_files import canonicalize_layers, reorder_layers


class TestMaterializeSankeyCompatFiles(unittest.TestCase):
    def test_canonicalize_chemprop_layers(self):
        layers = [
            "gnn_layer_1",
            "gnn_layer_2",
            "gnn_layer_3",
            "fingerprint",
            "ffn_layer_1",
            "ffn_layer_2",
        ]
        self.assertEqual(
            canonicalize_layers(layers),
            [
                "fingerprint",
                "gnn_layer_1",
                "gnn_layer_2",
                "gnn_layer_3",
                "ffn_layer_1",
                "ffn_layer_2",
            ],
        )

    def test_reorder_layers_respects_explicit_override(self):
        layers = ["gnn_layer_1", "fingerprint", "ffn_layer_1"]
        self.assertEqual(
            reorder_layers(layers, ["ffn_layer_1", "fingerprint"]),
            ["ffn_layer_1", "fingerprint", "gnn_layer_1"],
        )


if __name__ == "__main__":
    unittest.main()
