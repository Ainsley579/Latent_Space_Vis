#!/usr/bin/env python3
"""Create a tiny chemprop-style .pt for fast pipeline testing (epoch -> split -> embeddings -> layer -> tensor)."""
import numpy as np
import torch
from pathlib import Path

out = Path(__file__).resolve().parent.parent / "data" / "tiny_chemprop.pt"
out.parent.mkdir(parents=True, exist_ok=True)
np.random.seed(42)
n = 40
d = 8
# 2 epochs, 2 layers, test split only
data = {
    "1": {
        "test": {
            "embeddings": {
                "GNN layer 1": np.random.randn(n, d).astype(np.float32),
                "GNN layer 2": np.random.randn(n, d).astype(np.float32),
            },
            "id": list(range(n)),
        },
    },
    "2": {
        "test": {
            "embeddings": {
                "GNN layer 1": np.random.randn(n, d).astype(np.float32),
                "GNN layer 2": np.random.randn(n, d).astype(np.float32),
            },
            "id": list(range(n)),
        },
    },
}
torch.save(data, out)
print("Created", out)
