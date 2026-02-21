#!/usr/bin/env python3
"""Create a small synthetic .pt embedding file for pipeline testing."""
import numpy as np
from pathlib import Path

try:
    import torch
except ImportError:
    raise SystemExit("PyTorch required")

# Structure: 2 epochs, 3 layers, 50 samples, 8 dims
np.random.seed(42)
n_epochs, n_layers, n_samples, dim = 2, 3, 50, 8
data = {}
for ei in range(n_epochs):
    data[f"epoch_{ei}"] = {}
    for li, layer_name in enumerate(["emb", "layer1", "layer2"]):
        data[f"epoch_{ei}"][layer_name] = np.random.randn(n_samples, dim).astype(np.float32) * (1 + ei * 0.2 + li * 0.1)

out = Path(__file__).resolve().parent.parent / "data" / "synthetic_embeddings.pt"
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(data, out)
print(f"Saved {out} with structure: 2 epochs x 3 layers x {n_samples} samples x {dim} dims")
