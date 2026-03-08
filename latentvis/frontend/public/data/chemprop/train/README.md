# Latent Space Visualization - Frontend Data (Chemprop, train split)

This directory contains pipeline output for the **chemprop train split**:
embeddings from `/global/cfs/projectdirs/m636/nhadler/projects/esol/data/chemprop-embeddings.pt`
restricted to the **train** split.

The layout mirrors `frontend/public/data/chemprop` (test split) but lives under
`frontend/public/data/chemprop/train` so that:

- Train and test scatterplots are kept separate.
- Interactive `/generate-range` updates can write to `scatterplot_current/`
  without overwriting baseline `scatterplot/` JSONs.

All JSON schemas match the test case (scatter, topk, substructures, Sankey, clustering metrics).
