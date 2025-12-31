A notebook-friendly workflow for loading neural network embeddings, running GMM-based latent space analysis, and visualizing the results using interactive Scatter plots and Sankey diagrams.

The typical workflow in a notebook consists of three steps:
	1.	Set up imports and paths
	2.	Load embeddings and run the pipeline
	3.	Visualize results with interactive plots

## For Pipeline Usage
Refer to [README_PIPELIINE](https://github.com/Ainsley579/Latent_Space_Vis/blob/shiyi/scripts/GNNGMMAnalyzer/README_PIPELINE.md)

## Visualization in Notebook
Import funtions
```python
from latentvis import render_scatter, render_sankey
```
charts
```python
render_scatter(scatter_df, height=400)
render_sankey(sankey_df, height=450)
```
