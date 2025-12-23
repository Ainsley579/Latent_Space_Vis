#!/usr/bin/env python3
"""
Example usage of the embedding processing pipeline.

This script demonstrates how to use the general pipeline with sample data.
"""

import numpy as np
from pathlib import Path
from embedding_pipeline import EmbeddingPipeline

# Create sample embeddings (simulating a neural network with 3 layers)
np.random.seed(42)
n_samples = 200
n_features_emb = 100
n_features_layer1 = 80
n_features_layer2 = 60

embeddings = {
    'emb': np.random.randn(n_samples, n_features_emb),
    'output_block_0': np.random.randn(n_samples, n_features_layer1),
    'output_block_1': np.random.randn(n_samples, n_features_layer2),
    'output_block_2': np.random.randn(n_samples, n_features_layer2),
}

# Optional: Create ID mapping (embedding index -> original dataset index)
# This is useful when embeddings are a subset of the original dataset
id_mapping = {i: i + 1000 for i in range(n_samples)}

# Optional: Create color values for scatter plot visualization
# (e.g., prediction values, target values, etc.)
color_values = np.random.randn(n_samples)

# Optional: Add metadata
metadata = {
    'dataset': 'example_dataset',
    'model': 'example_model',
    'description': 'Example pipeline usage'
}

# Create output directory
output_dir = Path('./example_output')
output_dir.mkdir(exist_ok=True)

print("="*60)
print("Example: Embedding Processing Pipeline")
print("="*60)
print(f"Number of samples: {n_samples}")
print(f"Number of layers: {len(embeddings)}")
print(f"Output directory: {output_dir}")
print("="*60)

# Create and run pipeline
pipeline = EmbeddingPipeline(
    embeddings=embeddings,
    output_dir=output_dir,
    id_mapping=id_mapping,
    n_components=5,  # Or None for auto-detection
    color_values=color_values,
    metadata=metadata
)

# Run the complete pipeline
pipeline.run()

print("\n" + "="*60)
print("Pipeline completed successfully!")
print("="*60)
print(f"\nGenerated files in {output_dir}:")
print("  - scatter_plot_data.csv")
print("  - sankey_diagram_data.csv")
print("  - pipeline_metadata.json")
print("  - id_mapping.json")
print("\nYou can now use these files for visualization!")

