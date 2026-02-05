#!/usr/bin/env python3
"""
Split scatter_by_layer.csv from DimenetBenchmark into separate 10conf and noaug CSVs.
This ensures accurate data in Latent_Space_Vis backend.

Usage:
    python split_scatter_data.py \
        --source /path/to/DimenetBenchmark/data/scatter_plots/scatter_by_layer.csv \
        --output-dir /path/to/Latent_Space_Vis/latentvis/backend/data/scatter_plots/
"""

import argparse
import pandas as pd
from pathlib import Path


def split_scatter_data(source_csv, output_dir):
    """
    Split scatter_by_layer.csv into dimenet10conf and dimenet_noaug CSVs.
    
    Args:
        source_csv: Path to DimenetBenchmark/data/scatter_plots/scatter_by_layer.csv
        output_dir: Output directory for split CSVs
    """
    # Read source CSV
    print(f"Reading {source_csv}")
    df = pd.read_csv(source_csv)
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Model types: {df['model_type'].unique()}")
    print(f"Unique real_ids: {df['real_id'].nunique()}")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split by model type
    df_10conf = df[df['model_type'] == 'dimenet10conf'].copy()
    df_noaug = df[df['model_type'] == 'dimenet_noaug'].copy()
    
    # Reset index and drop model_type column
    df_10conf = df_10conf.drop(columns=['model_type']).reset_index(drop=True)
    df_noaug = df_noaug.drop(columns=['model_type']).reset_index(drop=True)
    
    # Save split CSVs
    out_10conf = output_dir / "scatter_by_layer.csv"
    out_noaug = output_dir / "scatter_noaug_by_layer.csv"
    
    df_10conf.to_csv(out_10conf, index=False)
    df_noaug.to_csv(out_noaug, index=False)
    
    print(f"\n✓ Saved {len(df_10conf)} 10conf rows to {out_10conf}")
    print(f"  Columns: {list(df_10conf.columns)}")
    print(f"  Sample (real_id=0, emb):")
    sample_10conf = df_10conf[(df_10conf['real_id'] == 0) & (df_10conf['layer_name'] == 'emb')]
    if len(sample_10conf) > 0:
        print(f"    x={sample_10conf.iloc[0]['x']}, y={sample_10conf.iloc[0]['y']}, cluster_id={sample_10conf.iloc[0]['cluster_id']}")
    
    print(f"\n✓ Saved {len(df_noaug)} noaug rows to {out_noaug}")
    print(f"  Columns: {list(df_noaug.columns)}")
    print(f"  Sample (real_id=0, emb):")
    sample_noaug = df_noaug[(df_noaug['real_id'] == 0) & (df_noaug['layer_name'] == 'emb')]
    if len(sample_noaug) > 0:
        print(f"    x={sample_noaug.iloc[0]['x']}, y={sample_noaug.iloc[0]['y']}, cluster_id={sample_noaug.iloc[0]['cluster_id']}")
    
    # Verify data consistency
    print(f"\n✓ Verification:")
    print(f"  10conf unique real_ids: {sorted(df_10conf['real_id'].unique())}")
    print(f"  noaug unique real_ids: {sorted(df_noaug['real_id'].unique())}")
    print(f"  Layers: {sorted(df_10conf['layer_name'].unique())}")
    
    # Warn if real_ids are not 0-151
    if df_10conf['real_id'].max() != 151:
        print(f"\n⚠️  WARNING: Expected real_ids 0-151, but max is {df_10conf['real_id'].max()}")


def main():
    parser = argparse.ArgumentParser(description="Split scatter_by_layer.csv by model type")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/DimenetBenchmark/data/scatter_plots/scatter_by_layer.csv"),
        help="Source CSV from DimenetBenchmark"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/data/scatter_plots/"),
        help="Output directory for split CSVs"
    )
    
    args = parser.parse_args()
    
    if not args.source.exists():
        print(f"Error: Source file not found: {args.source}")
        exit(1)
    
    split_scatter_data(args.source, args.output_dir)


if __name__ == "__main__":
    main()
