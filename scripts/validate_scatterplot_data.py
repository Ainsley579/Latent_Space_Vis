#!/usr/bin/env python3
"""
Validate that frontend scatterplot JSON data matches DimenetBenchmark source.

Checks:
1. All 18 JSON files exist (9 layers × 2 models)
2. Each file has exactly 152 points
3. Sample data values match DimenetBenchmark
4. real_id are consecutive 0-151
"""

import json
import csv
from pathlib import Path


def validate_json_files(json_dir):
    """Validate JSON files exist and have correct structure."""
    print("\n=== Validating JSON Files ===")
    
    json_dir = Path(json_dir)
    layers = ['emb', 'output_block_0', 'output_block_1', 'output_block_2', 
              'output_block_3', 'output_block_4', 'u_mlp_input', 'u_mlp_l1', 'u_mlp_l2']
    models = ['dimenet10conf', 'dimenet_noaug']
    
    total_points = 0
    issues = []
    
    for model in models:
        for layer in layers:
            filename = f"{model}_{layer}.json"
            filepath = json_dir / filename
            
            if not filepath.exists():
                issues.append(f"❌ Missing: {filename}")
                continue
            
            # Load and validate JSON
            with open(filepath) as f:
                data = json.load(f)
            
            # Check structure
            if not isinstance(data, list):
                issues.append(f"❌ {filename}: Expected list, got {type(data)}")
                continue
            
            if len(data) != 152:
                issues.append(f"⚠️  {filename}: Expected 152 points, got {len(data)}")
            
            # Check sample point
            if len(data) > 0:
                first = data[0]
                required_keys = {'real_id', 'x', 'y', 'cluster_id'}
                if not required_keys.issubset(first.keys()):
                    issues.append(f"❌ {filename}: Missing keys. Has {first.keys()}")
                
                # Validate real_ids are sequential
                real_ids = [int(p['real_id']) for p in data]
                if real_ids != list(range(152)):
                    issues.append(f"⚠️  {filename}: real_ids not sequential 0-151")
            
            print(f"✓ {filename}: {len(data)} points")
            total_points += len(data)
    
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ All JSON files validated successfully!")
    
    print(f"\nTotal points: {total_points} (expected: {152 * 18})")
    return len(issues) == 0


def compare_with_source(json_dir, csv_path, model_name):
    """Compare sample data from JSON with source CSV.
    
    Args:
        json_dir: Directory containing JSON files
        csv_path: Path to CSV file (model-specific)
        model_name: Model name (dimenet10conf or dimenet_noaug)
    """
    print("\n=== Comparing with Source Data ===")
    
    json_dir = Path(json_dir)
    csv_path = Path(csv_path)
    
    # Load source CSV (no model_type column since it's model-specific)
    csv_data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['layer_name'], int(row['real_id']))
            csv_data[key] = {
                'x': float(row['x']),
                'y': float(row['y']),
                'cluster_id': int(row['cluster_id'])
            }
    
    # Sample test cases
    test_cases = [
        ('emb', 0),
        ('emb', 1),
        ('output_block_0', 1),
        ('output_block_0', 50),
        ('u_mlp_l2', 151),
    ]
    
    all_match = True
    
    for layer, real_id in test_cases:
        # Load JSON
        json_file = json_dir / f"{model_name}_{layer}.json"
        with open(json_file) as f:
            json_data = json.load(f)
        
        json_point = next((p for p in json_data if int(p['real_id']) == real_id), None)
        
        if not json_point:
            print(f"❌ {model_name}/{layer}: Missing real_id {real_id}")
            all_match = False
            continue
        
        # Get source data
        source_key = (layer, real_id)
        source_point = csv_data.get(source_key)
        
        if not source_point:
            print(f"⚠️  {model_name}/{layer}: No source data for real_id {real_id}")
            continue
        
        # Compare
        x_match = abs(json_point['x'] - source_point['x']) < 1e-6
        y_match = abs(json_point['y'] - source_point['y']) < 1e-6
        c_match = json_point['cluster_id'] == source_point['cluster_id']
        
        if x_match and y_match and c_match:
            print(f"✓ {model_name}/{layer}/real_id={real_id}: MATCH")
        else:
            print(f"❌ {model_name}/{layer}/real_id={real_id}: MISMATCH")
            print(f"   JSON: x={json_point['x']}, y={json_point['y']}, cluster_id={json_point['cluster_id']}")
            print(f"   CSV:  x={source_point['x']}, y={source_point['y']}, cluster_id={source_point['cluster_id']}")
            all_match = False
    
    return all_match


def main():
    json_dir = Path("/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/frontend/public/data/scatterplot")
    csv_dir = Path("/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/data/scatter_plots")
    
    print("Validating scatterplot data...")
    
    json_ok = validate_json_files(json_dir)
    
    # Compare both CSVs separately
    print("\n\n" + "="*60)
    print("Comparing dimenet10conf data:")
    csv_10conf = csv_dir / "scatter_by_layer.csv"
    compare_ok_10conf = compare_with_source(json_dir, csv_10conf, "dimenet10conf")
    
    print("\n\n" + "="*60)
    print("Comparing dimenet_noaug data:")
    csv_noaug = csv_dir / "scatter_noaug_by_layer.csv"
    compare_ok_noaug = compare_with_source(json_dir, csv_noaug, "dimenet_noaug")
    
    print("\n\n" + "="*60)
    if json_ok and compare_ok_10conf and compare_ok_noaug:
        print("\n✅ ALL VALIDATIONS PASSED!")
        print("   Scatterplot data is accurate and matches DimenetBenchmark source.")
    else:
        print("\n❌ VALIDATION FAILED")
        print("   Please check the issues above.")


if __name__ == "__main__":
    main()
