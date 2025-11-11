#!/usr/bin/env python3
"""
Verify Generated Files

This script verifies that the generated files are correct by comparing:
1. Cluster assignments match between sample_component_paths and individual_real_id files
2. Transition counts match between aggregate files
3. Real IDs are correctly mapped
"""

import pandas as pd
import json
from pathlib import Path

def verify_version(version):
    """Verify files for a specific version"""
    print(f"\n{'='*60}")
    print(f"VERIFYING VERSION {version}")
    print(f"{'='*60}")
    
    base_dir = Path("/Users/chenjiaqing/Documents/GitHub/Latent_Space_Vis/frontend/public/data")
    
    # Load files
    sample_orig = pd.read_csv(base_dir / f'sample_component_paths_{version}.csv')
    sample_new = pd.read_csv(base_dir / f'sample_component_paths_{version}_with_real_id.csv')
    individual = pd.read_csv(base_dir / f'cluster_transition_matrices_{version}_individual_real_id.csv')
    aggregate_orig = pd.read_csv(base_dir / f'cluster_transition_matrices_{version}.csv')
    aggregate_new = pd.read_csv(base_dir / f'cluster_transition_matrices_{version}_with_real_id.csv')
    
    # Load mapping
    mapping_path = Path("/Users/chenjiaqing/Documents/GitHub/Latent_Space_Vis/scripts/GNNGMMAnalyzer/cas_to_original_mapping.json")
    with open(mapping_path, 'r') as f:
        mapping = {int(k): int(v) for k, v in json.load(f).items()}
    reverse_mapping = {v: k for k, v in mapping.items()}
    
    print(f"\nFile Sizes:")
    print(f"  sample_component_paths_{version}.csv: {len(sample_orig)} samples")
    print(f"  sample_component_paths_{version}_with_real_id.csv: {len(sample_new)} samples")
    print(f"  cluster_transition_matrices_{version}_individual_real_id.csv: {len(individual)} transitions")
    print(f"  cluster_transition_matrices_{version}.csv: {len(aggregate_orig)} transitions")
    print(f"  cluster_transition_matrices_{version}_with_real_id.csv: {len(aggregate_new)} transitions")
    
    # Check if sample_id in original corresponds to CAS index
    print(f"\n{'='*60}")
    print("VERIFICATION 1: Sample ID Mapping")
    print(f"{'='*60}")
    
    if version == 2:
        # Version 2: sample_id in original should be CAS index (0-151)
        # sample_id in new should be original index (mapped from CAS)
        print("\nChecking sample_id mapping for Version 2:")
        print("  Original sample_component_paths_2.csv uses CAS indices (0-151)")
        print("  New sample_component_paths_2_with_real_id.csv uses original indices (0-3477)")
        
        # Check a few samples
        matches = 0
        mismatches = 0
        for cas_id in range(min(10, len(sample_orig))):
            if cas_id in mapping:
                orig_id = mapping[cas_id]
                orig_row = sample_orig[sample_orig['sample_id'] == cas_id]
                new_row = sample_new[sample_new['sample_id'] == orig_id]
                
                if len(orig_row) > 0 and len(new_row) > 0:
                    # Compare cluster assignments
                    orig_emb = orig_row['emb'].values[0]
                    new_emb = new_row['emb'].values[0]
                    
                    if orig_emb == new_emb:
                        matches += 1
                    else:
                        mismatches += 1
                        if mismatches <= 3:
                            print(f"    ⚠ CAS {cas_id} -> Original {orig_id}: emb mismatch (orig={orig_emb}, new={new_emb})")
        
        print(f"\n  Cluster assignment matches: {matches}/{min(10, len(sample_orig))}")
        print(f"  Cluster assignment mismatches: {mismatches}/{min(10, len(sample_orig))}")
        
        if mismatches > 0:
            print("\n  ⚠ WARNING: Cluster assignments don't match!")
            print("  This could be due to:")
            print("    - Different GMM fits (random state, initialization)")
            print("    - Different alignment/permutation")
            print("    - Different embeddings used")
    
    # Check transition counts
    print(f"\n{'='*60}")
    print("VERIFICATION 2: Transition Count Comparison")
    print(f"{'='*60}")
    
    # Compare aggregate transition matrices
    print("\nComparing aggregate transition matrices:")
    print("  (Note: Counts may differ due to different GMM fits)")
    
    # Sample a few transitions
    sample_transitions = aggregate_orig.head(10)
    for _, row in sample_transitions.iterrows():
        from_layer = row['from_layer']
        to_layer = row['to_layer']
        from_cluster = row['from_cluster']
        to_cluster = row['to_cluster']
        
        orig_count = row['count']
        new_row = aggregate_new[
            (aggregate_new['from_layer'] == from_layer) &
            (aggregate_new['to_layer'] == to_layer) &
            (aggregate_new['from_cluster'] == from_cluster) &
            (aggregate_new['to_cluster'] == to_cluster)
        ]
        
        if len(new_row) > 0:
            new_count = new_row['count'].values[0]
            match = abs(orig_count - new_count) < 0.01
            if not match:
                print(f"    ⚠ {from_layer}->{to_layer}, {from_cluster}->{to_cluster}: orig={orig_count}, new={new_count}")
    
    # Check real ID consistency
    print(f"\n{'='*60}")
    print("VERIFICATION 3: Real ID Consistency")
    print(f"{'='*60}")
    
    print("\nChecking real ID consistency between files:")
    
    # Check if real_ids in individual file match sample_ids in sample_component_paths
    individual_real_ids = set(individual['real_id'].unique())
    sample_real_ids = set(sample_new['sample_id'].unique())
    
    print(f"  Real IDs in individual file: {len(individual_real_ids)} unique")
    print(f"  Real IDs in sample_component_paths: {len(sample_real_ids)} unique")
    print(f"  Overlap: {len(individual_real_ids & sample_real_ids)}")
    
    if individual_real_ids == sample_real_ids:
        print("  ✓ Real IDs match between files")
    else:
        print("  ⚠ Real IDs don't fully match")
        missing = sample_real_ids - individual_real_ids
        extra = individual_real_ids - sample_real_ids
        if missing:
            print(f"    Missing in individual: {sorted(list(missing))[:10]}")
        if extra:
            print(f"    Extra in individual: {sorted(list(extra))[:10]}")
    
    # Check real ID range
    print(f"\n  Real ID ranges:")
    print(f"    Individual file: {individual['real_id'].min()} - {individual['real_id'].max()}")
    print(f"    Sample paths file: {sample_new['sample_id'].min()} - {sample_new['sample_id'].max()}")
    
    # Verify mapping
    print(f"\n  Verifying CAS to original mapping:")
    cas_ids_in_individual = set()
    for real_id in individual['real_id'].unique():
        if real_id in reverse_mapping:
            cas_ids_in_individual.add(reverse_mapping[real_id])
    
    print(f"    CAS indices found: {len(cas_ids_in_individual)}")
    print(f"    Expected: 152 (for CAS embeddings)")
    
    return {
        'version': version,
        'sample_orig_count': len(sample_orig),
        'sample_new_count': len(sample_new),
        'individual_count': len(individual),
        'aggregate_orig_count': len(aggregate_orig),
        'aggregate_new_count': len(aggregate_new),
        'real_id_match': individual_real_ids == sample_real_ids
    }

def main():
    """Main verification function"""
    print("="*60)
    print("VERIFYING GENERATED FILES")
    print("="*60)
    
    results = []
    for version in [1, 2]:
        result = verify_version(version)
        results.append(result)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        print(f"\nVersion {result['version']}:")
        print(f"  Sample files: {result['sample_orig_count']} -> {result['sample_new_count']} samples")
        print(f"  Individual transitions: {result['individual_count']}")
        print(f"  Aggregate transitions: {result['aggregate_orig_count']} -> {result['aggregate_new_count']}")
        print(f"  Real ID consistency: {'✓' if result['real_id_match'] else '⚠'}")

if __name__ == "__main__":
    main()

