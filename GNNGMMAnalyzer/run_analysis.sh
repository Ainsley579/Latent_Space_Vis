#!/bin/bash
# Run Comprehensive GMM Analysis with Correct CAS Mapping

echo "Starting Comprehensive GMM Analysis with Correct CAS Mapping..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=""

# Change to the correct directory
cd /global/cfs/cdirs/m636/nhadler/projects/latent-spaces/GNNGMMAnalyzer

# Run the analysis
/global/homes/j/jiaqing/.conda/envs/losslandscapestda/bin/python comprehensive_gmm_analysis.py \
    --output_dir /global/cfs/cdirs/m636/nhadler/projects/latent-spaces/GNNGMMAnalyzerResults/dimenet10conf_checkpoint0_fixed_comprehensive_analysis_with_correct_cas_indices \
    --max_components 20

echo "Analysis completed!"
