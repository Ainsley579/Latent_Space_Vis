#!/usr/bin/env python3
"""
Run Comprehensive GMM Analysis with Correct CAS Mapping

This script runs the comprehensive analysis with the corrected CAS mapping.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_analysis():
    """Run the comprehensive GMM analysis"""
    
    print("Starting Comprehensive GMM Analysis with Correct CAS Mapping...")
    
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Define the command
    python_path = "/global/homes/j/jiaqing/.conda/envs/losslandscapestda/bin/python"
    output_dir = "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/GNNGMMAnalyzerResults/dimenet10conf_checkpoint0_fixed_comprehensive_analysis_with_correct_cas_indices"
    
    cmd = [
        python_path,
        "comprehensive_gmm_analysis.py",
        "--output_dir", output_dir,
        "--max_components", "20"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    try:
        # Run the analysis
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\nAnalysis completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError running analysis: {e}")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False

if __name__ == "__main__":
    success = run_analysis()
    if success:
        print("\n✅ Analysis completed successfully!")
        print("Check the output directory for results.")
    else:
        print("\n❌ Analysis failed. Check the error messages above.")

