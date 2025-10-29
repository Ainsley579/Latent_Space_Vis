#!/usr/bin/env python3
"""
Monitor the GMM analysis progress
"""

import os
import time
import glob
from pathlib import Path

def monitor_analysis():
    """Monitor the analysis progress"""
    
    results_dir = "/global/cfs/cdirs/m636/nhadler/projects/latent-spaces/GNNGMMAnalyzerResults"
    
    print("Monitoring GMM analysis progress...")
    print(f"Results directory: {results_dir}")
    
    # Look for new directories created today
    today = time.strftime("%Y%m%d")
    pattern = f"{results_dir}/dimenet10conf_checkpoint0_unique_cas_zca_{today}_*"
    
    print(f"Looking for directories matching: {pattern}")
    
    while True:
        # Find matching directories
        dirs = glob.glob(pattern)
        
        if dirs:
            # Get the most recent directory
            latest_dir = max(dirs, key=os.path.getctime)
            print(f"\nFound analysis directory: {latest_dir}")
            
            # Check for output files
            output_files = list(Path(latest_dir).glob("*.png"))
            csv_files = list(Path(latest_dir).glob("*.csv"))
            html_files = list(Path(latest_dir).glob("*.html"))
            npy_files = list(Path(latest_dir).glob("*.npy"))
            txt_files = list(Path(latest_dir).glob("*.txt"))
            
            total_files = len(output_files) + len(csv_files) + len(html_files) + len(npy_files) + len(txt_files)
            
            print(f"Progress: {total_files} files generated")
            print(f"  - PNG files: {len(output_files)}")
            print(f"  - CSV files: {len(csv_files)}")
            print(f"  - HTML files: {len(html_files)}")
            print(f"  - NPY files: {len(npy_files)}")
            print(f"  - TXT files: {len(txt_files)}")
            
            # List some key files
            if output_files:
                print(f"\nRecent output files:")
                for f in sorted(output_files)[-5:]:  # Show last 5
                    print(f"  - {f.name}")
            
            # Check if analysis is complete (look for summary file)
            summary_file = Path(latest_dir) / "analysis_summary.txt"
            if summary_file.exists():
                print(f"\n✅ Analysis appears to be complete!")
                print(f"📁 Results saved in: {latest_dir}")
                break
            else:
                print(f"\n⏳ Analysis still running... (no summary file yet)")
        else:
            print(f"\n⏳ No analysis directory found yet, waiting...")
        
        # Wait before checking again
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    try:
        monitor_analysis()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
