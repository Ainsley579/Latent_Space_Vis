#!/usr/bin/env python3
"""
Check Python Environment for Real ID Tracking Script

This script checks if all required dependencies are installed.
"""

import sys

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name} - NOT INSTALLED")
        return False

def main():
    """Check all required packages"""
    print("="*60)
    print("Checking Python Environment for Real ID Tracking Script")
    print("="*60)
    print()
    
    required_packages = [
        ("Python", "sys"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("torch", "torch"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scipy", "scipy"),
    ]
    
    all_installed = True
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            all_installed = False
    
    print()
    print("="*60)
    if all_installed:
        print("✅ All required packages are installed!")
        print()
        print("You can now run the real ID tracking script:")
        print("  python generate_real_id_tracking_csv.py --embeddings_dir <path> --output_dir <path>")
        print()
        print("Or use the helper script:")
        print("  bash run_real_id_tracking.sh")
    else:
        print("❌ Some required packages are missing!")
        print()
        print("Please install missing packages:")
        print("  pip install torch scikit-learn pandas numpy matplotlib seaborn scipy")
        print()
        print("Or if using conda:")
        print("  conda activate losslandscapestda")
        print("  conda install pytorch scikit-learn pandas numpy matplotlib seaborn scipy -c conda-forge")
    print("="*60)
    
    return 0 if all_installed else 1

if __name__ == "__main__":
    sys.exit(main())

