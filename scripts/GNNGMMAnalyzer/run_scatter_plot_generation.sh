#!/bin/bash
# Run Scatter Plot Data Generation Script
# This script generates CSV files for scatter plot visualization

set -e  # Exit on error

echo "=========================================="
echo "Scatter Plot Data Generator"
echo "=========================================="

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Set environment variables
export CUDA_VISIBLE_DEVICES=""

# Default paths
DEFAULT_EMBEDDINGS_DIR="$PROJECT_ROOT/scripts/dimenet10conf_embeddings"
DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/frontend/public/data"

# Parse command line arguments
EMBEDDINGS_DIR="${1:-$DEFAULT_EMBEDDINGS_DIR}"
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"

# Check if embeddings directory exists
if [ ! -d "$EMBEDDINGS_DIR" ]; then
    echo "❌ Error: Embeddings directory not found: $EMBEDDINGS_DIR"
    echo ""
    echo "Please provide the correct path to your embeddings directory:"
    echo "  Usage: $0 [embeddings_dir] [output_dir]"
    echo ""
    echo "Example:"
    echo "  $0 $DEFAULT_EMBEDDINGS_DIR $DEFAULT_OUTPUT_DIR"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found. Please install Python 3.7+"
    exit 1
fi

# Check if required Python packages are available
echo "Checking Python environment..."
python3 -c "import torch; import sklearn; import pandas; import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Required Python packages not found."
    echo ""
    echo "Please install the required packages:"
    echo "  pip install torch scikit-learn pandas numpy matplotlib seaborn scipy"
    echo ""
    echo "Or activate your conda environment:"
    echo "  conda activate losslandscapestda"
    exit 1
fi

echo "✓ Python environment check passed"
echo ""

# Change to script directory
cd "$SCRIPT_DIR"

# Build command
CMD="python3 generate_scatter_plot_data.py"
CMD="$CMD --embeddings_dir \"$EMBEDDINGS_DIR\""
CMD="$CMD --output_dir \"$OUTPUT_DIR\""

# Print configuration
echo "Configuration:"
echo "  Embeddings directory: $EMBEDDINGS_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo ""
echo "Running analysis..."
echo "=========================================="
echo ""

# Run the script
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Scatter plot data generation completed successfully!"
    echo "=========================================="
    echo ""
    echo "Generated files:"
    echo "  - $OUTPUT_DIR/scatter_plot_data_all_layers.csv"
    echo "  - $OUTPUT_DIR/scatter_plot_data_by_layer/*.csv (9 files)"
    echo "  - $OUTPUT_DIR/scatter_plot_metadata.json"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ Generation failed. Please check the error messages above."
    echo "=========================================="
    exit 1
fi

