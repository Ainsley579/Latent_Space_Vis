#!/bin/bash
# Run Real ID Tracking CSV Generation Script
# This script sets up the environment and runs the real ID tracking analysis

set -e  # Exit on error

echo "=========================================="
echo "Real ID Tracking CSV Generator"
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
N_COMPONENTS="${3:-}"  # Optional: number of components
LAYER_NAME="${4:-}"   # Optional: layer name for component analysis

# Check if embeddings directory exists
if [ ! -d "$EMBEDDINGS_DIR" ]; then
    echo "❌ Error: Embeddings directory not found: $EMBEDDINGS_DIR"
    echo ""
    echo "Please provide the correct path to your embeddings directory:"
    echo "  Usage: $0 [embeddings_dir] [output_dir] [n_components] [layer_name]"
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
CMD="python3 generate_real_id_tracking_csv.py"
CMD="$CMD --embeddings_dir \"$EMBEDDINGS_DIR\""
CMD="$CMD --output_dir \"$OUTPUT_DIR\""

if [ -n "$N_COMPONENTS" ]; then
    CMD="$CMD --n_components $N_COMPONENTS"
fi

if [ -n "$LAYER_NAME" ]; then
    CMD="$CMD --layer_name $LAYER_NAME"
fi

# Print configuration
echo "Configuration:"
echo "  Embeddings directory: $EMBEDDINGS_DIR"
echo "  Output directory: $OUTPUT_DIR"
if [ -n "$N_COMPONENTS" ]; then
    echo "  Number of components: $N_COMPONENTS"
fi
if [ -n "$LAYER_NAME" ]; then
    echo "  Layer for component analysis: $LAYER_NAME"
fi
echo ""
echo "Running analysis..."
echo "=========================================="
echo ""

# Run the script
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Analysis completed successfully!"
    echo "=========================================="
    echo ""
    echo "Generated files:"
    echo "  - $OUTPUT_DIR/real_id_layer_transitions/real_id_layer_transitions.csv"
    echo "  - $OUTPUT_DIR/real_id_component_transitions/real_id_component_transitions.csv"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ Analysis failed. Please check the error messages above."
    echo "=========================================="
    exit 1
fi

