#!/bin/bash
# Simple test runner for embedding pipeline tests

echo "Running embedding pipeline tests..."
echo "=================================="
echo ""

# Run pytest with verbose output
python -m pytest test_embedding_pipeline.py -v --tb=short

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
else
    echo ""
    echo "❌ Some tests failed. Check output above."
    exit 1
fi

