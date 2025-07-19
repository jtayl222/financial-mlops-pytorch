#!/bin/bash
set -e

echo "🔍 Checking Training Readiness for Demo Models"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "src/train_pytorch_model.py" ]; then
    echo "❌ Error: Not in project root directory"
    echo "   Please run from the seldon-system directory"
    exit 1
fi

echo "✅ Project structure verified"

# Check for processed data
PROCESSED_DATA_DIR="${PROCESSED_DATA_DIR:-/mnt/shared-data/processed}"
ALT_DATA_DIR="data/processed"

echo ""
echo "📊 Checking for training data..."

if [ -d "$PROCESSED_DATA_DIR" ] && [ -f "$PROCESSED_DATA_DIR/train_features.npy" ]; then
    echo "✅ Found processed data at: $PROCESSED_DATA_DIR"
    DATA_PATH="$PROCESSED_DATA_DIR"
elif [ -d "$ALT_DATA_DIR" ] && [ -f "$ALT_DATA_DIR/train_features.npy" ]; then
    echo "✅ Found processed data at: $ALT_DATA_DIR"
    DATA_PATH="$ALT_DATA_DIR"
    export PROCESSED_DATA_DIR="$ALT_DATA_DIR"
else
    echo "❌ No processed training data found!"
    echo ""
    echo "🔧 To generate training data, run:"
    echo "   python3 src/data_ingestion.py"
    echo "   python3 src/feature_engineering_pytorch.py"
    echo ""
    echo "💡 Or check if data exists in a different location:"
    echo "   ls -la /mnt/shared-data/processed/"
    echo "   ls -la data/processed/"
    
    # Try to find data files anywhere
    echo ""
    echo "🔍 Searching for training data files..."
    find . -name "train_features.npy" 2>/dev/null || echo "   No train_features.npy found"
    find . -name "*features*.npy" 2>/dev/null || echo "   No feature files found"
    
    exit 1
fi

# Check data file sizes and content
echo ""
echo "📋 Data verification:"
python3 -c "
import numpy as np
import os

data_dir = '$DATA_PATH'
files = ['train_features.npy', 'train_targets.npy', 'val_features.npy', 'val_targets.npy', 'test_features.npy', 'test_targets.npy']

for file in files:
    path = os.path.join(data_dir, file)
    if os.path.exists(path):
        data = np.load(path)
        print(f'   ✅ {file}: {data.shape} ({data.dtype})')
    else:
        print(f'   ❌ {file}: Missing')
        
# Check for NaN values
try:
    train_features = np.load(os.path.join(data_dir, 'train_features.npy'))
    train_targets = np.load(os.path.join(data_dir, 'train_targets.npy'))
    
    if np.isnan(train_features).any():
        print(f'   ⚠️  Warning: NaN values in training features')
    if np.isnan(train_targets).any():
        print(f'   ⚠️  Warning: NaN values in training targets')
    
    # Check class distribution
    unique, counts = np.unique(train_targets, return_counts=True)
    print(f'   📊 Target distribution: {dict(zip(unique.astype(int), counts))}')
    
except Exception as e:
    print(f'   ❌ Error checking data: {e}')
"

# Check MLflow connectivity
echo ""
echo "🔌 Checking MLflow connectivity..."
MLFLOW_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"

if command -v curl >/dev/null 2>&1; then
    if curl -s --connect-timeout 5 "$MLFLOW_URI/health" >/dev/null 2>&1; then
        echo "✅ MLflow accessible at: $MLFLOW_URI"
    else
        echo "⚠️  MLflow not accessible at: $MLFLOW_URI"
        echo "   Training will work but metrics won't be logged"
        echo "   To start MLflow: kubectl port-forward -n mlflow svc/mlflow 5000:5000"
    fi
else
    echo "⚠️  curl not available - cannot test MLflow connectivity"
fi

# Check Python dependencies
echo ""
echo "🐍 Checking Python dependencies..."
python3 -c "
import sys
required_packages = ['torch', 'numpy', 'pandas', 'sklearn', 'mlflow']

missing = []
for package in required_packages:
    try:
        __import__(package)
        print(f'   ✅ {package}')
    except ImportError:
        missing.append(package)
        print(f'   ❌ {package}')

if missing:
    print(f'\\n🔧 Install missing packages:')
    print(f'   pip install {\" \".join(missing)}')
    sys.exit(1)
"

# Estimate training time
echo ""
echo "⏱️  Training time estimates (CPU-only):"
echo "   🔵 Baseline model: ~5-10 minutes"
echo "   🟢 Enhanced model: ~10-15 minutes"
echo "   📊 Total time: ~15-25 minutes"

echo ""
echo "🎯 READINESS SUMMARY"
echo "==================="
echo "✅ Training data available at: $DATA_PATH"
echo "✅ Python dependencies satisfied"
echo "✅ Ready to train authentic demo models"
echo ""
echo "🚀 Next step: Run ./scripts/demo/train-demo-models.sh"