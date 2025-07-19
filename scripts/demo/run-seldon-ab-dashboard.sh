#!/bin/bash
# Production Seldon A/B Test Dashboard - Query actual experiment results

set -e

echo "🚀 Starting Seldon A/B Test Dashboard Generator"
echo "📊 Querying production experiment: financial-ab-test-experiment"
echo

# Check if virtual environment exists
if [ ! -d ".venv-live-dashboards" ]; then
    echo "❌ Virtual environment not found. Please run:"
    echo "   ./scripts/setup-live-dashboards.sh"
    exit 1
fi

# Activate virtual environment
echo "🔗 Activating Python virtual environment..."
source .venv-live-dashboards/bin/activate

# Check if required dependencies are installed
echo "📦 Checking dependencies..."
python -c "import aiohttp, matplotlib, pandas, numpy, seaborn, prometheus_client" 2>/dev/null || {
    echo "❌ Missing dependencies. Please run:"
    echo "   ./scripts/setup-live-dashboards.sh"
    exit 1
}

# Load environment variables if they exist
if [ -f ".env.live-dashboards" ]; then
    echo "📋 Loading environment variables..."
    set -a
    source .env.live-dashboards
    set +a
fi

# Set default environment variables
export PROMETHEUS_URL="${PROMETHEUS_URL:-http://192.168.1.85:30090}"
export SELDON_NAMESPACE="${SELDON_NAMESPACE:-seldon-system}"
export SELDON_EXPERIMENT_NAME="${SELDON_EXPERIMENT_NAME:-financial-ab-test-experiment}"

echo "📈 Configuration:"
echo "   Prometheus URL: $PROMETHEUS_URL"
echo "   Seldon Namespace: $SELDON_NAMESPACE"
echo "   Experiment Name: $SELDON_EXPERIMENT_NAME"
echo

# Check if experiment is running
echo "🔍 Checking if experiment is active..."
if command -v kubectl &> /dev/null; then
    EXPERIMENT_STATUS=$(kubectl get experiment $SELDON_EXPERIMENT_NAME -n $SELDON_NAMESPACE -o jsonpath='{.status.state}' 2>/dev/null || echo "NOT_FOUND")
    
    if [ "$EXPERIMENT_STATUS" = "NOT_FOUND" ]; then
        echo "⚠️  Experiment '$SELDON_EXPERIMENT_NAME' not found in namespace '$SELDON_NAMESPACE'"
        echo "   Dashboard will show simulated data"
    else
        echo "✅ Experiment status: $EXPERIMENT_STATUS"
    fi
else
    echo "⚠️  kubectl not available - cannot check experiment status"
fi

echo

# Run the dashboard generator
echo "🎨 Generating Seldon A/B test dashboard..."
python scripts/seldon-ab-dashboard.py

echo
echo "✅ Seldon dashboard generation completed!"
echo "🖼️  Check for generated PNG files in the current directory"
echo

# Show generated files
echo "📁 Generated files:"
ls -la seldon_ab_dashboard_*.png 2>/dev/null || echo "   No dashboard files found"