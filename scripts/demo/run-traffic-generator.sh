#!/bin/bash
# Generate Real Traffic to Seldon A/B Test Experiment

set -e

echo "🚀 Starting Real Traffic Generator for Seldon A/B Test"
echo "📊 This will send actual requests to your financial-ab-test-experiment"
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

# Load environment variables if they exist
if [ -f ".env.live-dashboards" ]; then
    echo "📋 Loading environment variables..."
    set -a
    source .env.live-dashboards
    set +a
fi

# Set default environment variables for traffic generation
export SELDON_NAMESPACE="${SELDON_NAMESPACE:-financial-mlops-pytorch}"
export SELDON_EXPERIMENT_NAME="${SELDON_EXPERIMENT_NAME:-financial-ab-test-experiment}"
export TRAFFIC_RATE="${TRAFFIC_RATE:-120}"           # 120 requests per minute (2/sec)
export TRAFFIC_DURATION="${TRAFFIC_DURATION:-10}"    # 10 minutes
export TRAFFIC_WORKERS="${TRAFFIC_WORKERS:-5}"       # 5 concurrent workers
export MODEL_TIMEOUT="${MODEL_TIMEOUT:-5.0}"         # 5 second timeout

# Try to discover Seldon endpoint
echo "🔍 Checking Seldon experiment status..."
if command -v kubectl &> /dev/null; then
    EXPERIMENT_STATUS=$(kubectl get experiment $SELDON_EXPERIMENT_NAME -n $SELDON_NAMESPACE -o jsonpath='{.status.state}' 2>/dev/null || echo "NOT_FOUND")
    
    if [ "$EXPERIMENT_STATUS" = "NOT_FOUND" ]; then
        echo "❌ Experiment '$SELDON_EXPERIMENT_NAME' not found in namespace '$SELDON_NAMESPACE'"
        echo "   Please ensure the experiment is deployed:"
        echo "   kubectl apply -k k8s/base"
        exit 1
    else
        echo "✅ Experiment status: $EXPERIMENT_STATUS"
    fi
    
    # Try to get the actual service endpoint
    SELDON_SERVICE=$(kubectl get svc -n $SELDON_NAMESPACE -l app.kubernetes.io/instance=$SELDON_EXPERIMENT_NAME -o name 2>/dev/null | head -1)
    if [ -n "$SELDON_SERVICE" ]; then
        echo "✅ Found Seldon service: $SELDON_SERVICE"
    fi
else
    echo "⚠️  kubectl not available - cannot verify experiment status"
fi

echo
echo "📈 Traffic Generation Configuration:"
echo "   Experiment: $SELDON_EXPERIMENT_NAME"
echo "   Namespace: $SELDON_NAMESPACE"
echo "   Rate: $TRAFFIC_RATE requests/minute"
echo "   Duration: $TRAFFIC_DURATION minutes"
echo "   Workers: $TRAFFIC_WORKERS concurrent"
echo "   Timeout: $MODEL_TIMEOUT seconds"
echo

# Confirm before starting
echo "⚠️  This will generate real traffic to your Seldon experiment."
echo "   Press Ctrl+C at any time to stop."
echo
read -p "🚀 Start traffic generation? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo
echo "🎯 Starting traffic generation..."
echo "📊 Monitor progress in the logs below"
echo

# Run the traffic generator
python scripts/seldon-traffic-generator.py

echo
echo "✅ Traffic generation completed!"
echo
echo "📊 Next steps:"
echo "1. Wait 2-3 minutes for metrics to appear in Prometheus"
echo "2. Run the dashboard to see real data:"
echo "   ./scripts/run-seldon-ab-dashboard.sh"
echo "3. Compare with previous simulated results"
echo