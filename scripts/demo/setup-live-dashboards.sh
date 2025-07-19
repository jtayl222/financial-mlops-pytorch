#!/bin/bash
#
# Setup Live Dashboards - Install dependencies and configure environment
#

set -e

echo "🚀 Setting up Live Dashboard Environment"

# Check if we're in the right directory
if [ ! -f "scripts/demo/live-dashboard-generator.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv-live-dashboards" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv-live-dashboards
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv-live-dashboards/bin/activate

# Upgrade pip
pip install --upgrade pip

# Note: requirements-live-dashboards.txt is now committed in scripts/demo/

# Install required packages
echo "📚 Installing Python dependencies..."
pip install -r scripts/demo/requirements-live-dashboards.txt

# Verify required environment variables are set
echo "🔍 Checking required environment variables..."

missing_vars=()

if [ -z "$MLFLOW_TRACKING_URI" ]; then
    missing_vars+=("MLFLOW_TRACKING_URI")
fi

if [ -z "$MLFLOW_DB_PASSWORD" ]; then
    missing_vars+=("MLFLOW_DB_PASSWORD")
fi

if [ -z "$MLFLOW_DB_USERNAME" ]; then
    missing_vars+=("MLFLOW_DB_USERNAME")
fi

if [ -z "$PROMETHEUS_URL" ]; then
    missing_vars+=("PROMETHEUS_URL")
fi

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo "❌ Error: Required environment variables not set:"
    printf '   %s\n' "${missing_vars[@]}"
    echo ""
    echo "💡 Set these variables and try again:"
    echo "   export MLFLOW_TRACKING_URI=http://your-mlflow-host:5000"
    echo "   export MLFLOW_DB_USERNAME=your-username"  
    echo "   export MLFLOW_DB_PASSWORD=your-password"
    echo "   export PROMETHEUS_URL=http://your-prometheus:9090"
    exit 1
fi

echo "✅ Using MLflow: $MLFLOW_TRACKING_URI"
echo "✅ Using Prometheus: $PROMETHEUS_URL"

# Create dashboard-specific configuration
echo "⚙️  Creating dashboard configuration..."
cat > .env.live-dashboards << EOF
# Dashboard Configuration
DASHBOARD_REFRESH_INTERVAL=30
DASHBOARD_PORT=8050
DASHBOARD_HOST=0.0.0.0

# Business Parameters for calculations
BASE_TRADING_VOLUME=10000000
ACCURACY_REVENUE_MULTIPLIER=0.005
LATENCY_COST_PER_MS=0.0001
ERROR_COST_MULTIPLIER=50
INFRASTRUCTURE_ANNUAL_COST=53000
EOF

echo "🔐 Created .env.live-dashboards using existing environment variables as defaults"
echo "⚠️  IMPORTANT: The file will use your existing MLflow environment variables automatically"

# Create helper scripts
echo "🛠️  Creating helper scripts..."

# Create live A/B dashboard script if it doesn't exist (now permanent)
if [ ! -f "scripts/demo/run-live-ab-dashboard.sh" ]; then
    echo "📝 Creating run-live-ab-dashboard.sh (permanent file)..."
    cat > scripts/demo/run-live-ab-dashboard.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Running Live A/B Testing Dashboard"
echo "Monitors currently running MLflow experiments in real-time"
echo ""
echo "⚠️  Note: This requires active experiments with status='RUNNING'"
echo "   If no experiments are running, use:"
echo "   ./scripts/demo/run-historical-ab-dashboard.sh"
echo ""

# Check for running experiments first
echo "🔍 Checking for running experiments..."
if command -v python3 > /dev/null; then
    RUNNING_COUNT=$(python3 -c "
import mlflow
import os
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://192.168.1.203:5000'))
client = mlflow.MlflowClient()
experiments = client.search_experiments()
running_count = 0
for exp in experiments:
    runs = client.search_runs([exp.experiment_id], filter_string='status = \"RUNNING\"')
    running_count += len(runs)
print(running_count)
" 2>/dev/null || echo "0")

    if [ "$RUNNING_COUNT" = "0" ]; then
        echo "⚠️  No running experiments found!"
        echo "   To start an experiment for live monitoring:"
        echo "   argo submit --from workflowtemplate/financial-training-pipeline-template -p model-variant=enhanced -n seldon-system"
        echo ""
        echo "   Or use historical analysis instead:"
        echo "   ./scripts/demo/run-historical-ab-dashboard.sh"
        echo ""
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
    else
        echo "✅ Found $RUNNING_COUNT running experiments"
    fi
fi

# Load environment
if [ -f ".env.live-dashboards" ]; then
    export $(cat .env.live-dashboards | grep -v '^#' | xargs)
fi

# Activate virtual environment
source .venv-live-dashboards/bin/activate

# Run the live monitoring dashboard
python3 scripts/demo/live-dashboard-generator.py
EOF
else
    echo "📋 Using existing run-live-ab-dashboard.sh (permanent file)"
fi

# Note: run-live-business-dashboard.sh is now a committed file

# Make scripts executable
if [ -f "scripts/demo/run-live-ab-dashboard.sh" ]; then
    chmod +x scripts/demo/run-live-ab-dashboard.sh
fi
if [ -f "scripts/demo/run-seldon-ab-dashboard.sh" ]; then
    chmod +x scripts/demo/run-seldon-ab-dashboard.sh
fi
# Note: run-live-business-dashboard.sh permissions set by git

# Note: interactive-live-dashboard.py is now a committed file

# Note: test-database-connection.py is now a committed file

echo ""
echo "✅ Live Dashboard Environment Setup Complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Activate virtual environment: source .venv-live-dashboards/bin/activate"
echo "2. Update .env.live-dashboards with your actual credentials"
echo "3. Test connections: python3 scripts/demo/test-database-connection.py"
echo "4. Analyze completed experiments: ./scripts/demo/run-historical-ab-dashboard.sh"
echo "5. Monitor live experiments: ./scripts/demo/run-live-ab-dashboard.sh (requires running experiments)"
echo "6. Monitor production Seldon A/B tests: ./scripts/demo/run-seldon-ab-dashboard.sh"
echo "7. Run live business dashboard: ./scripts/demo/run-live-business-dashboard.sh"
echo "8. Start interactive web dashboard: python3 scripts/demo/interactive-live-dashboard.py"
echo ""
echo "🌐 Web dashboard will be available at: http://localhost:8050"
echo ""
echo "⚠️  SECURITY NOTE: Change default passwords before production use!"