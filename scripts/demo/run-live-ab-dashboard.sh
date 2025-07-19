#!/bin/bash
set -e

echo "🚀 Running Live A/B Testing Dashboard"
echo "Monitors currently running MLflow experiments in real-time"
echo ""
echo "⚠️  Note: This requires active experiments with status='RUNNING'"
echo "   If no experiments are running, use:"
echo "   ./scripts/run-historical-ab-dashboard.sh"
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
        echo "   ./scripts/run-historical-ab-dashboard.sh"
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
python3 scripts/live-dashboard-generator.py