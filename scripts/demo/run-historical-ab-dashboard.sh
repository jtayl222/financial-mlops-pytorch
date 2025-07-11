#!/bin/bash
set -e

echo "📊 Running Historical A/B Testing Dashboard"
echo "Analyzes completed MLflow experiments and generates insights"
echo ""

# Load environment
if [ -f ".env.live-dashboards" ]; then
    export $(cat .env.live-dashboards | grep -v '^#' | xargs)
fi

# Activate virtual environment
source .venv-live-dashboards/bin/activate

# Run the historical analysis dashboard
python3 scripts/live-dashboard-generator.py