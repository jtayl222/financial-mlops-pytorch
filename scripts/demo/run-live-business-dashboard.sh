#!/bin/bash
set -e

echo "💰 Running Live Business Impact Dashboard"

# Load environment
if [ -f ".env.live-dashboards" ]; then
    export $(cat .env.live-dashboards | grep -v '^#' | xargs)
fi

# Activate virtual environment
source .venv-live-dashboards/bin/activate

# Run the dashboard
python3 scripts/demo/live-business-impact-dashboard.py
