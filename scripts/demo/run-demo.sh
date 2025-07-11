#!/bin/bash
# Financial MLOps Demo Script

echo "🚀 Starting Financial MLOps Demo"

# Reset any previous demo state
echo "🧹 Resetting demo environment..."
./scripts/demo/demo-reset.sh

# Setup dashboard environment
echo "📊 Setting up live dashboards..."
./scripts/demo/setup-live-dashboards.sh

# Run the A/B testing dashboard demo
echo "🧪 Running A/B testing dashboard..."
./scripts/demo/run-live-ab-dashboard.sh

echo "✅ Demo complete!"