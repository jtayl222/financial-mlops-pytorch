#!/bin/bash
#
# Demo Reset Script - Clean up temporary files between demo runs
#

set -e

echo "🧹 Resetting Demo Environment"
echo "=============================="

# Remove temporary environment file (recreated by setup script)
if [ -f ".env.live-dashboards" ]; then
    echo "🗑️  Removing temporary .env.live-dashboards"
    rm .env.live-dashboards
fi

# Remove requirements file (recreated by setup script)
if [ -f "requirements-live-dashboards.txt" ]; then
    echo "🗑️  Removing requirements-live-dashboards.txt"
    rm requirements-live-dashboards.txt
fi

# Remove generated helper scripts (recreated by setup script)
echo "🗑️  Removing generated helper scripts..."
# Note: run-live-ab-dashboard.sh is now permanent and not removed

if [ -f "scripts/run-live-business-dashboard.sh" ]; then
    rm scripts/run-live-business-dashboard.sh
fi

if [ -f "scripts/interactive-live-dashboard.py" ]; then
    rm scripts/interactive-live-dashboard.py
fi

if [ -f "scripts/test-database-connection.py" ]; then
    rm scripts/test-database-connection.py
fi

# Remove virtual environment (optional - takes time to recreate)
if [ "$1" = "--full-reset" ]; then
    if [ -d ".venv-live-dashboards" ]; then
        echo "🗑️  Removing virtual environment (--full-reset)"
        rm -rf .venv-live-dashboards
    fi
fi

# Remove generated dashboard images
echo "🗑️  Removing generated dashboard images..."
rm -f live_*_dashboard_*.png
rm -f business_impact_*.png
rm -f ab_testing_dashboard_*.png
rm -f monitoring_dashboard_*.png

# Remove any demo output files
rm -f demo_results_*.json
rm -f test_results_*.log

# Show what's left (should only be committed files)
echo ""
echo "✅ Demo Reset Complete!"
echo ""
echo "📋 Remaining untracked files:"
git ls-files --others --exclude-standard | grep -E '\.(py|sh|txt|env)$' || echo "   None (clean state)"

echo ""
echo "📝 Modified files (kept):"
git diff --name-only

echo ""
echo "🚀 Ready for fresh demo run:"
echo "   ./scripts/setup-live-dashboards.sh"