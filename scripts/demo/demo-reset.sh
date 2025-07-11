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

# Note: requirements-live-dashboards.txt is now committed in scripts/demo/ and not removed

# Remove generated helper scripts (recreated by setup script)
echo "🗑️  Removing generated helper scripts..."
# Note: run-live-ab-dashboard.sh is now permanent and not removed

# Note: run-live-business-dashboard.sh is now a committed file and not removed

# Note: interactive-live-dashboard.py is now a committed file and not removed

# Note: test-database-connection.py is now a committed file and not removed

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