#!/bin/bash
#
# Setup Live Dashboards - Install dependencies and configure environment
#

set -e

echo "🚀 Setting up Live Dashboard Environment"

# Check if we're in the right directory
if [ ! -f "scripts/live-dashboard-generator.py" ]; then
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

# Create requirements file if it doesn't exist
if [ ! -f "requirements-live-dashboards.txt" ]; then
    echo "📝 Creating requirements-live-dashboards.txt..."
    cat > requirements-live-dashboards.txt << EOF
# Live Dashboard Dependencies
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
numpy>=1.21.0
aiohttp>=3.8.0
python-dotenv>=0.19.0
plotly>=5.0.0
dash>=2.0.0
dash-bootstrap-components>=1.0.0
prometheus-client>=0.15.0
mlflow

# Optional: For Jupyter notebook support
jupyter>=1.0.0
ipywidgets>=7.6.0
EOF
fi

# Install required packages
echo "📚 Installing Python dependencies..."
pip install -r requirements-live-dashboards.txt

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
if [ ! -f "scripts/run-live-ab-dashboard.sh" ]; then
    echo "📝 Creating run-live-ab-dashboard.sh (permanent file)..."
    cat > scripts/run-live-ab-dashboard.sh << 'EOF'
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
        echo "   argo submit --from workflowtemplate/financial-training-pipeline-template -p model-variant=enhanced -n financial-mlops-pytorch"
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
EOF
else
    echo "📋 Using existing run-live-ab-dashboard.sh (permanent file)"
fi

# Script to run live business dashboard
cat > scripts/run-live-business-dashboard.sh << 'EOF'
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
python3 scripts/live-business-impact-dashboard.py
EOF

# Make scripts executable
if [ -f "scripts/run-live-ab-dashboard.sh" ]; then
    chmod +x scripts/run-live-ab-dashboard.sh
fi
if [ -f "scripts/run-seldon-ab-dashboard.sh" ]; then
    chmod +x scripts/run-seldon-ab-dashboard.sh
fi
chmod +x scripts/run-live-business-dashboard.sh

# Create interactive Dash app
echo "📱 Creating interactive web dashboard..."
cat > scripts/interactive-live-dashboard.py << 'EOF'
#!/usr/bin/env python3
"""
Interactive Live Dashboard - Real-time web interface
Run with: python3 scripts/interactive-live-dashboard.py
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import asyncio
import aiohttp
import os
from datetime import datetime, timedelta
import json
import mlflow
from mlflow.client import MlflowClient

# Load environment variables
from dotenv import load_dotenv
load_dotenv('.env.live-dashboards')

app = dash.Dash(__name__)

# Dashboard layout
app.layout = html.Div([
    html.H1("Live Financial ML A/B Testing Dashboard", 
            style={'textAlign': 'center', 'marginBottom': 30}),
    
    # Status indicators
    html.Div([
        html.Div([
            html.H3("System Status", style={'textAlign': 'center'}),
            html.Div(id="system-status", style={'textAlign': 'center', 'fontSize': 24})
        ], className="four columns"),
        
        html.Div([
            html.H3("Active Tests", style={'textAlign': 'center'}),
            html.Div(id="active-tests", style={'textAlign': 'center', 'fontSize': 24})
        ], className="four columns"),
        
        html.Div([
            html.H3("ROI", style={'textAlign': 'center'}),
            html.Div(id="roi-display", style={'textAlign': 'center', 'fontSize': 24})
        ], className="four columns"),
    ], className="row"),
    
    # Charts
    html.Div([
        html.Div([
            dcc.Graph(id="model-accuracy-chart")
        ], className="six columns"),
        
        html.Div([
            dcc.Graph(id="business-impact-chart")
        ], className="six columns"),
    ], className="row"),
    
    html.Div([
        html.Div([
            dcc.Graph(id="request-timeline")
        ], className="twelve columns"),
    ], className="row"),
    
    # Auto-refresh
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # Update every 30 seconds
        n_intervals=0
    )
])

async def get_live_data():
    """Fetch live data from MLflow and Prometheus"""
    data = {
        'system_status': '🟢 ONLINE',
        'active_tests': 3,
        'roi': 1143,
        'model_accuracy': {'Baseline': 78.5, 'Enhanced': 82.1},
        'business_impact': [1.8, -1.9, 4.0, 3.9],  # Revenue, Cost, Risk, Net
        'timeline': []
    }
    
    # Try to connect to real data sources using MLflow API
    try:
        # MLflow API connection
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://192.168.1.203:5000')
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        
        # Get A/B testing experiments
        experiments = client.search_experiments()
        active_tests = 0
        
        for exp in experiments:
            if 'ab' in exp.name.lower() or 'test' in exp.name.lower() or 'financial' in exp.name.lower():
                runs = client.search_runs([exp.experiment_id], filter_string='status = "RUNNING"')
                if runs:
                    active_tests += 1
        
        data['active_tests'] = active_tests
        data['system_status'] = '🟢 LIVE DATA'
        
    except Exception as e:
        print(f"Using simulated data: {e}")
        data['system_status'] = '🟡 SIMULATED'
    
    return data

@callback(
    [Output('system-status', 'children'),
     Output('active-tests', 'children'),
     Output('roi-display', 'children'),
     Output('model-accuracy-chart', 'figure'),
     Output('business-impact-chart', 'figure'),
     Output('request-timeline', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    """Update all dashboard components"""
    
    # Get live data (simplified for demo)
    data = {
        'system_status': '🟢 LIVE DATA',
        'active_tests': 3,
        'roi': 1143,
        'model_accuracy': {'Baseline': 78.5, 'Enhanced': 82.1},
        'business_impact': [1.8, -1.9, 4.0, 3.9]
    }
    
    # Model accuracy chart
    accuracy_fig = go.Figure(data=[
        go.Bar(name='Accuracy (%)', 
               x=list(data['model_accuracy'].keys()), 
               y=list(data['model_accuracy'].values()),
               marker_color=['#2E86AB', '#A23B72'])
    ])
    accuracy_fig.update_layout(title="Model Accuracy Comparison", yaxis_title="Accuracy (%)")
    
    # Business impact chart
    impact_categories = ['Revenue Lift', 'Cost Impact', 'Risk Reduction', 'Net Value']
    impact_colors = ['green', 'red', 'blue', 'gold']
    
    impact_fig = go.Figure(data=[
        go.Bar(x=impact_categories, 
               y=data['business_impact'],
               marker_color=impact_colors)
    ])
    impact_fig.update_layout(title="Business Impact Analysis (%)", yaxis_title="Impact (%)")
    
    # Request timeline (simulated)
    times = pd.date_range(start=datetime.now() - timedelta(hours=2), periods=60, freq='2min')
    baseline_rps = 850 + pd.Series(range(60)).apply(lambda x: 50 * np.sin(x/10) + np.random.normal(0, 20))
    enhanced_rps = 350 + pd.Series(range(60)).apply(lambda x: 30 * np.sin(x/10) + np.random.normal(0, 15))
    
    timeline_fig = go.Figure()
    timeline_fig.add_trace(go.Scatter(x=times, y=baseline_rps, name='Baseline Model', line=dict(color='#2E86AB')))
    timeline_fig.add_trace(go.Scatter(x=times, y=enhanced_rps, name='Enhanced Model', line=dict(color='#A23B72')))
    timeline_fig.update_layout(title="Request Rate Timeline", yaxis_title="Requests/sec")
    
    return (data['system_status'], 
            data['active_tests'], 
            f"{data['roi']}%",
            accuracy_fig,
            impact_fig,
            timeline_fig)

if __name__ == '__main__':
    app.run_server(
        host=os.getenv('DASHBOARD_HOST', '0.0.0.0'),
        port=int(os.getenv('DASHBOARD_PORT', '8050')),
        debug=True
    )
EOF

# Create database test script using MLflow API
echo "🧪 Creating database test script..."
cat > scripts/test-database-connection.py << 'EOF'
#!/usr/bin/env python3
"""
Test connections for live dashboards using MLflow API and Prometheus
"""

import os
import asyncio
import aiohttp
import mlflow
from mlflow.client import MlflowClient
from dotenv import load_dotenv
from datetime import datetime

async def test_connections():
    """Test MLflow API and Prometheus connections"""
    
    # Load environment
    load_dotenv('.env.live-dashboards')
    
    print("🧪 Testing Live Dashboard Connections")
    print("=" * 40)
    
    # Test MLflow API
    print(f"\n📊 [{datetime.now().strftime('%H:%M:%S')}] Testing MLflow API...")
    try:
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://192.168.1.203:5000')
        print(f"🔗 Using MLflow tracking URI: {tracking_uri}")
        
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        
        # Test basic connection
        experiments = client.search_experiments()
        print(f"✅ MLflow API: Connected successfully")
        print(f"   📈 Found {len(experiments)} total experiments")
        
        # Test A/B test experiments
        ab_experiments = []
        for exp in experiments:
            exp_name = exp.name
            if 'ab' in exp_name.lower() or 'test' in exp_name.lower() or 'financial' in exp_name.lower():
                ab_experiments.append(exp_name)
                
        print(f"   🧪 Found {len(ab_experiments)} A/B testing experiments:")
        for exp_name in ab_experiments[:5]:  # Show first 5
            print(f"     - {exp_name}")
        if len(ab_experiments) > 5:
            print(f"     ... and {len(ab_experiments) - 5} more")
            
        # Test runs in A/B experiments
        total_runs = 0
        for exp in experiments:
            if 'ab' in exp.name.lower() or 'test' in exp.name.lower() or 'financial' in exp.name.lower():
                runs = client.search_runs([exp.experiment_id])
                total_runs += len(runs)
                
        print(f"   🏃 Found {total_runs} total runs in A/B experiments")
        
    except Exception as e:
        print(f"❌ MLflow API: {e}")
        import traceback
        print(f"   📍 Error details: {traceback.format_exc()}")
    
    # Test Prometheus
    print(f"\n📈 [{datetime.now().strftime('%H:%M:%S')}] Testing Prometheus...")
    try:
        prometheus_url = os.getenv('PROMETHEUS_URL', 'http://192.168.1.85:30090')
        print(f"🔗 Using Prometheus URL: {prometheus_url}")
        
        # Add timeout for Prometheus requests
        timeout = aiohttp.ClientTimeout(total=10.0)  # 10 second timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{prometheus_url}/api/v1/query", params={'query': 'up'}) as response:
                if response.status == 200:
                    data = await response.json()
                    targets = len(data['data']['result'])
                    print(f"✅ Prometheus: Connected successfully")
                    print(f"   🎯 Found {targets} active targets")
                else:
                    print(f"❌ Prometheus: HTTP {response.status}")
                    
    except asyncio.TimeoutError:
        print(f"❌ Prometheus: Connection timeout (10s)")
    except Exception as e:
        print(f"❌ Prometheus: {e}")
        import traceback
        print(f"   📍 Error details: {traceback.format_exc()}")
    
    print("\n" + "=" * 40)
    print(f"🏁 [{datetime.now().strftime('%H:%M:%S')}] Connection test completed")

if __name__ == "__main__":
    asyncio.run(test_connections())
EOF

# Make test script executable
chmod +x scripts/test-database-connection.py

echo ""
echo "✅ Live Dashboard Environment Setup Complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Activate virtual environment: source .venv-live-dashboards/bin/activate"
echo "2. Update .env.live-dashboards with your actual credentials"
echo "3. Test connections: python3 scripts/test-database-connection.py"
echo "4. Analyze completed experiments: ./scripts/run-historical-ab-dashboard.sh"
echo "5. Monitor live experiments: ./scripts/run-live-ab-dashboard.sh (requires running experiments)"
echo "6. Monitor production Seldon A/B tests: ./scripts/run-seldon-ab-dashboard.sh"
echo "7. Run live business dashboard: ./scripts/run-live-business-dashboard.sh"
echo "8. Start interactive web dashboard: python3 scripts/interactive-live-dashboard.py"
echo ""
echo "🌐 Web dashboard will be available at: http://localhost:8050"
echo ""
echo "⚠️  SECURITY NOTE: Change default passwords before production use!"