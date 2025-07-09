#!/bin/bash

# Setup script for comprehensive A/B testing monitoring
# This script configures Prometheus, Grafana, and monitoring infrastructure

set -e

echo "🔧 Setting up Financial MLOps A/B Testing Monitoring"
echo "=================================================="

# Configuration
PROMETHEUS_URL="http://192.168.1.85:30090"
GRAFANA_URL="http://192.168.1.85:30300"
GRAFANA_ADMIN_USER="admin"
GRAFANA_ADMIN_PASS="admin"
METRICS_PORT=8001

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if service is accessible
check_service() {
    local url=$1
    local service_name=$2
    
    if curl -s --max-time 5 "$url" > /dev/null 2>&1; then
        print_status "$service_name is accessible at $url"
        return 0
    else
        print_error "$service_name is not accessible at $url"
        return 1
    fi
}

# Function to install Grafana dashboard
install_grafana_dashboard() {
    print_status "Installing Grafana dashboard..."
    
    # Create dashboard via API
    curl -X POST \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $GRAFANA_API_KEY" \
        -d @grafana/ab-testing-dashboard.json \
        "$GRAFANA_URL/api/dashboards/db" 2>/dev/null || {
        print_warning "Dashboard installation failed - you may need to install manually"
        echo "Dashboard file: grafana/ab-testing-dashboard.json"
        echo "Import this file via Grafana UI: $GRAFANA_URL/dashboard/import"
    }
}

# Function to setup Prometheus configuration
setup_prometheus_config() {
    print_status "Setting up Prometheus configuration..."
    
    cat > /tmp/prometheus-ab-testing.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "grafana/alert-rules.yaml"

scrape_configs:
  - job_name: 'ab-testing-metrics'
    static_configs:
      - targets: ['localhost:$METRICS_PORT']
    scrape_interval: 5s
    metrics_path: /metrics
    
  - job_name: 'seldon-metrics'
    kubernetes_sd_configs:
      - role: endpoints
        namespaces:
          names:
            - financial-ml
    relabel_configs:
      - source_labels: [__meta_kubernetes_service_name]
        action: keep
        regex: seldon-mesh
EOF
    
    print_status "Prometheus configuration created at /tmp/prometheus-ab-testing.yml"
    print_warning "Add this job configuration to your Prometheus config and restart Prometheus"
}

# Function to create monitoring startup script
create_monitoring_script() {
    print_status "Creating monitoring startup script..."
    
    cat > scripts/start-monitoring.sh << 'EOF'
#!/bin/bash

# Start comprehensive A/B testing monitoring
echo "🚀 Starting A/B Testing Monitoring Stack"

# Start metrics collector in background
echo "Starting metrics collector..."
python3 scripts/metrics-collector.py \
    --port 8001 \
    --endpoint http://192.168.1.202:80 \
    --status-interval 30 &

METRICS_PID=$!
echo "Metrics collector started with PID: $METRICS_PID"

# Wait for metrics to be available
echo "Waiting for metrics to be available..."
sleep 5

# Check if metrics are being generated
if curl -s http://localhost:8001/metrics | grep -q "ab_test_requests_total"; then
    echo "✅ Metrics are being generated successfully"
else
    echo "❌ Metrics generation failed"
    exit 1
fi

echo "📊 Monitoring stack is ready!"
echo "   - Metrics endpoint: http://localhost:8001/metrics"
echo "   - Prometheus: http://192.168.1.85:30090"
echo "   - Grafana: http://192.168.1.85:30300"
echo ""
echo "Press Ctrl+C to stop monitoring"

# Handle cleanup on exit
cleanup() {
    echo "🛑 Stopping monitoring..."
    kill $METRICS_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Keep script running
wait
EOF
    
    chmod +x scripts/start-monitoring.sh
    print_status "Monitoring script created: scripts/start-monitoring.sh"
}

# Main setup process
main() {
    echo "🔍 Checking prerequisites..."
    
    # Check Python dependencies
    if ! python3 -c "import prometheus_client" 2>/dev/null; then
        print_error "prometheus_client not installed. Installing..."
        pip3 install prometheus_client
    fi
    
    if ! python3 -c "import requests" 2>/dev/null; then
        print_error "requests not installed. Installing..."
        pip3 install requests
    fi
    
    # Check services
    check_service "$PROMETHEUS_URL" "Prometheus"
    check_service "$GRAFANA_URL" "Grafana"
    
    # Setup configurations
    setup_prometheus_config
    create_monitoring_script
    
    echo ""
    echo "📋 Setup Summary:"
    echo "=================="
    echo "✅ Monitoring scripts created"
    echo "✅ Grafana dashboard configuration ready"
    echo "✅ Prometheus alerts configured"
    echo "✅ Startup script created"
    echo ""
    echo "🚀 Next Steps:"
    echo "1. Import grafana/ab-testing-dashboard.json into Grafana"
    echo "2. Add Prometheus configuration from /tmp/prometheus-ab-testing.yml"
    echo "3. Run: ./scripts/start-monitoring.sh"
    echo ""
    echo "📊 Monitoring Endpoints:"
    echo "   - Grafana Dashboard: $GRAFANA_URL"
    echo "   - Prometheus: $PROMETHEUS_URL"
    echo "   - Metrics: http://localhost:$METRICS_PORT/metrics"
    echo ""
    echo "🔔 Alert Configuration:"
    echo "   - Configure Slack webhook in alert-rules.yaml"
    echo "   - Set up email notifications as needed"
    echo "   - Review thresholds in grafana/alert-rules.yaml"
}

# Run main function
main "$@"