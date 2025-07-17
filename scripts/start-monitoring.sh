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
