# MLOps Monitoring Stack

Monitoring components for the Financial MLOps Platform, extracted from exploratory work and integrated with existing Prometheus stack.

## Components

### Prometheus Alerts
- **File**: `prometheus/mlops-alert-rules.yaml`
- **Purpose**: PrometheusRule for MLOps-specific alerts
- **Namespace**: `monitoring`
- **Integration**: Works with your existing prometheus-pushgateway

### Grafana Dashboards
- **dashboard-configmap.yaml**: ConfigMap for dashboard auto-discovery
- **complete-ab-dashboard.json**: Full A/B testing dashboard
- **simple-ab-dashboard.json**: Simplified A/B testing dashboard

### Demo Scripts
- **../scripts/demo/advanced-ab-demo.py**: Advanced A/B testing demonstration

## Installation

### 1. Deploy Prometheus Alerts
```bash
kubectl apply -f monitoring/prometheus/mlops-alert-rules.yaml
```

### 2. Deploy Grafana Dashboard
```bash
kubectl apply -f monitoring/grafana/dashboard-configmap.yaml
```

### 3. Import Dashboards Manually
Import the JSON files through Grafana UI or:
```bash
# Copy to Grafana instance
kubectl cp monitoring/grafana/complete-ab-dashboard.json <grafana-pod>:/var/lib/grafana/dashboards/
```

## Alert Rules

### Model Performance Alerts
- **ModelAccuracyDegraded**: Fires when model accuracy < 55%
- **HighResponseTime**: Fires when 95th percentile response time > 200ms
- **ABTestTrafficImbalance**: Fires when traffic distribution is skewed

### Infrastructure Alerts
- **HighModelServerCPU**: Fires when CPU usage > 80%
- **HighModelServerMemory**: Fires when memory usage > 85%
- **ModelServerDown**: Fires when model server is unavailable

## Dashboard Features

### A/B Testing Metrics
- Traffic distribution between baseline and enhanced models
- Model accuracy comparison
- Response time analysis
- Error rate monitoring

### Infrastructure Metrics
- Resource utilization (CPU, memory)
- Pod status and health
- Network connectivity

## Integration with Existing Stack

Your existing monitoring stack includes:
- `prometheus-pushgateway-9b9bcc448-q5wn9` in monitoring namespace
- Standard Prometheus/Grafana setup

These components integrate seamlessly with your existing infrastructure.

## Usage

1. **Run A/B test demonstration**:
   ```bash
   python scripts/demo/advanced-ab-demo.py
   ```

2. **Monitor in Grafana**: Access the "Financial MLOps A/B Testing Dashboard"

3. **Check alerts**: View alerts in Prometheus AlertManager

## Extracted From

These components were extracted from exploratory work in the archived branch `maze-of-twisty-little-passages-all-alike` and represent the valuable monitoring infrastructure without the documentation overhead.