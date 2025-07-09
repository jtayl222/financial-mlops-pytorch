# A/B Testing Monitoring Setup Guide

This guide explains how to set up comprehensive monitoring for A/B testing experiments in the Financial MLOps platform.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   A/B Testing   │    │   Prometheus    │    │     Grafana     │
│   Metrics       │───▶│   Collection    │───▶│   Dashboard     │
│   Collection    │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Seldon Core   │    │   Alert Rules   │    │   Slack/Email   │
│   Experiments   │    │   & Policies    │    │   Notifications │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Setup

### 1. Install Dependencies

```bash
pip3 install prometheus_client requests numpy pandas matplotlib seaborn
```

### 2. Run Setup Script

```bash
./scripts/setup-monitoring.sh
```

### 3. Start Monitoring

```bash
./scripts/start-monitoring.sh
```

## Components

### 1. Metrics Collection (`scripts/metrics-collector.py`)

**Purpose**: Collects and exposes Prometheus metrics for A/B testing experiments

**Key Metrics**:
- `ab_test_requests_total`: Total requests processed by model
- `ab_test_response_time_seconds`: Response time distribution
- `ab_test_model_accuracy`: Model accuracy percentage
- `ab_test_traffic_percentage`: Traffic distribution
- `ab_test_business_impact`: Business impact metrics

**Usage**:
```bash
python3 scripts/metrics-collector.py --port 8001 --status-interval 30
```

### 2. Enhanced A/B Demo (`scripts/advanced-ab-demo.py`)

**Purpose**: Runs comprehensive A/B testing with integrated metrics collection

**New Features**:
- Real-time Prometheus metrics
- Live business impact calculation
- Concurrent request testing
- Comprehensive visualizations

**Usage**:
```bash
# Run with metrics enabled
python3 scripts/advanced-ab-demo.py --scenarios 200 --workers 5 --metrics-port 8002

# Run without metrics
python3 scripts/advanced-ab-demo.py --scenarios 200 --no-metrics
```

### 3. Grafana Dashboard (`grafana/ab-testing-dashboard.json`)

**Purpose**: Comprehensive visualization of A/B testing metrics

**Panels**:
- Traffic Distribution (pie chart)
- Model Accuracy Comparison (stat)
- Business Impact Summary (stat)
- Request Rate Over Time (time series)
- Response Time Distribution (time series)
- Model Performance Heatmap
- Business Impact Metrics (time series)
- Summary Table

**Import Instructions**:
1. Open Grafana: http://192.168.1.85:30300
2. Go to Dashboards → Import
3. Upload `grafana/ab-testing-dashboard.json`
4. Configure data source: Prometheus (http://192.168.1.85:30090)

### 4. Alert Rules (`grafana/alert-rules.yaml`)

**Purpose**: Automated monitoring and alerting for A/B testing experiments

**Alert Categories**:
- **Performance**: High response time, low accuracy
- **Business Impact**: Negative ROI, significant performance differences
- **Reliability**: High error rates, traffic imbalances
- **Operational**: Prediction drift, request rate anomalies

**Configuration**:
```yaml
# Example alert
- alert: ModelAccuracyDegraded
  expr: ab_test_model_accuracy < 75
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Model accuracy degraded"
```

## Monitoring Endpoints

### Local Services
- **Metrics Collector**: http://localhost:8001/metrics
- **A/B Demo Metrics**: http://localhost:8002/metrics

### Infrastructure Services
- **Prometheus**: http://192.168.1.85:30090
- **Grafana**: http://192.168.1.85:30300
- **Seldon Mesh**: http://192.168.1.202:80

## Key Metrics Explained

### Request Metrics
```prometheus
# Total requests by model and status
ab_test_requests_total{model_name="baseline-predictor",status="success"} 1500

# Response time distribution
ab_test_response_time_seconds_bucket{model_name="enhanced-predictor",le="0.1"} 850
```

### Business Metrics
```prometheus
# Model accuracy (percentage)
ab_test_model_accuracy{model_name="baseline-predictor"} 78.5

# Traffic distribution (percentage)
ab_test_traffic_percentage{model_name="enhanced-predictor"} 30.0

# Business impact metrics
ab_test_business_impact{model_name="enhanced-predictor",metric_type="revenue_impact"} 1.8
ab_test_business_impact{model_name="enhanced-predictor",metric_type="latency_cost"} 1.9
ab_test_business_impact{model_name="enhanced-predictor",metric_type="net_business_value"} 1.9
```

## Alert Configuration

### Slack Integration
1. Create Slack webhook URL
2. Update `grafana/alert-rules.yaml`:
   ```yaml
   notification_policies:
     - name: "critical-alerts"
       receivers:
         - slack-critical
   
   receivers:
     - name: slack-critical
       slack_configs:
         - api_url: "${SLACK_WEBHOOK_URL}"
           channel: "#mlops-alerts"
   ```

### Email Integration
1. Configure SMTP settings in Grafana
2. Update alert rules with email receivers
3. Set up distribution lists for different alert severities

## Troubleshooting

### Common Issues

**1. Metrics Not Appearing**
```bash
# Check if metrics endpoint is accessible
curl http://localhost:8001/metrics

# Verify Prometheus is scraping
curl http://192.168.1.85:30090/api/v1/targets
```

**2. Grafana Dashboard Not Loading**
- Verify data source configuration
- Check Prometheus connectivity
- Ensure time range is appropriate

**3. Alerts Not Firing**
- Check alert rule syntax
- Verify metric names and labels
- Test alert conditions manually

### Debugging Commands

```bash
# Check metrics generation
python3 scripts/test-metrics.py

# Verify Prometheus scraping
curl http://192.168.1.85:30090/api/v1/query?query=ab_test_requests_total

# Test alert rules
curl -X POST http://192.168.1.85:30090/api/v1/admin/tsdb/delete_series?match[]=ab_test_requests_total
```

## Production Deployment

### Security Considerations
1. Use authentication for Grafana
2. Secure Prometheus endpoints
3. Encrypt metrics in transit
4. Implement proper access controls

### Scaling
1. Configure Prometheus federation
2. Use remote storage for long-term retention
3. Implement metrics aggregation
4. Set up high-availability Grafana

### Maintenance
1. Regular backup of dashboards
2. Monitor metrics ingestion rate
3. Clean up old metrics data
4. Update alert thresholds based on experience

## Integration with CI/CD

### Automated Testing
```yaml
# Example GitHub Actions workflow
- name: Run A/B Test with Metrics
  run: |
    python3 scripts/advanced-ab-demo.py --scenarios 50 --no-viz
    
- name: Check Metrics Collection
  run: |
    curl http://localhost:8002/metrics | grep ab_test_requests_total
```

### Deployment Gates
- Use metrics to validate deployment success
- Implement automatic rollback based on alerts
- Generate reports for stakeholders

## Best Practices

1. **Metric Naming**: Use consistent naming conventions
2. **Label Design**: Keep labels low-cardinality
3. **Alert Fatigue**: Set appropriate thresholds
4. **Dashboard Organization**: Group related metrics
5. **Documentation**: Keep runbooks updated

## Additional Resources

- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Seldon Core Monitoring](https://docs.seldon.io/projects/seldon-core/en/latest/analytics/analytics.html)
- [A/B Testing Metrics](https://docs.company.com/ab-testing-metrics)

## Support

For issues with monitoring setup:
1. Check troubleshooting section above
2. Review logs in `/var/log/`
3. Contact MLOps team via Slack #mlops-support
4. Create issue in internal documentation repo