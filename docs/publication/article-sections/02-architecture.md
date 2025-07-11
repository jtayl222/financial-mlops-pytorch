# Technical Architecture: Production-Ready A/B Testing

## System Overview

Our A/B testing infrastructure follows GitOps principles with full observability:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Argo CD       │    │  Seldon Core    │    │   Prometheus    │
│   (GitOps)      │───▶│  (A/B Testing)  │───▶│   (Metrics)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     MLflow      │    │   Kubernetes    │    │    Grafana      │
│  (Registry)     │    │  (Platform)     │    │ (Visualization) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Seldon Core v2 Experiment Configuration

The heart of our A/B testing is a Seldon Experiment resource:

```yaml
apiVersion: mlops.seldon.io/v1alpha1
kind: Experiment
metadata:
  name: financial-ab-test-experiment
  namespace: financial-inference
spec:
  default: baseline-predictor
  candidates:
    - name: baseline-predictor
      weight: 70
    - name: enhanced-predictor
      weight: 30
  mirror:
    percent: 100
    name: traffic-mirror
```

**Key features:**
- **70/30 traffic split**: Conservative approach for financial models
- **Default fallback**: Automatic routing to baseline if enhanced fails
- **Traffic mirroring**: Copy requests for offline analysis

## Model Serving Infrastructure

Both models are deployed as Seldon `Server` resources:

```yaml
apiVersion: mlops.seldon.io/v1alpha1
kind: Server
metadata:
  name: baseline-predictor
spec:
  ingress:
    httpPort: 9000
    grpcPort: 9001
  protocol: v2
  modelUri: "gs://financial-inference-models/baseline/v1.2.0"
  requirements:
    - "torch==2.0.0"
    - "transformers==4.21.0"
  resources:
    requests:
      memory: "2Gi"
      cpu: "1000m"
    limits:
      memory: "4Gi"
      cpu: "2000m"
```

## Prometheus Metrics Collection

We collect comprehensive metrics for both models:

```python
# Request metrics
ab_test_requests_total{model_name="baseline-predictor",status="success"} 1851
ab_test_requests_total{model_name="enhanced-predictor",status="success"} 649

# Response time distribution
ab_test_response_time_seconds_bucket{model_name="baseline-predictor",le="0.05"} 1245
ab_test_response_time_seconds_bucket{model_name="enhanced-predictor",le="0.05"} 523

# Model accuracy
ab_test_model_accuracy{model_name="baseline-predictor"} 78.5
ab_test_model_accuracy{model_name="enhanced-predictor"} 82.1

# Business impact
ab_test_business_impact{model_name="enhanced-predictor",metric_type="net_business_value"} 3.3
```

## Traffic Distribution Strategy

Our traffic routing implements several safeguards:

### 1. Gradual Rollout
```
Phase 1: 95% baseline, 5% enhanced   (1 hour)
Phase 2: 85% baseline, 15% enhanced  (4 hours)
Phase 3: 70% baseline, 30% enhanced  (production)
```

### 2. Circuit Breaker
```python
if enhanced_model.error_rate > 0.05:
    route_all_traffic_to_baseline()
```

### 3. Performance Monitoring
```python
if enhanced_model.p95_latency > 200ms:
    alert_ops_team()
    consider_traffic_reduction()
```

## Request Flow

1. **Client Request** → Istio Gateway
2. **Istio Gateway** → Seldon Mesh
3. **Seldon Mesh** → Experiment Controller
4. **Experiment Controller** → Selected Model (weighted routing)
5. **Model Response** → Metrics Collection
6. **Metrics** → Prometheus → Grafana

## Infrastructure Requirements

### Kubernetes Cluster
- **5 nodes** (36 CPU cores, 260GB memory)
- **CNI**: Calico networking
- **LoadBalancer**: MetalLB
- **Storage**: 500GB persistent volumes

### Resource Allocation
```yaml
baseline-predictor:
  requests: { memory: "2Gi", cpu: "1000m" }
  limits: { memory: "4Gi", cpu: "2000m" }

enhanced-predictor:
  requests: { memory: "3Gi", cpu: "1500m" }
  limits: { memory: "6Gi", cpu: "3000m" }
```

---

*This architecture provides the foundation for safe, observable A/B testing at scale. Next, we'll show you the implementation details.*