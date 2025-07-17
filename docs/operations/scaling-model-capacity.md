# Scaling Model Serving Capacity

## Problem

Low success rates (26.8%) during high-volume A/B testing due to:
- Connection timeouts under load
- Limited model server replicas (only 1)
- Resource constraints (100m CPU, 1Gi memory)
- No autoscaling configured

## Solution Options

### Option 1: Immediate Manual Scaling (Quick Fix)

```bash
# Scale MLServer replicas immediately
kubectl scale statefulset mlserver -n financial-inference --replicas=3

# Scale Envoy proxy for better load distribution
kubectl scale deployment seldon-envoy -n financial-inference --replicas=2
```

**Impact**: 3x capacity increase within ~30 seconds

### Option 2: Update Runtime Configuration (Recommended)

Apply the production capacity patch:

```bash
kubectl apply -f k8s/overlays/production/increase-capacity-patch.yaml
```

This patch provides:
- **MLServer**: 3 replicas with 500m CPU, 2Gi memory each
- **Concurrent handling**: 4 parallel workers per replica
- **Envoy proxy**: 2 replicas for redundancy
- **HPA**: Auto-scaling from 3-10 replicas based on load

### Option 3: GitOps Update (Best Practice)

1. Update `k8s/base/seldon-runtime.yaml` with new capacity settings
2. Commit and push changes
3. Let ArgoCD apply the updates

```yaml
# Example update to seldon-runtime.yaml
spec:
  overrides:
  - name: mlserver
    replicas: 3  # Increased from 1
    resources:
      requests:
        cpu: 500m     # Increased from 100m
        memory: 2Gi   # Increased from 1Gi
```

## Capacity Calculations

**Current Setup**:
- 1 MLServer replica × 100m CPU = 0.1 CPU cores total
- 1 replica × ~10 concurrent requests = ~10 RPS max

**After Scaling**:
- 3 MLServer replicas × 500m CPU = 1.5 CPU cores total
- 3 replicas × 4 workers × ~10 requests = ~120 RPS capacity
- **12x increase in throughput**

## Monitoring After Scaling

```bash
# Watch pod scaling
kubectl get pods -n financial-inference -w

# Check resource usage
kubectl top pods -n financial-inference

# Monitor HPA status
kubectl get hpa mlserver-hpa -n financial-inference -w

# Test new capacity
python3 scripts/demo/advanced-ab-demo.py \
  --scenarios 500 \
  --workers 3 \
  --endpoint http://localhost:8082
```

## Expected Results

After applying capacity increases:
- **Success rate**: Should improve from 26.8% to >90%
- **Response time**: Maintain <20ms average
- **Throughput**: Handle 100+ RPS sustained
- **Auto-scaling**: Dynamically adjust 3-10 replicas

## Cost Considerations

**Resource Increase**:
- CPU: 0.1 → 1.5 cores (minimum)
- Memory: 1Gi → 6Gi (minimum)
- Potential: Up to 5 CPU cores, 20Gi memory at max scale

**AWS EKS Estimated Cost**:
- ~$0.10/hour for the additional capacity
- ~$72/month if running continuously

## Rollback Plan

If issues occur after scaling:

```bash
# Revert to single replica
kubectl scale statefulset mlserver -n financial-inference --replicas=1

# Or revert the entire configuration
kubectl delete -f k8s/overlays/production/increase-capacity-patch.yaml
```

## Best Practices

1. **Start small**: Scale to 3 replicas first, monitor, then increase
2. **Monitor metrics**: Watch CPU, memory, and success rates
3. **Set resource limits**: Prevent runaway resource consumption
4. **Use HPA**: Let Kubernetes handle dynamic scaling
5. **Load test**: Verify capacity improvements with realistic traffic

## Performance Tuning Parameters

Additional MLServer environment variables for fine-tuning:

```yaml
env:
- name: MLSERVER_PARALLEL_WORKERS
  value: "4"  # Concurrent request handlers
- name: MLSERVER_MAX_GRPC_WORKERS  
  value: "10" # gRPC thread pool size
- name: GUNICORN_WORKERS
  value: "2"  # HTTP server workers
- name: MLSERVER_ENABLE_DOCS
  value: "false"  # Disable Swagger UI in production
```

## Success Metrics

After scaling, you should see:
- ✅ A/B test success rate >90%
- ✅ P95 latency <50ms
- ✅ Zero timeout errors
- ✅ Smooth traffic distribution across replicas
- ✅ Grafana dashboards showing improved throughput