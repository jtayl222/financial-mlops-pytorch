# Testing Guide

This guide provides comprehensive testing procedures for the Financial MLOps platform to ensure reliability before advancing to production scenarios.

## Testing Strategy

### 1. Foundation Testing (Pre-Phase 2)
Test the basic infrastructure and model deployment before implementing A/B testing scenarios.

### 2. Integration Testing  
Verify end-to-end workflows from training to serving.

### 3. Performance Testing
Validate latency, throughput, and resource utilization requirements.

### 4. A/B Testing Validation
Ensure traffic splitting and experiment functionality works correctly.

## Pre-requisites

### Infrastructure Readiness
```bash
# Verify all required namespaces exist
kubectl get namespaces financial-ml financial-mlops-pytorch seldon-system

# Check Seldon components are running
kubectl get pods -n seldon-system
kubectl get deployments -n seldon-system

# Verify secrets are properly configured
kubectl get secrets -n financial-ml ml-platform
kubectl get secrets -n financial-mlops-pytorch ml-platform
```

### Cluster Resources
```bash
# Check available resources
kubectl top nodes
kubectl describe quota -n financial-ml
kubectl describe quota -n financial-mlops-pytorch
```

## Test Suite 1: Foundation Testing

### 1.1 Secret and Configuration Testing

```bash
# Test secret access from a pod
kubectl run secret-test --image=busybox --rm -it -n financial-ml \
  --overrides='{"spec":{"containers":[{"name":"test","image":"busybox","env":[{"name":"AWS_ACCESS_KEY_ID","valueFrom":{"secretKeyRef":{"name":"ml-platform","key":"AWS_ACCESS_KEY_ID"}}}],"command":["sh","-c","echo $AWS_ACCESS_KEY_ID && sleep 30"]}]}}' \
  -- /bin/sh

# Expected: Should print the MinIO access key
```

### 1.2 MLflow Connectivity Testing

```bash
# Test MLflow tracking server connectivity
kubectl run mlflow-test --image=python:3.9 --rm -it -n financial-mlops-pytorch \
  --overrides='{"spec":{"containers":[{"name":"test","image":"python:3.9","envFrom":[{"secretRef":{"name":"ml-platform"}}],"command":["sh","-c","pip install mlflow requests && python -c \"import mlflow; mlflow.set_tracking_uri(\\\"$MLFLOW_TRACKING_URI\\\"); print(mlflow.list_experiments())\" && sleep 30"]}]}}' \
  -- /bin/sh

# Expected: Should list available MLflow experiments
```

### 1.3 MinIO S3 Storage Testing

```bash
# Test S3 connectivity and model artifact access
kubectl run s3-test --image=amazon/aws-cli --rm -it -n financial-ml \
  --overrides='{"spec":{"containers":[{"name":"test","image":"amazon/aws-cli","envFrom":[{"secretRef":{"name":"ml-platform"}}],"command":["sh","-c","aws s3 ls s3://mlflow-artifacts/28/ --endpoint-url=$AWS_ENDPOINT_URL && sleep 30"]}]}}' \
  -- /bin/sh

# Expected: Should list model artifacts in the bucket
```

## Test Suite 2: Model Deployment Testing

### 2.1 Deploy Test Models

```bash
# Apply the base configuration
kubectl apply -k k8s/base

# Verify models are created
kubectl get models -n financial-ml
kubectl get experiments -n financial-ml

# Check model status
kubectl describe model baseline-predictor -n financial-ml
kubectl describe model enhanced-predictor -n financial-ml
```

### 2.2 Model Loading Verification

```bash
# Check if models are being scheduled
kubectl logs -n seldon-system deployment/seldon-v2-controller-manager | grep -i "baseline-predictor\|enhanced-predictor"

# Check MLServer logs for model loading
kubectl logs -n seldon-system mlserver-0 -c mlserver | grep -i "baseline\|enhanced"

# Verify model endpoints are ready
kubectl get models -n financial-ml -o wide
```

### 2.3 Model Health Checks

```bash
# Check model server health
kubectl exec -n seldon-system mlserver-0 -- curl -s http://localhost:9000/v2/health/live

# Check model readiness
kubectl exec -n seldon-system mlserver-0 -- curl -s http://localhost:9000/v2/health/ready

# List loaded models
kubectl exec -n seldon-system mlserver-0 -- curl -s http://localhost:9000/v2/models
```

## Test Suite 3: Inference Testing

### 3.1 Direct Model Testing

```bash
# Test baseline model inference
kubectl run inference-test --image=curlimages/curl --rm -it -n financial-ml \
  -- curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "features", "shape": [1, 10], "datatype": "FP32", "data": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]}]}' \
  http://baseline-predictor:9000/v2/models/baseline-predictor/infer

# Expected: Should return prediction results in JSON format
```

### 3.2 A/B Testing Experiment

```bash
# Test experiment endpoint
kubectl run experiment-test --image=curlimages/curl --rm -it -n financial-ml \
  -- curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "features", "shape": [1, 10], "datatype": "FP32", "data": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]}]}' \
  http://financial-ab-test-experiment:9000/v2/models/financial-ab-test-experiment/infer

# Expected: Should route to either baseline or enhanced model based on weights
```

### 3.3 Traffic Distribution Testing

```bash
# Send multiple requests to verify traffic splitting
for i in {1..20}; do
  kubectl run test-request-$i --image=curlimages/curl --rm -n financial-ml \
    -- curl -s -X POST \
    -H "Content-Type: application/json" \
    -d '{"inputs": [{"name": "features", "shape": [1, 10], "datatype": "FP32", "data": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]}]}' \
    http://financial-ab-test-experiment:9000/v2/models/financial-ab-test-experiment/infer
done

# Monitor logs to verify traffic distribution matches experiment weights (70%/30%)
```

## Test Suite 4: Performance Testing

### 4.1 Latency Testing

```bash
# Create a load testing script
cat > load_test.sh << 'EOF'
#!/bin/bash
MODEL_ENDPOINT="http://financial-ab-test-experiment:9000/v2/models/financial-ab-test-experiment/infer"
REQUESTS=100
CONCURRENCY=10

kubectl run load-test --image=curlimages/curl --rm -it -n financial-ml \
  -- sh -c "
    for i in \$(seq 1 $REQUESTS); do
      time curl -s -X POST \
        -H 'Content-Type: application/json' \
        -d '{\"inputs\": [{\"name\": \"features\", \"shape\": [1, 10], \"datatype\": \"FP32\", \"data\": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]}]}' \
        $MODEL_ENDPOINT > /dev/null 2>&1
    done
  "
EOF

chmod +x load_test.sh
./load_test.sh

# Target: <100ms p95 latency
```

### 4.2 Resource Utilization

```bash
# Monitor resource usage during load test
kubectl top pods -n seldon-system
kubectl top pods -n financial-ml

# Check if resource quotas are being respected
kubectl describe quota -n financial-ml
```

## Test Suite 5: Workflow Integration Testing

### 5.1 Training Pipeline Testing

```bash
# Submit a training workflow
argo submit --from workflowtemplate/financial-training-pipeline-template \
  -p model-variant=baseline \
  -n financial-mlops-pytorch \
  --watch

# Verify successful completion
argo get <workflow-name> -n financial-mlops-pytorch

# Check if new model artifacts are created in MLflow
mlflow experiments search
```

### 5.2 End-to-End Pipeline

```bash
# Test complete pipeline: data -> training -> deployment -> inference
# 1. Submit data pipeline
argo submit --from workflowtemplate/financial-data-pipeline-template -n financial-mlops-pytorch

# 2. Submit training pipeline
argo submit --from workflowtemplate/financial-training-pipeline-template \
  -p model-variant=test \
  -n financial-mlops-pytorch

# 3. Update model deployment with new artifacts
# 4. Test inference on new model
```

## Test Suite 6: Monitoring and Observability

### 6.1 Metrics Collection

```bash
# Check Seldon metrics
kubectl port-forward -n seldon-system mlserver-0 8082:8082 &
curl http://localhost:8082/metrics | grep seldon

# Check custom metrics
curl http://localhost:8082/metrics | grep financial
```

### 6.2 Logging Verification

```bash
# Check model prediction logs
kubectl logs -n seldon-system mlserver-0 -c mlserver | grep prediction

# Check experiment traffic logs
kubectl logs -n seldon-system deployment/seldon-v2-controller-manager | grep experiment
```

## Test Automation Script

Create a comprehensive test runner:

```bash
cat > run_tests.sh << 'EOF'
#!/bin/bash
set -e

echo "🧪 Starting Financial MLOps Platform Tests"

# Test Suite 1: Foundation
echo "📋 Test Suite 1: Foundation Testing"
./tests/foundation_tests.sh

# Test Suite 2: Model Deployment
echo "📋 Test Suite 2: Model Deployment Testing"  
./tests/model_deployment_tests.sh

# Test Suite 3: Inference
echo "📋 Test Suite 3: Inference Testing"
./tests/inference_tests.sh

# Test Suite 4: Performance
echo "📋 Test Suite 4: Performance Testing"
./tests/performance_tests.sh

# Test Suite 5: Integration
echo "📋 Test Suite 5: Workflow Integration Testing"
./tests/integration_tests.sh

# Test Suite 6: Monitoring
echo "📋 Test Suite 6: Monitoring Testing"
./tests/monitoring_tests.sh

echo "✅ All tests completed successfully!"
EOF

chmod +x run_tests.sh
```

## Success Criteria

### Foundation Tests (Must Pass)
- ✅ All secrets accessible in target namespaces
- ✅ MLflow connectivity established
- ✅ MinIO S3 storage accessible
- ✅ Seldon components healthy

### Model Deployment Tests (Must Pass)
- ✅ Models deploy successfully to financial-ml namespace
- ✅ Models show as "Ready" in status
- ✅ Model endpoints respond to health checks

### Inference Tests (Must Pass)
- ✅ Direct model inference returns valid predictions
- ✅ A/B experiment routes traffic correctly
- ✅ Traffic distribution matches configured weights (70%/30%)

### Performance Tests (Target)
- 🎯 <100ms p95 latency for inference requests
- 🎯 >99% uptime during load testing
- 🎯 Resource usage within defined quotas

### Integration Tests (Must Pass)
- ✅ Training workflows complete successfully
- ✅ Model artifacts stored in MLflow
- ✅ End-to-end pipeline functionality

## Troubleshooting Test Failures

### Common Issues and Solutions

**Models not loading:**
```bash
# Check secret configuration
kubectl describe secret ml-platform -n financial-ml

# Check model storage URI accessibility
kubectl run debug --image=amazon/aws-cli --rm -it -n financial-ml --envFrom secretRef:ml-platform -- aws s3 ls s3://mlflow-artifacts/28/ --endpoint-url=$AWS_ENDPOINT_URL
```

**Inference requests failing:**
```bash
# Check service endpoints
kubectl get svc -n financial-ml

# Check model server logs
kubectl logs -n seldon-system mlserver-0 -c mlserver --tail=50
```

**Performance issues:**
```bash
# Check resource constraints
kubectl describe pod mlserver-0 -n seldon-system

# Check node resources
kubectl describe node
```

## Next Steps After Testing

Once all tests pass:
1. **Update docs/plan.md** with current status
2. **Proceed to Phase 2**: Train multiple model variants
3. **Implement advanced A/B testing** scenarios
4. **Add comprehensive monitoring** and alerting

This testing framework ensures a solid foundation before advancing to more complex MLOps scenarios.