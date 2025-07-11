# Infrastructure Reproduction Testing

## Overview

This document captures the process and lessons learned from full infrastructure reproduction testing - completely deleting and rebuilding the entire MLOps platform from documentation.

## Test Procedure

### 1. Complete Infrastructure Deletion
```bash
# Delete all application resources
kubectl delete -k k8s/base

# Handle stuck resources with finalizers (see troubleshooting docs)
kubectl patch model baseline-predictor -n financial-inference --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true
kubectl patch model enhanced-predictor -n financial-inference --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true
kubectl patch experiment financial-ab-test-experiment -n financial-inference --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true
kubectl patch server mlserver -n financial-inference --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true
```

### 2. Infrastructure Rebuild
```bash
# Rebuild from documentation
kubectl apply -k k8s/base

# Apply platform-managed secrets
kubectl apply -f infrastructure/manifests/sealed-secrets/financial-inference/seldon-rclone-sealed-secret.yaml
```

### 3. Pipeline Validation
```bash
# Test data pipeline
argo submit --from workflowtemplate/financial-data-pipeline-template -n financial-mlops-pytorch

# Test training pipeline
argo submit --from workflowtemplate/financial-training-pipeline-template -p model-variant=baseline -n financial-mlops-pytorch
```

## Key Discoveries

### MLflow Model URI Structure
**Critical Finding**: MLflow stores models with this exact pattern:
```
s3://mlflow-artifacts/{experiment_id}/models/m-{run_id}/artifacts/
```

**Not** directly under run IDs as initially assumed. This pattern is consistent across all MLflow deployments.

### Secret Management
- **RClone Secret**: Platform team manages `seldon-rclone-gs-public` via sealed secrets
- **Model Secrets**: Remove `secretName` references from Model specs to avoid "Secret does not have 1 key" errors
- **Authentication**: MLflow requires basic auth via environment variables

### Model Deployment Dependencies
1. MLServer must be running (3/3 containers ready)
2. RClone secret must exist in correct JSON format
3. Model URIs must point to directories containing `MLmodel` files
4. No `secretName` references in Model specs when using RClone config

## Automation Script

Created `scripts/update_model_uris.py` that:
- Uses MLflow `/ajax-api/` endpoints (not `/api/`)
- Authenticates with `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`
- Searches by `params.model_variant` for specific model variants
- Updates Kubernetes YAML files with correct URI format

### Usage
```bash
# Update specific variant
python3 scripts/update_model_uris.py --model-variant enhanced

# Update all variants
python3 scripts/update_model_uris.py

# Use different endpoint
python3 scripts/update_model_uris.py --mlflow-endpoint "http://192.168.1.203:5000"
```

## Validation Checklist

### ✅ Infrastructure Health
- [ ] MLServer pod running (3/3 containers)
- [ ] Seldon scheduler, envoy, modelgateway running
- [ ] MLflow tracking server accessible
- [ ] MinIO storage accessible
- [ ] Argo workflows controller running

### ✅ Pipeline Functionality
- [ ] Data ingestion pipeline completes successfully
- [ ] Feature engineering pipeline completes successfully  
- [ ] Model training pipeline completes successfully
- [ ] Models logged to MLflow with correct experiment structure

### ✅ Model Deployment
- [ ] Models show `READY=True` status
- [ ] Experiments show `EXPERIMENT READY=True` status
- [ ] MLServer agent logs show successful model loading
- [ ] Model endpoints are accessible for inference

## Time to Recovery

**Total Time**: ~45 minutes from deletion to full operational state
- Infrastructure deletion: 5 minutes
- Infrastructure rebuild: 10 minutes  
- Pipeline execution: 15 minutes
- Model deployment: 10 minutes
- Troubleshooting/validation: 5 minutes

## Lessons Learned

### Documentation Effectiveness
- **✅ CLAUDE.md**: Sufficient for basic infrastructure rebuild
- **✅ Migration checklists**: Covered all major steps
- **✅ Troubleshooting docs**: Finalizer cleanup procedures were essential
- **⚠️ Model URIs**: Required trial-and-error to discover correct format

### Process Improvements
1. **Add MLflow model URI validation** to training pipeline
2. **Automate model deployment** as part of training workflow
3. **Create health check script** for post-deployment validation
4. **Document platform secret dependencies** more clearly

### Production Readiness
This reproduction test validates that:
- Infrastructure-as-code approach is working
- Documentation is sufficient for disaster recovery
- MLOps pipelines are resilient and reproducible
- Platform team coordination points are well-defined

## Recommended Automation

### CI/CD Integration
```bash
# Add to training pipeline final step
- name: update-model-deployments
  template: update-model-uris
  arguments:
    parameters:
    - name: model-variant
      value: "{{workflow.parameters.model-variant}}"
```

### Health Check Script
```bash
#!/bin/bash
# scripts/health-check.sh
kubectl get models -n financial-inference
kubectl get experiments -n financial-inference  
kubectl get workflows -n financial-mlops-pytorch
argo list -n financial-mlops-pytorch | head -5
```

This reproduction test demonstrates the maturity of our MLOps infrastructure and validates our disaster recovery capabilities.