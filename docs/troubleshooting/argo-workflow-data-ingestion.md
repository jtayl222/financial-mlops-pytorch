# Argo Workflow Data Ingestion Troubleshooting

## Overview

This document covers troubleshooting the financial data ingestion pipeline using Argo Workflows with the `financial-data-pipeline-template`.

## Common Commands

### Workflow Submission and Monitoring

```bash
# Submit IBB data ingestion workflow
argo submit --from workflowtemplate/financial-data-pipeline-template \
  -p ingestion-start-date="2018-01-01" \
  -p ingestion-end-date="2023-12-31" \
  -p tickers="IBB" \
  -n seldon-system

# Expected output:
# Name:                financial-data-pipeline-template-rcgqr
# Namespace:           seldon-system
# ServiceAccount:      unset
# Status:              Pending
# Parameters:          
#   ingestion-start-date: 2018-01-01
#   ingestion-end-date: 2023-12-31
#   tickers:           IBB

# Monitor workflow progress
argo list -n seldon-system

# Watch workflow execution logs
argo logs -f financial-data-pipeline-template-xxxxx -n seldon-system

# Get detailed workflow status
argo get financial-data-pipeline-template-xxxxx -n seldon-system
```

## Expected Successful Output

### Workflow Execution Log Sample

```
# Step 1: Data Ingestion
2025-07-15 02:35:31,779 - INFO - Starting data ingestion process.
2025-07-15 02:35:31,780 - INFO - Configured TICKERS: ['IBB']
2025-07-15 02:35:33,394 - INFO - Downloading data for ticker: IBB
2025-07-15 02:35:34,248 - INFO - Successfully downloaded 1509 rows for IBB.
2025-07-15 02:35:34,301 - INFO - Data for IBB saved to /mnt/shared-data/raw/IBB_raw_2018-01-01_2023-12-31.csv

# Step 2: Feature Engineering
2025-07-15 02:35:57,859 - INFO - Starting feature engineering and PyTorch data preparation process.
2025-07-15 02:36:00,670 - INFO - Processing file: /mnt/shared-data/raw/IBB_raw_2018-01-01_2023-12-31.csv
2025-07-15 02:36:00,795 - INFO - Train features shape: (2024, 35)
2025-07-15 02:36:01,583 - INFO - Number of training sequences: 2015
2025-07-15 02:36:02,244 - INFO - Feature engineering and PyTorch data preparation completed successfully.

# MLflow Integration
🏃 View run at: http://mlflow.mlflow.svc.cluster.local:5000/#/experiments/0/runs/xxxxx
🧪 View experiment at: http://mlflow.mlflow.svc.cluster.local:5000/#/experiments/0
```

### Key Success Indicators

- ✅ "Successfully downloaded X rows for IBB"
- ✅ "Feature engineering and PyTorch data preparation completed successfully"
- ✅ MLflow tracking URLs displayed
- ✅ Workflow status shows "Succeeded" in `argo list`

## Common Issues and Solutions

### 1. Workflow Shows "Pending" Status

**Symptoms:**
- Workflow remains in "Pending" state indefinitely
- No pods are created

**Debug Commands:**
```bash
# Check if workflow templates are properly configured
kubectl describe workflowtemplate financial-data-pipeline-template -n seldon-system

# Verify service account permissions
kubectl get serviceaccount argo-workflow-sa -n seldon-system

# Check for resource constraints
kubectl describe nodes

# Verify Argo Workflows controller is running
kubectl get pods -n argo
```

**Solutions:**
- Ensure service account has proper permissions
- Check if cluster has sufficient resources
- Verify Argo Workflows is properly installed

### 2. Wrong Tickers Being Used (Hardcoded Values)

**Symptoms:**
- Log shows: "Configured TICKERS: ['AAPL', 'MSFT']" instead of "['IBB']"
- Data downloaded for wrong symbols

**Debug Commands:**
```bash
# Check if template uses parameters correctly
kubectl get workflowtemplate financial-data-pipeline-template -n seldon-system -o yaml | grep -A 10 -B 5 TICKERS
```

**Solutions:**
```bash
# Ensure the template uses workflow parameters
# In k8s/base/financial-data-pipeline.yaml:
# - name: TICKERS
#   value: "{{workflow.parameters.tickers}}"

# Re-apply the updated template
kubectl apply -f k8s/base/financial-data-pipeline.yaml
```

### 3. Container/Image Issues

**Symptoms:**
- "ModuleNotFoundError" or import errors
- "ImagePullBackOff" status

**Debug Commands:**
```bash
# Check container image being used
kubectl describe workflowtemplate financial-data-pipeline-template -n seldon-system | grep image

# Check pod events for pull errors
kubectl describe pod <workflow-pod-name> -n seldon-system

# Verify image exists and is accessible
docker pull jtayl22/financial-predictor:latest
```

**Solutions:**
- Ensure the container image contains all required dependencies
- Verify image repository access and credentials
- Update image tag if needed

### 4. Storage/PVC Issues

**Symptoms:**
- "PersistentVolumeClaim not found" errors
- Data not persisting between workflow steps

**Debug Commands:**
```bash
# Check if persistent volumes are bound
kubectl get pv,pvc -n seldon-system

# Verify storage class exists
kubectl get storageclass

# Check PVC details
kubectl describe pvc shared-data-pvc -n seldon-system
```

**Solutions:**
```bash
# Create missing PVCs if needed
kubectl apply -f k8s/base/  # This should include PVC definitions

# Check storage class configuration
kubectl describe storageclass <your-storage-class>
```

### 5. MLflow Connectivity Issues

**Symptoms:**
- No MLflow tracking URLs in logs
- "Connection refused" errors to MLflow

**Debug Commands:**
```bash
# Check MLflow service in its namespace
kubectl get svc -n mlflow

# Test internal connectivity
kubectl run test-pod --image=curlimages/curl -it --rm -- \
  curl http://mlflow.mlflow.svc.cluster.local:5000/health

# Check MLflow pod status
kubectl get pods -n mlflow
```

**Solutions:**
```bash
# Port-forward to access MLflow directly
kubectl port-forward -n mlflow svc/mlflow 5000:5000

# Verify MLflow configuration in workflow
kubectl get workflowtemplate financial-data-pipeline-template -n seldon-system -o yaml | grep -i mlflow
```

### 6. Data Verification Issues

**Symptoms:**
- Workflow succeeds but no data files found
- Empty or corrupted data files

**Debug Commands:**
```bash
# Check data in persistent storage
kubectl exec -it -n seldon-system <any-pod-with-storage> -- \
  ls -la /mnt/shared-data/raw/

# Verify file contents
kubectl exec -it -n seldon-system <any-pod-with-storage> -- \
  head -10 /mnt/shared-data/raw/IBB_raw_2018-01-01_2023-12-31.csv

# Check processed features
kubectl exec -it -n seldon-system <any-pod-with-storage> -- \
  ls -la /mnt/shared-data/processed/
```

**Expected File Structure:**
```
/mnt/shared-data/
├── raw/
│   └── IBB_raw_2018-01-01_2023-12-31.csv
└── processed/
    ├── train_features.npy
    ├── train_targets.npy
    ├── val_features.npy
    ├── val_targets.npy
    ├── test_features.npy
    ├── test_targets.npy
    └── combined_processed_data.csv
```

## Log Analysis

### Normal Workflow Progression

1. **Workflow Creation**: Status changes from "Pending" to "Running"
2. **Data Ingestion Step**: Downloads ticker data from financial APIs
3. **Feature Engineering Step**: Processes raw data and creates ML-ready datasets
4. **MLflow Logging**: Experiments and artifacts logged to MLflow
5. **Completion**: Status changes to "Succeeded"

### Error Patterns to Watch For

- **API Rate Limiting**: "429 Too Many Requests" from financial data providers
- **Network Issues**: "Connection timeout" or "DNS resolution failed"
- **Resource Limits**: "OOMKilled" or "Evicted" pod status
- **Permission Issues**: "Forbidden" errors accessing storage or services

## Related Documentation

- [Seldon v2 API 404 Debugging](../troubleshooting/seldon-v2-api-404-debugging.md)
- [NGINX Ingress Cross-Namespace Routing](../troubleshooting/nginx-ingress-cross-namespace-routing.md)
- [Demo Instructions](../demo/DEMO_INSTRUCTIONS.md)

## Quick Recovery Commands

```bash
# Delete failed workflow and retry
argo delete financial-data-pipeline-template-xxxxx -n seldon-system

# Clean up stuck pods
kubectl delete pods --field-selector=status.phase=Failed -n seldon-system

# Restart Argo controller if needed
kubectl rollout restart deployment argo-workflow-controller -n argo

# Resubmit workflow with fresh parameters
argo submit --from workflowtemplate/financial-data-pipeline-template \
  -p tickers="IBB" -n seldon-system
```