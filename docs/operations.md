# Operations Guide

This guide provides operational procedures and troubleshooting information for the Financial MLOps platform.

## Development Workflows

### Model Training Variants

Train different model variants for A/B testing:

```bash
# Train baseline model
MODEL_VARIANT=baseline python src/train_pytorch_model.py

# Train enhanced model for A/B testing
MODEL_VARIANT=enhanced python src/train_pytorch_model.py

# Train lightweight model for edge deployment
MODEL_VARIANT=lightweight python src/train_pytorch_model.py
```

### Local Development

Run training locally with environment variables:

```bash
cd ~/REPOS/financial-mlops-pytorch

MODEL_VARIANT=baseline \
SCALER_DIR=/mnt/shared-artifacts/scalers \
MLFLOW_TRACKING_URI=http://<MLFLOW_HOST>:<PORT> \
MLFLOW_S3_ENDPOINT_URL=http://<MINIO_HOST>:<PORT> \
PROCESSED_DATA_DIR=/mnt/shared-data/processed \
RAW_DATA_DIR=/mnt/shared-data/raw \
MODEL_SAVE_DIR=/mnt/shared-artifacts/models \
python3 src/train_pytorch_model.py
```

## Pipeline Operations

### Submit Training Workflows

```bash
# Submit individual model training workflows
argo submit --from workflowtemplate/financial-training-pipeline-template \
  -p model-variant=baseline \
  -n financial-mlops-pytorch

argo submit --from workflowtemplate/financial-training-pipeline-template \
  -p model-variant=enhanced \
  -n financial-mlops-pytorch

argo submit --from workflowtemplate/financial-training-pipeline-template \
  -p model-variant=lightweight \
  -n financial-mlops-pytorch

# Submit data pipeline
argo submit --from workflowtemplate/financial-data-pipeline-template \
  -n financial-mlops-pytorch --watch
```

### Monitor Workflows

```bash
# List workflows
argo list -n financial-mlops-pytorch

# Watch workflow progress
argo get <workflow-name> -n financial-mlops-pytorch

# View workflow logs
argo logs <workflow-name> -n financial-mlops-pytorch
```

## Model Deployment

### Check Model Status

```bash
# Check if models are loading
kubectl get models -n financial-inference

# Check experiment status  
kubectl get experiments -n financial-inference

# Check model details
kubectl describe model baseline-predictor -n financial-inference
```

### Test Model Endpoints

```bash
# Test A/B endpoint
curl -H "Host: financial-predictor.local" http://<CLUSTER_IP>/predict

# Test specific model
curl -X POST -H "Content-Type: application/json" \
  -H "Host: financial-predictor.local" \
  -d '{"instances": [[1.0, 2.0, 3.0]]}' \
  http://<CLUSTER_IP>/v2/models/baseline-predictor/infer
```

## Infrastructure Operations

### Container Image Management

```bash
# Build and push main application image
docker build -t <REGISTRY>/<USERNAME>/financial-predictor:latest . --push

# Build and push Jupyter image  
docker build -t <REGISTRY>/<USERNAME>/financial-predictor-jupyter:latest -f jupyter/Dockerfile . --push
```

### MinIO Operations

```bash
# Set up MinIO client
mc alias set minio http://<MINIO_HOST>:<PORT> <ACCESS_KEY> <SECRET_KEY>

# List model artifacts
mc ls minio/mlflow-artifacts/28/
mc ls minio/mlflow-artifacts/29/

# List all experiments
mc ls minio/mlflow-artifacts/ --recursive
```

### MLflow Operations

```bash
# List all experiments
mlflow experiments search

# View experiment details
mlflow experiments describe --experiment-id 28
```

## Troubleshooting

### Model Loading Issues

If models are not loading or showing as "not ready":

```bash
# 1. Check namespace configuration
kubectl get models -n financial-inference
kubectl describe model baseline-predictor -n financial-inference

# 2. Verify secret exists
kubectl get secret ml-platform -n financial-inference

# 3. Check Seldon controller logs
kubectl logs -n seldon-system deployment/seldon-v2-controller-manager | grep baseline-predictor

# 4. Check MLServer logs
kubectl logs -n seldon-system mlserver-0 -c mlserver | tail -50

# 5. Check scheduler logs
kubectl logs -n seldon-system deployment/seldon-scheduler
```

### ArgoCD Operations

```bash
# Get ArgoCD admin password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d && echo

# Login to ArgoCD
argocd login <ARGOCD_HOST>:<PORT>

# Delete problematic applications
argocd app delete seldon-deployments --cascade=true
```

### Namespace Management

```bash
# Clean deployment (development only)
kubectl delete ns financial-mlops-pytorch
kubectl create ns financial-mlops-pytorch

# Apply configurations
kubectl apply -k k8s/base
```

## Model Storage Structure

MLflow stores artifacts in the following structure:

```
s3://mlflow-artifacts/<experiment_id>/<run_id>/artifacts/
s3://mlflow-artifacts/<experiment_id>/models/<model_id>/artifacts/
```

Example MLflow experiments:
- **28**: financial-mlops-pytorch-baseline
- **29**: financial-mlops-pytorch-enhanced  
- **30**: financial-mlops-pytorch-lightweight

## Environment Configuration

Configure environment-specific endpoints in your deployment:

```bash
# Example environment variables
export MLFLOW_HOST="mlflow.example.com"
export MINIO_HOST="minio.example.com" 
export ARGOCD_HOST="argocd.example.com"
export CLUSTER_IP="<your-cluster-external-ip>"
```

Development environment example:
- **MLflow**: http://mlflow.local:30800
- **MinIO**: http://minio.local:30900
- **ArgoCD**: http://argocd.local:30080
- **Model Endpoint**: http://cluster.local (with Host header: financial-predictor.local)

## Status Validation

After deployment, validate the system:

```bash
# Check all Seldon components
kubectl get pods -n seldon-system

# Check model deployment status
kubectl get models,experiments -n financial-inference

# Check training workflows
kubectl get workflows -n financial-mlops-pytorch

# Verify persistent volumes
kubectl get pvc -n financial-mlops-pytorch
```

## Recovery Procedures

### Model Rollback

```bash
# Route 100% traffic to stable model
kubectl patch experiment financial-ab-test-experiment -n financial-inference --type='merge' \
  -p='{"spec":{"candidates":[{"name":"baseline-predictor","weight":100},{"name":"enhanced-predictor","weight":0}]}}'
```

### Workflow Recovery

```bash
# Retry failed workflow
argo retry <workflow-name> -n financial-mlops-pytorch

# Resubmit workflow
argo resubmit <workflow-name> -n financial-mlops-pytorch
```