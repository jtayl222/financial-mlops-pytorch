# Quick Reference Guide

**Financial MLOps platform with PyTorch, Seldon Core v2, and Argo Workflows.**

## Environment Overview

- **Container Orchestration**: Kubernetes (k3s v1.33.1+ recommended)
- **CNI**: Calico for network policies
- **LoadBalancer**: MetalLB for service exposure
- **Namespaces**: `seldon-system` (serving), `seldon-system` (training)
- **Storage**: Persistent volumes for shared data and artifacts

## Quick Commands

```bash
# Verify cluster health
kubectl get nodes -o wide
kubectl get pods -n kube-system

# Deploy infrastructure
kubectl apply -k k8s/base

# Check deployment status
kubectl get models,experiments -n seldon-system
kubectl get workflows -n seldon-system

# Train models
argo submit --from workflowtemplate/financial-training-pipeline-template \
  -p model-variant=baseline -n seldon-system

# Update model URIs after training
python3 scripts/update_model_uris.py

# Update specific model from specific experiment (after extracting experiment ID from Argo logs)
python3 scripts/update_model_uris.py --experiment-id 28 --model-variant baseline
```

## GitOps with Argo CD

### Architecture
- **Main Application**: `financial-mlops-infrastructure` (k8s/base/ → seldon-system namespace)
- **Auto-sync**: Enabled with prune and self-heal
- **Status Ignoring**: Models/experiments have dynamic status fields ignored during sync

### Access
- **Web UI**: http://192.168.1.85:30080
- **CLI**: `kubectl port-forward svc/argocd-server -n argocd 8080:443`

### Commands
```bash
# Automated model deployment with GitOps
./scripts/gitops-model-update.sh enhanced

# Check GitOps applications
kubectl get applications -n argocd

# Manual sync if needed
kubectl patch application financial-mlops-infrastructure -n argocd -p '{"operation":{"sync":{}}}' --type=merge

# View sync history
argocd app history financial-mlops-infrastructure
```

### GitOps Workflow
1. Update manifests in k8s/base/
2. Commit and push to git
3. Argo CD auto-detects changes (within 3 minutes)
4. Applies necessary changes to cluster

## Model Deployment Troubleshooting

1. **"no matching servers available"** → Check MLServer pod status
2. **"connection timeout"** → Verify network policies and service connectivity
3. **"storage error"** → Validate S3/MinIO configuration and credentials
4. **"MLmodel not found"** → Verify URI format: `s3://bucket/experiment/models/m-runid/artifacts/`

## Model Variants

- **baseline**: Standard LSTM (64 hidden, 2 layers)
- **enhanced**: Advanced LSTM (128 hidden, 3 layers) 
- **lightweight**: Optimized LSTM (32 hidden, 1 layer)

## Key Files

- `docs/operations/`: Operational procedures and guides
- `docs/troubleshooting/`: Issue resolution documentation
- `k8s/base/`: Core Kubernetes manifests
- `kubernetes/`: GitOps configurations for Argo CD

## Architecture Decisions

### Dedicated MLServer
- Run MLServer in `seldon-system` namespace for isolation
- Use dedicated Server resource with model-specific capabilities
- Avoids cross-namespace dependencies and improves security

### Git Conventions
- Prefer larger commits with multiple related changes
- Keep commit messages concise and descriptive
- Batch related fixes/features into single meaningful commits

## Resource Requirements

- **Training**: 4-8Gi memory, 2-4 CPU cores per training job
- **Serving**: 1-2Gi memory, 500m-1 CPU per model instance
- **Storage**: 20Gi+ for shared data, 10Gi+ for model artifacts

## Monitoring & Health Checks

```bash
# Check infrastructure health
kubectl get pods -A | grep -E "(mlflow|seldon|argo)"

# Monitor training progress  
argo list -n seldon-system

# Verify model serving
kubectl get models -n seldon-system
kubectl describe model baseline-predictor -n seldon-system
```

## Security Considerations

- Network policies enforce namespace isolation
- Sealed secrets for credential management
- RBAC for service account permissions
- No sensitive data in git repository

## Development Workflow

1. **Local Development**: Train and test models locally
2. **Containerization**: Build and push container images
3. **Pipeline Execution**: Submit Argo Workflows for training
4. **Model Deployment**: Update URIs and apply via GitOps
5. **Validation**: Test model endpoints and A/B experiments

## Support Resources

- [Reproduction Testing Guide](reproduction-testing.md)
- [GitOps Setup Documentation](gitops-setup.md)
- [Migration Checklists](../migration/)
- [Troubleshooting Guides](../troubleshooting/)