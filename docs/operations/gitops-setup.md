# GitOps Setup with Argo CD

## Overview

This document describes the GitOps setup for the Financial MLOps platform using Argo CD for automated deployment and synchronization.

## Architecture

### Argo CD Applications

1. **`financial-mlops-infrastructure`** (Main Application)
   - **Source**: `k8s/base/` directory
   - **Destination**: `financial-inference` namespace  
   - **Auto-sync**: Enabled with prune and self-heal
   - **Purpose**: Manages core MLOps infrastructure

2. **`seldon-deployments`** (Component Application)
   - **Source**: `kubernetes/seldon-deployments/`
   - **Destination**: `financial-inference` namespace
   - **Purpose**: Seldon Core models and experiments

3. **`argo-workflows-pipelines`** (Component Application)
   - **Source**: `kubernetes/argo-workflows/`
   - **Destination**: `financial-mlops-pytorch` namespace
   - **Purpose**: Training and data pipelines

4. **`mlflow-infra`** (Placeholder Application)
   - **Source**: `kubernetes/mlflow/`
   - **Destination**: `mlflow` namespace
   - **Purpose**: MLflow-specific configurations (platform-managed)

## GitOps Directory Structure

```
kubernetes/
├── argo-workflows/
│   └── kustomization.yaml          # Pipeline resources
├── seldon-deployments/
│   └── kustomization.yaml          # Model deployment resources
├── mlflow/
│   └── kustomization.yaml          # MLflow configurations
└── financial-mlops-application.yaml # Main Argo CD application
```

## Key Features

### Automated Synchronization
- **Auto-sync**: Changes to git automatically deploy to cluster
- **Self-heal**: Argo CD corrects manual cluster changes
- **Prune**: Removes resources deleted from git

### Status Ignoring
Models and experiments have dynamic status fields that are ignored during sync:
```yaml
ignoreDifferences:
- group: mlops.seldon.io
  kind: Model
  jsonPointers:
  - /status
- group: mlops.seldon.io  
  kind: Experiment
  jsonPointers:
  - /status
```

### Sync Options
- `CreateNamespace=true`: Auto-create target namespaces
- `PrunePropagationPolicy=foreground`: Ordered resource deletion
- `PruneLast=true`: Delete dependent resources last

## Accessing Argo CD

### Web UI
- **URL**: http://192.168.1.85:30080
- **Credentials**: Use Kubernetes service account or configured OIDC

### CLI Access
```bash
# Port forward for CLI access
kubectl port-forward svc/argocd-server -n argocd 8080:443

# Login with CLI
argocd login localhost:8080
```

## GitOps Workflow

### 1. Development Workflow
```bash
# Make changes to k8s/base/ manifests
vim k8s/base/financial-predictor-ab-test.yaml

# Commit changes
git add k8s/base/
git commit -m "update: model URIs from latest training"
git push origin main
```

### 2. Automatic Deployment
- Argo CD detects git changes within 3 minutes
- Compares desired state (git) vs actual state (cluster)
- Applies necessary changes to achieve desired state
- Reports sync status and health

### 3. Model URI Updates Integration
```bash
# After training completion
python3 scripts/update_model_uris.py

# Commit automated changes
git add k8s/base/financial-predictor-ab-test.yaml
git commit -m "update: model URIs from MLflow experiment 29"
git push origin main

# Argo CD automatically deploys updated models
```

## Monitoring and Troubleshooting

### Application Status
```bash
# Check all applications
kubectl get applications -n argocd

# Check specific application
kubectl describe application financial-mlops-infrastructure -n argocd
```

### Sync Status
- **Synced**: Git and cluster are in sync
- **OutOfSync**: Changes detected, sync pending/in progress
- **Unknown**: Unable to determine status

### Health Status
- **Healthy**: All resources are running properly
- **Progressing**: Resources are starting/updating
- **Degraded**: Some resources have issues
- **Missing**: Required resources not found

### Manual Sync
```bash
# Force sync specific application
kubectl patch application financial-mlops-infrastructure -n argocd -p '{"operation":{"sync":{}}}' --type=merge

# Or use Argo CD CLI
argocd app sync financial-mlops-infrastructure
```

## Security Considerations

### Repository Access
- Uses HTTPS with GitHub repository
- Requires read access to `jtayl222/financial-mlops-pytorch`
- Webhook setup for immediate sync (optional)

### RBAC
- Argo CD service account has cluster-admin permissions
- Application-specific RBAC in target namespaces
- Principle of least privilege for application controllers

### Secrets Management
- Sealed secrets managed by platform team
- No sensitive data stored in git repository
- Argo CD doesn't manage secret creation/rotation

## Best Practices

### 1. Environment Separation
```bash
# Production branch protection
k8s/overlays/production/  # Production overrides
k8s/overlays/staging/     # Staging overrides
k8s/base/                 # Base configuration
```

### 2. Progressive Deployment
- Use staging environment for validation
- Blue-green deployments for models
- Canary releases via Seldon experiments

### 3. Rollback Strategy
```bash
# View application history
argocd app history financial-mlops-infrastructure

# Rollback to previous revision
argocd app rollback financial-mlops-infrastructure 0
```

### 4. Drift Detection
- Argo CD automatically detects configuration drift
- Self-heal corrects unauthorized changes
- Alerts on persistent drift issues

## Integration with CI/CD

### Training Pipeline Integration
Add to final training pipeline step:
```yaml
- name: update-deployments
  template: git-commit-template
  arguments:
    parameters:
    - name: files
      value: "k8s/base/financial-predictor-ab-test.yaml"
    - name: message
      value: "update: model URIs from training run {{workflow.parameters.model-variant}}"
```

### Webhook Setup
Configure GitHub webhook for immediate sync:
```bash
# Webhook URL
https://argocd.your-domain.com/api/webhook

# Events: push, pull_request (merged)
```

## Disaster Recovery

### Full Environment Recreation
1. Ensure git repository is available
2. Deploy Argo CD to new cluster
3. Apply main application manifest
4. Argo CD recreates entire environment from git

### Recovery Time
- **Application sync**: 1-3 minutes
- **Full deployment**: 5-10 minutes  
- **Model training + deployment**: 15-20 minutes

This GitOps setup provides automated, auditable, and reproducible deployments for the entire MLOps platform.