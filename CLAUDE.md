# CLAUDE.md

**Financial MLOps pipeline with PyTorch, Seldon Core v2, and Argo Workflows.**

## Current Environment (CRITICAL: Post-Migration)
- **CNI**: Calico (MIGRATED FROM Flannel - base branch was Flannel)
- **LoadBalancer**: MetalLB (MIGRATED FROM NodePort - affects service configs)
- **Cluster**: Fresh k3s v1.33.1+k3s1 with Calico networking (5 nodes, 36 CPU cores, ~260GB memory)
- **Namespaces**: `financial-ml` (serving), `financial-mlops-pytorch` (training)
- **Key Issue**: Base configs designed for Flannel+NodePort, now Calico+MetalLB
- **Resource Quotas**: 50 CPUs, 100Gi requests, 200Gi limits per namespace (generous for primary app)
- **Technical Debt**: ResourceQuota removed due to Argo v3.6.10 + K8s v1.33 strict validation (see TECHNICAL-DEBT.md)

## Quick Commands
```bash
# FIRST: Verify environment (run when debugging networking issues)
kubectl get pods -n kube-system | grep calico   # Confirms Calico CNI
kubectl get svc -A | grep LoadBalancer          # Confirms MetalLB
kubectl get nodes -o wide                       # Shows cluster state

# Deploy infrastructure
kubectl apply -k k8s/base

# Check status
kubectl get models,experiments -n financial-ml
kubectl get svc -A | grep LoadBalancer

# Train models
argo submit --from workflowtemplate/financial-training-pipeline-template \
  -p model-variant=baseline -n financial-mlops-pytorch
```

## Model Deployment Issues?
1. **"no matching servers available"** → Platform team: missing seldon-config
2. **"connection timeout"** → Platform team: cluster networking 
3. **"storage error"** → Check MLflow LoadBalancer IPs

## Key Files
- `NETWORK-POLICY-GUIDELINES.md`: App vs platform team responsibilities
- `PLATFORM-REQUESTS/`: Templates for platform team escalation
- `k8s/base/network-policy.yaml`: Application-level network policies

## Model Variants
- `baseline`: Standard LSTM (64 hidden, 2 layers)
- `enhanced`: Advanced LSTM (128 hidden, 3 layers) 
- `lightweight`: Optimized LSTM (32 hidden, 1 layer)

## Git Conventions
- **Prefer larger commits** with multiple related changes
- **Keep commit messages short and focused** - avoid verbose descriptions
- Batch related fixes/features into single meaningful commits

## Architecture Decision: Dedicated MLServer
- **Run MLServer in financial-ml namespace** for better isolation (not in seldon-system)
- Use dedicated Server resource with correct capabilities for our models
- Avoids cross-namespace dependencies and improves security boundaries

## Article Documentation
- **Document this experience** in a technical article about MLOps migration
- See `ARTICLE-OUTLINE.md` for table of contents and key points
- See `METALLB-MIGRATION-NOTES.md` for context on why MetalLB was adopted
- Capture lessons learned about CNI migration, network policies, and team collaboration

## Preserved Work
- **Git stash**: `stash@{0}` contains extensive previous work before reset
- **Temp files**: `/tmp/preserve-files/` has detailed docs and configs
- Much valuable work was done and preserved - can be recovered if needed