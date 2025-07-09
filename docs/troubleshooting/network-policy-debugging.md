# Network Policy Debugging Guide

## DNS Resolution Issues

### Symptom
```
Temporary failure in name resolution
Failed to establish a new connection: [Errno -3]
```

### Root Cause
Missing egress rules for DNS resolution (port 53) to kube-system namespace.

### Solution
Add DNS egress rules to network policy:
```yaml
egress:
- to:
  - namespaceSelector:
      matchLabels:
        name: kube-system
  ports:
  - protocol: UDP
    port: 53
  - protocol: TCP
    port: 53
```

## Kubernetes API Server Access

### Symptom
```
dial tcp 10.43.0.1:443: i/o timeout
Error (exit code 64): Post "https://10.43.0.1:443/apis/argoproj.io/v1alpha1/..."
```

### Root Cause
Argo workflow pods cannot communicate with Kubernetes API server for status updates.

### Solution
Add API server egress rules:
```yaml
egress:
- to: []
  ports:
  - protocol: TCP
    port: 443
  - protocol: TCP
    port: 6443
```

## Cross-Namespace Communication

### Symptom
```
connection refused
no route to host
```

### Root Cause
Network policies blocking cross-namespace communication for MLOps workflows.

### Solution
Add specific namespace egress rules:
```yaml
egress:
- to:
  - namespaceSelector:
      matchLabels:
        name: target-namespace
  ports:
  - protocol: TCP
    port: 5000  # Example: MLflow
```

## Testing Commands

### DNS Resolution Test
```bash
kubectl run test-dns --image=busybox --rm -it --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"test-dns","image":"busybox","command":["nslookup","kubernetes.default"],"resources":{"requests":{"cpu":"100m","memory":"128Mi"},"limits":{"cpu":"200m","memory":"256Mi"}}}]}}' \
  -n financial-mlops-pytorch
```

### API Server Connectivity Test
```bash
kubectl run test-api --image=curlimages/curl --rm --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"test-api","image":"curlimages/curl","command":["curl","-k","-m","5","https://10.43.0.1:443/healthz"],"resources":{"requests":{"cpu":"100m","memory":"128Mi"},"limits":{"cpu":"200m","memory":"256Mi"}}}]}}' \
  -n financial-mlops-pytorch
```

### Service Connectivity Test
```bash
kubectl run test-service --image=curlimages/curl --rm --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"test-service","image":"curlimages/curl","command":["curl","-m","5","http://mlflow.mlflow.svc.cluster.local:5000"],"resources":{"requests":{"cpu":"100m","memory":"128Mi"},"limits":{"cpu":"200m","memory":"256Mi"}}}]}}' \
  -n financial-mlops-pytorch
```

## Environment Detection

### Verify CNI Implementation
```bash
kubectl get pods -n kube-system | grep calico   # Confirms Calico CNI
kubectl get pods -n kube-system | grep flannel  # Check for Flannel (should be empty)
```

### Check LoadBalancer Configuration
```bash
kubectl get svc -A | grep LoadBalancer          # Confirms MetalLB
kubectl get nodes -o wide                       # Shows cluster networking
```

### Network Policy Status
```bash
kubectl get networkpolicies -A                  # List all network policies
kubectl describe networkpolicy <policy-name> -n <namespace>  # Check specific policy
```