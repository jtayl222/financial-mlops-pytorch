# Technical Debt

## Argo Workflows ResourceQuota Issue

### Background
During k3s cluster rebuild (July 2025), the platform team upgraded:
- **Kubernetes**: v1.29-1.30 → v1.33.1+k3s1  
- **Argo Workflows**: Unknown → v3.6.10

### Problem
Workflows that ran successfully for 2 weeks began failing with:
```
failed quota: financial-mlops-pytorch-quota: must specify limits.memory for: init,wait; requests.cpu for: init,wait; requests.memory for: init,wait
```

### Root Cause
**Argo Workflows v3.6.10 + Kubernetes v1.33** enforces stricter ResourceQuota validation requiring explicit resource specifications for Argo's system containers (`init`, `wait`).

### Current Workaround (Technical Debt)
**Temporarily removed ResourceQuota** from `financial-mlops-pytorch` namespace:
```bash
kubectl delete resourcequota financial-mlops-pytorch-quota -n financial-mlops-pytorch
```

### Impact
- **Security**: No resource limits enforcement on training workloads
- **Stability**: Risk of resource exhaustion from runaway workflows
- **Operations**: Manual resource monitoring required

### Proper Solutions (Pick One)

#### Option 1: Update Workflow Templates
Add `podSpecPatch` to all WorkflowTemplates:
```yaml
spec:
  podSpecPatch: |
    containers:
    - name: init
      resources:
        requests: {cpu: 100m, memory: 128Mi}
        limits: {cpu: 500m, memory: 256Mi}
    - name: wait
      resources:
        requests: {cpu: 100m, memory: 128Mi}
        limits: {cpu: 500m, memory: 256Mi}
```

#### Option 2: Global Argo Configuration
Configure controller with default resources for system containers.

#### Option 3: Replace with LimitRanges
Use LimitRanges instead of ResourceQuotas for more flexible enforcement:
```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: default-limits
spec:
  limits:
  - default:
      cpu: 500m
      memory: 256Mi
    defaultRequest:
      cpu: 100m
      memory: 128Mi
    type: Container
```

### Priority
**Medium** - Affects resource governance but doesn't block functionality.

### Owner
Application team (workflow templates owned by us, not platform team).

### Related Files
- `k8s/base/financial-data-pipeline.yaml`
- `k8s/base/training-pipeline.yaml`
- `k8s/base/rbac.yaml` (contains ResourceQuota)

### Notes
- Pipelines work perfectly - this is purely a validation issue
- Previous environment likely had different Argo/K8s versions without strict validation
- Consider this when migrating to future Kubernetes versions

## Cluster DNS Resolution Issue

### Background
After k3s cluster rebuild with Calico CNI, workflow pods cannot resolve service DNS names.

### Problem
Workflows fail with DNS resolution errors:
```
socket.gaierror: [Errno -3] Temporary failure in name resolution
Failed to establish connection to 'mlflow.mlflow.svc.cluster.local'
```

### Root Cause Assessment
**Likely Platform Issue**:
- ✅ External LoadBalancer IP works (`192.168.1.207:5000`)
- ✅ Service exists with correct ClusterIP  
- ❌ DNS resolution fails from workflow pods
- ❌ Cross-namespace service discovery broken

### Current Workaround (Technical Debt)
**Using LoadBalancer IPs instead of service DNS**:
```yaml
# Workaround - should be service DNS
MLFLOW_TRACKING_URI: "http://192.168.1.207:5000"
# Proper - but currently broken
MLFLOW_TRACKING_URI: "http://mlflow.mlflow.svc.cluster.local:5000"
```

### Impact
- **Architecture violation**: External IPs instead of service discovery
- **Fragility**: LoadBalancer IP changes break workflows  
- **Performance**: Traffic routes externally instead of cluster-internal

### Platform Team Action Required
1. **CoreDNS configuration** - Verify cluster DNS service
2. **Calico networking** - Check cross-namespace DNS forwarding
3. **Network policies** - Ensure service discovery is permitted

### Priority
**High** - Breaks fundamental Kubernetes service discovery patterns.

### Owner
**Platform team** - cluster DNS and networking infrastructure.