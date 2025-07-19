# Platform Team Request: Seldon Core Configuration

## Request Type
**Infrastructure Configuration** - Missing cluster-wide Seldon Core server definitions

## Issue Description
The `seldon-config` ConfigMap is missing from the `seldon-system` namespace, preventing models from scheduling to inference servers.

## Impact
- **Critical**: All model deployments fail with "no matching servers are available"
- **Affected Teams**: Any team using Seldon Core v2 for model serving
- **Business Impact**: Model serving pipeline completely blocked

## Current State
```bash
kubectl get configmaps -n seldon-system
# Output shows missing seldon-config ConfigMap
```

## Required Configuration
The platform team needs to create/restore the `seldon-config` ConfigMap in `seldon-system` namespace with inference server definitions.

### Minimum Required Servers
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: seldon-config
  namespace: seldon-system
data:
  predictor_servers: |
    {
      "MLSERVER": {
        "grpc": {
          "image": "docker.io/seldonio/mlserver:1.6.1",
          "defaultImageVersion": "1.6.1"
        },
        "rest": {
          "image": "docker.io/seldonio/mlserver:1.6.1", 
          "defaultImageVersion": "1.6.1"
        },
        "protocols": ["seldon", "v2"],
        "capabilities": ["sklearn", "xgboost", "mlflow", "pytorch", "torch", "scikit-learn", "python", "numpy"]
      },
      "TRITON": {
        "grpc": {
          "image": "nvcr.io/nvidia/tritonserver:24.04-py3",
          "defaultImageVersion": "24.04-py3"
        },
        "rest": {
          "image": "nvcr.io/nvidia/tritonserver:24.04-py3",
          "defaultImageVersion": "24.04-py3"
        },
        "protocols": ["v2"],
        "capabilities": ["tensorflow", "pytorch", "onnx", "python"]
      }
    }
```

## Validation Steps
After configuration is applied:

1. **Verify ConfigMap exists:**
   ```bash
   kubectl get configmap seldon-config -n seldon-system
   ```

2. **Restart Seldon operator:**
   ```bash
   kubectl rollout restart deployment/seldon-v2-controller-manager -n seldon-system
   ```

3. **Test model scheduling:**
   ```bash
   kubectl apply -f <model-definition>
   kubectl get models -n <namespace>
   # Should show models transitioning from False to True
   ```

## Dependencies
- **Seldon Core v2** infrastructure properly installed
- **MLServer and Triton** container images accessible
- **RBAC permissions** for Seldon operator to read ConfigMap

## Priority
**P0 - Critical** - Blocks all model serving capabilities

## Contact
- **Requesting Team**: seldon-system
- **Technical Contact**: [Team Lead]
- **Business Contact**: [Product Owner]

---
**Request Date**: 2025-07-07  
**Status**: Pending Platform Team Response