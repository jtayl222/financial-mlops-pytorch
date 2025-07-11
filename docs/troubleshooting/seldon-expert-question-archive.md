# Seldon Core v2 Model Scheduling Issue - Expert Help Needed

## Problem Summary
**Models fail to schedule with "no matching servers are available" despite having a running MLServer with correct capabilities in the same namespace.**

## Environment
- **Seldon Core**: v2.9.0 with custom agent (`jtayl22/seldon-agent:2.9.0-pr6582-test`)
- **Kubernetes**: Fresh k3s v1.33.1+k3s1 with Calico CNI + MetalLB
- **MLServer**: v1.6.1
- **Architecture**: Financial MLOps pipeline with namespace isolation

## Current State

### Working Components
```bash
# MLServer is healthy and ready
kubectl get servers -n financial-inference
NAME       READY   REPLICAS   LOADED MODELS   AGE
mlserver   True    1          0               45m

# MLServer pod running successfully
kubectl get pods -n financial-inference | grep mlserver
mlserver-0                             3/3     Running   0          45m

# seldon-config ConfigMap exists and looks correct
kubectl get configmap seldon-config -n seldon-system
NAME            DATA   AGE
seldon-config   1      90m
```

### Failing Models
```bash
kubectl get models -n financial-inference
NAME                 READY   DESIRED REPLICAS   AVAILABLE REPLICAS   AGE
baseline-predictor   False                                           5m
enhanced-predictor   False                                           5m

kubectl describe model baseline-predictor -n financial-inference
Status:
  Conditions:
    Message:               ScheduleFailed
    Reason:                Failed to schedule model as no matching servers are available
    Status:                False
    Type:                  ModelReady
```

## Configuration Details

### Model Definition
```yaml
apiVersion: mlops.seldon.io/v1alpha1
kind: Model
metadata:
  name: baseline-predictor
  namespace: financial-inference
spec:
  storageUri: s3://mlflow-artifacts/28/models/m-d6d788df1b5849b3a3df1d04434c17b9/artifacts/
  requirements:
  - mlflow
  - torch
  - numpy
  - scikit-learn
  secretName: ml-platform
  server: mlserver
```

### MLServer Definition
```yaml
apiVersion: mlops.seldon.io/v1alpha1
kind: Server
metadata:
  name: mlserver
  namespace: financial-inference
spec:
  serverConfig: mlserver
  capabilities: ["pytorch", "torch", "sklearn", "scikit-learn", "xgboost", "mlflow", "python", "numpy"]
  replicas: 1
```

### seldon-config ConfigMap
```json
{
    "MLSERVER": {
        "capabilities": [
            "sklearn", "xgboost", "mlflow", "pytorch", "torch", "scikit-learn", "python", "numpy"
        ],
        "grpc": {
            "defaultImageVersion": "1.6.1",
            "image": "docker.io/seldonio/mlserver:1.6.1"
        },
        "protocols": ["seldon", "v2"],
        "rest": {
            "defaultImageVersion": "1.6.1", 
            "image": "docker.io/seldonio/mlserver:1.6.1"
        }
    }
}
```

## What We've Tried

### 1. Network Policy Updates
- Updated network policies for Calico CNI (migrated from Flannel)
- Ensured cross-namespace communication between financial-inference and seldon-system
- Added LoadBalancer ingress rules for MetalLB

### 2. Controller Restarts
- Restarted `seldon-v2-controller-manager` multiple times
- Restarted `seldon-scheduler` in financial-inference namespace
- Deleted and recreated models to trigger fresh scheduling

### 3. Server Configuration
- Created dedicated Server resource in financial-inference namespace
- Ensured capabilities include all model requirements (`mlflow`, `torch`, `scikit-learn`, `numpy`)
- Verified MLServer pod is running and healthy (3/3 Ready)

### 4. Infrastructure Verification
- Confirmed seldon-config ConfigMap exists with correct capabilities
- Verified MLServer agent using fixed image with networking patches
- Checked that all Seldon components are running in both namespaces

## Architecture Context

### Namespace Strategy
- **financial-inference**: Model serving namespace (where models and MLServer should run)
- **seldon-system**: Seldon Core control plane and shared MLServer
- **Goal**: Dedicated MLServer in financial-inference for better isolation

### Current Setup
```bash
# SeldonRuntime in financial-inference namespace
kubectl get seldonruntime -n financial-inference
NAME                 AGE
financial-inference-runtime 2h

# Dedicated Server resource
kubectl get servers -n financial-inference  
NAME       READY   REPLICAS   LOADED MODELS   AGE
mlserver   True    1          0               45m
```

## Key Questions

1. **Server Discovery**: How does the Seldon scheduler discover Server resources vs global seldon-config?
2. **Namespace Isolation**: Should we use SeldonRuntime OR dedicated Server resources, or both?
3. **Capabilities Matching**: Are we missing something in the capability matching logic?
4. **Storage Access**: Could the S3 storage URI be blocking scheduling (even though it's just scheduling, not loading)?

## Logs Analysis

### Scheduler Logs Show Server Recognition
```bash
kubectl logs seldon-scheduler-0 -n financial-inference --tail=5
time="2025-07-07T16:30:09Z" level=info msg="Server notification mlserver expectedReplicas 1 shared false" func=ServerNotify source=SchedulerServer
```

### MLServer Logs Show Health
```bash
kubectl logs mlserver-0 -n financial-inference -c mlserver --tail=3
INFO: 192.168.1.105:36660 - "GET /v2/health/live HTTP/1.1" 200 OK
```

## Expected Behavior
Models should schedule to the local MLServer in the financial-inference namespace and progress from ScheduleFailed to ModelReady status.

## Similar Working Setup?
Are there example configurations or common patterns for:
- Dedicated MLServer per namespace with Seldon Core v2?
- Proper capability matching between Model requirements and Server capabilities?
- Debugging "no matching servers" errors when servers appear to be available?

Any insights into what we might be missing would be greatly appreciated!