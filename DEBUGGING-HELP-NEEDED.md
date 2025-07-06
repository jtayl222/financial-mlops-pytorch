# 🆘 **Seeking Second Opinion: Seldon Core v2 Model Scheduling Issue**

## **Background**
Implementing enterprise MLOps with Seldon Core v2 using dedicated SeldonRuntime per namespace (following Netflix/Spotify patterns). Successfully deployed infrastructure but hitting a model scheduling roadblock.

## **Architecture**
- **Namespace Isolation**: `financial-ml` (models) + `financial-mlops-pytorch` (training)
- **SeldonRuntime**: Dedicated runtime per namespace with scheduler, envoy, modelgateway
- **Secrets**: Package-based delivery from infrastructure team (working)
- **RBAC/NetworkPolicies**: Proper security controls in place

## **Current State** ✅
```bash
# All infrastructure components healthy
kubectl get pods -n financial-ml
NAME                                   READY   STATUS    RESTARTS   AGE
hodometer-7db8996f79-lrpp8             1/1     Running   0          5m
mlserver-0                             3/3     Running   0          5m  
seldon-envoy-5d9f4c78f9-ql922          1/1     Running   0          5m
seldon-modelgateway-766f559c94-tmnfx   1/1     Running   0          5m
seldon-scheduler-0                     1/1     Running   0          5m

# Scheduler running and accessible
kubectl logs seldon-scheduler-0 -n financial-ml
time="2025-07-05T02:42:08Z" level=info msg="Scheduler server running on 9004 mtls:false"
time="2025-07-05T02:42:08Z" level=info msg="Agent server running on 9005 mtls:false"
```

## **The Problem** ❌
Models won't schedule - getting intermittent connection errors:

```bash
kubectl describe model baseline-predictor -n financial-ml
Status:
  Conditions:
    Message: ModelProgressing
    Reason:  rpc error: code = Unavailable desc = connection error: desc = "transport: Error while dialing: dial tcp 10.43.9.156:9004: connect: connection refused"
    Status:  False
    Type:    ModelReady
```

**Controller logs show mixed signals:**
```
2025-07-05T02:43:16.993Z INFO schedulerClient Connected to scheduler {"host": "seldon-scheduler.financial-ml", "port": 9004}
2025-07-05T02:43:17.005Z ERROR Reconciler error {...} "rpc error: code = Unavailable desc = connection error: desc = \"transport: Error while dialing: dial tcp 10.43.9.156:9004: i/o timeout\""
```

## **Configuration Details**

**SeldonRuntime:**
```yaml
apiVersion: mlops.seldon.io/v1alpha1
kind: SeldonRuntime
metadata:
  name: financial-ml-runtime
  namespace: financial-ml
spec:
  config:
    agentConfig: {rclone: {}}
    kafkaConfig: {}
    serviceConfig: {}
    tracingConfig: {}
  overrides:
  - name: seldon-scheduler
    replicas: 1
  - name: seldon-envoy  
    replicas: 1
  - name: seldon-modelgateway
    replicas: 1
  - name: seldon-dataflow-engine
    replicas: 0  # Disabled (no Kafka)
  - name: seldon-pipelinegateway
    replicas: 0  # Disabled
  - name: mlserver
    replicas: 1
```

**Model Config:**
```yaml
apiVersion: mlops.seldon.io/v1alpha1
kind: Model
metadata:
  name: baseline-predictor
  namespace: financial-ml
spec:
  storageUri: s3://mlflow-artifacts/28/models/m-d6d788df1b5849b3a3df1d04434c17b9/artifacts/
  requirements: [mlflow, torch, numpy, scikit-learn]
  secretName: ml-platform
  server: mlserver
```

**Network Setup:**
- Service: `seldon-scheduler.financial-ml.svc.cluster.local:9004`
- Controller connects from `seldon-system` namespace to `financial-ml` 
- NetworkPolicy allows `seldon-system` ↔ `financial-ml` traffic
- No CPU limits on ResourceQuota (removed for SeldonRuntime compatibility)

## **What We've Tried**
1. ✅ **Clean slate**: Deleted/recreated namespaces completely
2. ✅ **RBAC fixes**: Proper permissions for model deployment  
3. ✅ **Resource quotas**: Removed CPU limits requirement
4. ✅ **Manual MLServer**: Deployed with proper config volumes
5. ✅ **Secret validation**: Infrastructure packages applied correctly
6. ⚠️ **Port testing**: Controller connects intermittently but times out

## **Key Questions**
1. **Architecture**: Is dedicated SeldonRuntime per namespace the right approach for v2? Or should we use shared `seldon-system` components?

2. **MLServer Registration**: Does the `mlserver` override in SeldonRuntime automatically create MLServer pods, or do we need manual deployment?

3. **Cross-namespace connectivity**: Should the global controller (seldon-system) be able to connect to dedicated schedulers (financial-ml)? Any special networking requirements?

4. **Debugging approach**: What's the best way to validate scheduler→MLServer connectivity and model scheduling pipeline?

## **Environment**
- **Seldon Core**: v2.9.0
- **Kubernetes**: K3s cluster  
- **Controller**: `CLUSTERWIDE=true` (cluster-wide watching enabled)
- **Networking**: Istio + NetworkPolicies + team-based namespace isolation

## **Success Criteria**
Need models to transition from `READY=False` to `READY=True` with successful scheduling on MLServer instances.

**Any insights on Seldon v2 dedicated runtime patterns or debugging approaches would be hugely appreciated!** 🙏

---
*Context: Building this as enterprise MLOps portfolio for job applications - architecture and documentation are solid, just need this final scheduling piece working.*

## **Additional Context**

### **Repository Structure**
This is part of a comprehensive enterprise MLOps implementation:
- ✅ **Team-based namespace isolation** following Fortune 500 patterns
- ✅ **Package-based secret management** with infrastructure team collaboration
- ✅ **Comprehensive documentation** (LESSONS-LEARNED.md, INFRASTRUCTURE-REQUIREMENTS.md)
- ✅ **Enterprise security controls** (RBAC, NetworkPolicies, ResourceQuotas)
- ✅ **GitOps patterns** with Kustomize configuration management

### **Architecture Decisions Made**
- **From**: Cross-namespace scheduler sharing (complex, fragile)
- **To**: Dedicated SeldonRuntime per namespace (industry best practice)
- **Result**: Clean isolation but current scheduling roadblock

### **Files for Reference**
- `k8s/base/seldon-runtime.yaml` - SeldonRuntime configuration
- `k8s/base/financial-predictor-ab-test.yaml` - Model definitions
- `k8s/base/rbac.yaml` - Security controls and ResourceQuota
- `LESSONS-LEARNED.md` - Architectural decisions and enterprise patterns
- `INFRASTRUCTURE-REQUIREMENTS.md` - Platform requirements and secret management

### **Debugging Commands**
```bash
# Check all components
kubectl get pods,svc,models,experiments -n financial-ml

# Monitor controller logs
kubectl logs -f seldon-v2-controller-manager-756d5d4c4d-dcjrr -n seldon-system | grep financial-ml

# Check scheduler activity
kubectl logs -f seldon-scheduler-0 -n financial-ml

# Test direct connectivity
kubectl port-forward seldon-scheduler-0 9004:9004 -n financial-ml

# Validate secrets
kubectl get secrets -n financial-ml
kubectl describe secret ml-platform -n financial-ml
```

### **What Works**
- All infrastructure components deploy successfully
- Secrets are properly applied and accessible
- NetworkPolicies allow cross-namespace communication
- Controller can intermittently connect to scheduler
- MLServer pods are healthy with all containers ready

### **What Doesn't Work**
- Models remain in `READY=False` state
- Inconsistent scheduler connectivity from controller
- No visible MLServer registration in scheduler logs
- No model loading activity despite proper S3/MLflow access

This represents a solid enterprise MLOps foundation - just need help getting over this final scheduling hurdle! 🎯