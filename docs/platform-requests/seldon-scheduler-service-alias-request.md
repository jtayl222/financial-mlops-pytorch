# Platform Team Request: Seldon Scheduler Service Alias

**Request ID:** PLAT-2025-07-12-001  
**Requesting Team:** Financial MLOps Application Team  
**Target Namespace:** `seldon-system` (Platform Team owned)  
**Priority:** High - Blocking trade show demo  
**Type:** Service Configuration  

## Problem Statement

The `seldon-v2-controller-manager` in `seldon-system` cannot connect to the central scheduler, blocking A/B testing functionality for our financial inference models.

**Root Cause:** Service naming mismatch identified by expert runbook analysis.

## Technical Details

### Current State:
- Controller manager expects service named: `seldon-scheduler.seldon-system.svc:9004`
- Actual service name in cluster: `seldon-system-scheduler.seldon-system.svc:9004`
- **Impact:** Controller manager fails with "Scheduler not ready" gRPC errors

### Evidence:
```bash
kubectl -n seldon-system logs deploy/seldon-v2-controller-manager | grep "Scheduler not ready"
# Shows: "rpc error: code = Canceled desc = grpc: the client connection is closing"
```

## Requested Change

**Option A: Create Service Alias (Recommended)**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: seldon-scheduler
  namespace: seldon-system
  labels:
    managed-by: platform-team
    request-id: PLAT-2025-07-12-001
spec:
  type: ExternalName
  externalName: seldon-system-scheduler.seldon-system.svc.cluster.local
  ports:
  - name: grpc
    port: 9004
    protocol: TCP
```

**Alternative Option B: Environment Variable Override**
```bash
# Update controller manager deployment
kubectl -n seldon-system set env deploy/seldon-v2-controller-manager \
  SELDON_SCHEDULER_HOST=seldon-system-scheduler \
  SELDON_SCHEDULER_PORT=9004

kubectl -n seldon-system rollout restart deploy/seldon-v2-controller-manager
```

## Validation Steps

After applying the fix, Platform Team can verify success with:

```bash
# 1. Controller manager connects successfully
kubectl -n seldon-system logs deploy/seldon-v2-controller-manager -c manager | grep "Successfully connected"

# 2. Scheduler shows model registration activity
kubectl -n seldon-system logs deploy/seldon-scheduler | grep "Register model"

# 3. Application team can then test inference endpoint
curl -H "Host: financial-predictor.local" \
     -H "seldon-model: financial-ab-test-experiment.experiment" \
     -H "Content-Type: application/json" \
     http://192.168.1.249/v2/models/baseline-predictor_1/infer \
     --data '{"inputs":[{"name":"input_data","shape":[1,10,35],"datatype":"FP32","data":[[...]]}]}'
```

## Impact Assessment

### Before Fix:
- ❌ All inference requests return 404
- ❌ A/B testing completely blocked
- ❌ Trade show demo using local fallback

### After Fix:
- ✅ Controller manager → scheduler connectivity restored
- ✅ Cross-namespace model discovery working
- ✅ Envoy routes configured automatically
- ✅ A/B testing endpoint responds with 200

## Dependencies

**None** - This is a self-contained change in `seldon-system` namespace.

**Network Policies:** Already configured to allow cross-namespace traffic.

**RBAC:** Controller manager already has cluster-wide permissions.

## Timeline

**Requested by:** 2025-07-12  
**Business Need:** Trade show demo preparation  
**Preferred Resolution:** Within 24 hours  

## Contact Information

**Application Team Lead:** Financial MLOps Team  
**Technical Context:** All infrastructure is working except controller manager → scheduler connectivity  
**Reference Document:** [seldon-v2-api-404-debugging.md](../troubleshooting/seldon-v2-api-404-debugging.md)

## Expert Validation

This fix pattern comes from Seldon expert runbook analysis. The issue is a common configuration pattern where the controller manager uses default DNS names that don't match actual service names in production deployments.

**Expert Quote:** *"Create/alias the Service name `seldon-scheduler` in `seldon-system` or set `SELDON_SCHEDULER_HOST` env-var on `seldon-v2-controller-manager`. Once the operator logs 'register model', 404s disappear."*

---

**Platform Team:** Please apply either Option A or Option B above. Option A is recommended as it maintains default DNS expectations for all components.