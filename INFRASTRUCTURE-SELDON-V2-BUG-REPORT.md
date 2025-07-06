# 🐛 **Infrastructure Team: Seldon Core v2.9 Critical Bug - Model Scheduling Failure**

## **Summary**
MLServer agent in Seldon Core v2.9.0 cannot connect to MLServer gRPC endpoint within the same pod, preventing all model scheduling and causing complete model serving failure.

## **Environment**
- **Cluster**: Production K3s cluster (5 nodes)
- **Seldon Core Version**: v2.9.0
- **MLServer Version**: 1.6.1
- **Architecture**: Dedicated SeldonRuntime per namespace
- **Network**: Cross-namespace communication working, NetworkPolicies resolved

## **Impact**
- ✅ **Infrastructure**: All components healthy (scheduler, envoy, modelgateway, runtime)
- ✅ **Agent Registration**: MLServer agent successfully subscribes to scheduler
- ❌ **Model Scheduling**: All models fail with "Failed to schedule model as no matching servers are available"
- ❌ **Production Impact**: Complete model serving outage

## **Root Cause Analysis**

### **Agent Connection Issue**
MLServer agent attempts to connect to MLServer gRPC endpoint using `0.0.0.0:9500` instead of `localhost:9500`, causing connection failures:

```
level=error msg="Waiting for Inference Server service to become ready" 
error="rpc error: code = Unavailable desc = connection error: desc = \"transport: Error while dialing: dial tcp 0.0.0.0:9500: connect: connection refused\""
```

### **Scheduler Behavior**
Despite agent registering successfully, scheduler reports zero capacity:
```
level=info msg="Received subscribe request from mlserver:0" func=Subscribe source=AgentServer
level=warning msg="Empty server for test-model:1 so ignoring event" func=updateServerModelStatus
level=warning msg="Failed to schedule model as no matching servers are available"
```

## **Suspected Fix Applied**
We believe **PR #6582** has been applied to resolve this issue:
**URL**: https://github.com/SeldonIO/seldon-core/pull/6582

### **Expected Fix Details**
The PR should add support for `SELDON_SERVER_HOST` environment variable in the agent CLI configuration:

```go
// Expected changes in scheduler/cmd/agent/cli/cli.go
const (
    envServerHost = "SELDON_SERVER_HOST"  // Added
    // ... existing constants
)

func updateFlagsFromEnv() {
    // ... existing calls
    maybeUpdateInferenceHost()  // Added
    // ... existing calls
}

func maybeUpdateInferenceHost() {  // Added function
    if isFlagPassed("inference-host") {
        return
    }
    
    inferenceHostFromEnv, found := getEnvString(envServerHost)
    if !found {
        return
    }
    
    log.Infof("Setting inference-host from %s to %s", envServerHost, inferenceHostFromEnv)
    InferenceHost = inferenceHostFromEnv
}
```

## **Current Configuration**

### **Environment Variables Set**
```yaml
env:
- name: SELDON_SERVER_HOST
  value: "localhost"
- name: SELDON_SERVER_GRPC_PORT
  value: "9500"
- name: MLSERVER_GRPC_ENDPOINT
  value: "localhost:9500"
```

### **Expected vs Actual Behavior**

**Expected** (with PR #6582):
```
level=info msg="Setting inference-host from SELDON_SERVER_HOST to localhost"
level=info msg="Inference server ready"
```

**Actual** (current):
```
# No "Setting inference-host" message appears
level=error msg="...dial tcp 0.0.0.0:9500: connect: connection refused"
```

## **Test Case**

### **Reproduction Steps**
1. Deploy SeldonRuntime with MLServer:
```yaml
apiVersion: mlops.seldon.io/v1alpha1
kind: SeldonRuntime
metadata:
  name: test-runtime
  namespace: test-namespace
spec:
  overrides:
  - name: mlserver
    replicas: 1
```

2. Deploy model:
```yaml
apiVersion: mlops.seldon.io/v1alpha1
kind: Model
metadata:
  name: test-model
  namespace: test-namespace
spec:
  storageUri: s3://bucket/model/
  server: mlserver
  secretName: ml-platform
```

3. Verify environment variables:
```bash
kubectl exec mlserver-0 -c agent -n test-namespace -- env | grep SELDON_SERVER_HOST
# Should return: SELDON_SERVER_HOST=localhost
```

### **Expected Results** (if PR #6582 applied)
```bash
# Agent logs should show:
kubectl logs mlserver-0 -c agent -n test-namespace | grep "Setting inference-host"
# Expected: level=info msg="Setting inference-host from SELDON_SERVER_HOST to localhost"

# Model should be ready:
kubectl get model test-model -n test-namespace
# Expected: test-model   True    1            1
```

### **Actual Results** (current failure)
```bash
# Agent logs show no inference-host setting:
kubectl logs mlserver-0 -c agent -n test-namespace | grep "Setting inference-host"
# Actual: (no output)

# Model scheduling fails:
kubectl get model test-model -n test-namespace  
# Actual: test-model   False

kubectl describe model test-model -n test-namespace | grep Message
# Actual: Message: Failed to schedule model as no matching servers are available
```

## **Verification Commands**

### **Check Agent Binary for Fix**
```bash
# Verify if PR #6582 code is in the agent binary
kubectl exec mlserver-0 -c agent -n test-namespace -- strings /bin/agent | grep "Setting inference-host"
# Expected: "Setting inference-host from %s to %s" (if fix applied)
```

### **Check MLServer Health**
```bash
# MLServer gRPC is healthy:
kubectl logs mlserver-0 -c mlserver -n test-namespace | grep "gRPC server running"
# Returns: gRPC server running on http://0.0.0.0:9500
```

### **Check Agent Registration**
```bash
# Agent registers successfully:
kubectl logs seldon-scheduler-0 -n test-namespace | grep "Received subscribe request from mlserver"
# Returns: level=info msg="Received subscribe request from mlserver:0"
```

## **Infrastructure Team Actions Needed**

### **1. Verify PR #6582 Application**
- [ ] Confirm PR #6582 has been merged into deployed v2.9.0 build
- [ ] Check if agent binary includes the `maybeUpdateInferenceHost()` function
- [ ] Verify environment variable processing includes `SELDON_SERVER_HOST`

### **2. If PR Not Applied**
- [ ] Apply PR #6582 to Seldon Core v2.9.0
- [ ] Rebuild agent container image
- [ ] Update agent image in cluster

### **3. If PR Already Applied**
- [ ] Investigate why `maybeUpdateInferenceHost()` function is not being called
- [ ] Check for build issues or missing dependencies
- [ ] Verify environment variable precedence and flag handling

## **Debugging Information Available**

### **Current Agent Image**
```bash
kubectl get pod mlserver-0 -n test-namespace -o jsonpath='{.spec.containers[?(@.name=="agent")].image}'
# Returns: docker.io/seldonio/seldon-agent:2.9.0
```

### **Complete Environment Variables**
```bash
kubectl exec mlserver-0 -c agent -n test-namespace -- env | grep SELDON_
# Full environment available for inspection
```

### **Full Agent Logs**
```bash
kubectl logs mlserver-0 -c agent -n test-namespace
# Complete startup and error logs available
```

## **Business Impact**

### **Severity**: **CRITICAL**
- **Production Outage**: All model serving unavailable
- **Migration Blocked**: Cannot complete Seldon v1→v2 migration
- **Enterprise Impact**: Financial ML platform offline

### **Timeline**
- **Issue Identified**: July 5, 2025
- **Root Cause**: Agent→MLServer intra-pod gRPC connectivity
- **Expected Fix**: PR #6582 implementation
- **Resolution Needed**: Immediate (production deployment waiting)

## **Related Resources**

### **Seldon Core Issues**
- **GitHub PR**: https://github.com/SeldonIO/seldon-core/pull/6582
- **Issue Type**: Agent configuration bug
- **Component**: MLServer agent CLI environment variable handling

### **Documentation**
- Complete infrastructure logs and configurations available
- Reproducible test case with minimal setup
- Environment variable verification commands provided

## **Success Criteria**

✅ **Agent logs show**: `"Setting inference-host from SELDON_SERVER_HOST to localhost"`  
✅ **Model deployment succeeds**: `kubectl get model test-model` shows `True`  
✅ **No gRPC connection errors**: Agent connects to `localhost:9500` successfully  
✅ **Scheduler capacity reporting**: No more "Empty server" warnings  

---

**Priority**: **P0 - Critical Production Issue**  
**Contact**: MLOps Engineering Team  
**Required**: Immediate infrastructure team investigation and resolution