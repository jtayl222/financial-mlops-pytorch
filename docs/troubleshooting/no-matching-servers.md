# Troubleshooting: "No Matching Servers Available"

## Symptom

Models fail to schedule with the error:

```
Reason: Failed to schedule model as no matching servers are available
Status: False
Type:   ModelReady
```

**Model Status:**
```bash
$ kubectl get models -n seldon-system
NAME                 READY   DESIRED REPLICAS   AVAILABLE REPLICAS   AGE
baseline-predictor   False                                           15m
```

**Model Spec:**
```yaml
spec:
  requirements:
    - mlflow
    - torch
    - numpy
    - scikit-learn
  server: mlserver
  storageUri: s3://mlflow-artifacts/29/models/m-d64ffcb77a684fbfa8597e439c920a07/artifacts/
```

**Server Capabilities:**
```bash
$ kubectl get pod/mlserver-0 -o yaml | grep SELDON_SERVER_CAPABILITIES -A1
- name: SELDON_SERVER_CAPABILITIES
  value: mlflow,torch,scikit-learn,numpy
```

## Root Cause Analysis

Despite the model requirements matching the server capabilities exactly, the Seldon scheduler cannot find an available server. This indicates a **scheduler connectivity issue** rather than a capability mismatch.

### Common Causes

1. **Scheduler-Agent Communication Failure**
   - MLServer agent cannot connect to the Seldon scheduler
   - Scheduler doesn't know about available servers
   - Network policies blocking communication

2. **DNS Resolution Issues**
   - Agent trying to connect to wrong scheduler IP
   - Cached DNS entries pointing to non-existent services
   - External DNS conflicts (see `seldon-network-issues.md`)

3. **Service Discovery Problems**
   - Server not registered with scheduler
   - Scheduler restart required to refresh server registry
   - Timing issues during startup

## Diagnostic Commands

### 1. Check Server Status
```bash
# Check if server is ready
kubectl get servers -n seldon-system

# Check server details
kubectl describe server mlserver -n seldon-system
```

### 2. Check Scheduler Logs
```bash
# Look for server registration messages
kubectl logs -n seldon-system seldon-scheduler-0 | grep -i "server"

# Check for connectivity errors
kubectl logs -n seldon-system seldon-scheduler-0 | grep -i "error"
```

### 3. Check Agent Connectivity
```bash
# Check if agent can connect to scheduler
kubectl logs -n seldon-system mlserver-0 -c agent | grep -i "scheduler"

# Look for connection errors
kubectl logs -n seldon-system mlserver-0 -c agent | grep -i "error"

# Common error patterns:
# "Scheduler not ready" - Agent cannot connect to scheduler
# "dial tcp X.X.X.X:9005: i/o timeout" - Network connectivity issue  
# "connect: operation not permitted" - Network policy blocking
```

### 4. Check Network Connectivity
```bash
# Test scheduler service resolution
kubectl run debug --rm -it --image=busybox -- nslookup seldon-scheduler.seldon-system

# Check if scheduler ports are accessible
kubectl get svc seldon-scheduler -n seldon-system
```

## Resolution Strategies

### Strategy 1: Fix Network Policy (Root Cause - Most Common)

**This is the most common root cause.** The MLServer agent cannot connect to the scheduler due to missing intra-namespace communication rules:

```bash
# Check current network policy
kubectl describe networkpolicy allow-seldon-scheduler-ingress -n seldon-system

# The policy MUST include podSelector: {} to allow same-namespace communication
```

**Required Network Policy Fix:**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-seldon-scheduler-ingress
  namespace: seldon-system
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: seldon-scheduler
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: seldon-system
    - podSelector: {}  # ← THIS LINE IS CRITICAL - allows intra-namespace communication
    ports:
    - protocol: TCP
      port: 9005  # Agent connection port
    # ... other ports
```

**Apply the fix and restart MLServer:**
```bash
# Apply network policy fix
kubectl apply -f k8s/base/network-policy.yaml

# Restart MLServer to pick up changes
kubectl delete pod mlserver-0 -n seldon-system

# Verify connectivity
kubectl logs -n seldon-system mlserver-0 -c agent | grep "Load model.*success"
```

### Strategy 2: Delete and Recreate Models (Timing Issues)

Models created before the server is fully ready often get stuck in failed state:

```bash
# Delete models and experiment
kubectl delete -f k8s/base/financial-predictor-ab-test.yaml

# Wait for server to be fully ready
kubectl wait --for=condition=Ready server/mlserver -n seldon-system --timeout=120s

# Recreate models with fresh scheduling
kubectl apply -f k8s/base/financial-predictor-ab-test.yaml
```

### Strategy 3: Restart Components (Secondary Fix)

If network policy and model recreation don't work, try component restart:

```bash
# Restart the MLServer to refresh scheduler connection
kubectl delete pod mlserver-0 -n seldon-system

# If that doesn't work, restart the scheduler
kubectl delete pod seldon-scheduler-0 -n seldon-system

# Wait for pods to restart and check status
kubectl get pods -n seldon-system
```

### Strategy 4: Check Network Policies (Verification)

Ensure communication is allowed between components:

```bash
# Check existing network policies
kubectl get networkpolicies -n seldon-system

# Verify scheduler ingress policy exists
kubectl describe networkpolicy allow-seldon-scheduler-ingress -n seldon-system
```

If missing, apply the network policy:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-seldon-scheduler-ingress
  namespace: seldon-system
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: seldon-scheduler
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: seldon-system
    - podSelector: {}
    ports:
    - protocol: TCP
      port: 9002
    - protocol: TCP
      port: 9004
    - protocol: TCP
      port: 9005
    - protocol: TCP
      port: 9008
```

### Strategy 5: Verify Service Discovery

Check if the scheduler can see the server:

```bash
# Check scheduler's view of available servers
kubectl logs -n seldon-system seldon-scheduler-0 | grep "Server notification"

# Should see messages like:
# "Server notification mlserver expectedReplicas 1 shared false"

# Check for scheduling failures
kubectl logs -n seldon-system seldon-scheduler-0 | grep -E "(Failed to schedule|Empty server)"

# Common patterns indicating the issue:
# "Failed to schedule model as no matching servers are available"
# "Empty server for model-name so ignoring event"
```

### Strategy 6: DNS Resolution Fix

If DNS issues persist (especially with external domain conflicts):

```bash
# Check if external DNS is interfering
nslookup seldon-scheduler.seldon-system

# If resolving to external IP, restart CoreDNS
kubectl delete pod -l k8s-app=kube-dns -n kube-system
```

### Strategy 6: Capability Verification

Verify exact capability matching:

```bash
# Check model requirements
kubectl get model baseline-predictor -n seldon-system -o jsonpath='{.spec.requirements}'

# Check server capabilities
kubectl get server mlserver -n seldon-system -o jsonpath='{.spec.capabilities}'

# Ensure exact match (order doesn't matter, but names must be identical)
```

## Prevention Guidelines

### 1. Proper Startup Sequence
```bash
# Always deploy in this order:
1. SeldonRuntime (creates scheduler)
2. Server resources (creates MLServer)
3. Model resources (requires both above)
```

### 2. Health Checks
```bash
# Verify scheduler is ready before deploying models
kubectl wait --for=condition=Ready pod/seldon-scheduler-0 -n seldon-system --timeout=120s

# Verify server is ready
kubectl wait --for=condition=Ready server/mlserver -n seldon-system --timeout=120s
```

### 3. Monitoring
Set up alerts for:
- Server registration failures
- Scheduler connectivity issues
- Model scheduling failures

## Success Indicators

When resolved, you should see:

```bash
# Models become ready
$ kubectl get models -n seldon-system
NAME                 READY   DESIRED REPLICAS   AVAILABLE REPLICAS   AGE
baseline-predictor   True    1                  1                    20m

# Server shows loaded models
$ kubectl get servers -n seldon-system  
NAME       READY   REPLICAS   LOADED MODELS   AGE
mlserver   True    1          2               25m

# Scheduler logs show successful registration
$ kubectl logs seldon-scheduler-0 -n seldon-system | grep "Server notification"
"Server notification mlserver expectedReplicas 1 shared false"
```

## Related Issues

- [Seldon Architecture Confusion](seldon-architecture-confusion.md) - Choosing the right pattern
- [Seldon Network Issues](../seldon-network-issues.md) - DNS conflicts and connectivity
- [Model Loading Issues](seldon-model-loading.md) - MLflow and storage problems

## Common Mistakes

1. **Mixed Architecture**: Using centralized + distributed patterns
2. **Wrong Server Reference**: Pointing to servers in other namespaces
3. **Missing NetworkPolicy**: Blocking scheduler-agent communication
4. **Capability Mismatch**: Typos in capability names (e.g., "sklearn" vs "scikit-learn")
5. **Deployment Order**: Creating models before scheduler is ready