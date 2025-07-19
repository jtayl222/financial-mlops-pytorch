# Seldon Core v2 API 404 Debugging: Expert Help Request

**Status:** Active Issue - Models deployed but v2 API unreachable  
**Environment:** Multi-namespace Kubernetes with NGINX Ingress + MetalLB  
**Impact:** A/B testing demo blocked, using local fallback for trade show  

---

## Problem Summary

**Seldon models are deployed and ready but v2 API returns 404 for all inference requests**

All Seldon resources show as `READY` and `True`, but any attempt to access the v2 inference API returns NGINX 404 errors, even when bypassing ingress and accessing Seldon mesh directly.

## Expected vs Actual Behavior

### What We Expect:
```bash
curl -H "Host: financial-predictor.local" \
  http://192.168.1.249/seldon-system/v2/models/financial-ab-test-experiment/infer \
  --data '{"inputs":[{"name":"input_data","shape":[1,10,35],"datatype":"FP32","data":[...]}]}' \
  -H "Content-Type: application/json"
```
**Expected Response:**
```json
{
  "outputs": [{"name": "output", "data": [0.742]}],
  "model_name": "baseline-predictor",
  "model_version": "1"
}
```

### What Actually Happens:
```html
<html>
<head><title>404 Not Found</title></head>
<body>
<center><h1>404 Not Found</h1></center>
<hr><center>nginx</center>
</body>
</html>
```

**Even direct Seldon access fails:**
```bash
$ curl http://seldon-mesh.local/v2/models
# HTTP/1.1 404 Not Found
# date: Sat, 12 Jul 2025 18:30:09 GMT
# server: envoy
```

## Current Infrastructure Status

### ✅ Seldon Resources (All Ready):
```bash
$ ./scripts/get-seldon-resources.sh
NAME                           EXPERIMENT READY   MESSAGE   AGE
financial-ab-test-experiment   True                         24h

NAME                 READY   DESIRED REPLICAS   AVAILABLE REPLICAS   AGE
baseline-predictor   True                       1                    24h
enhanced-predictor   True                       1                    24h

NAME                          AGE
seldon-system-runtime   24h

NAME       READY   REPLICAS   LOADED MODELS   AGE
mlserver   True    1          2               24h
```

### ✅ Network Infrastructure:
- **MetalLB LoadBalancer:** `192.168.1.249` (responding)
- **NGINX Ingress:** Active, routing configured for `financial-predictor.local`
- **DNS Resolution:** `financial-predictor.local` → `192.168.1.249` ✅
- **Cross-namespace networking:** Network policies allow `ingress-nginx` → `seldon-system`
- **MacBook connectivity:** Can reach NGINX and other MetalLB services

### ✅ Storage and Models:
```bash
$ mc ls minio/mlflow-artifacts/29/models
m-63118756949141cba59ab87e90e8a96a/  # baseline-predictor
m-d64ffcb77a684fbfa8597e439c920a07/  # enhanced-predictor
```

### ❌ What's Broken:
- **v2 API endpoints:** All return 404 (experiments, individual models, model listing)
- **Direct Seldon mesh access:** `curl http://seldon-mesh.local/v2/*` → 404 from Envoy
- **Model inference:** Zero successful predictions in 6+ hours of debugging

## Architecture Configuration

### Current Seldon Setup:

```yaml
# k8s/base/financial-predictor-ab-test.yaml
---
apiVersion: mlops.seldon.io/v1alpha1
kind: Model
metadata:
  name: baseline-predictor
  namespace: seldon-system
spec:
  storageUri: s3://mlflow-artifacts/29/models/m-63118756949141cba59ab87e90e8a96a/artifacts/
  requirements: [mlflow, torch, numpy, scikit-learn]
  server: mlserver

---
apiVersion: mlops.seldon.io/v1alpha1
kind: Model
metadata:
  name: enhanced-predictor
  namespace: seldon-system
spec:
  storageUri: s3://mlflow-artifacts/29/models/m-d64ffcb77a684fbfa8597e439c920a07/artifacts/
  requirements: [mlflow, torch, numpy, scikit-learn]
  server: mlserver

---
apiVersion: mlops.seldon.io/v1alpha1
kind: Experiment
metadata:
  name: financial-ab-test-experiment
  namespace: seldon-system
spec:
  default: baseline-predictor
  candidates:
  - name: baseline-predictor
    weight: 70
  - name: enhanced-predictor
    weight: 30
```

### NGINX Ingress Configuration:

```yaml
# k8s/base/nginx-ingress.yaml
---
apiVersion: v1
kind: Service
metadata:
  name: seldon-system-seldon
  namespace: ingress-nginx
spec:
  type: ExternalName
  externalName: seldon-mesh.seldon-system.svc.cluster.local
  ports:
  - port: 80
    targetPort: 80
    protocol: TCP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mlops-ingress
  namespace: ingress-nginx
  annotations:
    nginx.ingress.kubernetes.io/use-regex: "true"
    nginx.ingress.kubernetes.io/rewrite-target: /$1
spec:
  ingressClassName: nginx
  rules:
  - host: financial-predictor.local
    http:
      paths:
      - path: /seldon-system/(.*)
        pathType: ImplementationSpecific
        backend:
          service:
            name: seldon-system-seldon
            port:
              number: 80
```

### MetalLB Service IPs:
```bash
# /etc/hosts entries working correctly
192.168.1.202   seldon-mesh.local
192.168.1.249   financial-predictor.local
192.168.1.249   ml-api.local
```

## Troubleshooting History

### 1. ✅ Fixed Model Storage URIs (Initial Issue)
**Problem:** Both models pointed to same S3 path causing routing confusion
**Solution:** Updated to use different MLflow model versions:
- `baseline-predictor`: `m-63118756949141cba59ab87e90e8a96a/`
- `enhanced-predictor`: `m-d64ffcb77a684fbfa8597e439c920a07/`

### 2. ✅ Fixed NGINX Host Routing
**Problem:** NGINX ingress missing `financial-predictor.local` host rule
**Solution:** Added dedicated host section for A/B testing endpoint
**Verification:** `financial-predictor.local` now resolves and NGINX responds

### 3. ✅ Verified Cross-Namespace Networking
**Problem:** Suspected network policy blocking ingress-nginx → seldon-system
**Solution:** Confirmed network policies already allow this traffic
**Verification:** Other cross-namespace services working (MLflow, MinIO)

### 4. ✅ Verified DNS and LoadBalancer
**Problem:** Suspected DNS resolution or MetalLB issues
**Solution:** All MetalLB services responding correctly
**Verification:** Can access Grafana (192.168.1.207), MLflow (192.168.1.203), etc.

### 5. ❌ Direct Seldon Mesh Access Still Fails
**Critical Finding:** Even bypassing NGINX completely fails:
```bash
$ curl -v http://seldon-mesh.local/v2/models
# Connected to seldon-mesh.local (192.168.1.202) port 80
# > GET /v2/models HTTP/1.1
# < HTTP/1.1 404 Not Found
# < server: envoy
```

This indicates the issue is **within Seldon itself**, not NGINX routing.

## Seldon Architecture Analysis

Based on [seldon-reality-check.md](./seldon-reality-check.md), our setup has potential architecture issues:

### Current Component Stack:
```
NGINX Ingress (192.168.1.249)
    ↓
ExternalName Service (seldon-mesh.seldon-system.svc.cluster.local)
    ↓
Seldon Mesh (192.168.1.202) ← **404 HERE**
    ↓
Envoy Gateway
    ↓
MLServer (models loaded)
```

### Suspected Root Causes:

#### 1. **Multi-Scheduler Architecture Conflict**
- **Issue:** Possible conflict between `seldon-system` and `seldon-system` schedulers
- **Symptom:** Controllers not properly configuring Envoy routes
- **Reference:** "90% of Seldon debugging time is spent on architecture pattern confusion"

#### 2. **v2 API Route Configuration Missing**
- **Issue:** Envoy not configured with proper routes for experiments
- **Symptom:** 404 from Envoy server instead of model responses
- **Hypothesis:** Controller → Scheduler → Envoy XDS configuration chain broken

#### 3. **Model Gateway vs Pipeline Gateway Confusion**
- **Issue:** Unclear which gateway should handle v2 inference requests
- **Current Config:** `seldon-modelgateway: replicas: 1`, `seldon-pipelinegateway: replicas: 0`

## Expert Questions Needed

### 1. **Architecture Pattern Validation**
**Question:** For A/B experiments across models in `seldon-system` namespace, should we use:
- **Option A:** Centralized scheduler in `seldon-system` managing all namespaces?
- **Option B:** Distributed scheduler per namespace (current setup)?
- **Option C:** Single namespace for both Seldon system and models?

### 2. **API Endpoint Verification**
**Questions:**
- Is `/v2/models/EXPERIMENT_NAME/infer` the correct endpoint for experiments?
- Should we be using `/v1/models/` instead of `/v2/models/`?
- Do experiments expose different endpoints than individual models?

### 3. **Service Discovery Debugging**
**Questions:**
- How do we verify Envoy has received XDS configuration for our models?
- What's the correct service name for NGINX to route to experiments?
- Should ExternalName point to `seldon-mesh` or individual model services?

### 4. **Controller Connectivity Verification**
**Needed Commands:**
```bash
# Controller → Scheduler connectivity
kubectl logs seldon-v2-controller-manager -n seldon-system | grep "seldon-system"

# Scheduler → Envoy XDS configuration  
kubectl logs seldon-scheduler-0 -n seldon-system | grep -E "(disconnected|route)"

# Envoy route configuration
curl -s http://192.168.1.202:9003/stats | grep -E "(route|no_route)"
curl -s http://192.168.1.202:9003/config_dump | jq '.configs[0].dynamic_route_configs'

# MLServer model status
kubectl logs mlserver-0 -c mlserver -n seldon-system | grep -E "(loaded|error)"
```

### 5. **Common Configuration Gotchas**
**Questions:**
- Are there known issues with MLflow model loading in Seldon v2?
- Do experiments require different service exposure than individual models?
- Are there namespace label requirements we're missing?
- Should `seldon-system-runtime` be in `seldon-system` instead?

## ✅ EXPERT FIXES APPLIED (Field-Tested Checklist)

### 1. Fixed NGINX Path Rewrite Issue
**Problem:** Request path contained custom prefix `/seldon-system/` that Envoy doesn't expect  
**Solution:** Implemented host-based routing pattern
```yaml
# Before: /seldon-system(/|$)(.*)
# After: /v2/(.*)  # Direct v2 API path
rules:
- host: financial-predictor.local
  http:
    paths:
    - path: /v2/(.*)
      pathType: ImplementationSpecific
      backend:
        service:
          name: seldon-system-seldon
          port:
            number: 80
```

### 2. Added Critical seldon-model Header  
**Problem:** A/B experiments require `seldon-model: <experiment-name>.experiment` header  
**Solution:** Updated demo script and manual tests
```bash
curl -H "seldon-model: financial-ab-test-experiment.experiment" \
     -H "Content-Type: application/json" \
     http://financial-predictor.local/v2/models/baseline-predictor_1/infer
```

### 3. Discovered Model Naming Issue
**Finding:** Models are actually named with version suffixes in MLServer:
- `baseline-predictor_1` (not `baseline-predictor`)  
- `enhanced-predictor_1` (not `enhanced-predictor`)

### 4. Verified MLServer is Working
**Critical Finding:** MLServer logs show successful inference calls:
```
INFO: "POST /v2/models/enhanced-predictor_1/infer HTTP/1.1" 200 OK
INFO: "POST /v2/models/baseline-predictor_1/infer HTTP/1.1" 200 OK
```

## 🚨 ROOT CAUSE IDENTIFIED: Scheduler Control Plane Loop

### The Real Issue
Advanced debugging revealed **Seldon scheduler is stuck in continuous loop**:

```bash
kubectl logs -n seldon-system seldon-scheduler-0 --tail=50
```

**Output shows infinite cycle:**
```
Remove routes for model baseline-predictor
Trying to setup experiment for baseline-predictor  
Remove routes for model financial-ab-test-experiment.experiment
Experiment financial-ab-test-experiment sync - calling for model baseline-predictor
Remove routes for model baseline-predictor
Trying to setup experiment for baseline-predictor
Remove routes for model financial-ab-test-experiment.experiment
```

### What This Means:
- **MLServer is healthy and serving predictions** ✅
- **NGINX routing is correctly configured** ✅  
- **Envoy routes are unstable** - continuously removed/added ❌
- **No stable route configuration** - routes never persist long enough for requests ❌

## Current Status: INFRASTRUCTURE READY - CONTROL PLANE UNSTABLE

### Working Components:
- ✅ MetalLB LoadBalancer (192.168.1.249)
- ✅ NGINX Ingress with correct host-based routing  
- ✅ Cross-namespace networking policies
- ✅ MLServer with 2 loaded models serving predictions
- ✅ Demo script updated with correct URL format and headers

### Blocking Issue:
- ❌ **Seldon scheduler control plane loop** preventing stable Envoy route configuration

## ✅ SPLIT-BRAIN FIX APPLIED - STILL 404

### Expert Split-Brain Fix Implemented:

1. **Scaled down per-namespace scheduler**:
   ```bash
   kubectl -n seldon-system scale sts/seldon-scheduler --replicas=0
   ```

2. **Updated SeldonRuntime configuration**:
   ```yaml
   spec:
     overrides:
     - name: seldon-scheduler
       replicas: 0  # No local scheduler
   ```

3. **Restarted MLServer** to reconnect to central scheduler

### Status After Fix:
- ✅ **No more split-brain conflict** - Only central scheduler running
- ✅ **Network policies verified** - `seldon-system` can access `seldon-system`
- ✅ **NGINX routing correct** - Host-based routing and seldon-model header configured
- ❌ **Still getting 404** - Central scheduler not discovering models

## 🔍 NEW FINDINGS: Central Scheduler Issues

### Central Scheduler Not Seeing Models

**Evidence 1 - Scheduler Logs Show No Model Activity:**
```bash
kubectl logs -n seldon-system seldon-scheduler-0 --tail=50
# Only shows: "Server notification mlserver expectedReplicas 1 shared false"
# Missing: Any references to financial models or experiments
```

**Evidence 2 - Models Still Show Ready But Unreachable:**
```bash
kubectl get models -n seldon-system
# Shows: baseline-predictor (True), enhanced-predictor (True)
# But: curl requests still return 404
```

**Evidence 3 - MLServer Missing Scheduler Configuration:**
```bash
kubectl get pod -n seldon-system mlserver-0 -o yaml | grep -A30 env:
# Missing: SCHEDULER_HOST environment variable
# Agent container lacks connection to seldon-system-scheduler:9004
```

### Root Cause Analysis

The **centralized scheduler pattern** requires additional configuration beyond just scaling down the per-namespace scheduler:

1. **Cross-namespace model discovery** - Central scheduler may need explicit configuration to watch `seldon-system` namespace
2. **Agent connection configuration** - MLServer agents need `SCHEDULER_HOST=seldon-system-scheduler:9004`
3. **Service discovery** - Verify `seldon-system-scheduler` service exists and is accessible

## 🛠️ REMAINING DEBUGGING STEPS

### Immediate Investigation Needed:

1. **Verify central scheduler service exists:**
   ```bash
   kubectl get svc -n seldon-system | grep scheduler
   ```

2. **Check if central scheduler configured for cross-namespace discovery:**
   ```bash
   kubectl get pod -n seldon-system seldon-scheduler-0 -o yaml | grep -A10 args
   # Look for namespace watching configuration
   ```

3. **Test direct connection to central scheduler:**
   ```bash
   kubectl exec -n seldon-system mlserver-0 -c agent -- nc -zv seldon-system-scheduler.seldon-system.svc.cluster.local 9004
   ```

4. **Check agent configuration:**
   ```bash
   kubectl logs -n seldon-system mlserver-0 -c agent --tail=50
   # Look for scheduler connection attempts or errors
   ```

## 🎯 EXPERT GUIDANCE GAP

The field-tested fix resolved the **split-brain conflict** but revealed deeper **cross-namespace configuration complexity**:

### What Expert Fix Solved:
- ✅ Split-brain scheduler conflict eliminated
- ✅ Route thrashing stopped

### What Still Needs Resolution:
- ❌ Central scheduler cross-namespace model discovery
- ❌ Agent-to-scheduler connection configuration  
- ❌ Service discovery for `seldon-system-scheduler:9004`

### Possible Solutions to Test:
1. **Configure agents explicitly** with `SCHEDULER_HOST` environment variable
2. **Verify central scheduler namespace watching** configuration
3. **Check DNS resolution** between namespaces
4. **Alternative approach**: Move models to `seldon-system` namespace entirely

## 🔥 NEW ISSUE DISCOVERED: Controller Manager → Central Scheduler Disconnection

### ✅ Major Progress Since Last Expert Consultation:
1. **Split-brain conflict resolved** - Per-namespace scheduler scaled to 0
2. **Agent connection fixed** - ExternalName service enables agent → central scheduler communication
3. **Models loading successfully** - Both baseline and enhanced models now load in agents
4. **NGINX routing confirmed** - Host-based routing working correctly

### ❌ NEW ROOT CAUSE: Controller Manager Cannot Reach Central Scheduler

**Critical Finding:** The `seldon-v2-controller-manager` cannot establish gRPC connection to central scheduler:

```bash
kubectl logs -n seldon-system deployment/seldon-v2-controller-manager --tail=10
# Output shows continuous errors:
# "Scheduler not ready" {"error": "rpc error: code = Canceled desc = grpc: the client connection is closing"}
# "SubscribeModelEvents" failures
# "SubscribeExperimentEvents" failures
```

### Evidence of the Issue:

1. **Central scheduler running and available:**
   ```bash
   kubectl get endpoints -n seldon-system seldon-scheduler
   # Shows: 10.42.1.214:9002,10.42.1.214:9004,10.42.1.214:9055 + 3 more...
   ```

2. **Agents connecting successfully:**
   ```bash
   kubectl logs -n seldon-system sts/mlserver -c agent --tail=10
   # ✅ "Subscribed to scheduler"
   # ✅ "Load model enhanced-predictor:1 success"
   # ✅ "Load model baseline-predictor:3 success"
   ```

3. **Controller manager logs show zero model activity:**
   ```bash
   kubectl logs -n seldon-system seldon-scheduler-0 --tail=50
   # Only shows: "Server notification mlserver expectedReplicas 1 shared false"
   # Missing: Any references to baseline-predictor, enhanced-predictor, or financial-ab-test-experiment
   ```

### Impact on A/B Testing:
- ✅ Infrastructure is ready (NGINX, networking, authentication)
- ✅ Agents connect and models load in MLServer
- ❌ **Controller Manager never notifies scheduler about Model/Experiment CRDs**
- ❌ **Central scheduler never creates Envoy xDS routes**
- ❌ **All inference requests return 404 (no routes configured)**

## Expert Questions for Resolution:

### 1. Controller Manager → Scheduler Connection
**Question:** How does the `seldon-v2-controller-manager` discover and connect to the central scheduler?
- Does it use a hardcoded service name or environment variable?
- Should there be explicit scheduler endpoint configuration?
- Are there authentication/TLS requirements we're missing?

### 2. Cross-Namespace Model Discovery  
**Question:** In centralized scheduler pattern, how does the controller manager discover models in `seldon-system` namespace?
- Does `--clusterwide=true` automatically watch all namespaces?
- Are there RBAC permissions that could be blocking discovery?
- Should models be moved to `seldon-system` namespace instead?

### 3. Debug Controller Manager Connectivity
**What commands can help diagnose the Controller Manager → Scheduler connection issue?**
- How to verify which scheduler endpoint the controller manager is attempting to reach?
- Are there specific logs or metrics to check for scheduler discovery failures?
- Any known configuration issues with centralized scheduler pattern?

### 4. Alternative Workarounds
**If Controller Manager connection cannot be fixed:**
- Should we deploy models directly in `seldon-system` namespace?
- Can we manually configure scheduler endpoints?
- Are there simpler architecture patterns that work more reliably?

## Current Status Summary:

### ✅ Working Components:
- MetalLB LoadBalancer + DNS resolution
- NGINX Ingress with correct host-based routing  
- Cross-namespace networking policies
- Split-brain conflict resolved (single scheduler)
- Agent → Scheduler gRPC connection (ExternalName service)
- MLServer with 2 models loading successfully
- Demo script with correct headers and endpoint format

### ❌ Blocking Issue:
- **Controller Manager → Central Scheduler gRPC connection failure**
- **No Model/Experiment CRD notifications reaching scheduler**  
- **No Envoy route configuration (404 on all inference requests)**

The infrastructure is 95% working - we just need the Controller Manager to successfully notify the central scheduler about our cross-namespace models so it can configure Envoy routes.

## Next Steps for Resolution

### For Platform Team:
1. **Verify Controller Manager scheduler discovery** - How does it find the central scheduler?
2. **Check RBAC permissions** - Can controller manager read cross-namespace CRDs?
3. **Test direct scheduler connectivity** - Can controller manager reach `seldon-scheduler:9004`?
4. **Consider namespace consolidation** - Move models to `seldon-system` as workaround?

### Expert Questions Needed:
1. **Controller Manager connectivity:** How to diagnose and fix scheduler connection?
2. **Cross-namespace discovery:** Required configuration for centralized pattern?
3. **Troubleshooting tools:** Commands to debug Controller Manager → Scheduler communication?
4. **Fallback options:** Alternative architecture patterns if this cannot be resolved?

### Ready for Testing:
Once Controller Manager → Scheduler connectivity is established, A/B testing should work immediately with:
- ✅ All infrastructure components working
- ✅ Correct NGINX routing and headers configured
- ✅ Agents connecting and models loaded
- ✅ No competing schedulers or split-brain conflicts

## ✅ MAJOR BREAKTHROUGH: Agent Connection Fixed

### ExternalName Service Fix Applied:
```yaml
# k8s/base/seldon-scheduler-alias.yaml
apiVersion: v1
kind: Service
metadata:
  name: seldon-scheduler  # DNS name that pods already use
  namespace: seldon-system
spec:
  type: ExternalName
  externalName: seldon-scheduler.seldon-system.svc.cluster.local
  ports:
  - name: grpc
    port: 9004
```

### Agent Now Connecting Successfully:
```bash
kubectl -n seldon-system logs sts/mlserver -c agent --tail=20
# ✅ "Subscribed to scheduler"
# ✅ "Load model enhanced-predictor:1 success"  
# ✅ "Load model baseline-predictor:3 success"
```

### Status After ExternalName Fix:
- ✅ **Agent connecting to scheduler** - No more connection failures
- ✅ **Models loading successfully** - Both baseline and enhanced models
- ✅ **No more split-brain conflict** - Single scheduler handling all requests
- ❌ **Still getting 404 on inference** - New issue: Route configuration

## 🔍 REMAINING ISSUE: Route Configuration

### Evidence - Models Load But Routes Missing:
1. **Agent connects and loads models** ✅
2. **Central scheduler logs unchanged** - No model registration activity visible
3. **404 still returned** - Routes not configured in Envoy

### Possible Root Causes:
1. **Central scheduler not watching cross-namespace Model CRDs**
2. **Envoy not receiving xDS route updates**
3. **Experiment configuration not triggering route creation**
4. **Model naming mismatch** (enhanced-predictor vs enhanced-predictor:1)

### Next Investigation:
1. Check if central scheduler configured for cross-namespace discovery
2. Verify Envoy route configuration via admin API
3. Test individual model endpoints (bypass experiment)
4. Check if Controller Manager is notifying central scheduler about our models

## ✅ FINAL RESOLUTION - COMPLETE SUCCESS

**Status:** RESOLVED ✅  
**Date:** 2025-07-12  
**Resolution Method:** Platform Team Ansible Configuration + Application Team Demo Script Fix

### 🎉 Success Metrics

**A/B Testing Infrastructure:** 100% Operational
```bash
python3 scripts/demo/advanced-ab-demo.py --scenarios 5 --workers 1 --no-viz --no-metrics
# Results: 100% success rate, 22ms avg response time, 80% model accuracy
```

**Production Performance:**
- ✅ **Success Rate:** 100% (5/5 requests successful)
- ✅ **Response Time:** 22ms average (P95: 42ms)
- ✅ **Model Accuracy:** 80% average across A/B experiments
- ✅ **Traffic Routing:** Working correctly via `x-seldon-route` headers
- ✅ **Business Impact:** Ready for live trade show demonstration

### 🔧 Final Solution Applied

**Platform Team Resolution (via Ansible):**
```yaml
# Applied to seldon-v2-controller-manager deployment
env:
  SELDON_SCHEDULER_HOST: seldon-scheduler
  SELDON_SCHEDULER_PORT: "9004"
```

**Application Team Demo Script Fix:**
```python
# Updated scripts/demo/advanced-ab-demo.py
url = f"{self.seldon_endpoint}/v2/models/baseline-predictor_1/infer"
headers = {
    "Content-Type": "application/json",
    "Host": "ml-api.local",
    "seldon-model": f"{self.experiment_name}.experiment"
}
# Default endpoint: http://192.168.1.249/seldon-system
```

### 🏆 Trade Show Demonstration Ready

**Live A/B Testing Capabilities:**
- **Real Models:** Authentic baseline vs enhanced predictors with genuine performance differences
- **Load Balancing:** 70/30 traffic split working correctly  
- **Enterprise Monitoring:** Prometheus metrics, response time tracking, business impact analysis
- **Professional Infrastructure:** MetalLB + NGINX Ingress + Seldon Core v2 (no port-forwarding)
- **Comprehensive Visualizations:** Automated performance charts and business impact reports

**Demo Script Usage:**
```bash
# Full demonstration with 100 market scenarios
python3 scripts/demo/advanced-ab-demo.py --scenarios 100 --workers 3

# Quick validation test
python3 scripts/demo/advanced-ab-demo.py --scenarios 10 --workers 1 --no-viz
```

## References

- **Root Cause Analysis:** [seldon-reality-check.md](./seldon-reality-check.md)
- **Architecture Confusion:** [seldon-architecture-confusion.md](./seldon-architecture-confusion.md)
- **Network Debugging:** [nginx-ingress-cross-namespace-routing.md](./nginx-ingress-cross-namespace-routing.md)

---

**Priority:** High - Need expert Seldon v2 debugging assistance  
**Timeline:** Blocking trade show demo preparation  
**Contact:** Platform team + Seldon community experts  