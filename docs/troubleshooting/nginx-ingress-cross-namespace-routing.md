# NGINX Ingress Cross-Namespace Routing Issue

## Problem Statement
NGINX Ingress Controller is installed and operational, but ExternalName services for cross-namespace routing are not resolving properly, resulting in 404 errors for seldon-system namespace services.

## Current State

### ✅ Working Components
- NGINX Ingress Controller: `192.168.1.249` (responds to requests)
- Backend services: `seldon-mesh.seldon-system` accessible via port-forward
- Individual models: `baseline-predictor`, `enhanced-predictor` work via localhost:8082
- Ingress responds: Returns nginx 404 (not connection refused)

### ❌ Failing Components
- Cross-namespace routing: `ml-api.local/seldon-system/*` → 404
- ExternalName service resolution: May not be resolving DNS correctly
- Path rewriting: `/$2` rewrite rule may have issues

## Architecture Overview

```
External Request → NGINX Ingress (192.168.1.249) → ExternalName Service → seldon-system namespace
                                 ↓
                          ml-api.local/seldon-system/v2/models
                                 ↓
                          seldon-system-seldon (ExternalName)
                                 ↓
                          seldon-mesh.seldon-system.svc.cluster.local
```

## Debugging Steps Performed

### 1. Basic Connectivity Tests
```bash
# NGINX Ingress responds
curl http://ml-api.local/seldon-system/v2/models
# Result: nginx 404 (not connection refused) ✅

# Backend works via port-forward  
kubectl port-forward -n seldon-system svc/seldon-mesh 8082:80
curl http://localhost:8082/v2/models/baseline-predictor
# Result: JSON model metadata ✅
```

### 2. Ingress Configuration Verification
```bash
kubectl get ingress -n ingress-nginx mlops-ingress -o yaml
```

**Current Configuration:**
```yaml
spec:
  rules:
  - host: ml-api.local
    http:
      paths:
      - backend:
          service:
            name: seldon-system-seldon
            port:
              number: 80
        path: /seldon-system(/|$)(.*)
        pathType: Prefix
```

**Annotations:**
```yaml
nginx.ingress.kubernetes.io/rewrite-target: /$2
nginx.ingress.kubernetes.io/ssl-redirect: "false"
nginx.ingress.kubernetes.io/cors-allow-origin: "*"
```

### 3. ExternalName Service Verification
```bash
kubectl get svc -n ingress-nginx seldon-system-seldon
```

**Output:**
```
NAME                         TYPE           EXTERNAL-IP
seldon-system-seldon   ExternalName   seldon-mesh.seldon-system.svc.cluster.local
```

## Potential Root Causes

### Theory 1: DNS Resolution Issue
**Hypothesis**: ExternalName service cannot resolve `seldon-mesh.seldon-system.svc.cluster.local`

**Test Commands Needed:**
```bash
# Test DNS resolution from ingress-nginx namespace
kubectl run debug-pod -n ingress-nginx --image=busybox --rm -it -- sh
nslookup seldon-mesh.seldon-system.svc.cluster.local

# Test from a pod in the ingress-nginx namespace
kubectl exec -n ingress-nginx deployment/ingress-nginx-controller -- nslookup seldon-mesh.seldon-system.svc.cluster.local
```

### Theory 2: Path Rewriting Issue
**Hypothesis**: The rewrite rule `/$2` is not correctly transforming paths

**Current Flow:**
```
/seldon-system/v2/models/baseline-predictor
            ↓ regex: /seldon-system(/|$)(.*)
            ↓ rewrite: /$2  
            ↓ result: /v2/models/baseline-predictor
```

**Expected Backend Path:** `/v2/models/baseline-predictor` ✅

**Test Commands Needed:**
```bash
# Test without rewrite to see raw path forwarding
# Temporarily modify ingress to remove rewrite-target annotation
kubectl annotate ingress -n ingress-nginx mlops-ingress nginx.ingress.kubernetes.io/rewrite-target-
```

### Theory 3: Network Policies Blocking Cross-Namespace Traffic
**Hypothesis**: Network policies prevent ingress-nginx → seldon-system communication

**Test Commands Needed:**
```bash
# Check network policies in both namespaces
kubectl get networkpolicy -n ingress-nginx
kubectl get networkpolicy -n seldon-system

# Test with network policies temporarily disabled
kubectl delete networkpolicy -n seldon-system --all
```

### Theory 4: Service Port Mismatch
**Hypothesis**: Port mapping is incorrect between ExternalName and target service

**Current Configuration:**
- ExternalName service: port 80
- Target service: `seldon-mesh.seldon-system` port ?

**Test Commands Needed:**
```bash
# Verify target service ports
kubectl get svc -n seldon-system seldon-mesh
kubectl describe svc -n seldon-system seldon-mesh
```

## Systematic Debugging Plan

### Phase 1: DNS Resolution Verification
1. Create debug pod in ingress-nginx namespace
2. Test DNS resolution of target service
3. Verify CoreDNS configuration for cross-namespace resolution

### Phase 2: Path Routing Verification  
1. Enable ingress controller debug logging
2. Test with simplified ingress rules
3. Verify path rewriting behavior

### Phase 3: Network Policy Testing
1. Temporarily disable network policies
2. Test direct pod-to-pod communication
3. Identify blocking policies

### Phase 4: Service Discovery Testing
1. Verify target service configuration
2. Test alternative routing methods (Endpoints vs ExternalName)
3. Compare with working namespace routing

## Platform Team Action Items

### Immediate Investigation Required:
1. **DNS Resolution**: Verify CoreDNS cross-namespace service discovery
2. **Network Policies**: Check if seldon-system policies block ingress traffic
3. **NGINX Ingress Logs**: Enable debug logging to see routing decisions

### Commands for Platform Team:
```bash
# 1. Check ingress controller logs
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller

# 2. Enable debug logging
kubectl patch configmap -n ingress-nginx ingress-nginx-controller --patch '{"data":{"error-log-level":"debug"}}'

# 3. Test DNS resolution
kubectl run test-dns -n ingress-nginx --image=busybox --rm -it -- nslookup seldon-mesh.seldon-system.svc.cluster.local

# 4. Check CoreDNS configuration
kubectl get configmap -n kube-system coredns -o yaml

# 5. Verify service discovery works
kubectl run test-curl -n ingress-nginx --image=curlimages/curl --rm -it -- curl http://seldon-mesh.seldon-system.svc.cluster.local/v2/models
```

## ✅ SOLUTION FOUND

### Root Cause: Missing IngressClass specification
The ingress was being ignored because it didn't specify `ingressClassName: nginx`.

### Solution Applied:
1. **Added IngressClass**: `spec.ingressClassName: nginx`
2. **Fixed Path Patterns**: Updated to use `ImplementationSpecific` pathType with regex
3. **Corrected Rewrite Rules**: Used `/$1` to match single capture group

### Working Configuration:
```yaml
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
  - host: ml-api.local
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

### Verification:
```bash
curl http://ml-api.local/seldon-system/v2/models/baseline-predictor
# Result: ✅ JSON model metadata
```

## Workaround (Current)
Port-forwarding works reliably:
```bash
kubectl port-forward -n seldon-system svc/seldon-mesh 8082:80
curl http://localhost:8082/v2/models/baseline-predictor  # ✅ Works
```

## Success Criteria
```bash
curl http://ml-api.local/seldon-system/v2/models/baseline-predictor
# Expected: JSON model metadata (not 404)
```

---
*This debug guide provides systematic investigation steps for platform team resolution.*