# Finding the Correct Seldon URL

## Problem
We have working Seldon components but can't find the correct endpoint URL for our experiment. Getting 404s on standard paths.

## ✅ SOLUTION: NGINX Ingress (Current Approach)
**NGINX Ingress Controller provides unified external access** (implemented 2025-07-11):

### Production-Ready External Access
```bash
# NGINX Ingress endpoint (no port-forwarding needed)
curl http://ml-api.local/seldon-system/v2/models

# Models accessible via ingress
curl http://ml-api.local/seldon-system/v2/models/baseline-predictor
curl http://ml-api.local/seldon-system/v2/models/enhanced-predictor
curl http://ml-api.local/seldon-system/v2/models/financial-ab-test-experiment
```

### Architecture
```
External → NGINX Ingress (192.168.1.249) → /seldon-system/* → seldon-system namespace
```

## Legacy Methods (For Reference)

### Direct LoadBalancer Access (Limited)
- Seldon mesh at `192.168.1.202:80` (seldon-system namespace only)
- Does **not** provide access to seldon-system namespace

## Exploration Strategy

### 1. Basic Health/Status Endpoints
Try common health check and status paths:

```bash
# Basic root
curl -s http://192.168.1.202/

# Health checks
curl -s http://192.168.1.202/health
curl -s http://192.168.1.202/healthz
curl -s http://192.168.1.202/ready
curl -s http://192.168.1.202/status

# Metrics
curl -s http://192.168.1.202/metrics
curl -s http://192.168.1.202/stats
```

### 2. API Discovery
Try standard API discovery paths:

```bash
# OpenAPI/Swagger docs
curl -s http://192.168.1.202/docs
curl -s http://192.168.1.202/swagger
curl -s http://192.168.1.202/api/docs
curl -s http://192.168.1.202/v2/docs

# API versions
curl -s http://192.168.1.202/v1
curl -s http://192.168.1.202/v2
curl -s http://192.168.1.202/api

# Model listing
curl -s http://192.168.1.202/v2/models
curl -s http://192.168.1.202/models
```

### 3. Namespace-Specific Paths
Previously thought pattern was `/seldon/<namespace>/<model-name>/v2/docs`:

```bash
# Our specific experiment
curl -s http://192.168.1.202/seldon/seldon-system/financial-ab-test-experiment/v2/docs
curl -s http://192.168.1.202/seldon/seldon-system/financial-ab-test-experiment/
curl -s http://192.168.1.202/seldon/seldon-system/

# Individual models
curl -s http://192.168.1.202/seldon/seldon-system/baseline-predictor/v2/docs
curl -s http://192.168.1.202/seldon/seldon-system/enhanced-predictor/v2/docs

# Namespace only
curl -s http://192.168.1.202/seldon/seldon-system/
curl -s http://192.168.1.202/seldon/
```

### 4. Alternative Patterns
Try other common routing patterns:

```bash
# With namespace prefix
curl -s http://192.168.1.202/seldon-system/v2/models
curl -s http://192.168.1.202/ns/seldon-system/v2/models

# With inference prefix
curl -s http://192.168.1.202/inference/financial-ab-test-experiment
curl -s http://192.168.1.202/predict/financial-ab-test-experiment

# KServe-style paths
curl -s http://192.168.1.202/v1/models/financial-ab-test-experiment
curl -s http://192.168.1.202/v1/models/financial-ab-test-experiment:predict
```

### 5. Port-Specific Checks
The seldon-mesh has multiple ports (80, 9003). Try 9003:

```bash
curl -s http://192.168.1.202:9003/
curl -s http://192.168.1.202:9003/v2/models
curl -s http://192.168.1.202:9003/health
```

## LEGACY SOLUTION ⚠️ (Port-Forward Method)

### Working Pattern Discovered (Before NGINX Ingress)
The issue was **namespace routing**. MetalLB provides external access, but the seldon-system LoadBalancer (192.168.1.202) only routes to seldon-system components, not our seldon-system namespace components.

**Why Port-Forward Was Needed:**
- ✅ MetalLB: External IP access works (192.168.1.202)  
- ❌ Namespace isolation: LoadBalancer doesn't cross namespaces
- ❌ Missing cross-namespace routing configuration

**Legacy approach** (still works for debugging):

```bash
# Port forward to seldon-system mesh
kubectl port-forward -n seldon-system svc/seldon-mesh 8082:80

# Working endpoints:
curl http://localhost:8082/v2/models/baseline-predictor     # ✅ 200 OK
curl http://localhost:8082/v2/models/enhanced-predictor     # ✅ 200 OK
curl http://localhost:8082/v2/models/financial-ab-test-experiment  # ✅ 200 OK
```

### Key Findings
1. **Individual models work**: `baseline-predictor` and `enhanced-predictor` return model metadata
2. **Namespace isolation**: seldon-system LoadBalancer ≠ seldon-system components
3. **NGINX Ingress solves this**: Production-ready cross-namespace routing
4. **Correct inference URL pattern**: `/v2/models/{model-name}/infer`

### Current State
- ✅ **NGINX Ingress**: Production external access (ml-api.local)
- ✅ **Port-forward**: Still available for debugging
- ✅ **Traffic generators**: Updated to use ingress endpoints

### After URL Resolution - Common Next Issue
Once you find working URLs, you may encounter **500 Internal Server Error** instead of 404s. This typically indicates **payload format issues**:

```bash
# 404 = URL/routing problem (this document)
# 500 = Payload/data format problem (separate troubleshooting)
```

Check MLServer logs for specific errors:
```bash
kubectl logs -n seldon-system mlserver-0 -c mlserver --tail=20
```

### Swagger Documentation Access
The expected pattern `http://<endpoint>/seldon/<namespace>/<model-name>/v2/docs` **does not work** with our LoadBalancer setup:

```bash
# FAILED - Returns 404
curl http://192.168.1.202/seldon/seldon-system/baseline-predictor/v2/docs
```

**Alternative approaches for API docs:**
1. **Port-forward method** (if docs endpoint exists):
   ```bash
   kubectl port-forward -n seldon-system svc/seldon-mesh 8082:80
   curl http://localhost:8082/v2/docs  # Check if available
   ```

2. **Direct model inspection** (working method):
   ```bash
   curl http://localhost:8082/v2/models/baseline-predictor  # Returns input/output schema
   ```

## Expected Outcomes

We should find:
- Health/status endpoints to confirm the service is running
- API documentation or model listing endpoints
- The correct path pattern for our experiment: `financial-ab-test-experiment`
- Working inference endpoints for traffic generation

## Current Components Status
- ✅ Models: `baseline-predictor`, `enhanced-predictor` (Ready=True)
- ✅ Experiment: `financial-ab-test-experiment` (Ready=True)
- ✅ Server: `mlserver` (2 loaded models)
- ✅ **External access**: Found working pattern via port-forward