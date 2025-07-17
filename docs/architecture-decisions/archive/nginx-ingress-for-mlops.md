# NGINX Ingress Controller for MLOps: What, Why, and How

## What is NGINX Ingress Controller?

NGINX Ingress Controller is a **Kubernetes-native load balancer** that manages external access to services in a cluster. It acts as the **single entry point** for all external traffic and routes requests to the appropriate backend services based on rules you define.

### Core Components
```
External Traffic → NGINX Ingress Controller → Kubernetes Services → Pods
```

**Key Features:**
- **Path-based routing**: `/financial-inference/` → financial-inference namespace
- **Host-based routing**: `ml-api.company.com` → ML services
- **TLS termination**: HTTPS handling at the edge
- **Load balancing**: Distribute traffic across multiple pods
- **Rate limiting**: Protect services from overload

## Why NGINX Ingress for MLOps?

### 1. **Solves Our Current Architecture Gap**

**Current Setup (Problematic):**
```
External → MetalLB LoadBalancer (192.168.1.202) → seldon-system only
        → Port-forward required for financial-inference namespace
```

**With NGINX Ingress (Industry Standard):**
```
External → NGINX Ingress → /financial-inference/* → financial-inference namespace
                        → /trading-models/*     → trading namespace  
                        → /risk-analysis/*      → risk namespace
```

### 2. **Industry Best Practices Alignment**

Based on 2025 MLOps research, the standard architecture pattern is:

✅ **Recommended Pattern:**
- Single external endpoint
- Ingress controller for intelligent routing
- Namespace isolation with cross-namespace access
- Path-based service discovery

❌ **Anti-Pattern (Our Current):**
- LoadBalancer per namespace (expensive, unscalable)
- Port-forwarding for production access
- No unified API gateway

### 3. **MLOps-Specific Benefits**

**Model Serving Challenges Solved:**
- **Multi-tenant model serving**: Different teams/models in separate namespaces
- **A/B testing**: Route traffic splits to different model versions
- **Canary deployments**: Gradual rollout of new models
- **API versioning**: `/v1/models/` vs `/v2/models/`
- **Authentication/authorization**: Centralized security policies

**Example MLOps Routing:**
```yaml
# Route to different model serving frameworks
/seldon/financial-inference/*  → Seldon Core models
/kserve/trading-models/*       → KServe models  
/bentoml/risk-analysis/*       → BentoML models
/mlflow/experimentation/*      → MLflow models
```

## How to Implement NGINX Ingress

### Step 1: Install NGINX Ingress Controller

```bash
# Add NGINX Ingress Helm repo
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

# Install NGINX Ingress with MetalLB integration
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.service.type=LoadBalancer \
  --set controller.service.loadBalancerIP=192.168.1.210
```

### Step 2: Create Ingress Resources for Seldon

```yaml
# k8s/ingress/seldon-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: seldon-mlops-ingress
  namespace: ingress-nginx
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /$2
    nginx.ingress.kubernetes.io/ssl-redirect: "false"
    nginx.ingress.kubernetes.io/cors-allow-origin: "*"
spec:
  rules:
  - host: ml-api.local  # or your domain
    http:
      paths:
      # Financial inference models
      - path: /financial-inference(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: seldon-mesh
            port:
              number: 80
        # Route to financial-inference namespace
      
      # Future namespaces
      - path: /trading-models(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: seldon-mesh
            port:
              number: 80
        # Would route to trading-models namespace
      
      # MLflow experiment tracking
      - path: /mlflow(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: mlflow
            port:
              number: 5000
        # Route to mlflow namespace
```

### Step 3: Configure Cross-Namespace Service References

```yaml
# k8s/ingress/external-services.yaml
apiVersion: v1
kind: Service
metadata:
  name: financial-inference-seldon
  namespace: ingress-nginx
spec:
  type: ExternalName
  externalName: seldon-mesh.financial-inference.svc.cluster.local
  ports:
  - port: 80
    targetPort: 80
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-external
  namespace: ingress-nginx
spec:
  type: ExternalName
  externalName: mlflow.mlflow.svc.cluster.local
  ports:
  - port: 5000
    targetPort: 5000
```

### Step 4: Update DNS/Hosts

```bash
# Add to /etc/hosts for local testing
192.168.1.210 ml-api.local

# Or configure DNS for production
ml-api.yourcompany.com → 192.168.1.210
```

## Implementation Plan for Our Environment

### Phase 1: Basic NGINX Setup
1. **Install NGINX Ingress** with MetalLB integration (new IP: 192.168.1.210)
2. **Create basic ingress** for financial-inference namespace
3. **Test external access** without port-forwarding
4. **Update traffic generator** to use ingress endpoint

### Phase 2: Multi-Namespace Routing  
1. **Add ingress rules** for additional namespaces
2. **Configure cross-namespace** service discovery
3. **Implement path-based routing** for different model types
4. **Add monitoring** and observability

### Phase 3: Production Hardening
1. **TLS termination** with Let's Encrypt or corporate certs
2. **Authentication/authorization** integration
3. **Rate limiting** and DDoS protection
4. **Monitoring and alerting** for ingress performance

## Benefits for Our MLOps Platform

### 1. **Immediate Improvements**
- **No more port-forwarding** for production access
- **Single API endpoint** for all ML services (`ml-api.company.com`)
- **Scalable architecture** for multiple namespaces/teams
- **Industry-standard patterns** for future team members

### 2. **Future Capabilities**
- **API Gateway features**: Authentication, rate limiting, monitoring
- **Blue/green deployments**: Traffic switching between model versions  
- **Canary releases**: Gradual rollout of new models
- **Multi-cluster support**: Route to different K8s clusters

### 3. **Developer Experience**
```bash
# Before (development only)
kubectl port-forward -n financial-inference svc/seldon-mesh 8082:80
curl http://localhost:8082/v2/models/financial-ab-test-experiment

# After (production ready)
curl http://ml-api.company.com/financial-inference/v2/models/financial-ab-test-experiment
```

## Architecture Comparison

### Current Architecture (Anti-Pattern)
```
Internet → MetalLB → seldon-system only
                  ❌ financial-inference (port-forward required)
                  ❌ trading-models (would need separate LoadBalancer)
                  ❌ risk-analysis (would need separate LoadBalancer)
```

### Proposed Architecture (Best Practice)
```
Internet → NGINX Ingress (192.168.1.210) → /financial-inference/* → financial-inference namespace
                                        → /trading-models/*     → trading-models namespace
                                        → /risk-analysis/*      → risk-analysis namespace
                                        → /mlflow/*             → mlflow namespace
                                        → /prometheus/*         → monitoring namespace
```

## Cost and Resource Considerations

### MetalLB IP Usage
- **Current**: 3 IPs used (seldon-system, mlflow, minio)
- **Without Ingress**: Would need 10+ IPs for 10 namespaces (expensive)
- **With Ingress**: 1 additional IP (192.168.1.210) serves unlimited namespaces

### Resource Requirements
- **NGINX Ingress Controller**: ~100m CPU, 200Mi memory
- **High availability**: 2-3 replicas recommended
- **Performance**: Can handle thousands of requests/second

## Security Benefits

### 1. **Centralized Security Policies**
```yaml
# Rate limiting
nginx.ingress.kubernetes.io/rate-limit: "100"

# CORS policies  
nginx.ingress.kubernetes.io/cors-allow-origin: "https://company.com"

# Authentication
nginx.ingress.kubernetes.io/auth-url: "https://auth.company.com/validate"
```

### 2. **TLS Termination**
- **Single certificate** for all ML services
- **Automatic HTTPS** enforcement
- **Perfect Forward Secrecy** support

### 3. **Network Isolation**
- **Ingress policies** control external access
- **Backend services** remain in private networks
- **WAF integration** possible for advanced protection

## Migration Strategy

### Option 1: Gradual Migration (Recommended)
1. **Deploy NGINX Ingress** alongside existing MetalLB setup
2. **Test with new endpoint** (`ml-api.company.com`)
3. **Migrate services gradually** namespace by namespace
4. **Keep existing LoadBalancers** until migration complete
5. **Sunset old endpoints** after validation

### Option 2: Big Bang Migration
1. **Deploy NGINX Ingress** 
2. **Update all service** configurations simultaneously
3. **Switch DNS** to new endpoint
4. **Remove old LoadBalancers**

**Recommendation**: Gradual migration reduces risk and allows for rollback.

## Conclusion

NGINX Ingress Controller solves our **fundamental architectural gap** and aligns our MLOps platform with **2025 industry best practices**. It provides:

- ✅ **Scalable external access** without LoadBalancer proliferation
- ✅ **Production-ready patterns** for multi-tenant ML serving
- ✅ **Future-proof architecture** for additional namespaces/teams
- ✅ **Cost-effective solution** using existing MetalLB infrastructure
- ✅ **Industry-standard approach** for Kubernetes MLOps platforms

**Next Steps**: Implement Phase 1 with basic NGINX setup and test external access to financial-inference namespace without port-forwarding.