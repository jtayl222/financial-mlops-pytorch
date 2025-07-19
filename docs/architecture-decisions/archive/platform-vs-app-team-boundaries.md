# Platform vs Application Team Responsibilities: NGINX Ingress Boundary Analysis

## Current Responsibility Model

### Platform Team Responsibilities ✅
- **Infrastructure**: MetalLB, K8s cluster, CNI (Calico)
- **Shared services**: MLflow, MinIO, sealed-secrets
- **Base networking**: Load balancer IPs, DNS configuration
- **Security policies**: Cluster-wide RBAC, admission controllers

### Application Team Responsibilities ✅  
- **Application deployment**: Seldon models, experiments, training pipelines
- **Namespace management**: Create/delete their own namespaces
- **Application networking**: Network policies within their namespaces
- **Resource management**: Within allocated quotas

### Gray Area (Current Tension) ⚠️
- **Network policies**: App team manages, but affects platform networking
- **Cross-namespace communication**: App needs, platform controls
- **External access**: App needs external endpoints, platform controls ingress

## NGINX Ingress: Platform or Application Responsibility?

### Analysis Framework

**Platform Team Scope:**
- Cluster-wide infrastructure components
- Shared services affecting multiple teams
- Security boundaries and policies
- Network infrastructure (LoadBalancers, DNS)

**Application Team Scope:**
- Team-specific deployments
- Application-level configuration
- Business logic and ML models
- Team namespace management

### NGINX Ingress Responsibility Matrix

| Component | Platform Team | App Team | Rationale |
|-----------|---------------|----------|-----------|
| **Ingress Controller Installation** | ✅ | ❌ | Cluster-wide component, requires cluster-admin |
| **Ingress Controller Configuration** | ✅ | ❌ | Global policies, security, LoadBalancer IP |
| **Global Ingress Policies** | ✅ | ❌ | Rate limiting, CORS, TLS policies |
| **Application Ingress Rules** | ❌ | ✅ | App-specific routing, paths, backends |
| **TLS Certificates** | ✅ | ❌ | Security infrastructure, domain control |
| **DNS Configuration** | ✅ | ❌ | Infrastructure, external connectivity |

## Recommended Responsibility Split

### Platform Team: NGINX Infrastructure
```yaml
# Platform team manages
# 1. Ingress Controller Installation
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --set controller.service.loadBalancerIP=192.168.1.210

# 2. Global Configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-configuration
  namespace: ingress-nginx
data:
  # Global rate limiting
  rate-limit: "1000"
  # Global CORS policy
  enable-cors: "true"
  # TLS configuration
  ssl-protocols: "TLSv1.2 TLSv1.3"
```

### Application Team: Ingress Routes
```yaml
# App team manages in their namespace
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: seldon-system-ingress
  namespace: seldon-system  # App team's namespace
  annotations:
    # App-specific configurations only
    nginx.ingress.kubernetes.io/rewrite-target: /$2
spec:
  rules:
  - host: ml-api.company.com  # Platform team provides this domain
    http:
      paths:
      - path: /seldon-system(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: seldon-mesh  # App team's service
            port:
              number: 80
```

## Implementation Strategy

### Option A: Platform-Managed Ingress (Recommended)
**Platform team provides:**
- NGINX Ingress Controller installation and configuration
- Global domain (`ml-api.company.com`)
- TLS certificates and security policies
- Template Ingress resources for app teams

**Application team self-service:**
- Copy template Ingress resource to their namespace
- Modify paths and backend services for their needs
- Deploy and manage their own Ingress rules
- Request platform team for new domains/certificates if needed

### Option B: Shared Ingress Management
**Platform team provides:**
- NGINX Ingress Controller
- Centralized Ingress resource with routing rules
- App teams submit PRs to add their routes

**Problems with this approach:**
- ❌ App teams can't independently manage their routes
- ❌ Platform team becomes bottleneck for app changes
- ❌ Conflicts with "independent namespace management" goal

### Option C: App-Managed Ingress
**Application team manages:**
- Their own Ingress controllers per namespace
- Complete control over routing and policies

**Problems with this approach:**
- ❌ Resource waste (multiple ingress controllers)
- ❌ LoadBalancer IP exhaustion  
- ❌ Conflicts with platform team's infrastructure control

## Recommended Implementation: Option A

### Platform Team Deliverables
```bash
# 1. Install NGINX Ingress (one-time)
kubectl apply -f platform/ingress-controller/

# 2. Provide domain and TLS
# DNS: ml-api.company.com → 192.168.1.210
# TLS: Wildcard cert for *.ml-api.company.com

# 3. Create Ingress template for app teams
```

### Application Team Self-Service
```yaml
# k8s/ingress/seldon-system-ingress.yaml
# App team copies template, modifies for their needs
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: seldon-system-routes
  namespace: seldon-system
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /$2
spec:
  tls:
  - hosts:
    - ml-api.company.com
    secretName: platform-tls-cert  # Platform-provided
  rules:
  - host: ml-api.company.com
    http:
      paths:
      - path: /seldon-system(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: seldon-mesh
            port:
              number: 80
```

## Network Policy Boundary Clarification

### Current Network Policy Issue
The seldon-system team managing network policies is indeed **crossing the platform boundary** because:

- Network policies affect **cluster-wide connectivity**
- **Security implications** beyond single namespace
- **Platform team responsibility** for network security

### Recommended Network Policy Split

**Platform Team: Infrastructure Network Policies**
```yaml
# Platform manages policies affecting platform services
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-ingress-to-apps
  namespace: seldon-system
spec:
  podSelector: {}
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx  # Platform-controlled
```

**Application Team: Application Network Policies**
```yaml
# App team manages internal app communication only
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: seldon-internal-communication
  namespace: seldon-system
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: seldon-scheduler
  ingress:
  - from:
    - podSelector: {}  # Same namespace only
    ports:
    - port: 9005
```

## Responsibility Boundaries Summary

### Platform Team (Infrastructure)
- ✅ NGINX Ingress Controller installation/configuration
- ✅ Global security policies and TLS certificates
- ✅ LoadBalancer IPs and DNS configuration
- ✅ Cross-namespace network policies
- ✅ Platform service connectivity (MLflow, MinIO)

### Application Team (Business Logic)
- ✅ Application Ingress rules in their namespaces
- ✅ Application-specific routing and backends
- ✅ Internal application network policies
- ✅ Namespace lifecycle management
- ✅ Model deployments and experiments

### Shared/Templates (Self-Service)
- 📋 Ingress resource templates provided by platform
- 📋 Network policy templates for common patterns
- 📋 Documentation and examples for app teams

## Benefits of This Approach

### For Platform Team
- **Maintains control** over cluster infrastructure
- **Reduces support burden** through self-service templates
- **Security boundaries** clearly defined
- **Scalable model** for multiple app teams

### For Application Team  
- **Independence** for application-specific routing
- **Self-service** capabilities within boundaries
- **Clear ownership** of their namespace resources
- **Fast iteration** without platform team bottlenecks

### For Organization
- **Clear responsibility matrix** reduces conflicts
- **Scalable model** for additional ML teams
- **Security compliance** through platform controls
- **Developer productivity** through appropriate autonomy

## Migration Plan

### Phase 1: Platform Team Setup
1. Install NGINX Ingress Controller
2. Configure global policies and TLS
3. Create Ingress templates and documentation

### Phase 2: Application Team Migration
1. App team copies Ingress template
2. Configures routes for their services
3. Tests external access through ingress
4. Removes dependency on port-forwarding

### Phase 3: Network Policy Cleanup
1. Platform team takes ownership of cross-namespace policies
2. App team retains internal application policies
3. Remove app team's cluster-level network policy permissions

## Conclusion

**NGINX Ingress sits primarily in Platform Team territory** because it's cluster infrastructure, but with **clear self-service boundaries** for application teams.

This maintains the goal of **independent namespace management** while respecting **platform team infrastructure control**. The application team gets the autonomy they need for their ML workloads without compromising cluster security or scalability.