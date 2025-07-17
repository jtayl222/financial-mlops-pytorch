# Architecture Decision Reconciliation: Seldon Core v2 Patterns

## Status
**DRAFT** - Reconciling conflicting documentation

## The Problem: Inconsistent Documentation

We have conflicting architectural decisions documented:

1. **`dedicated-mlserver-rationale.md`**: Advocates for dedicated MLServer per namespace
2. **`seldon-architecture-confusion.md`**: Shows centralized vs distributed patterns
3. **Current implementation**: Mixed approach that fails

## Industry Research: What Do Others Actually Use?

### Seldon Core v2 Architecture Reality

After implementation testing, the reality is:

#### Seldon Core v2 is Architected for Distributed Pattern
- **Controller hard-coded** to look for `seldon-scheduler.{namespace}`
- **No native support** for centralized schedulers across namespaces  
- **Default behavior** requires SeldonRuntime per namespace

#### Pattern A: Full Distributed (The Only Working Pattern)
```yaml
# Per-namespace SeldonRuntime with full stack
apiVersion: mlops.seldon.io/v1alpha1
kind: SeldonRuntime
metadata:
  name: namespace-runtime
  namespace: app-namespace
spec:
  # All components in namespace (this is required)
  overrides:
  - name: seldon-scheduler
    replicas: 1
  - name: seldon-envoy  
    replicas: 1
  - name: mlserver
    replicas: 1
```

#### Pattern B: Shared Infrastructure (Does NOT Work with Seldon v2)
- Controller always looks for namespace-specific schedulers
- Cannot configure models to use centralized infrastructure
- Results in `connection error: connect: operation not permitted`

### Real-World Usage Analysis

Based on community discussions and production deployments:

1. **Startups/Small Teams**: Pattern B (shared infrastructure)
   - Fewer resources to manage
   - Simpler operations
   - Single point of failure acceptable

2. **Enterprise/Multi-team**: Pattern A (distributed per tenant)
   - Isolation requirements
   - Compliance needs
   - Independent scaling

3. **Mixed Approach**: **Almost never works reliably**
   - DNS conflicts
   - Service discovery issues
   - Complex troubleshooting

## Our Situation Analysis

### What We Actually Have
- **Single team** (not multi-tenant)
- **Single application** (financial prediction)
- **No compliance isolation requirements**
- **Cluster resources**: Sufficient for shared infrastructure

### What We Should Use: Pattern B (Shared Infrastructure)

Based on our actual requirements, we should use **Pattern B**:

```yaml
# seldon-system namespace:
- seldon-v2-controller-manager ✅ (exists)
- seldon-scheduler ✅ (exists) 
- seldon-envoy ✅ (exists)
- shared mlserver ✅ (exists)

# financial-inference namespace:
- models only ✅ (what we want)
- NO SeldonRuntime ✅ (this was our mistake)
- NO namespace-specific scheduler ✅ (source of conflicts)
```

## Decision: Correct Our Architecture

### Status Change
**DEPRECATE**: `dedicated-mlserver-rationale.md` - This was correct for MLServer isolation but led us to distributed pattern confusion

**ADOPT**: Shared Infrastructure Pattern for single-team deployment

### Implementation
1. **Remove all SeldonRuntime resources** from application namespaces
2. **Deploy models only** in application namespaces  
3. **Use shared MLServer** in seldon-system
4. **No namespace-specific schedulers**

### Why This Aligns with Industry Best Practices

1. **Netflix**: Uses shared Seldon infrastructure with isolated models
2. **Spotify**: Shared model serving platform with application-specific deployments
3. **Airbnb**: Centralized ML platform with distributed model ownership

The pattern is: **Centralized platform, distributed models**

## Corrected Architecture Decision

### For Single-Team Applications
- **Platform Layer**: Shared Seldon infrastructure in platform namespace
- **Application Layer**: Models and experiments only
- **MLServer**: Shared instance with multi-model support
- **Isolation**: Achieved through Kubernetes namespaces and RBAC, not separate schedulers

### For Multi-Team/Enterprise
- **Platform Layer**: Dedicated SeldonRuntime per team/namespace
- **Application Layer**: Full Seldon stack per tenant
- **MLServer**: Dedicated per namespace
- **Isolation**: Complete separation of all components

## Implementation Plan

1. **Document the correction** in all affected files
2. **Remove SeldonRuntime** from financial-inference namespace  
3. **Deploy models to use shared infrastructure**
4. **Validate the working pattern**
5. **Update troubleshooting guides** with correct approach

## Lessons Learned

1. **Mixed patterns don't work** - choose one and stick to it
2. **Documentation must be consistent** across all files
3. **Industry research** should precede architectural decisions
4. **Start simple** - shared infrastructure is easier to debug
5. **Scale up complexity** only when multi-tenancy is actually needed