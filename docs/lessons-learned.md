# Lessons Learned: Seldon Core v2 Model Deployment

## Summary

This document captures key lessons learned during the implementation of enterprise MLOps with Seldon Core v2, team-based namespace isolation, and package-based secret management.

## Architecture Decisions

### ✅ What Worked: Team-Based Namespace Isolation

**Implementation:**
```yaml
financial-ml/           # Model serving and inference
financial-mlops-pytorch/ # Training pipelines and development  
seldon-system/          # Shared infrastructure (if needed)
```

**Benefits:**
- Clear ownership boundaries between teams
- Independent resource quotas and RBAC policies
- Simplified cost allocation and billing
- Enhanced security through namespace-level isolation

### ✅ What Worked: Package-Based Secret Management

**Pattern:**
1. Infrastructure team creates secrets using SealedSecrets
2. Secrets packaged into tar.gz files with kustomization overlays
3. Development teams extract packages to `k8s/manifests/` (gitignored)
4. Teams apply secrets independently to their namespaces

**Commands:**
```bash
# Infrastructure team delivers packages
# financial-mlops-pytorch-ml-secrets-20250704.tar.gz
# financial-mlops-pytorch-models-secrets-20250704.tar.gz

# Development team applies
tar xzf financial-mlops-pytorch-ml-secrets-20250704.tar.gz -C k8s/manifests/financial-mlops-pytorch
tar xzf financial-mlops-pytorch-models-secrets-20250704.tar.gz -C k8s/manifests/financial-ml
kubectl apply -k k8s/manifests/financial-ml/production
kubectl apply -k k8s/manifests/financial-mlops-pytorch/production
```

**Benefits:**
- Development autonomy while maintaining security compliance
- Complete environment reset capability for testing
- Consistent secret structure across environments
- Clear separation between infrastructure and application concerns

## Technical Challenges & Solutions

### ❌ Challenge: Cross-Namespace Scheduler Connectivity

**Problem:** Initially tried to share `seldon-system` scheduler across namespaces using ExternalName services and manual MLServer deployment.

**Issues Encountered:**
- Models couldn't find matching servers ("no matching servers are available")
- Complex cross-namespace networking requirements
- Agent registration failures with shared scheduler
- RBAC permission complexities

**Root Cause:** Seldon v2 architecture expects scheduler and model servers in same namespace for optimal isolation.

### ✅ Solution: Dedicated SeldonRuntime per Namespace

**Industry Best Practice Implementation:**
```yaml
apiVersion: mlops.seldon.io/v1alpha1
kind: SeldonRuntime
metadata:
  name: financial-ml-runtime
  namespace: financial-ml
spec:
  config:
    agentConfig:
      rclone: {}
    kafkaConfig: {}
    serviceConfig: {}
    tracingConfig: {}
  overrides:
  - name: hodometer
    replicas: 1
  - name: seldon-scheduler
    replicas: 1
  - name: seldon-envoy
    replicas: 1
  - name: seldon-dataflow-engine
    replicas: 1
  - name: seldon-modelgateway
    replicas: 1
  - name: seldon-pipelinegateway
    replicas: 1
  - name: mlserver
    replicas: 1
  seldonConfig: default
```

**Results:**
- Complete infrastructure isolation per team
- Eliminates cross-namespace networking complexity
- Follows enterprise patterns used by Netflix, Spotify, Uber
- Enables independent scaling and fault isolation

### ❌ Challenge: ResourceQuota CPU Limits

**Problem:** SeldonRuntime components don't include CPU limits by default, causing pod creation failures:
```
Error creating: pods "seldon-scheduler-0" is forbidden: failed quota: financial-ml-quota: must specify limits.cpu for: scheduler
```

**Solutions:**
1. **Development/Testing:** Remove CPU limits requirement from ResourceQuota
2. **Production:** Work with platform team to configure resource limits in SeldonRuntime

**Learning:** Seldon v2 SeldonRuntime CRD doesn't support `resources` field in `overrides` for configuring CPU/memory limits.

### ❌ Challenge: MLServer Registration 

**Current State:** SeldonRuntime deployed successfully with all infrastructure components, but MLServer instances may need additional configuration to properly register with scheduler.

**Investigation Needed:**
- Whether MLServer automatically deploys when models are created
- If additional MLServer configuration is required
- How model scheduling works with dedicated runtime

## Configuration Insights

### Seldon v2 Controller Configuration

**Required for Cluster-Wide Operation:**
```yaml
env:
- name: CLUSTERWIDE
  value: "true"
args:
- --clusterwide=true
```

### NetworkPolicy Configuration

**Team Isolation with Inter-Namespace Communication:**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: financial-ml-isolation
  namespace: financial-ml
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: seldon-system  # Allow infrastructure access
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: seldon-system
```

## Enterprise MLOps Patterns

### Successful Patterns

1. **Namespace-per-Team**: Clear ownership and isolation boundaries
2. **Package-based Secrets**: Infrastructure security with development autonomy  
3. **Dedicated Runtime**: Complete infrastructure isolation per team
4. **GitOps Integration**: Kustomize-based configuration management
5. **Resource Quotas**: Prevent resource exhaustion with appropriate limits

### Anti-Patterns to Avoid

1. **Shared Runtime Components**: Creates operational dependencies
2. **Cross-Namespace Scheduler Sharing**: Complex networking and RBAC
3. **Embedded Secrets**: Violates security separation of concerns
4. **Manual Secret Management**: Breaks development autonomy

## Industry Comparison

### What We Implemented vs Industry Standards

**✅ Our Implementation:**
- Team-based namespace isolation
- Package-based secret delivery  
- Dedicated runtime per namespace
- GitOps configuration management

**✅ Industry Leaders (Netflix, Spotify, Uber):**
- Microservice isolation with namespace-per-team
- Platform abstraction with team autonomy
- Security compliance with developer productivity
- Infrastructure as code patterns

**Alignment:** Our approach follows established enterprise MLOps patterns used by major tech companies.

## Next Steps

### Immediate Actions Needed

1. **Complete Model Deployment Testing:** Verify models schedule properly with dedicated runtime
2. **MLServer Configuration:** Ensure MLServer instances register correctly with scheduler
3. **End-to-End Validation:** Test full pipeline from training to inference

### Future Enhancements

1. **Resource Limits Configuration:** Work with platform team for production-ready resource constraints
2. **Monitoring Integration:** Add observability for dedicated runtime components
3. **Auto-scaling Configuration:** Implement HPA for model servers
4. **Multi-Environment Strategy:** Extend pattern to staging/production environments

## Key Takeaways

1. **Industry Best Practice:** Dedicated runtime per namespace is the enterprise standard
2. **Security & Autonomy:** Package-based secrets enable both compliance and developer productivity
3. **Complexity Trade-offs:** Dedicated runtime eliminates cross-namespace complexity
4. **Resource Management:** Consider ResourceQuota implications when deploying Seldon v2
5. **Platform Evolution:** Seldon v2 architecture favors namespace-isolated deployments

This approach provides a solid foundation for enterprise MLOps that scales with organizational growth while maintaining security and operational excellence.