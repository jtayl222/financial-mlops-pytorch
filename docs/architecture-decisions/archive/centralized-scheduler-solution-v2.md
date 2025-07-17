# Centralized Scheduler Solution v2: Helm-based Deployment

## Status
**ADOPTED** - Production deployment pattern using Helm charts for centralized scheduler architecture

## Executive Summary

This document provides the definitive guide for deploying the centralized scheduler pattern described in `SELDON-PRODUCTION-ARCHITECTURE.md` using Helm charts. This approach ensures consistency between the control plane (seldon-system) and application namespaces while maintaining the recommended production architecture.

## Architecture Overview

The centralized scheduler pattern consists of:
- **Control Plane**: Single scheduler in `seldon-system` namespace (Helm-managed)
- **Application Namespaces**: SeldonRuntime configurations that disable local schedulers
- **Configuration Alignment**: Both control plane and applications use the same SeldonConfig

## Key Insight: Helm Chart Configuration

The Helm chart deploys a **SeldonRuntime named `seldon`** in the `seldon-system` namespace that manages the centralized scheduler. This runtime uses the `default` SeldonConfig, which becomes the source of truth for the entire cluster.

### Helm Chart Values
```yaml
# From: helm get values -n seldon-system seldon-core-v2-runtime --all
scheduler:
  disable: false      # Enables centralized scheduler
  replicas: 1         # Single scheduler instance
  serviceType: LoadBalancer
seldonConfig: default  # Uses default SeldonConfig
```

## Implementation Steps

### 1. Control Plane Deployment (seldon-system)

**Prerequisites**: 
- Helm chart already deployed with centralized scheduler enabled
- Verify scheduler is running:

```bash
# Check scheduler deployment
kubectl get pods -n seldon-system | grep scheduler
# Expected: seldon-scheduler-0   1/1   Running

# Check SeldonConfig
kubectl get seldonconfig default -n seldon-system
# Expected: default config exists
```

### 2. Application Namespace Configuration

**Key Principle**: Application SeldonRuntimes must use `seldonConfig: default` to match the Helm-deployed scheduler.

#### SeldonRuntime Configuration
```yaml
apiVersion: mlops.seldon.io/v1alpha1
kind: SeldonRuntime
metadata:
  name: financial-inference-runtime
  namespace: financial-inference
spec:
  overrides:
  - name: hodometer
    replicas: 1
  - name: seldon-envoy
    replicas: 1
  - name: seldon-dataflow-engine
    replicas: 0
  - name: seldon-modelgateway
    replicas: 1
  - name: seldon-pipelinegateway
    replicas: 0
  # Critical: Disable local scheduler
  - name: seldon-scheduler
    replicas: 0
  # MLServer configuration
  - name: mlserver
    replicas: 3
  # Must match Helm chart configuration
  seldonConfig: default
```

#### Server Configuration
```yaml
apiVersion: mlops.seldon.io/v1alpha1
kind: Server
metadata:
  name: mlserver
  namespace: financial-inference
spec:
  # Use default mlserver ServerConfig from seldon-system
  serverConfig: mlserver
  # Fixed capability strings (sklearn not scikit-learn)
  capabilities: ["mlflow", "torch", "sklearn", "numpy"]
  replicas: 1
```

#### Model Configuration
```yaml
apiVersion: mlops.seldon.io/v1alpha1
kind: Model
metadata:
  name: baseline-predictor
  namespace: financial-inference
spec:
  server: mlserver
  # Fixed capability strings to match server
  requirements: ["mlflow", "torch", "sklearn", "numpy"]
  storageUri: "s3://mlflow-artifacts/path/to/model"
```

### 3. ExternalName Service Pattern

Since the Helm chart creates the scheduler service in `seldon-system`, application namespaces should redirect to it:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: seldon-scheduler
  namespace: financial-inference
spec:
  type: ExternalName
  externalName: seldon-scheduler.seldon-system.svc.cluster.local
  ports:
  - name: xds
    port: 9002
    targetPort: 9002
  - name: scheduler
    port: 9004
    targetPort: 9004
  - name: agent
    port: 9005
    targetPort: 9005
```

## Configuration Alignment Matrix

| Component | Control Plane (seldon-system) | Application (financial-inference) |
|-----------|-------------------------------|-----------------------------------|
| **Scheduler** | Enabled via Helm (`replicas: 1`) | Disabled (`replicas: 0`) |
| **SeldonConfig** | `default` (Helm-managed) | `default` (must match) |
| **ServerConfig** | `mlserver` (Helm-created) | References `mlserver` |
| **Service** | `seldon-scheduler` (LoadBalancer) | `seldon-scheduler` (ExternalName) |

## Common Misconfigurations

### ❌ Wrong SeldonConfig Reference
```yaml
# WRONG: Creates config mismatch
spec:
  seldonConfig: centralized  # Different from Helm chart
```

### ❌ Wrong Capability Strings
```yaml
# WRONG: Uses scikit-learn instead of sklearn
spec:
  requirements: ["mlflow", "torch", "scikit-learn", "numpy"]
```

### ❌ Custom ServerConfig in Application Namespace
```yaml
# WRONG: Creates unnecessary complexity
spec:
  serverConfig: centralized-scheduler-config  # Should use existing mlserver
```

## Validation Steps

### 1. Verify Scheduler Connectivity
```bash
# Check agent logs for successful subscription
kubectl logs mlserver-0 -n financial-inference -c agent | grep "Subscribed to scheduler"
# Expected: "Subscribed to scheduler"

# Check scheduler receiving notifications
kubectl logs -n seldon-system seldon-scheduler-0 | grep "Server notification mlserver"
# Expected: Regular server notification messages
```

### 2. Verify Configuration Alignment
```bash
# Check SeldonRuntime configuration
kubectl get seldonruntime financial-inference-runtime -n financial-inference -o yaml | grep seldonConfig
# Expected: seldonConfig: default

# Check Server configuration
kubectl get server mlserver -n financial-inference -o yaml | grep serverConfig
# Expected: serverConfig: mlserver
```

### 3. Verify Model Scheduling
```bash
# Check server loaded models
kubectl get server mlserver -n financial-inference
# Expected: LOADED MODELS > 0

# Check model status
kubectl get models -n financial-inference
# Expected: READY: True
```

## Troubleshooting Guide

### Issue: "No matching servers available"
**Root Cause**: Capability string mismatch
**Solution**: Ensure models use `sklearn` not `scikit-learn`

### Issue: "ServerConfig not found"
**Root Cause**: Referencing non-existent ServerConfig
**Solution**: Use `serverConfig: mlserver` (exists in seldon-system)

### Issue: "Scheduler not ready"
**Root Cause**: SeldonConfig mismatch
**Solution**: Ensure `seldonConfig: default` in application SeldonRuntime

### Issue: Agent connection failures
**Root Cause**: Missing ExternalName service
**Solution**: Create ExternalName service pointing to seldon-system scheduler

## Network Policies

Ensure cross-namespace communication is enabled:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: financial-inference-app-policy
  namespace: financial-inference
spec:
  podSelector: {}
  policyTypes:
  - Egress
  egress:
  # Allow communication to seldon-system for scheduler
  - to:
    - namespaceSelector:
        matchLabels:
          name: seldon-system
    ports:
    - protocol: TCP
      port: 9002  # xDS
    - protocol: TCP
      port: 9004  # Scheduler
    - protocol: TCP
      port: 9005  # Agent
```

## Migration from Custom Configurations

If you have existing custom SeldonConfigs or ServerConfigs:

### From Custom SeldonConfig
```bash
# Old approach
seldonConfig: centralized

# New approach (matches Helm chart)
seldonConfig: default
```

### From Custom ServerConfig
```bash
# Old approach
serverConfig: centralized-scheduler-config

# New approach (uses Helm-created config)
serverConfig: mlserver
```

## Best Practices

### 1. Configuration Consistency
- Always use `seldonConfig: default` in application namespaces
- Always use `serverConfig: mlserver` for MLServer deployments
- Use exact capability strings from MLServer documentation

### 2. Namespace Isolation
- Deploy one SeldonRuntime per application namespace
- Use ExternalName services for scheduler redirection
- Maintain proper network policies

### 3. Monitoring and Validation
- Monitor scheduler logs for server notifications
- Check agent logs for successful subscriptions
- Validate model scheduling through server status

## Production Deployment Checklist

- [ ] Helm chart deployed with centralized scheduler enabled
- [ ] Application SeldonRuntime uses `seldonConfig: default`
- [ ] Server references `serverConfig: mlserver`
- [ ] Models use correct capability strings (`sklearn` not `scikit-learn`)
- [ ] ExternalName service redirects to centralized scheduler
- [ ] Network policies allow cross-namespace communication
- [ ] Agent logs show "Subscribed to scheduler"
- [ ] Scheduler logs show server notifications
- [ ] Models successfully scheduled and loaded

## References

- [SELDON-PRODUCTION-ARCHITECTURE.md](./SELDON-PRODUCTION-ARCHITECTURE.md) - Target architecture
- [Seldon Core v2 Helm Charts](https://github.com/SeldonIO/seldon-core/tree/v2/helm-charts)
- [MLServer Capabilities Documentation](https://docs.seldon.io/projects/seldon-core/en/latest/contents/servers/index.html)

---

**This solution provides the definitive approach for deploying the centralized scheduler pattern using Helm charts, ensuring consistency between control plane and application configurations.**