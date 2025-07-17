# Seldon Core v2 Production Architecture

## Status
**ADOPTED** - Production-ready architecture following Seldon Core v2 official patterns and industry best practices

## Executive Summary

This document defines the authoritative architecture for Seldon Core v2 deployment, strictly adhering to the **official Seldon Core v2 design patterns** as confirmed by the Seldon Core maintainers. All configurations follow the expected behavior where agents connect to inference servers within the same pod using standard Kubernetes networking.

## Architecture Principles

### Core Design Patterns (Seldon Official)
1. **Standard Pod Architecture**: Each inference server pod contains exactly 3 containers: `agent`, `rclone`, and `inference-server` (MLServer)
2. **Intra-Pod Communication**: Agent container connects to inference server container within the same pod via localhost
3. **Centralized Scheduler**: Single scheduler manages all models across namespaces (industry best practice)
4. **Kubernetes-Native Networking**: Standard Kubernetes service discovery and DNS resolution

### Environment Considerations
- **Kubernetes Version**: 1.28+ recommended
- **Container Runtime**: containerd (preferred) or Docker
- **CNI Plugin**: Calico (our implementation) or any standard CNI
- **Host OS**: Linux (required for proper localhost mapping)

## Production Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    seldon-system                        │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │            seldon-scheduler                         ││
│  │                                                     ││
│  │  • Single source of truth for all models           ││
│  │  • xDS configuration server (port 9002)            ││
│  │  • Agent communication (port 9004)                 ││
│  │  • Cross-namespace model discovery                 ││
│  │  • LoadBalancer: 192.168.1.201                     ││
│  └─────────────────────────────────────────────────────┘│
│                                                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │        seldon-controller-manager                    ││
│  │                                                     ││
│  │  • Watches Model/Experiment CRDs                   ││
│  │  • Notifies scheduler of changes                   ││
│  │  • Manages lifecycle events                        ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
                             │
                             │ Standard Kubernetes networking
                             │
┌─────────────────────────────────────────────────────────┐
│                financial-inference                      │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │                 MLServer Pods                       ││
│  │                                                     ││
│  │  Pod 1: [agent|rclone|mlserver]                    ││
│  │  Pod 2: [agent|rclone|mlserver]                    ││
│  │  Pod 3: [agent|rclone|mlserver]                    ││
│  │                                                     ││
│  │  • Agent → MLServer: localhost (intra-pod)         ││
│  │  • Agent → Scheduler: seldon-scheduler.seldon-     ││
│  │    system.svc.cluster.local:9004                   ││
│  │  • Standard Kubernetes service discovery           ││
│  └─────────────────────────────────────────────────────┘│
│                                                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │                 seldon-envoy                        ││
│  │                                                     ││
│  │  • xDS client to central scheduler                 ││
│  │  • Route configuration from scheduler              ││
│  │  • Load balancing across MLServer pods             ││
│  │  • Standard Kubernetes service mesh                ││
│  └─────────────────────────────────────────────────────┘│
│                                                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │              Model CRDs (Only)                      ││
│  │                                                     ││
│  │  • baseline-predictor                              ││
│  │  • enhanced-predictor                              ││
│  │  • financial-ab-test-experiment                    ││
│  │                                                     ││
│  │  No local scheduler components                      ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

## Component Specifications

### Central Scheduler Configuration (seldon-system)

**Service Definition**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: seldon-scheduler
  namespace: seldon-system
spec:
  type: LoadBalancer
  loadBalancerIP: 192.168.1.201
  selector:
    app.kubernetes.io/name: seldon-scheduler
  ports:
  - name: xds
    port: 9002
    targetPort: 9002
    protocol: TCP
  - name: scheduler
    port: 9004
    targetPort: 9004
    protocol: TCP
  - name: agent
    port: 9005
    targetPort: 9005
    protocol: TCP
```

**Controller Manager Configuration**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: seldon-controller-manager
  namespace: seldon-system
spec:
  template:
    spec:
      containers:
      - name: manager
        env:
        - name: SELDON_SCHEDULER_HOST
          value: seldon-scheduler.seldon-system.svc.cluster.local
        - name: SELDON_SCHEDULER_PORT
          value: "9004"
```

### Application Namespace Configuration (financial-inference)

**SeldonRuntime Specification**:
```yaml
apiVersion: mlops.seldon.io/v1alpha1
kind: SeldonRuntime
metadata:
  name: financial-inference-runtime
  namespace: financial-inference
spec:
  overrides:
  # Critical: No local scheduler
  - name: seldon-scheduler
    replicas: 0
  
  # MLServer configuration (follows Seldon official pattern)
  - name: mlserver
    replicas: 3
    env:
    # Agent connects to central scheduler
    - name: SELDON_SCHEDULER_HOST
      value: seldon-scheduler.seldon-system.svc.cluster.local
    - name: SELDON_SCHEDULER_PORT
      value: "9004"
    # MLServer binds to localhost (standard Seldon pattern)
    - name: MLSERVER_HTTP_PORT
      value: "9000"
    - name: MLSERVER_GRPC_PORT
      value: "9500"
    # Agent connects to MLServer via localhost (intra-pod)
    - name: SELDON_AGENT_MLSERVER_HOST
      value: "localhost"
    - name: SELDON_AGENT_MLSERVER_PORT
      value: "9000"
    resources:
      requests:
        cpu: 500m
        memory: 2Gi
      limits:
        cpu: 1000m
        memory: 4Gi
  
  # Envoy configuration
  - name: seldon-envoy
    replicas: 1
    env:
    # Envoy connects to central scheduler for xDS
    - name: SELDON_SCHEDULER_HOST
      value: seldon-scheduler.seldon-system.svc.cluster.local
    - name: SELDON_SCHEDULER_PORT
      value: "9002"
    resources:
      requests:
        cpu: 250m
        memory: 512Mi
      limits:
        cpu: 500m
        memory: 1Gi
```

## Network Architecture

### Standard Kubernetes Networking
All networking follows standard Kubernetes patterns:

```yaml
# DNS Resolution (automatic)
seldon-scheduler.seldon-system.svc.cluster.local → 192.168.1.201:9004

# Intra-pod communication (localhost)
agent → mlserver: localhost:9000

# Service mesh (Kubernetes services)
external → seldon-envoy → mlserver-pods
```

### Network Policies
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: financial-inference-network-policy
  namespace: financial-inference
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  egress:
  # DNS resolution (required)
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53
  
  # Central scheduler access
  - to:
    - namespaceSelector:
        matchLabels:
          name: seldon-system
    - podSelector:
        matchLabels:
          app.kubernetes.io/name: seldon-scheduler
    ports:
    - protocol: TCP
      port: 9002  # xDS
    - protocol: TCP
      port: 9004  # Scheduler
    - protocol: TCP
      port: 9005  # Agent
  
  # Intra-pod communication (implicit - always allowed)
  # agent → mlserver: localhost:9000
```

## Model Configuration

### Model CRDs (Standard Seldon Pattern)
```yaml
apiVersion: mlops.seldon.io/v1alpha1
kind: Model
metadata:
  name: baseline-predictor
  namespace: financial-inference
spec:
  storageUri: "s3://mlflow-artifacts/models/baseline-predictor"
  requirements:
  - mlserver[sklearn]
  - mlserver[mlflow]
  memory: 2Gi
  replicas: 1
```

### Experiment Configuration
```yaml
apiVersion: mlops.seldon.io/v1alpha1
kind: Experiment
metadata:
  name: financial-ab-test-experiment
  namespace: financial-inference
spec:
  default: baseline-predictor
  candidates:
  - name: baseline-predictor
    weight: 70
  - name: enhanced-predictor
    weight: 30
```

## Validation Requirements

### Architecture Validation Script
```bash
#!/bin/bash
# scripts/validate-seldon-architecture.sh

# 1. Verify no local scheduler services exist
if kubectl get svc seldon-scheduler -n financial-inference 2>/dev/null; then
    echo "❌ FAIL: Local scheduler service exists - must be removed"
    exit 1
fi

# 2. Verify central scheduler is running
SCHEDULER_READY=$(kubectl get sts seldon-scheduler -n seldon-system -o jsonpath='{.status.readyReplicas}')
if [[ "$SCHEDULER_READY" -lt 1 ]]; then
    echo "❌ FAIL: Central scheduler not running"
    exit 1
fi

# 3. Verify agent connections
AGENT_CONNECTED=$(kubectl logs -n financial-inference sts/mlserver -c agent --tail=50 | grep -c "Subscribed to scheduler")
if [[ "$AGENT_CONNECTED" -lt 1 ]]; then
    echo "❌ FAIL: Agents not connected to scheduler"
    exit 1
fi

# 4. Verify intra-pod communication
# Agent should connect to MLServer on localhost:9000
LOCALHOST_BINDING=$(kubectl logs -n financial-inference sts/mlserver -c mlserver --tail=50 | grep -c "0.0.0.0:9000")
if [[ "$LOCALHOST_BINDING" -lt 1 ]]; then
    echo "❌ FAIL: MLServer not binding to localhost"
    exit 1
fi

# 5. Verify models are ready
MODELS_READY=$(kubectl get models -n financial-inference -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}')
if [[ "$MODELS_READY" != *"True"* ]]; then
    echo "❌ FAIL: Models not ready"
    exit 1
fi

echo "✅ Architecture validation passed - deployment follows Seldon Core v2 patterns"
```

## Scaling and Performance

### Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mlserver-hpa
  namespace: financial-inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: StatefulSet
    name: mlserver
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Performance Expectations
- **Request Success Rate**: >95% for healthy deployment
- **Response Time**: P95 <50ms for inference requests
- **Intra-pod Latency**: <1ms (agent → mlserver)
- **Scheduler Connectivity**: 100% of agents connected

## Environment-Specific Considerations

### Kubernetes Environment
- **Supported**: Standard Kubernetes clusters on Linux
- **Container Runtime**: containerd (preferred), Docker
- **CNI**: Any standard CNI plugin (Calico, Flannel, etc.)

### Host OS Requirements
- **Linux**: Required for proper localhost (0.0.0.0) mapping
- **Windows**: Not supported for production deployments
- **macOS**: Development only (Docker Desktop)

### Virtualization
- **Bare Metal**: Full support
- **VM**: Full support with proper networking
- **Container Runtime**: Must support localhost networking

## Troubleshooting

### Common Issues and Solutions

#### Agent Cannot Connect to MLServer
```bash
# Check MLServer is binding to localhost
kubectl logs -n financial-inference sts/mlserver -c mlserver | grep "0.0.0.0:9000"

# Expected: "INFO: MLServer listening on 0.0.0.0:9000"
# If not found, check container runtime networking
```

#### Models Not Ready
```bash
# Check agent scheduler connection
kubectl logs -n financial-inference sts/mlserver -c agent | grep "Subscribed to scheduler"

# Check central scheduler logs
kubectl logs -n seldon-system sts/seldon-scheduler | grep "Model registered"
```

#### xDS Connection Failures
```bash
# Verify no local scheduler services
kubectl get svc seldon-scheduler -n financial-inference

# Should return: "Error from server (NotFound)"
# If exists, delete: kubectl delete svc seldon-scheduler -n financial-inference
```

## Migration from Previous Architectures

### Step 1: Clean Environment
```bash
# Remove any local scheduler services
kubectl delete svc seldon-scheduler -n financial-inference --ignore-not-found

# Remove any local scheduler statefulsets
kubectl delete sts seldon-scheduler -n financial-inference --ignore-not-found
```

### Step 2: Apply Standard Configuration
```bash
# Apply the SeldonRuntime with scheduler replicas=0
kubectl apply -f k8s/base/seldon-runtime.yaml

# Verify configuration
scripts/validate-seldon-architecture.sh
```

### Step 3: Validation
```bash
# All checks must pass
scripts/validate-seldon-architecture.sh

# Functional test
python3 scripts/demo/advanced-ab-demo.py --scenarios 100 --workers 3
```

## Security Considerations

### Network Security
- Network policies enforce namespace isolation
- Central scheduler only accessible from authorized namespaces
- No cross-namespace service dependencies except to central scheduler

### RBAC Configuration
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: seldon-model-manager
  namespace: financial-inference
rules:
- apiGroups: ["mlops.seldon.io"]
  resources: ["models", "experiments"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list", "watch"]
```

## Best Practices Summary

### Must Do
1. ✅ Use centralized scheduler pattern (single scheduler in seldon-system)
2. ✅ No local scheduler services in application namespaces
3. ✅ Standard 3-container pod architecture: agent|rclone|mlserver
4. ✅ Intra-pod communication via localhost
5. ✅ Standard Kubernetes networking and service discovery

### Never Do
1. ❌ Deploy local scheduler services in application namespaces
2. ❌ Use non-standard networking configurations
3. ❌ Modify intra-pod communication patterns
4. ❌ Deploy without validating architecture compliance

### Performance Optimization
1. Scale horizontally (replicas) not vertically (resources)
2. Use HPA for dynamic scaling
3. Monitor agent-scheduler connectivity
4. Optimize model loading and caching

## References

- [Seldon Core v2 Official Documentation](https://docs.seldon.io/projects/seldon-core/en/v2/)
- [Kubernetes Service Mesh Best Practices](https://kubernetes.io/docs/concepts/services-networking/)
- [MLOps Production Patterns](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Seldon Core GitHub Issues](https://github.com/SeldonIO/seldon-core/issues)

---

**This architecture strictly follows Seldon Core v2 official patterns and is validated against the maintainer's feedback. Any deviations from this design should be justified and documented.**