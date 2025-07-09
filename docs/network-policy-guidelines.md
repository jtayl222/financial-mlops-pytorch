# Network Policy Guidelines

## Responsibility Matrix

### Platform Team Responsibilities
- **Cluster-wide network policies** (CoreDNS access, kube-system connectivity)
- **Cross-namespace communication** policies between different teams
- **Infrastructure-level security** policies (ingress controllers, service mesh)
- **Default deny/allow** rules at cluster level
- **Service mesh configuration** (Istio, Envoy gateway rules)
- **Seldon Core configuration** (seldon-config ConfigMap defining inference servers)
- **Cluster-wide service discovery** and server capability definitions

### Application Team (financial-mlops-pytorch) Responsibilities  
- **Application-specific** network policies within our namespaces
- **Service-to-service** communication within financial-ml/financial-mlops-pytorch
- **External service access** requirements (documented for platform approval)
- **Pod-to-pod** communication rules within our application scope

## Current Network Architecture

### Namespaces Managed
- `financial-ml` - Model serving and experiments (Seldon Models/Experiments)
- `financial-mlops-pytorch` - Training workloads (Argo Workflows)

### Required Cross-Namespace Communication
1. **financial-mlops-pytorch** → **financial-ml**
   - Training workflows need to deploy models to serving namespace
   - MLflow artifacts sharing between training and serving

2. **financial-ml** → **seldon-system**
   - Model scheduling and server discovery
   - Seldon runtime communication

3. **Both namespaces** → **kube-system**
   - DNS resolution (CoreDNS)
   - Essential cluster services

### External Dependencies
- **MLflow LoadBalancer**: `192.168.1.208:5000`
- **MinIO LoadBalancer**: `192.168.1.200:9000`  
- **Model Inference LoadBalancer**: `192.168.1.209:80`

## Communication Protocol with Platform Team

### When to Engage Platform Team
1. **Cross-namespace policies** affecting other teams
2. **Cluster-wide DNS/networking** issues
3. **LoadBalancer/Ingress** configuration
4. **Service mesh** integration requirements

### Documentation Required
- Clear description of communication requirements
- Source and destination namespaces/services
- Ports and protocols needed
- Business justification for access

### Example Request Format
```
Request: Allow financial-ml namespace to access seldon-scheduler in seldon-system
Source: financial-ml namespace (all pods)
Destination: seldon-system/seldon-scheduler:9005
Protocol: TCP
Justification: Model scheduling requires direct communication with Seldon scheduler
```

## Application-Level Network Policies

Our `k8s/base/network-policy.yaml` should focus on:
- **Intra-namespace** communication rules
- **Application-specific** security requirements
- **Pod selector** based policies for our services

## Security Principles

1. **Least Privilege**: Only allow necessary communication
2. **Defense in Depth**: Application policies complement platform policies
3. **Clear Documentation**: All network requirements must be documented
4. **Platform Coordination**: No conflicts with cluster-wide policies

## Troubleshooting Process

1. **Application Issue**: Check our network policies first
2. **Cross-Namespace Issue**: Coordinate with platform team
3. **Cluster-Wide Issue**: Escalate to platform team immediately
4. **DNS/Infrastructure**: Always platform team responsibility

---

**Last Updated**: 2025-07-07  
**Contact**: financial-mlops-pytorch team  
**Platform Contact**: Infrastructure/Platform team