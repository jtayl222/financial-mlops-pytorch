# Architecture Decision: Dedicated MLServer per Namespace

## Status
**Accepted** - Implemented in financial-ml namespace

## Context
During the CNI migration from Flannel to Calico, we encountered issues with Seldon Core v2 model deployment. The question arose: should we use a shared cluster-wide MLServer or deploy dedicated MLServer instances per namespace?

## Decision
Deploy dedicated MLServer StatefulSets within application namespaces rather than relying on shared cluster-wide servers.

## Rationale

### Security Benefits
- **Namespace Isolation**: Each application team has full control over their inference server
- **Credential Separation**: S3/MLflow credentials scoped to specific namespaces
- **Network Policy Enforcement**: Easier to implement fine-grained network controls
- **Resource Isolation**: Memory and CPU limits per application workload

### Operational Advantages
- **Independent Scaling**: Scale inference capacity based on application-specific load
- **Deployment Autonomy**: Application teams can update server configurations independently
- **Debugging Simplification**: Isolated logs and metrics per application
- **Version Management**: Different applications can use different MLServer versions

### Reduced Cross-Team Dependencies
- **Platform Team Workload**: Fewer shared infrastructure components to manage
- **Application Team Velocity**: Direct control over model serving infrastructure
- **Incident Isolation**: Issues in one application don't affect others
- **Configuration Flexibility**: Each team can optimize for their specific use case

## Implementation Details

### MLServer Configuration
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mlserver
  namespace: financial-ml  # Application-specific namespace
spec:
  # ... StatefulSet configuration
  template:
    spec:
      containers:
      - name: mlserver
        env:
        - name: MLSERVER_MODELS_DIR
          value: /mnt/agent/models
        - name: SELDON_SERVER_CAPABILITIES
          value: mlflow,torch,scikit-learn,numpy
```

### Resource Allocation
- **Memory**: 1Gi request, 1Gi limit per container
- **CPU**: 100m-200m request, varies by workload
- **Storage**: 1Gi PVC for model artifacts
- **Capabilities**: Scoped to application requirements (mlflow, torch, sklearn, numpy)

### Network Configuration
- **Service**: ClusterIP within namespace
- **Network Policy**: Allow ingress from seldon-system and financial-mlops-pytorch
- **DNS**: `mlserver.financial-ml.svc.cluster.local`

## Alternatives Considered

### Shared Cluster-Wide MLServer
**Pros**: 
- Centralized management
- Resource sharing efficiency
- Platform team control

**Cons**: 
- Security boundaries crossed
- Complex credential management  
- Single point of failure
- Cross-team coordination overhead

### MLServer per Model
**Pros**:
- Ultimate isolation
- Fine-grained resource control

**Cons**:
- Resource overhead
- Management complexity
- Over-engineering for current scale

## Consequences

### Positive
- ✅ **Security**: Strong namespace isolation achieved
- ✅ **Performance**: Predictable resource allocation per application
- ✅ **Debugging**: Clear ownership and isolated troubleshooting
- ✅ **Scalability**: Independent scaling decisions per team

### Negative
- ❌ **Resource Usage**: Higher memory/CPU overhead vs shared approach
- ❌ **Management**: Each team manages their own MLServer instance
- ❌ **Consistency**: Potential drift in configurations across teams

### Neutral
- 🟡 **Complexity**: Moved from centralized to distributed complexity
- 🟡 **Monitoring**: Need namespace-specific monitoring vs centralized

## Validation

### Success Metrics
- **Model Deployment Success Rate**: >95% for models in dedicated namespaces
- **Incident Isolation**: Zero cross-application impacts from MLServer issues
- **Deployment Velocity**: Application teams can deploy without platform coordination
- **Resource Utilization**: Acceptable overhead vs shared infrastructure

### Current Results (Post-Implementation)
- ✅ Models successfully deployed: baseline-predictor, enhanced-predictor
- ✅ A/B experiment operational with 70/30 traffic split
- ✅ Zero cross-namespace security incidents
- ✅ Independent troubleshooting and debugging achieved

## Future Considerations

### Scaling Decisions
- **Resource Monitoring**: Track utilization to optimize resource allocation
- **Cost Analysis**: Compare dedicated vs shared approaches at scale
- **Multi-Cluster**: Consider implications for multi-cluster deployments

### Standardization Opportunities
- **Configuration Templates**: Provide standard MLServer configurations
- **Best Practices**: Document optimal resource allocation patterns
- **Monitoring Standards**: Consistent observability across namespaces

### Migration Path
If future requirements necessitate shared infrastructure:
1. **Pilot Program**: Test shared MLServer with non-critical workloads
2. **Migration Strategy**: Gradual transition with rollback capabilities  
3. **Security Review**: Ensure namespace isolation maintained
4. **Performance Validation**: Compare serving latency and throughput

## Related Decisions
- [Network Policy Strategy](network-policy-strategy.md)
- [Platform Team Coordination](../platform-coordination/responsibility-matrix.md)
- [Resource Quota Management](../troubleshooting/resource-quota-issues.md)