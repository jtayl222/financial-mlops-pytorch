# Platform vs Application Team Responsibility Matrix

## Network Policies

| Component | Application Team | Platform Team | Notes |
|-----------|------------------|---------------|-------|
| Application-level policies | ✅ Owns | 🤝 Consults | Within namespace boundaries |
| Cross-namespace policies | 🤝 Requests | ✅ Owns | Requires coordination |
| Cluster-wide policies | ❌ No access | ✅ Owns | Infrastructure-level |
| DNS egress rules | ✅ Implements | 🤝 Guides | Standard patterns |

## Secrets Management

| Component | Application Team | Platform Team | Notes |
|-----------|------------------|---------------|-------|
| Application secrets | ✅ Owns | 🤝 Provides tooling | MLflow, S3 credentials |
| Infrastructure secrets | 🤝 Requests | ✅ Owns | RClone, registry credentials |
| Sealed secrets | ✅ Creates | ✅ Provides keys | Shared responsibility |
| Secret rotation | 🤝 Coordinates | ✅ Executes | Planned maintenance |

## Service Configuration

| Component | Application Team | Platform Team | Notes |
|-----------|------------------|---------------|-------|
| Application services | ✅ Owns | 🤝 Reviews | LoadBalancer, ClusterIP |
| LoadBalancer IPs | 🤝 Requests | ✅ Assigns | MetalLB pool management |
| Ingress routes | ✅ Configures | ✅ Provides controller | Shared implementation |
| Service mesh | ❌ No access | ✅ Owns | If implemented |

## Seldon Core Components

| Component | Application Team | Platform Team | Notes |
|-----------|------------------|---------------|-------|
| Model CRDs | ✅ Owns | 🤝 Supports | Model, Experiment definitions |
| MLServer config | ✅ Configures | 🤝 Provides base | Namespace-specific servers |
| seldon-config | 🤝 Requests | ✅ Owns | Cluster-wide server definitions |
| RClone secrets | 🤝 Requests | ✅ Provides | Infrastructure credentials |

## Monitoring & Observability

| Component | Application Team | Platform Team | Notes |
|-----------|------------------|---------------|-------|
| Application metrics | ✅ Owns | 🤝 Collects | Custom business metrics |
| Infrastructure metrics | 🤝 Consumes | ✅ Owns | Node, network, storage |
| Log aggregation | 🤝 Configures | ✅ Provides | Shared logging infrastructure |
| Alerting rules | ✅ Application | ✅ Infrastructure | Separate scopes |

## Resource Management

| Component | Application Team | Platform Team | Notes |
|-----------|------------------|---------------|-------|
| Resource quotas | 🤝 Requests | ✅ Sets | Based on capacity planning |
| Namespace limits | 🤝 Requests | ✅ Enforces | Cluster resource protection |
| Pod resources | ✅ Specifies | 🤝 Validates | Within quota limits |
| Storage classes | 🤝 Consumes | ✅ Provides | NFS, local, cloud options |

## Escalation Scenarios

### Application Team → Platform Team

| Scenario | Escalation Required | Template |
|----------|-------------------|----------|
| "No matching servers available" | ✅ Yes | seldon-config-request.md |
| DNS resolution failures | ✅ Yes | network-policy-escalation.md |
| LoadBalancer IP conflicts | ✅ Yes | loadbalancer-request.md |
| Resource quota exceeded | ✅ Yes | quota-increase-request.md |
| Cross-namespace access | ✅ Yes | network-policy-request.md |

### Platform Team → Application Team

| Scenario | Coordination Required | Action |
|----------|---------------------|---------|
| Cluster maintenance | ✅ Yes | Advance notice, migration support |
| Security policy updates | ✅ Yes | Impact assessment, testing support |
| Resource quota changes | ✅ Yes | Capacity planning, timeline coordination |
| Network policy updates | ✅ Yes | Testing, validation support |

## Communication Protocols

### Standard Requests
1. **Create issue** in shared repository
2. **Use provided templates** for consistency
3. **Include impact assessment** (P0-P3 priority)
4. **Provide technical context** (logs, configs)
5. **Suggest validation steps** for verification

### Emergency Escalations
1. **Direct communication** to platform team
2. **Include business impact** in initial message
3. **Provide immediate workarounds** if available
4. **Follow up with formal request** for permanent fix

### Knowledge Sharing
1. **Document lessons learned** in shared repository
2. **Update troubleshooting guides** based on incidents
3. **Share best practices** across teams
4. **Maintain decision records** for architectural choices

## Decision Authority

### Application Team Decisions
- Application architecture within namespace
- Resource allocation within quota
- Deployment strategies and schedules
- Application-specific monitoring and alerting

### Platform Team Decisions  
- Cluster infrastructure choices
- Security policies and enforcement
- Resource quota allocation
- Infrastructure service configurations

### Joint Decisions
- Cross-namespace communication patterns
- Shared service configurations
- Disaster recovery procedures
- Security incident response