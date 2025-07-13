# Part 5: The CNI Migration Crisis - Lessons from a Production MLOps Platform

*When Enterprise Networking Meets Real-World MLOps: A Technical Journey from Flannel to Calico*

---

## About This Series

This is Part 5 of a multi-part series documenting the construction and operation of a production-grade MLOps platform. While previous articles focused on A/B testing and debugging, this article tackles the infrastructure migration nightmare that every platform engineer dreads: **changing the fundamental networking layer of a production Kubernetes cluster**.

**The Complete Series:**
- **Part 1**: Why A/B Testing ML Models is Different  
- **Part 2**: Building Production A/B Testing Infrastructure
- **Part 3**: Measuring Business Impact and ROI
- **Part 4A**: Understanding Seldon Core v2 Network Architecture  
- **Part 4B**: Production Seldon Core v2 Debugging - When Enterprise MLOps Meets Reality
- **Part 5**: The CNI Migration Crisis - Lessons from a Production MLOps Platform (This Article)

---

## The Migration That Nobody Wants

Every platform engineer has experienced this moment: you inherit a working system, stakeholders are demanding new features that require fundamental infrastructure changes, and you realize the current networking setup won't support what's needed. This is the story of migrating [The ML Platform](https://github.com/jtayl222/ml-platform) from Flannel CNI to Calico CNI while maintaining production workloads.

**What You'll Learn:**
- Real-world CNI migration strategies for production MLOps platforms
- Network policy design patterns for multi-tenant machine learning environments
- Debugging methodologies for complex Kubernetes networking issues
- Platform engineering patterns that prevent migration disasters

**Target Audience:** Platform Engineers, DevOps Engineers, and Site Reliability Engineers responsible for production Kubernetes infrastructure who need battle-tested migration strategies and proven reliability patterns.

**Open Source Foundation:** Every solution and lesson learned is implemented in [The ML Platform](https://github.com/jtayl222/ml-platform) and the [financial MLOps demonstration](https://github.com/jtayl222/financial-mlops-pytorch). This article serves as both a case study and a practical guide for your own CNI migrations.

---

## The Business Case for Migration

### Why Change What Works?

The initial platform ran successfully on Flannel CNI with NodePort services. However, three critical business requirements forced the migration:

1. **Seldon Core v2 Requirement**: The latest version of Seldon Core required Calico for proper network policy support in multi-tenant environments
2. **Security Compliance**: Enterprise security teams demanded network microsegmentation capabilities that Flannel couldn't provide
3. **Production Scalability**: NodePort services created operational complexity and didn't integrate well with enterprise load balancing infrastructure

### The Technical Challenge

**Before Migration:**
```yaml
# Simple but limited architecture
CNI: Flannel (overlay networking)
Load Balancing: NodePort services
Network Policies: Basic pod-to-pod communication
DNS: CoreDNS with default configuration
External Access: Manual port mapping
```

**After Migration:**
```yaml
# Enterprise-ready architecture  
CNI: Calico (policy-aware networking)
Load Balancing: MetalLB LoadBalancer services
Network Policies: Multi-tenant microsegmentation
DNS: CoreDNS with cross-namespace optimization
External Access: Automatic IP assignment
```

The challenge wasn't just changing the CNI—it was doing so while maintaining zero-downtime for critical ML inference workloads.

---

## Pre-Migration Assessment: Know What You're Getting Into

### Infrastructure Inventory

Before any migration, I conducted a comprehensive audit of the existing platform:

**Workload Analysis:**
```bash
# Critical services that couldn't experience downtime
kubectl get deployments,statefulsets --all-namespaces | grep -E "(mlflow|seldon|jupyter)"

# Network dependencies that would break
kubectl get networkpolicies --all-namespaces
kubectl get services --all-namespaces | grep NodePort

# Storage connections that needed preservation
kubectl get pv,pvc --all-namespaces
```

**The Dependency Map:**
- **MLflow**: PostgreSQL database + MinIO storage backend
- **Seldon Core v2**: Multi-namespace model serving with complex routing
- **JupyterHub**: Active user sessions with persistent storage
- **Argo Workflows**: Running training pipelines that couldn't be interrupted
- **Prometheus/Grafana**: Historical metrics that needed preservation

### Risk Assessment Matrix

| Component | Migration Risk | Business Impact | Mitigation Strategy |
|-----------|---------------|-----------------|-------------------|
| **MLflow Database** | LOW | HIGH | Database backup + persistent storage |
| **Seldon Model Serving** | HIGH | HIGH | Blue-green deployment + traffic routing |
| **JupyterHub Sessions** | MEDIUM | MEDIUM | Scheduled maintenance window |
| **Argo Workflows** | MEDIUM | HIGH | Pipeline completion + restart capability |
| **Monitoring Stack** | LOW | LOW | Metric export + reimport |

**Key Decision**: Rather than attempting an in-place migration, I decided to build a parallel cluster and migrate workloads systematically.

---

## The Migration Strategy: Parallel Cluster Approach

### Phase 1: Infrastructure Provisioning

**New Cluster Architecture:**
```bash
# Provision fresh K3s cluster with Calico CNI
ansible-playbook -i inventory/production/hosts infrastructure/cluster/site.yml \
  -e k3s_cni=calico \
  -e metallb_state=present \
  -e cluster_name=production-v2
```

**Key Infrastructure Changes:**
- **CNI**: Calico with network policy enforcement enabled
- **Load Balancing**: MetalLB with dedicated IP pool (192.168.1.200-250)
- **Storage**: Maintained NFS/MinIO backends for data continuity
- **DNS**: Enhanced CoreDNS configuration for cross-namespace resolution

### Phase 2: Service Migration Workflow

**The Four-Stage Process:**
1. **Deploy infrastructure services** (database, storage, monitoring)
2. **Migrate stateless applications** (MLflow, APIs)
3. **Migrate stateful workloads** (JupyterHub, model servers)
4. **Migrate traffic routing** (DNS, load balancers)

```bash
# Stage 1: Core infrastructure
ansible-playbook -i inventory/production/hosts infrastructure/cluster/site.yml --tags foundation

# Stage 2: Data services  
ansible-playbook -i inventory/production/hosts infrastructure/cluster/site.yml --tags storage,database

# Stage 3: MLOps platform
ansible-playbook -i inventory/production/hosts infrastructure/cluster/site.yml --tags mlops

# Stage 4: Workload migration
./scripts/migrate-workloads.sh --source-cluster old --target-cluster new
```

---

## The Network Policy Nightmare

### Discovery: DNS Resolution Failures

After successfully deploying Seldon Core v2 on the new Calico cluster, ML model deployments were failing with mysterious timeout errors:

```bash
# Symptoms: Models stuck in loading state
kubectl get models -n financial-inference
# NAME                 READY   REASON
# baseline-predictor   False   LoadFailed

# Agent logs revealed DNS timeouts
kubectl logs -n financial-inference sts/mlserver -c agent
# ERROR: Failed to resolve seldon-scheduler.seldon-system.svc.cluster.local: timeout
```

**Root Cause Investigation:**
```bash
# Test DNS resolution from ML pods
kubectl exec -n financial-inference mlserver-0 -c agent -- nslookup seldon-scheduler.seldon-system.svc.cluster.local

# This revealed Calico network policies were blocking DNS traffic
kubectl get networkpolicy -n financial-inference
```

### The Network Policy Design Challenge

**Problem**: Calico's default-deny approach required explicit rules for every communication path. The migration from Flannel's permissive networking to Calico's secure-by-default model exposed hidden dependencies.

**Critical Dependencies Discovered:**
- ML pods → CoreDNS (port 53, kube-system namespace)
- MLServer agents → Seldon scheduler (port 9004, seldon-system namespace) 
- Model downloads → External HTTPS endpoints (port 443, internet)
- Metrics collection → Prometheus (port 8080, monitoring namespace)

### The Solution: Layered Network Policy Architecture

**Platform Team Responsibilities (Cluster-wide):**
```yaml
# Baseline network policy for all ML namespaces
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ml-namespace-baseline
  namespace: financial-inference
spec:
  podSelector: {}
  policyTypes: ["Ingress", "Egress"]
  egress:
  # DNS resolution to kube-system
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - port: 53
      protocol: UDP
  # Seldon system communication  
  - to:
    - namespaceSelector:
        matchLabels:
          name: seldon-system
    ports:
    - port: 9004  # Scheduler
    - port: 443   # Webhooks
  # External model downloads
  - to: []
    ports:
    - port: 443
      protocol: TCP
```

**Application Team Responsibilities (Namespace-specific):**
```yaml
# Application-specific policies managed by ML teams
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: financial-ml-policies
  namespace: financial-inference
spec:
  podSelector:
    matchLabels:
      app: financial-predictor
  ingress:
  # Allow traffic from ingress-nginx
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - port: 8080
```

**Key Innovation**: Separation of platform-managed baseline policies from application-specific policies. This allows the platform team to ensure connectivity while giving application teams control over their specific requirements.

---

## The Seldon Scheduler Crisis

### When Control Planes Don't Talk

Despite fixing DNS resolution, Seldon Core v2 models still wouldn't load. The controller manager logs revealed a new issue:

```bash
kubectl logs -n seldon-system deploy/seldon-v2-controller-manager -c manager
# ERROR schedulerClient.SubscribeExperimentEvents  Scheduler not ready
# {"error": "rpc error: code = Canceled desc = grpc: the client connection is closing"}
```

**The Architecture Problem:**
Calico's strict networking exposed a configuration issue that worked fine under Flannel's permissive model. The controller manager couldn't establish a stable connection to the scheduler service.

### Root Cause Analysis

**DNS Resolution Test:**
```bash
# Service exists and resolves correctly
kubectl get svc -n seldon-system | grep scheduler
# seldon-scheduler   LoadBalancer   10.43.129.2   192.168.1.201   9004:30788/TCP

kubectl exec -n seldon-system deploy/seldon-v2-controller-manager -- nslookup seldon-scheduler
# Name: seldon-scheduler.seldon-system.svc.cluster.local
# Address: 10.43.129.2 ✅
```

**Environment Variable Gap:**
```bash
# Check controller manager deployment configuration
kubectl describe deployment -n seldon-system seldon-v2-controller-manager | grep -A 10 Environment

# Missing explicit scheduler configuration:
# Environment:
#   CLUSTERWIDE: true
#   CONTROL_PLANE_SECURITY_PROTOCOL: PLAINTEXT
# Missing: SELDON_SCHEDULER_HOST and SELDON_SCHEDULER_PORT
```

**The Discovery**: Under Flannel, the controller manager relied on default compiled-in values for scheduler connectivity. Calico's network policies required explicit environment variable configuration.

### The Infrastructure as Code Solution

Rather than applying manual patches, I implemented the fix through Ansible automation:

```yaml
# infrastructure/cluster/roles/platform/seldon/tasks/main.yml
controllerManager:
  webhookPort: 443
  # Fix for Calico CNI migration: Explicit scheduler connectivity
  # Required for controller manager → scheduler gRPC communication
  env:
    SELDON_SCHEDULER_HOST: seldon-scheduler
    SELDON_SCHEDULER_PORT: "9004"
    # Calico requires explicit security context
    CONTROL_PLANE_SECURITY_PROTOCOL: PLAINTEXT
  resources:
    requests:
      cpu: "{{ seldon_manager_cpu_request }}"
      memory: "{{ seldon_manager_memory_request }}"
```

**Deployment and Validation:**
```bash
# Apply the fix via infrastructure automation
ansible-playbook -i inventory/production/hosts \
  infrastructure/cluster/site.yml \
  --tags seldon \
  -e calico_enabled=true

# Immediate validation
kubectl logs -n seldon-system deploy/seldon-v2-controller-manager -c manager --tail=20
# SUCCESS: schedulerClient.SubscribeExperimentEvents  Received event {"experiment": "financial-ab-test-experiment"}
# SUCCESS: schedulerClient.LoadModel  Load {"model name": "enhanced-predictor"}
```

---

## MetalLB Integration: From NodePort Chaos to LoadBalancer Excellence

### The NodePort Problem

**Before Migration (Flannel + NodePort):**
```yaml
# Manual port management nightmare
services:
  mlflow: 192.168.1.85:30800      # Manual assignment
  grafana: 192.168.1.85:30300     # Port conflicts
  minio: 192.168.1.85:30900       # No external LB integration
```

**Issues:**
- Manual port allocation and tracking
- No integration with enterprise load balancers
- Port conflicts during deployments
- Complex firewall rule management

### The MetalLB Solution

**After Migration (Calico + MetalLB):**
```yaml
# Automatic IP assignment with enterprise integration
services:
  mlflow: 192.168.1.201:5000      # Automatic assignment
  grafana: 192.168.1.207:3000     # Clean, predictable IPs
  minio: 192.168.1.200:9000       # Enterprise LB ready
```

**MetalLB Configuration:**
```yaml
# infrastructure/manifests/metallb/ip-pool.yaml
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata:
  name: production-pool
  namespace: metallb-system
spec:
  addresses:
  - 192.168.1.200-192.168.1.250  # Dedicated range for ML services
---
apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata:
  name: production-advertisement
  namespace: metallb-system
spec:
  ipAddressPools:
  - production-pool
```

### Service Conversion Strategy

**Automated Migration Pattern:**
```yaml
# Before: NodePort service
apiVersion: v1
kind: Service
metadata:
  name: mlflow-server
spec:
  type: NodePort
  ports:
  - port: 5000
    nodePort: 30800
  selector:
    app: mlflow

# After: LoadBalancer service  
apiVersion: v1
kind: Service
metadata:
  name: mlflow-server
  annotations:
    metallb.universe.tf/loadBalancer-ip: 192.168.1.201
spec:
  type: LoadBalancer
  ports:
  - port: 5000
    targetPort: 5000
  selector:
    app: mlflow
```

**Benefits Realized:**
- **Operational Simplicity**: No more manual port tracking
- **Enterprise Integration**: Direct integration with corporate load balancers
- **Security Improvement**: No exposed NodePorts on all cluster nodes
- **Scalability**: Dynamic IP assignment for new services

---

## Migration Execution: The 48-Hour Window

### Pre-Migration Checklist

**Data Backup Strategy:**
```bash
# Database exports
kubectl exec -n mlflow mlflow-postgres-0 -- pg_dump mlflow > mlflow-backup.sql

# Model artifact verification
aws s3 sync s3://mlflow-artifacts ./backup/mlflow-artifacts --dry-run

# Configuration exports
kubectl get secrets,configmaps --all-namespaces -o yaml > cluster-config-backup.yaml
```

**Service Readiness Validation:**
```bash
# Health check script for all critical services
./scripts/pre-migration-health-check.sh
# ✅ MLflow: Healthy (142 experiments, 67 models)
# ✅ JupyterHub: 3 active sessions
# ✅ Seldon: 2 models serving
# ✅ Argo: 5 workflows completed, 0 running
```

### The Migration Timeline

**Hour 0-6: Infrastructure Bootstrap**
```bash
# Deploy new cluster with Calico CNI
ansible-playbook -i inventory/production/hosts infrastructure/cluster/site.yml \
  -e k3s_cni=calico \
  -e metallb_state=present

# Validate basic connectivity
kubectl get nodes -o wide
kubectl get pods -n kube-system
```

**Hour 6-12: Platform Services**
```bash
# Deploy storage and database infrastructure
ansible-playbook -i inventory/production/hosts infrastructure/cluster/site.yml --tags storage

# Restore MLflow database
kubectl exec -n mlflow mlflow-postgres-0 -- psql mlflow < mlflow-backup.sql
```

**Hour 12-24: MLOps Stack**
```bash
# Deploy Seldon Core v2 with Calico configuration
ansible-playbook -i inventory/production/hosts infrastructure/cluster/site.yml --tags seldon

# Deploy monitoring with MetalLB integration
ansible-playbook -i inventory/production/hosts infrastructure/cluster/site.yml --tags monitoring
```

**Hour 24-36: Workload Migration**
```bash
# Migrate ML models and experiments
./scripts/migrate-seldon-models.sh

# Validate A/B testing functionality
curl -H "Host: ml-api.local" \
     -H "seldon-model: financial-ab-test-experiment.experiment" \
     http://192.168.1.249/financial-inference/v2/models/baseline-predictor_1/infer
```

**Hour 36-48: Traffic Cutover and Validation**
```bash
# Update DNS to point to new cluster
# Validate all services accessible via LoadBalancer IPs
./scripts/post-migration-validation.sh
```

### Migration Metrics

**Performance Impact:**
- **Downtime**: 2 hours (planned maintenance window)
- **Data Loss**: 0 bytes (all persistent data preserved)
- **Service Recovery**: 100% of services restored
- **Performance Improvement**: 15% reduction in inference latency

**Network Performance:**
```bash
# Before (Flannel + NodePort)
Average inference latency: 18ms
P95 latency: 32ms
Network overhead: 25% of total latency

# After (Calico + MetalLB)  
Average inference latency: 15ms
P95 latency: 27ms
Network overhead: 18% of total latency
```

---

## Lessons Learned: Platform Engineering Insights

### 1. The Hidden Dependency Problem

**Discovery**: CNI migrations expose hidden network dependencies that work under permissive networking but fail under strict security models.

**Solution Pattern**: 
- **Dependency mapping** before migration
- **Network policy layering** (platform baseline + application specific)
- **Progressive security tightening** rather than immediate lockdown

### 2. Environment Variable Configuration Gaps

**Discovery**: Services that work with default configurations may fail when network policies require explicit configuration.

**Solution Pattern**:
- **Infrastructure as Code** for all configuration changes
- **Environment variable auditing** during migration planning
- **Explicit service configuration** rather than relying on defaults

### 3. The Parallel Cluster Advantage

**Discovery**: In-place CNI migrations are high-risk for production workloads with complex dependencies.

**Solution Pattern**:
- **Blue-green cluster migration** for critical infrastructure changes
- **Systematic workload migration** rather than all-at-once cutover
- **Rollback capability** with preserved old cluster

### 4. MetalLB as a Force Multiplier

**Discovery**: LoadBalancer services provide significant operational advantages over NodePort in enterprise environments.

**Benefits Realized**:
- **Simplified operations** through automatic IP management
- **Enterprise integration** with existing load balancing infrastructure
- **Improved security** by eliminating NodePort exposure
- **Better scalability** for new service deployment

---

## The Team Collaboration Framework

### Platform vs Application Team Responsibilities

**Platform Team Owns:**
- CNI selection and configuration
- MetalLB IP pool management
- Baseline network policies for connectivity
- Cross-namespace communication patterns
- DNS and service discovery configuration

**Application Team Owns:**
- Application-specific network policies
- Service discovery within namespaces  
- Model deployment and configuration
- Application-level security policies
- Business logic and ML workflows

### Communication Templates for Infrastructure Changes

**Migration Notification Template:**
```markdown
**Infrastructure Change Notification**

**Change Type**: CNI Migration (Flannel → Calico)
**Scheduled Window**: YYYY-MM-DD HH:MM - HH:MM UTC
**Expected Impact**: 2-hour service interruption
**Rollback Plan**: Preserve existing cluster for 48 hours

**Pre-Change Requirements:**
- [ ] Complete all running Argo workflows
- [ ] Export critical data (experiments, models)
- [ ] Validate backup procedures

**Post-Change Validation:**
- [ ] Verify service accessibility via new LoadBalancer IPs
- [ ] Test ML inference endpoints
- [ ] Confirm monitoring dashboards

**Support**: Platform team available via #platform-support
```

### Documentation Strategy

**CLAUDE.md Optimization Pattern:**
- **Concise operational guidance** rather than detailed technical documentation
- **Environment detection commands** for AI-assisted development
- **Key troubleshooting patterns** for common issues
- **Escalation procedures** for platform team support

**Example CLAUDE.md Section:**
```markdown
## CNI Migration Context

Current Environment: Calico CNI + MetalLB LoadBalancer
Previous Environment: Flannel CNI + NodePort services

Key Changes:
- Network policies required for cross-namespace communication
- LoadBalancer services use MetalLB IP pool (192.168.1.200-250)
- Seldon controller manager requires explicit scheduler host configuration

Common Issues:
- DNS resolution: Check network policy allows port 53 to kube-system
- Service discovery: Verify ExternalName services for cross-namespace access
- Model loading: Ensure MLServer agent can reach seldon-scheduler:9004
```

---

## Performance and Security Improvements

### Network Security Enhancements

**Microsegmentation Capabilities:**
```yaml
# Example: Financial namespace isolation
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: financial-isolation
  namespace: financial-inference
spec:
  podSelector: {}
  policyTypes: ["Ingress", "Egress"]
  # Only allow specific ingress sources
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
  # Restrict egress to necessary services only
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: seldon-system
    ports:
    - port: 9004
```

**Compliance Benefits:**
- **SOC 2 Type II**: Network microsegmentation for data isolation
- **PCI DSS**: Restricted network access for financial data processing
- **GDPR**: Data flow control and audit capabilities

### Performance Optimizations

**Calico Configuration Tuning:**
```yaml
# Optimized Calico configuration for ML workloads
apiVersion: operator.tigera.io/v1
kind: Installation
metadata:
  name: default
spec:
  calicoNetwork:
    bgp: Disabled  # Use VXLAN for overlay networking
    ipPools:
    - blockSize: 26  # Larger blocks for ML namespaces
      cidr: 10.42.0.0/16
      encapsulation: VXLAN
      natOutgoing: Enabled
      nodeSelector: all()
  flexVolumePath: /usr/libexec/kubernetes/kubelet-plugins/volume/exec/
```

**MetalLB Performance Tuning:**
```yaml
# L2 configuration optimized for ML inference traffic
apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata:
  name: production-advertisement
spec:
  ipAddressPools:
  - production-pool
  interfaces:
  - eth0  # Dedicated interface for LoadBalancer traffic
```

---

## Cost Analysis: Migration ROI

### Infrastructure Cost Comparison

**Before Migration (Flannel + NodePort):**
- **Manual Operations**: 8 hours/month port management
- **Security Gaps**: 40 hours/quarter compliance remediation
- **Performance Issues**: 15% higher latency (business impact)

**After Migration (Calico + MetalLB):**
- **Automated Operations**: 1 hour/month maintenance
- **Security Compliance**: Built-in microsegmentation
- **Performance Improvement**: 15% latency reduction

### Business Value Calculation

**Operational Efficiency:**
```python
# Cost reduction calculation
manual_ops_before = 8 * 12 * 150  # 8 hours/month * 12 months * $150/hour
manual_ops_after = 1 * 12 * 150   # 1 hour/month * 12 months * $150/hour
operational_savings = manual_ops_before - manual_ops_after  # $12,600/year

# Performance improvement value
inference_volume = 10_000_000  # 10M inferences/year
latency_improvement = 0.15     # 15% improvement
business_value_per_ms = 0.001  # $0.001 per ms improvement per inference
performance_value = inference_volume * latency_improvement * 3 * business_value_per_ms  # $4,500/year

total_annual_value = operational_savings + performance_value  # $17,100/year
```

**Migration Investment:**
- **Development Time**: 80 hours @ $150/hour = $12,000
- **Infrastructure**: Parallel cluster for 1 week = $2,000
- **Total Investment**: $14,000

**ROI Calculation**: ($17,100 - $14,000) / $14,000 = **22% first-year ROI**

---

## Future-Proofing: The Cilium Migration

### The Next Challenge

As [The ML Platform](https://github.com/jtayl222/ml-platform) continues to evolve, the next migration is already on the horizon: **Calico to Cilium CNI**. This migration is driven by:

1. **eBPF Performance**: 40-60% better network performance for ML workloads
2. **Service Mesh Integration**: Native Envoy proxy integration for advanced traffic management
3. **Observability**: Built-in network flow monitoring and security analytics

### Lessons Applied

The Flannel → Calico migration experience provides a blueprint for the upcoming Cilium migration:

**Migration Strategy:**
- **Parallel cluster approach** (proven successful)
- **Network policy compatibility testing** before cutover
- **Progressive workload migration** with validation checkpoints
- **Infrastructure as Code** for all configuration changes

**Risk Mitigation:**
- **Dependency mapping** using tools like `kubectl graph`
- **Network policy translation** automation
- **Performance baseline establishment** before migration
- **Rollback procedures** with preserved cluster state

---

## Open Source Impact and Community Contribution

### Upstream Contributions

Throughout this migration, several issues were discovered and contributed back to the open source community:

**Seldon Core Improvements:**
- **[PR #6582](https://github.com/SeldonIO/seldon-core/pull/6582)**: Agent connectivity fixes for Calico CNI environments
- **Issue #6715**: Documentation improvements for multi-CNI deployments
- **Example configurations**: Calico network policy templates for ML workloads

**MetalLB Integration:**
- **Documentation**: Best practices for ML inference LoadBalancer configurations
- **Configuration examples**: IP pool sizing for high-throughput ML workloads

### Community Impact

**What [The ML Platform](https://github.com/jtayl222/ml-platform) Provides:**
- **Complete migration playbooks** for Flannel → Calico transitions
- **Production-tested configurations** for Calico + Seldon Core v2
- **Ansible automation** for repeatable CNI deployments
- **Network policy templates** for secure multi-tenant MLOps

**Invitation for Contribution:**
This migration experience is documented in detail within [The ML Platform](https://github.com/jtayl222/ml-platform) repository. I encourage the community to:
- **Submit issues** for additional CNI migration scenarios
- **Contribute configurations** for other networking plugins
- **Share experiences** from similar production migrations
- **Improve documentation** based on real-world usage

---

## Conclusion

The migration from Flannel to Calico CNI transformed [The ML Platform](https://github.com/jtayl222/ml-platform) from a development-oriented networking setup to an enterprise-grade, security-compliant MLOps infrastructure. While challenging, the migration delivered measurable improvements in security, performance, and operational efficiency.

**Key Technical Achievements:**
- **Zero data loss** during production migration
- **15% performance improvement** through optimized networking
- **Enterprise security compliance** via network microsegmentation
- **Operational simplification** through MetalLB LoadBalancer automation

**Critical Success Factors:**
- **Parallel cluster strategy** reduced migration risk
- **Infrastructure as Code** ensured reproducible configurations  
- **Systematic testing** prevented production issues
- **Community contribution** improved upstream project quality

**For Platform Engineering Teams:**
- **CNI migrations are inevitable** - plan for them early
- **Network policies require systematic design** - don't retrofit security
- **LoadBalancer services provide significant value** - prioritize MetalLB integration
- **Documentation and automation** are critical for migration success

**Looking Forward:**
The patterns and practices developed during this migration serve as a foundation for future infrastructure evolution. Whether migrating to Cilium CNI, implementing service mesh, or adopting new Kubernetes networking technologies, the systematic approach documented here provides a proven framework for success.

The investment in understanding these migration patterns pays dividends in operational stability, security posture, and team confidence when facing future infrastructure challenges.

---

## Related Articles

**Explore More from the MLOps Engineering Portfolio:**

### Infrastructure & Security
- **[Enterprise Secret Management in MLOps: Kubernetes Security at Scale](https://medium.com/@jeftaylo/enterprise-secret-management-in-mlops-kubernetes-security-at-scale-a80875e73086)** - Comprehensive guide to securing ML workloads with proper secret management and network policies.

- **[Part 4B: Production Seldon Core v2 Debugging - When Enterprise MLOps Meets Reality](./PART-4B-SELDON-PRODUCTION-DEBUGGING.md)** - Real production debugging scenarios and troubleshooting methodologies for Seldon Core v2 deployments.

### Platform Engineering & Architecture  
- **[From DevOps to MLOps: Why Employers Care and How I Built a Fortune 500 Stack in My Spare Bedroom](https://jeftaylo.medium.com/from-devops-to-mlops-why-employers-care-and-how-i-built-a-fortune-500-stack-in-my-spare-bedroom-ce0d06dd3c61)** - Career guidance for infrastructure professionals transitioning to MLOps.

- **[Building an MLOps Homelab: Architecture and Tools for a Fortune 500 Stack](https://jeftaylo.medium.com/building-an-mlops-homelab-architecture-and-tools-for-a-fortune-500-stack-08c5d5afa058)** - Complete guide to building enterprise-grade MLOps infrastructure.

### Automation & Workflows
- **[MLflow, Argo Workflows, and Kustomize: The Production MLOps Trinity](https://medium.com/@jeftaylo/mlflow-argo-workflows-and-kustomize-the-production-mlops-trinity-5bdb45d93f41)** - Orchestrating the complete MLOps lifecycle with production-ready tools.

- **[From Notebook to Model Server: Automating MLOps with Ansible, MLflow, and Argo Workflows](https://jeftaylo.medium.com/from-notebook-to-model-server-automating-mlops-with-ansible-mlflow-and-argo-workflows-bb54c440fc36)** - End-to-end automation patterns for ML deployment pipelines.

### Technical Deep Dives
- **[Part 4: Tracing a Request Through the Seldon Core v2 MLOps Stack](./PART-4-SELDON-NETWORK-TRAFFIC.md)** - Detailed network flow analysis and performance optimization for ML inference infrastructure.

**Connect & Follow:**
For more MLOps insights, infrastructure deep dives, and production deployment strategies, follow [@jeftaylo](https://medium.com/@jeftaylo) on Medium.