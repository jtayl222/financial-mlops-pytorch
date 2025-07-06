# 🎯 **Seldon Core v2 Expert Consultation Request**

## **Context & Background**

We're implementing an enterprise MLOps platform migration from Seldon Core v1.17.0 to v2.9.0, following industry best practices for multi-tenant namespace isolation (Netflix, Spotify, Uber patterns). We've successfully resolved major technical hurdles but have encountered architectural questions requiring expert guidance.

## **Current Architecture**

### **Environment**
- **Platform**: K3s cluster (5 nodes)
- **Seldon Core**: v2.9.0 
- **Deployment Pattern**: Dedicated SeldonRuntime per namespace
- **Enterprise Context**: Financial ML platform with strict tenant isolation

### **Namespace Strategy**
```yaml
# Namespace 1: financial-ml (production models)
apiVersion: mlops.seldon.io/v1alpha1
kind: SeldonRuntime
metadata:
  name: financial-ml-runtime
  namespace: financial-ml
spec:
  overrides:
  - name: seldon-scheduler
    replicas: 1
  - name: mlserver  
    replicas: 1
  # ... other components

# Namespace 2: financial-mlops-pytorch (model development)  
# Similar SeldonRuntime configuration
```

## **Technical Progress & Achievements**

### **✅ Successfully Resolved**
1. **Cross-namespace networking** - Controller in `seldon-system` ↔ Schedulers in tenant namespaces
2. **RBAC configuration** - Proper service account permissions for MLServer agents
3. **Secret management** - Package-based secrets with infrastructure team collaboration
4. **NetworkPolicy issues** - Resolved traffic blocking between components
5. **Critical bug fix** - Agent `SELDON_SERVER_HOST` environment variable handling ([related to PR #6582](https://github.com/SeldonIO/seldon-core/pull/6582))

### **Current Technical Status**
```bash
# ✅ Infrastructure healthy
kubectl get pods -n financial-ml
# NAME                                   READY   STATUS    RESTARTS   AGE
# hodometer-7db8996f79-lrpp8             1/1     Running   0          35h
# seldon-envoy-5d9f4c78f9-ql922          1/1     Running   0          35h  
# seldon-modelgateway-766f559c94-tmnfx   1/1     Running   0          35h
# seldon-scheduler-0                     1/1     Running   0          35h
# mlserver-0                             3/3     Running   0          2h

# ✅ Agent registration successful
kubectl logs seldon-scheduler-0 -n financial-ml | grep subscribe
# time="2025-07-06T14:00:00Z" level=info msg="Received subscribe request from mlserver:0" 

# ❌ Zero capacity reporting
kubectl logs seldon-scheduler-0 -n financial-ml | grep "Empty server"
# time="2025-07-06T14:01:36Z" level=warning msg="Empty server for test-model-simple:1 so ignoring event"
```

## **Primary Question: Architecture Best Practices**

### **Current Challenge**
We have **MLServer agents successfully subscribing to schedulers** but **reporting zero capacity**, preventing model scheduling. The agent logs show:

```bash
# Agent connects to scheduler successfully:
time="2025-07-06T14:00:00Z" level=info msg="Subscribed to scheduler"

# But MLServer health checks fail:
time="2025-07-06T13:59:59Z" level=info msg="Waiting for Inference Server service to become ready"
error="rpc error: code = Unavailable desc = connection error: desc = \"transport: Error while dialing: dial tcp 10.42.0.70:9500: connect: connection refused\""

# Agent proceeds to "ready" anyway:
time="2025-07-06T14:00:00Z" level=info msg="All critical agent subservices ready"

# Result: Zero capacity reported to scheduler
time="2025-07-06T14:01:36Z" level=warning msg="Empty server for test-model-simple:1 so ignoring event"
```

### **Architectural Questions for Expert**

#### **1. Multi-Namespace MLServer Architecture**
**Question**: What is the recommended architecture for MLServer deployment in multi-tenant Seldon v2 environments?

**Options we've considered**:
- **A)** Dedicated MLServer per namespace (current approach)
- **B)** Shared MLServer in `seldon-system` with cross-namespace model references 
- **C)** Hybrid approach with both shared and dedicated servers

**Current findings**:
- Cross-namespace server references (`spec.server: "seldon-system/mlserver"`) fail because schedulers can only schedule to locally registered servers
- Each SeldonRuntime appears to operate independently within its namespace

#### **2. Agent Capacity Reporting Mechanism**
**Question**: How should MLServer agents handle capacity reporting when intra-pod connectivity is restricted?

**Technical context**:
- MLServer binds to `0.0.0.0:9500` (all interfaces)
- Agent attempts connection via pod IP (`10.42.0.70:9500`) 
- Connection fails due to platform networking restrictions
- Agent reports "ready" but with zero capacity

**Is there a configuration to**:
- Skip MLServer health checks for capacity reporting?
- Use alternative capacity detection mechanisms?
- Configure agent to assume MLServer availability?

#### **3. Enterprise Deployment Patterns**
**Question**: For enterprise environments with strict namespace isolation, what are the recommended patterns?

**Our requirements**:
- **Tenant isolation**: Complete separation between `financial-ml` and `financial-mlops-pytorch`
- **Security**: No cross-tenant data access
- **Scalability**: Independent scaling per namespace
- **Compliance**: Audit trails per tenant

**Current approach validation**:
```yaml
# Per-namespace SeldonRuntime
spec:
  overrides:
  - name: seldon-scheduler
    replicas: 1      # Dedicated scheduler per namespace
  - name: mlserver
    replicas: 1      # Dedicated MLServer per namespace
  - name: seldon-modelgateway  
    replicas: 1      # Dedicated gateway per namespace
```

#### **4. Networking Requirements**
**Question**: What are the minimum networking requirements for Seldon v2 components?

**Current connectivity matrix**:
```
✅ Global Controller → Namespace Schedulers (cross-namespace)
✅ Agent → Scheduler (intra-namespace) 
✅ External → Services (via NodePort/LoadBalancer)
❌ Agent → MLServer (intra-pod) - BLOCKED BY PLATFORM
```

**Specific questions**:
- Is intra-pod localhost connectivity mandatory for proper operation?
- Can agents operate with external MLServer connections only?
- Are there alternative communication patterns for restricted environments?

## **Platform Constraints**

### **K3s Networking Limitations**
- **CNI**: Flannel (default K3s configuration)
- **Restriction**: Intra-pod localhost connectivity blocked by platform
- **Similar issues**: Previously resolved with JupyterHub by disabling NetworkPolicies
- **Infrastructure team**: Working on networking solution but timeline uncertain

### **Current Workarounds Attempted**
```yaml
# 1. Pod IP communication (partially working)
env:
- name: SELDON_SERVER_HOST
  valueFrom:
    fieldRef:
      fieldPath: status.podIP  # Agent connects to pod IP instead of localhost

# 2. Binding configuration
env:  
- name: MLSERVER_HOST
  value: "0.0.0.0"  # MLServer binds to all interfaces

# 3. NetworkPolicy permissive rules
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: mlserver-allow-intra-pod
spec:
  podSelector:
    matchLabels:
      seldon-server-name: mlserver
  policyTypes: [Ingress, Egress]
  ingress: [{}]  # Allow all
  egress: [{}]   # Allow all
```

**Results**: Agent subscription works, but capacity reporting still fails

## **Specific Expert Guidance Needed**

### **1. Immediate Fix**
Given our platform networking constraints, what configuration changes can enable model scheduling with the current setup?

### **2. Architecture Validation** 
Is our per-namespace SeldonRuntime approach aligned with Seldon v2 best practices for enterprise multi-tenancy?

### **3. Alternative Patterns**
Are there proven deployment patterns for restricted networking environments that we should consider?

### **4. Capacity Reporting**
Can MLServer agents be configured to report capacity without successful intra-pod health checks?

## **Success Criteria**

```bash
# Target outcome:
kubectl get model test-model-simple -n financial-ml
# NAME                READY   DESIRED REPLICAS   AVAILABLE REPLICAS   AGE  
# test-model-simple   True    1                  1                    5m

# With successful scheduling:
kubectl logs seldon-scheduler-0 -n financial-ml | grep "test-model-simple"
# time="2025-07-06T14:XX:XX" level=info msg="Model test-model-simple scheduled successfully"
```

## **Current Model Configuration**
```yaml
apiVersion: mlops.seldon.io/v1alpha1
kind: Model
metadata:
  name: test-model-simple
  namespace: financial-ml
spec:
  storageUri: s3://mlflow-artifacts/28/models/m-d6d788df1b5849b3a3df1d04434c17b9/artifacts/
  requirements:
  - mlflow
  - torch  
  - numpy
  - scikit-learn
  secretName: ml-platform
  server: mlserver
```

## **Additional Context**

### **Migration Status**
- **Phase**: 95% complete - only model scheduling remains
- **Timeline**: Enterprise migration deadline approaching
- **Impact**: Production ML workload deployment blocked

### **Team Experience**
- **Seldon v1**: 2+ years production experience
- **Seldon v2**: New adoption - following migration guides
- **Platform**: Kubernetes-native team with extensive k8s experience

### **Documentation References**
- [Seldon Core v2 Architecture](https://docs.seldon.io/projects/seldon-core/en/v2/contents/getting-started/your-first-model.html)
- [Multi-tenancy patterns](https://docs.seldon.io/projects/seldon-core/en/v2/contents/advanced/multi-tenancy.html)
- Migration from v1→v2 (following official migration guide)

---

## **Request Summary**

We need **Seldon Core v2 architectural guidance** for:
1. **Multi-namespace MLServer deployment patterns** in enterprise environments
2. **Agent capacity reporting alternatives** for restricted networking
3. **Validation of our current approach** vs. recommended best practices
4. **Immediate configuration fixes** to enable model scheduling

**Priority**: High - blocking production migration completion  
**Contact**: Enterprise MLOps Engineering Team  
**Timeline**: Seeking expert consultation within 48-72 hours

Thank you for your expertise in helping us complete this critical Seldon v2 migration!