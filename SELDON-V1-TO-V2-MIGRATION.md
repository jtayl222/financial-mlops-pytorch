# 🚀 **Seldon Core v1.17.0 → v2.9.0 Migration Guide**

> **Real-world migration experience**: This guide documents actual challenges and solutions from migrating an enterprise MLOps platform from Seldon Core v1.17.0 to v2.9.0.

## **Executive Summary**

Seldon Core v2 represents a **complete architectural rewrite** - not an incremental upgrade. Expect a full reimplementation rather than configuration updates.

**Migration Complexity**: 🔴 **HIGH** - Plan for weeks, not days  
**Breaking Changes**: 🔴 **EXTENSIVE** - Everything changes  
**Benefits**: ✅ Better performance, simplified operations, enterprise features

---

## **🏗️ Core Architectural Changes**

### **v1.17.0 Architecture**
```yaml
# Simple SeldonDeployment
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: model-deployment
spec:
  predictors:
  - name: default
    replicas: 1
    graph:
      name: model
      implementation: MLFLOW_SERVER
```

### **v2.9.0 Architecture** 
```yaml
# Three separate resources required
apiVersion: mlops.seldon.io/v1alpha1
kind: SeldonRuntime        # Runtime infrastructure
---
apiVersion: mlops.seldon.io/v1alpha1  
kind: Model               # Model definition
---
apiVersion: mlops.seldon.io/v1alpha1
kind: Experiment          # Traffic routing
```

**Key Difference**: v1 had single resource deployment, v2 requires runtime + model + experiment pattern.

---

## **📋 Migration Checklist**

### **Phase 1: Infrastructure Preparation**
- [ ] **Deploy Seldon v2 Operator** (separate from v1)
- [ ] **Create SeldonRuntime** per namespace 
- [ ] **Configure RBAC** for cross-namespace communication
- [ ] **Set up NetworkPolicies** (critical - see networking section)
- [ ] **Configure ResourceQuotas** (generous limits required)

### **Phase 2: Model Migration**
- [ ] **Rewrite SeldonDeployments** as Model + Experiment resources
- [ ] **Update model serving framework** (MLServer replaces custom containers)
- [ ] **Migrate storage configurations** (S3/GCS/etc)
- [ ] **Update secret management** (new secret structure)

### **Phase 3: Networking & Security**
- [ ] **Fix NetworkPolicy conflicts** (biggest blocker)
- [ ] **Configure service mesh compatibility** (Istio/etc)
- [ ] **Update monitoring/observability** (new metrics endpoints)
- [ ] **Test cross-namespace connectivity**

---

## **🔥 Critical Migration Challenges**

### **1. NetworkPolicy Hell** 🔴 **MAJOR BLOCKER**

**Problem**: v2 requires cross-namespace communication between:
- Global controller (`seldon-system`) 
- Dedicated schedulers (per namespace)

**Symptoms**:
```bash
# Connection refused errors
rpc error: code = Unavailable desc = connection error: 
desc = "transport: Error while dialing: dial tcp 10.43.9.156:9004: connect: connection refused"
```

**Solution**:
```bash
# Option A: Disable NetworkPolicies temporarily
kubectl delete networkpolicy --all -n your-namespace

# Option B: Fix namespace labels
kubectl label namespace seldon-system name=seldon-system

# Option C: Create permissive NetworkPolicy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-cross-namespace
spec:
  podSelector: {}
  policyTypes: ["Ingress", "Egress"]
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: seldon-system
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: seldon-system
```

### **2. RBAC Configuration** 🟡 **MODERATE**

**Problem**: MLServer agents need extensive Kubernetes permissions.

**Solution**:
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: your-ml-namespace
rules:
- apiGroups: ["mlops.seldon.io"]
  resources: ["models", "experiments"]
  verbs: ["get", "list", "create", "update", "patch", "delete"]
- apiGroups: [""]
  resources: ["configmaps", "secrets", "services", "pods"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: mlserver-rolebinding
subjects:
- kind: ServiceAccount
  name: default  # Or dedicated service account
roleRef:
  kind: Role
  name: your-role
```

### **3. Resource Requirements** 🟡 **MODERATE**

**Problem**: v2 requires significantly more resources than v1.

**Before (v1)**:
```yaml
resources:
  requests:
    cpu: 100m
    memory: 512Mi
```

**After (v2)**:
```yaml
# Per namespace components
resources:
  requests:
    cpu: "2"      # 20x increase
    memory: 4Gi   # 8x increase
```

**Solution**: Update ResourceQuotas generously:
```yaml
apiVersion: v1
kind: ResourceQuota
spec:
  hard:
    requests.cpu: "50"
    requests.memory: 100Gi
    limits.memory: 200Gi
```

### **4. MLServer Agent Issues** 🟡 **MODERATE**

**Problem**: MLServer agent crashes due to missing RClone configuration.

**Symptoms**:
```
level=fatal msg="Failed to initialise rclone config listener" 
error="secrets \"seldon-rclone-gs-public\" not found"
```

**Solution**:
```bash
# Create dummy RClone secret
kubectl create secret generic seldon-rclone-gs-public \
  -n your-namespace --from-literal=dummy=dummy
```

---

## **🎯 Best Practices**

### **Namespace Strategy**
```yaml
# Recommended: Dedicated SeldonRuntime per namespace
apiVersion: mlops.seldon.io/v1alpha1
kind: SeldonRuntime
metadata:
  name: team-runtime
  namespace: team-namespace
spec:
  overrides:
  - name: seldon-scheduler
    replicas: 1
  - name: seldon-envoy
    replicas: 1
  - name: seldon-modelgateway
    replicas: 1
  - name: seldon-dataflow-engine
    replicas: 0  # Disable if no Kafka
  - name: seldon-pipelinegateway
    replicas: 0
  - name: mlserver
    replicas: 1
```

### **Model Definition Pattern**
```yaml
apiVersion: mlops.seldon.io/v1alpha1
kind: Model
metadata:
  name: your-model
  namespace: your-namespace
spec:
  storageUri: s3://bucket/path/to/model/
  requirements:
  - mlflow
  - torch
  - numpy
  secretName: your-s3-secret
  server: mlserver
```

### **Gradual Migration Strategy**
1. **Run v1 and v2 in parallel** (different namespaces)
2. **Migrate one model at a time**
3. **Validate functionality before decommissioning v1**
4. **Keep v1 as fallback during transition**

---

## **🐛 Common Debugging Commands**

```bash
# Check overall health
kubectl get seldonruntime -A
kubectl get models -A
kubectl get experiments -A

# Debug networking
kubectl logs seldon-v2-controller-manager-xxx -n seldon-system | grep your-namespace
kubectl logs seldon-scheduler-0 -n your-namespace

# Test connectivity
kubectl run debug --image=nicolaka/netshoot --rm --restart=Never \
  -n seldon-system -- nc -zv scheduler.your-namespace.svc.cluster.local 9004

# Check MLServer registration
kubectl logs mlserver-0 -c agent -n your-namespace
kubectl logs mlserver-0 -c mlserver -n your-namespace

# Monitor model scheduling
kubectl describe model your-model -n your-namespace
```

---

## **⏱️ Migration Timeline**

**Small deployment (1-3 models)**: 1-2 weeks  
**Medium deployment (10+ models)**: 3-4 weeks  
**Large deployment (50+ models)**: 6-8 weeks  

**Time breakdown**:
- Infrastructure setup: 30%
- Networking/RBAC fixes: 40%
- Model migration: 20%
- Testing/validation: 10%

---

## **🔍 Key Differences Summary**

| Aspect | v1.17.0 | v2.9.0 |
|--------|---------|--------|
| **Deployment** | Single SeldonDeployment | Runtime + Model + Experiment |
| **Networking** | Simple pod-to-pod | Cross-namespace controller-scheduler |
| **Resource Usage** | Lightweight | Resource-intensive |
| **RBAC** | Minimal | Extensive permissions required |
| **Storage** | Direct model loading | Agent-based with RClone |
| **Observability** | Prometheus/Grafana | Built-in telemetry + external |
| **Scaling** | Manual replica management | Automatic server scaling |

---

## **🎉 Success Criteria**

✅ **SeldonRuntime shows all components Ready**  
✅ **Models transition from `READY=False` to `READY=True`**  
✅ **MLServer agents register with scheduler**  
✅ **Cross-namespace connectivity working**  
✅ **No "connection refused" errors in controller logs**  
✅ **Models successfully serve predictions**

---

## **🚨 Migration Warnings**

1. **⚠️ Not backward compatible** - Complete rewrite required
2. **⚠️ Significantly higher resource requirements**
3. **⚠️ Complex networking requirements** - NetworkPolicies often conflict
4. **⚠️ Limited documentation** - Expect trial-and-error debugging
5. **⚠️ Breaking API changes** - All client code needs updates

---

## **📚 References**

- [Seldon Core v2 Documentation](https://docs.seldon.io/projects/seldon-core/en/v2/)
- [v1 to v2 Migration Guide](https://docs.seldon.io/projects/seldon-core/en/v2/contents/getting-started/migration.html)
- [NetworkPolicy Debugging](https://kubernetes.io/docs/concepts/services-networking/network-policies/)

---

**💡 Pro Tip**: Start with a clean namespace and minimal NetworkPolicies. Add security controls gradually after confirming basic functionality works.