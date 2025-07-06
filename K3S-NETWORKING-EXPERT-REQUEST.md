# 🚨 **K3s Networking Expert Consultation Request**

## **Issue Summary**
We've successfully resolved the major Seldon Core v2 migration issues but are blocked by a **K3s intra-pod networking restriction** that prevents containers within the same pod from communicating via localhost.

## **Environment**
- **Cluster**: K3s v1.32.5+k3s1 / v1.33.1+k3s1 (5 nodes)
- **CNI**: Flannel (K3s default)
- **Container Runtime**: containerd 2.0.5-k3s1
- **OS**: Ubuntu 24.04.2 LTS
- **Application**: Seldon Core v2.9.0 MLOps platform

## **Problem Statement**

### **What Works**
✅ **Inter-pod communication**: Cross-namespace networking functions perfectly  
✅ **External access**: Services accessible from outside the cluster  
✅ **Pod-to-service**: Pods can reach other pods via Kubernetes services  
✅ **Agent subscription**: Agent successfully connects to scheduler in different namespace

### **What Fails**
❌ **Intra-pod localhost connectivity**: Container A cannot connect to Container B within same pod via `127.0.0.1` or `localhost`

## **Specific Technical Details**

### **Pod Architecture**
```
┌─────────────────────────────────────────────────────────┐
│ MLServer Pod (mlserver-0)                               │
│ ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│ │   rclone    │  │    agent     │  │    mlserver     │  │
│ │   :5572     │  │              │  │ HTTP: :9000     │  │
│ │             │  │              │  │ gRPC: :9500◄────┼──┼─ FAILS HERE
│ └─────────────┘  └──────────────┘  └─────────────────┘  │
│                           │                             │
│                           └─────────────────────────────┼─ dial tcp 127.0.0.1:9500: 
│                         (tries to connect)              │  connect: connection refused
└─────────────────────────────────────────────────────────┘
```

### **Error Pattern**
```bash
# MLServer successfully binds to all interfaces:
[mlserver.grpc] INFO - gRPC server running on http://0.0.0.0:9500

# Agent fails to connect via localhost:
level=error msg="Waiting for Inference Server service to become ready"
error="rpc error: code = Unavailable desc = connection error: desc = \"transport: Error while dialing: dial tcp 127.0.0.1:9500: connect: connection refused\""
```

### **Verification Evidence**
```bash
# MLServer container is healthy and accessible via service:
kubectl get svc mlserver -n financial-ml
# NAME       TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)    AGE
# mlserver   ClusterIP   10.43.116.121  <none>        9500/TCP   1d

# External gRPC connectivity works:
kubectl port-forward mlserver-0 9500:9500 -n financial-ml
# Successfully connects from outside pod

# Localhost connectivity fails from within pod:
kubectl exec mlserver-0 -c agent -n financial-ml -- curl localhost:9000
# Connection refused (if curl were available)
```

## **Attempted Solutions**

### **1. NetworkPolicy Approaches**
```yaml
# Tried explicit permissive NetworkPolicy:
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
**Result**: No effect - still connection refused

### **2. MLServer Binding Configuration**
```yaml
# Tried different binding addresses:
- name: MLSERVER_HOST
  value: "0.0.0.0"     # Default - fails
- name: MLSERVER_HOST  
  value: "127.0.0.1"   # Localhost only - fails
- name: MLSERVER_HOST
  value: "::1"         # IPv6 localhost - not tested
```
**Result**: Both `0.0.0.0` and `127.0.0.1` binding fail for localhost connectivity

### **3. Agent Connection Configuration**
```yaml
# Tried different connection targets:
- name: SELDON_SERVER_HOST
  value: "localhost"   # Fails
- name: SELDON_SERVER_HOST
  value: "127.0.0.1"   # Fails  
- name: SELDON_SERVER_HOST
  value: "::1"         # Fails (IPv6)
```
**Result**: All localhost variants fail with "connection refused"

### **4. Service Mesh Investigation**
- **Istio**: Not blocking intra-pod traffic (verified)
- **Flannel CNI**: Default K3s configuration
- **iptables**: Not examined in detail

## **Similar Issues Resolved**

### **JupyterHub Success Case**
We previously resolved similar intra-pod connectivity issues with JupyterHub by disabling NetworkPolicies:
```yaml
singleuser:
  networkPolicy:
    enabled: false
```
This suggests the issue is **platform-level networking restrictions** rather than application configuration.

### **Seldon Core v2 Bug Fix**
We successfully resolved the primary Seldon Core v2 bug (agent ignoring `SELDON_SERVER_HOST` environment variable) by implementing missing source code functionality. This confirms our application configuration is correct.

## **Questions for K3s Networking Expert**

### **1. K3s Container Networking**
- Does K3s/containerd have container isolation that blocks localhost connectivity between containers in the same pod?
- Are there K3s flags or configurations that control intra-pod networking behavior?

### **2. CNI Configuration**
- Could Flannel CNI settings be preventing localhost traffic within pods?
- Are there Flannel configuration options to allow intra-pod localhost communication?

### **3. iptables/Netfilter Rules**
- Could K3s be installing iptables rules that block localhost traffic between containers?
- What commands should we run to diagnose container-level networking restrictions?

### **4. Alternative Solutions**
- Should we use pod IP addresses instead of localhost for intra-pod communication?
- Are there K3s-specific networking patterns for multi-container pod communication?
- Could this be resolved by changing the pod's `dnsPolicy` or `hostNetwork` settings?

## **Diagnostic Commands**

### **Network Configuration**
```bash
# Check K3s cluster configuration:
kubectl get nodes -o wide
kubectl describe node [node-name] | grep -A10 "System Info"

# Check CNI configuration:
ls /var/lib/rancher/k3s/server/manifests/
cat /var/lib/rancher/k3s/server/manifests/flannel.yaml

# Check pod networking:
kubectl exec mlserver-0 -c agent -n financial-ml -- ip addr show
kubectl exec mlserver-0 -c agent -n financial-ml -- ip route show
```

### **Container Runtime**
```bash
# Check containerd configuration:
crictl ps | grep mlserver
crictl inspect [container-id]

# Check namespace isolation:
kubectl exec mlserver-0 -c agent -n financial-ml -- cat /proc/1/cgroup
```

## **Expected Outcome**

We need **intra-pod localhost connectivity** to work so that:
1. ✅ Agent container can connect to MLServer gRPC at `localhost:9500`
2. ✅ Agent reports proper capacity to scheduler  
3. ✅ Models can be scheduled and deployed successfully
4. ✅ Seldon Core v1→v2 migration completes

## **Business Impact**

**Severity**: **P1 Critical**
- **Production Migration Blocked**: Cannot complete Seldon v1→v2 upgrade
- **MLOps Platform Offline**: All model serving capabilities unavailable
- **Timeline Impact**: Enterprise migration deadline at risk

## **Current Workaround**

No viable workaround identified. External service access works, but agent requires localhost connectivity for proper capacity reporting and model lifecycle management.

---

## **Contact Information**

**Team**: Enterprise MLOps Engineering  
**Environment**: Production K3s cluster  
**Urgency**: High - blocking critical migration

**Request**: Expert consultation on K3s intra-pod networking configuration to enable localhost connectivity between containers within the same pod.

## **Additional Context**

### **Successful Components**
- ✅ Cross-namespace networking (controller ↔ scheduler)
- ✅ Service discovery and DNS resolution  
- ✅ External pod access via services
- ✅ RBAC and secret management
- ✅ Storage and persistence

### **Only Remaining Issue**
- ❌ Intra-pod localhost connectivity (agent ↔ MLServer)

This appears to be a **K3s platform configuration issue** rather than an application bug, as evidenced by similar JupyterHub networking challenges that required platform-level networking policy changes.