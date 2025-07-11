# Platform Team Escalation: Calico CNI Networking Issue

## Request Type
**Infrastructure Networking** - Cluster-level connectivity issue after Flannel to Calico migration

## Issue Description
Models in `financial-inference` namespace cannot connect to Seldon scheduler, causing deployment failures with timeout errors.

## Error Details
```
rpc error: code = Unavailable desc = connection error: desc = "transport: Error while dialing: dial tcp 10.43.51.131:9004: i/o timeout"
```

## Impact
- **Critical**: All model deployments fail to schedule
- **Affected Services**: Seldon Core v2 model serving
- **Business Impact**: Complete model serving pipeline blocked

## Environment Context
- **Previous**: Flannel CNI + NodePort services
- **Current**: Calico CNI + MetalLB LoadBalancer services
- **Cluster**: Fresh k3s installation (v1.33.1+k3s1)

## Investigation Completed

### ✅ Application-Level Actions Taken
1. **Network policies updated** for Calico environment (`k8s/base/network-policy.yaml`)
2. **RBAC and resource quotas** verified and working
3. **Cross-namespace communication** configured correctly
4. **Services and pods** are running healthy

### ✅ Confirmed Working
- All Seldon components running in both `financial-inference` and `seldon-system`
- MLServer and Triton servers operational in `seldon-system`
- Application-level network policies applied successfully
- LoadBalancer services available via MetalLB

### ❌ Platform-Level Issue
- Models cannot reach scheduler at `10.43.51.131:9004`
- Connection timeouts suggest cluster networking problem
- Likely related to Calico CNI configuration or DNS resolution

## Technical Details

### Model Status
```bash
kubectl describe model baseline-predictor -n financial-inference
# Shows persistent "ModelProgressing" with scheduler timeout
```

### Network Architecture
- **Source**: Models in `financial-inference` namespace
- **Target**: Seldon scheduler in same namespace (port 9004)
- **Expected**: Direct communication within namespace
- **Actual**: Connection timeout

### Seldon Components
```bash
kubectl get pods -n financial-inference
# hodometer, seldon-envoy, seldon-modelgateway, seldon-scheduler all running
```

## Platform Team Action Required

### 1. Cluster-Wide Network Policy Review
Verify that Calico installation allows:
- **Intra-namespace communication** within `financial-inference`
- **DNS resolution** from application namespaces to `kube-system`
- **Service discovery** within namespaces

### 2. Calico CNI Configuration
Check if Calico network policies or configuration are blocking:
- Pod-to-pod communication within namespaces
- Service endpoint connectivity
- gRPC traffic on custom ports (9004, 9005)

### 3. Service Discovery
Verify that:
- Services are properly registered in CoreDNS
- Network routes are correctly configured for pod IPs
- No firewall rules blocking internal traffic

## Validation Steps

After platform team applies fixes:

1. **Test model connectivity:**
   ```bash
   kubectl get models -n financial-inference
   # Should show Ready: True
   ```

2. **Verify no timeout errors:**
   ```bash
   kubectl describe model baseline-predictor -n financial-inference
   # Should show "ModelReady" instead of timeout
   ```

3. **Confirm end-to-end functionality:**
   ```bash
   # Models should successfully schedule and become available
   kubectl get experiments -n financial-inference
   ```

## Application Team Status
- **Network policies**: ✅ Updated for Calico environment  
- **Application config**: ✅ Ready for platform networking fix
- **Monitoring**: ✅ Ready to test after cluster fix

## Priority
**P0 - Critical** - Blocks all model serving capabilities

## References
- `NETWORK-POLICY-GUIDELINES.md` - Responsibility matrix
- `k8s/base/network-policy.yaml` - Application-level policies (working)
- Feature branch: `feature/fix-seldon-model-deployment`

---
**Request Date**: 2025-07-07  
**Status**: Awaiting Platform Team Response  
**Contact**: financial-mlops-pytorch team