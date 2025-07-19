# CNI Migration Checklist - Flannel to Calico

## Pre-Migration Assessment

### 1. Document Current State
- [ ] Record existing CNI implementation: `kubectl get pods -n kube-system | grep -E "(flannel|calico)"`
- [ ] Document LoadBalancer configuration: `kubectl get svc -A | grep LoadBalancer`
- [ ] List all network policies: `kubectl get networkpolicies -A`
- [ ] Export current configurations: `kubectl get all -A -o yaml > pre-migration-backup.yaml`

### 2. Identify Dependencies
- [ ] Map cross-namespace communication requirements
- [ ] Document external service access patterns
- [ ] List DNS dependencies (kube-system, etc.)
- [ ] Inventory LoadBalancer service requirements

### 3. Environment Context
- [ ] Cluster version: `kubectl version`
- [ ] Node configuration: `kubectl get nodes -o wide`
- [ ] Resource quotas: `kubectl get resourcequota -A`
- [ ] Storage classes: `kubectl get storageclass`

## Migration Preparation

### 1. Backup Critical Data
- [ ] Git commit all current configurations
- [ ] Export secrets (if possible): `kubectl get secrets -A`
- [ ] Document service endpoints and IPs
- [ ] Backup MLflow experiment data
- [ ] Preserve trained model artifacts

### 2. Update Network Policies
- [ ] Add DNS resolution rules (port 53 to kube-system)
- [ ] Add Kubernetes API access (port 443/6443)
- [ ] Update cross-namespace communication rules
- [ ] Add MLflow namespace egress rules

### 3. Service Configuration Updates
- [ ] Change external IPs to internal service names
- [ ] Update MLFLOW_TRACKING_URI to use cluster DNS
- [ ] Modify storage endpoints for internal routing
- [ ] Update any hardcoded IP addresses

## Post-Migration Validation

### 1. Infrastructure Verification
- [ ] Confirm CNI: `kubectl get pods -n kube-system | grep calico`
- [ ] Verify LoadBalancer: `kubectl get svc -A | grep LoadBalancer`
- [ ] Test node communication: `kubectl get nodes -o wide`
- [ ] Check cluster DNS: `kubectl get svc -n kube-system kube-dns`

### 2. Network Connectivity Testing
- [ ] DNS resolution: Test with busybox pod
- [ ] API server access: Test with curl pod  
- [ ] Cross-namespace communication: Test service-to-service
- [ ] External LoadBalancer access: Test from outside cluster

### 3. Application Validation
- [ ] MLflow tracking server accessibility
- [ ] MinIO/S3 storage connectivity
- [ ] Argo workflows execution
- [ ] Seldon model deployment
- [ ] Model serving endpoints

### 4. MLOps Pipeline Testing
- [ ] Submit data ingestion workflow
- [ ] Run feature engineering pipeline
- [ ] Execute model training
- [ ] Deploy model to Seldon
- [ ] Test A/B experiment serving
- [ ] Verify MLflow experiment tracking

## Common Migration Issues

### Network Policy Problems
- **DNS failures**: Add port 53 egress to kube-system
- **API timeouts**: Add port 443 egress for Kubernetes API
- **Service access**: Use internal service names, not external IPs

### Resource Constraints
- **Quota violations**: Update resource quotas for new components
- **OOMKilled pods**: Adjust memory limits in podSpecPatch
- **CPU throttling**: Review CPU requests and limits

### Service Discovery Issues
- **LoadBalancer routing**: MetalLB external IPs may not be routable internally
- **DNS resolution**: Service names may change between CNI implementations
- **Port conflicts**: LoadBalancer port assignments may differ

### Secret/Configuration Format
- **RClone config**: Seldon expects specific JSON format
- **Multi-key secrets**: Some components expect single-key secrets
- **Environment variables**: Service endpoints may need updates

## Known Issues During Cleanup

### Seldon Resources with Finalizers
**Symptom**: Namespace stuck in "Terminating" state
**Cause**: Seldon Model/Experiment resources have finalizers that prevent deletion
**Solution**:
```bash
# Check stuck resources
kubectl api-resources --verbs=list --namespaced -o name | xargs -n 1 kubectl get --show-kind --ignore-not-found -n seldon-system

# Remove finalizers from stuck resources (safe with error handling)
kubectl patch model baseline-predictor -n seldon-system --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true
kubectl patch model enhanced-predictor -n seldon-system --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true
kubectl patch experiment financial-ab-test-experiment -n seldon-system --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true
kubectl patch server mlserver -n seldon-system --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true

# Force delete namespaces if still stuck
kubectl delete namespace seldon-system --grace-period=0 --force 2>/dev/null || true
kubectl delete namespace seldon-system --grace-period=0 --force 2>/dev/null || true
```

## Rollback Procedures

### 1. Quick Rollback
- [ ] Restore pre-migration cluster state (if possible)
- [ ] Revert network policy changes
- [ ] Restore original service configurations
- [ ] Re-apply backup YAML configurations

### 2. Partial Rollback
- [ ] Revert specific network policies
- [ ] Restore previous service endpoints
- [ ] Rollback problematic application deployments
- [ ] Preserve working components

### 3. Documentation
- [ ] Document what failed and why
- [ ] Record successful migration steps
- [ ] Update troubleshooting guides
- [ ] Share lessons learned with team

## Success Criteria

### Infrastructure
✅ CNI successfully changed to Calico  
✅ LoadBalancer services operational with MetalLB  
✅ All network policies functional  
✅ Cross-namespace communication working  

### Applications  
✅ MLflow tracking server accessible  
✅ Argo workflows executing successfully  
✅ Seldon models deploying and serving  
✅ A/B experiments operational  

### Operations
✅ Monitoring and logging functional  
✅ Resource quotas appropriate  
✅ Security policies enforced  
✅ Backup and recovery validated