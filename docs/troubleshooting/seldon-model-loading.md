# Seldon Model Loading Troubleshooting

## "No Matching Servers Available"

### Symptom
```
Failed to schedule model as no matching servers are available
ModelReady: False - ScheduleFailed
```

### Root Causes & Solutions

#### 1. Missing MLServer Pod
**Check**: `kubectl get pods -n financial-inference | grep mlserver`
**Solution**: Ensure MLServer StatefulSet is running and ready (3/3 containers)

#### 2. Missing RClone Secret
**Check**: `kubectl get secret seldon-rclone-gs-public -n financial-inference`
**Solution**: Create rclone secret with correct JSON format

#### 3. Wrong RClone Secret Format
**Symptom**: MLServer agent crashes with JSON parsing errors
**Check**: 
```bash
kubectl get secret seldon-rclone-gs-public -n financial-inference -o jsonpath='{.data.rclone\.conf}' | base64 -d
```
**Correct Format**:
```json
{
  "name": "s3",
  "type": "s3", 
  "parameters": {
    "provider": "Minio",
    "access_key_id": "minioadmin",
    "secret_access_key": "minioadmin123",
    "endpoint": "http://minio.minio.svc.cluster.local:9000",
    "region": "us-east-1"
  }
}
```

#### 4. Secret Key Count Issues
**Symptom**: `Secret does not have 1 key ml-platform`
**Root Cause**: Agent expects exactly 1 key in referenced secret
**Solution**: Remove `secretName` from Model spec if using rclone config:
```bash
kubectl patch model baseline-predictor -n financial-inference --type='merge' -p='{"spec":{"secretName":null}}'
```

## MLServer Container Issues

### Agent Container Crashes
**Check Logs**: `kubectl logs mlserver-0 -n financial-inference -c agent`

#### Common Error Patterns:
1. **RClone Config Parse Error**: Fix JSON format (see above)
2. **Missing Secret**: Create `seldon-rclone-gs-public` 
3. **S3 Access Denied**: Verify MinIO credentials and bucket access

### MLServer Container Not Ready
**Check**: 
```bash
kubectl describe pod mlserver-0 -n financial-inference
kubectl logs mlserver-0 -n financial-inference -c mlserver
```

#### Common Issues:
1. **Environment tarball not found**: Normal warning, can be ignored
2. **No models in repository**: Expected until models are loaded
3. **Memory/CPU limits**: Check resource constraints

## Storage URI Issues

### Incorrect S3 Path
**Symptom**: Model loads but fails to find artifacts
**Check**: Verify storage URI matches actual MLflow artifacts:
```bash
mc ls minio/mlflow-artifacts/28/models/
```
**Fix**: Update Model storageUri to match actual path

### Service Resolution Issues  
**Symptom**: `No route to host` for external LoadBalancer IPs
**Solution**: Use internal service names in URIs:
- ❌ `s3://192.168.1.203:9000/bucket`
- ✅ `s3://mlflow-artifacts/path` (with internal endpoint in rclone config)

## Validation Commands

### Check Model Status
```bash
kubectl get models -n financial-inference
kubectl describe model baseline-predictor -n financial-inference
```

### Monitor Model Loading
```bash
kubectl logs mlserver-0 -n financial-inference -c agent -f
```

### Test Model Endpoints
```bash
# Check if model is serving
kubectl get svc -n financial-inference
curl -X POST http://<model-service>/v2/models/<model-name>/infer -d '{"inputs":[...]}'
```

### Experiment Status
```bash
kubectl get experiments -n financial-inference
kubectl describe experiment financial-ab-test-experiment -n financial-inference
```

## Cleanup Issues

### Stuck Namespaces (Finalizer Problems)
**Symptom**: Namespace remains in "Terminating" state after deletion
**Root Cause**: Seldon resources have finalizers preventing cleanup

#### Check Stuck Resources
```bash
kubectl api-resources --verbs=list --namespaced -o name | xargs -n 1 kubectl get --show-kind --ignore-not-found -n financial-inference
```

#### Remove Finalizers
```bash
# Models
kubectl patch model baseline-predictor -n financial-inference --type='merge' -p='{"metadata":{"finalizers":null}}'
kubectl patch model enhanced-predictor -n financial-inference --type='merge' -p='{"metadata":{"finalizers":null}}'

# Experiments  
kubectl patch experiment financial-ab-test-experiment -n financial-inference --type='merge' -p='{"metadata":{"finalizers":null}}'

# Servers
kubectl patch server mlserver -n financial-inference --type='merge' -p='{"metadata":{"finalizers":null}}'
```

#### Force Namespace Deletion
```bash
kubectl delete namespace financial-inference --grace-period=0 --force
```