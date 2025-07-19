# Infrastructure Requirements

This document outlines the infrastructure requirements and secrets that must be provided by the infrastructure repository for this MLOps application to function properly.

## Namespace Strategy

The infrastructure should implement **team-based namespace isolation** following industry best practices:

- `seldon-system` - Financial ML models and experiments  
- `seldon-system` - Application workloads (Argo Workflows, training jobs)
- `seldon-system` - Seldon Mesh controller and shared infrastructure

This approach provides:
- Clear ownership boundaries
- Team-specific resource quotas and RBAC
- Independent scaling and configuration
- Simplified cost allocation

## Required Secrets

### 1. ML Platform Secret (seldon-system namespace)

**Secret Name:** `ml-platform`  
**Namespace:** `seldon-system`  
**Type:** `Opaque`

This secret contains S3/MinIO credentials for model storage access. Required keys:

```yaml
stringData:
  AWS_ACCESS_KEY_ID: minioadmin
  AWS_SECRET_ACCESS_KEY: minioadmin123  
  AWS_DEFAULT_REGION: us-east-1
  AWS_ENDPOINT_URL: http://minio.minio.svc.cluster.local:9000
```

### 2. ML Platform Secret (seldon-system namespace)

**Secret Name:** `ml-platform`  
**Namespace:** `seldon-system`  
**Type:** `Opaque`

This secret contains full MLOps platform credentials for training workflows. Required keys:

```yaml
stringData:
  # S3/MinIO credentials
  AWS_ACCESS_KEY_ID: minioadmin
  AWS_SECRET_ACCESS_KEY: minioadmin123
  AWS_DEFAULT_REGION: us-east-1  
  AWS_ENDPOINT_URL: http://minio.minio.svc.cluster.local:9000
  
  # MLflow credentials
  MLFLOW_TRACKING_URI: http://mlflow.mlflow.svc.cluster.local:5000
  MLFLOW_TRACKING_USERNAME: mlflow
  MLFLOW_TRACKING_PASSWORD: my-secure-mlflow-tracking-password
  MLFLOW_S3_ENDPOINT_URL: http://minio.minio.svc.cluster.local:9000
  MLFLOW_FLASK_SERVER_SECRET_KEY: 6EF6B30F9E557F948C402C89002C7C8A
  
  # RClone credentials  
  RCLONE_CONFIG_S3_TYPE: s3
  RCLONE_CONFIG_S3_PROVIDER: Minio
  RCLONE_CONFIG_S3_ACCESS_KEY_ID: minioadmin
  RCLONE_CONFIG_S3_SECRET_ACCESS_KEY: minioadmin123
  RCLONE_CONFIG_S3_REGION: us-east-1
  RCLONE_CONFIG_S3_ENDPOINT: http://minio.minio.svc.cluster.local:9000
```

### 3. Container Registry Secret (seldon-system namespace)

**Secret Name:** `ghcr`  
**Namespace:** `seldon-system`  
**Type:** `kubernetes.io/dockerconfigjson`

This secret contains GitHub Container Registry credentials for pulling private images.

## Secret Management Strategy

### Package-Based Delivery

The infrastructure team delivers secrets through package-based deployment for development autonomy:

- **Infrastructure Repository**: Creates and manages secrets using SealedSecrets
- **Package Creation**: Secrets are packaged into tar.gz files with kustomization overlays  
- **Development Application**: Developers extract packages to `k8s/manifests/` directory (gitignored)
- **Namespace Application**: Each team applies secrets to their respective namespaces

This approach enables:
- **Development Autonomy**: Teams can deploy and test without infrastructure dependencies
- **Security Compliance**: Infrastructure maintains centralized secret management
- **Environment Consistency**: Same secret structure across dev/staging/production
- **Complete Reset Capability**: Teams can recreate entire environments for end-to-end testing

### Deployment Workflow

```bash
# Infrastructure team creates and delivers packages (done once)
# Creates: seldon-system-ml-secrets-20250704.tar.gz
# Creates: seldon-system-models-secrets-20250704.tar.gz

# Development team extracts and applies packages
tar xzf seldon-system-ml-secrets-20250704.tar.gz -C k8s/manifests/seldon-system
tar xzf seldon-system-models-secrets-20250704.tar.gz -C k8s/manifests/seldon-system

tree k8s/manifests/
# k8s/manifests/
# ├── seldon-system/
# │   └── production/
# │       ├── kustomization.yaml
# │       └── ml-platform-secret.yaml
# └── seldon-system/
#     └── production/
#         ├── ghcr-secret.yaml
#         ├── kustomization.yaml
#         └── ml-platform-secret.yaml

kubectl apply -k k8s/manifests/seldon-system/production
kubectl apply -k k8s/manifests/seldon-system/production
```

### Secret Lifecycle

- **Creation**: Infrastructure team manages SealedSecrets in infrastructure repository
- **Distribution**: Package delivery through tar.gz extraction to development repositories  
- **Application**: Development teams apply packages to target namespaces
- **Rotation**: Infrastructure team updates packages, development teams re-apply

## Seldon Configuration

### Controller Configuration

The Seldon v2 controller must be configured for **cluster-wide watching** to support team-based namespace isolation:

```yaml
env:
- name: CLUSTERWIDE
  value: "true"
args:
- --clusterwide=true
```

### Runtime Deployment Patterns

**Industry Best Practice: Dedicated SeldonRuntime per Namespace**

Instead of sharing runtime components across namespaces, deploy dedicated SeldonRuntime instances for complete isolation:

```yaml
apiVersion: mlops.seldon.io/v1alpha1
kind: SeldonRuntime
metadata:
  name: seldon-system-runtime
  namespace: seldon-system
spec:
  config:
    agentConfig:
      rclone: {}
    kafkaConfig: {}
    serviceConfig: {}
    tracingConfig: {}
  overrides:
  - name: hodometer
    replicas: 1
  - name: seldon-scheduler
    replicas: 1
  - name: seldon-envoy
    replicas: 1
  - name: seldon-dataflow-engine
    replicas: 1
  - name: seldon-modelgateway
    replicas: 1
  - name: seldon-pipelinegateway
    replicas: 1
  - name: mlserver
    replicas: 1
  seldonConfig: default
```

**Benefits of Dedicated Runtime:**
- ✅ **Complete Isolation**: Each team has independent infrastructure
- ✅ **Independent Scaling**: Runtime components scale per team needs  
- ✅ **Fault Isolation**: Issues in one team don't affect others
- ✅ **Security Boundaries**: Clear separation of concerns
- ✅ **Operational Independence**: Teams can manage their own runtime

**Resource Quota Considerations:**

SeldonRuntime components may not include CPU limits by default. For environments with ResourceQuotas requiring CPU limits, either:

1. **Remove CPU limits requirement** (recommended for development):
```yaml
apiVersion: v1
kind: ResourceQuota
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    limits.memory: 40Gi  # CPU limits removed
    count/models.mlops.seldon.io: "10"
```

2. **Configure runtime with resource limits** (requires infrastructure team support)

**Known Limitations:**

- SeldonRuntime CRD doesn't support `resources` field in `overrides` for CPU/memory limits
- MLServer instances may need separate deployment configuration
- Cross-namespace networking requires careful NetworkPolicy configuration

## Security Requirements

The infrastructure repository must implement security controls defined in `k8s/base/rbac.yaml`:

- **RBAC** - Role-based access control for model deployment
- **NetworkPolicies** - Namespace isolation and controlled inter-namespace communication  
- **ResourceQuotas** - Resource limits to prevent resource exhaustion

See `k8s/base/rbac.yaml` for the complete security configuration that should be applied by the infrastructure team.