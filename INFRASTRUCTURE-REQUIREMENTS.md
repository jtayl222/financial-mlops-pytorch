# Infrastructure Requirements

This document outlines the infrastructure requirements and secrets that must be provided by the infrastructure repository for this MLOps application to function properly.

## Namespace Strategy

The infrastructure should implement **team-based namespace isolation** following industry best practices:

- `financial-ml` - Financial ML models and experiments  
- `financial-mlops-pytorch` - Application workloads (Argo Workflows, training jobs)
- `seldon-system` - Seldon Mesh controller and shared infrastructure

This approach provides:
- Clear ownership boundaries
- Team-specific resource quotas and RBAC
- Independent scaling and configuration
- Simplified cost allocation

## Required Secrets

### 1. ML Platform Secret (financial-ml namespace)

**Secret Name:** `ml-platform`  
**Namespace:** `financial-ml`  
**Type:** `Opaque`

This secret contains S3/MinIO credentials for model storage access. Required keys:

```yaml
stringData:
  AWS_ACCESS_KEY_ID: minioadmin
  AWS_SECRET_ACCESS_KEY: minioadmin123  
  AWS_DEFAULT_REGION: us-east-1
  AWS_ENDPOINT_URL: http://minio.minio.svc.cluster.local:9000
```

### 2. ML Platform Secret (financial-mlops-pytorch namespace)

**Secret Name:** `ml-platform`  
**Namespace:** `financial-mlops-pytorch`  
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

### 3. Container Registry Secret (financial-mlops-pytorch namespace)

**Secret Name:** `ghcr`  
**Namespace:** `financial-mlops-pytorch`  
**Type:** `kubernetes.io/dockerconfigjson`

This secret contains GitHub Container Registry credentials for pulling private images.

## Deployment Notes

- Models and experiments are deployed to `seldon-system` namespace because the Seldon controller is configured with `CLUSTERWIDE=false`
- The infrastructure team should manage all secrets using SealedSecrets or another secure secret management solution
- Secret rotation and updates should be handled by the infrastructure repository
- The application repository only references these secrets by name and expects them to exist

## Seldon Configuration

The Seldon v2 controller must be configured for **cluster-wide watching** to support team-based namespace isolation:

```yaml
env:
- name: CLUSTERWIDE
  value: "true"
args:
- --clusterwide=true
```

## Security Requirements

The infrastructure repository must implement security controls defined in `k8s/base/rbac.yaml`:

- **RBAC** - Role-based access control for model deployment
- **NetworkPolicies** - Namespace isolation and controlled inter-namespace communication  
- **ResourceQuotas** - Resource limits to prevent resource exhaustion

See `k8s/base/rbac.yaml` for the complete security configuration that should be applied by the infrastructure team.