# financial-mlops-pytorch

```
# b3175a7bc9d7473aaa3ea61316de1889

docker build -t jtayl22/financial-predictor:latest . --push

# Build and push the Jupyter image
docker build -t jtayl22/financial-predictor-jupyter:latest -f jupyter/Dockerfile . --push

RUN THIS COMMAND IN the ml-platform repository root directory
./scripts/package-ml-secrets.sh financial-mlops-pytorch dev,production financial-team@company.com

kustomize build k8s/applications/financial-mlops-pytorch/overlays/dev | kubectl apply -f -

argo submit --from workflowtemplate/financial-data-pipeline-template -n financial-mlops-pytorch --watch

cp -r ../ml-platform/infrastructure/packages/financial-mlops-pytorch/* k8s/manifests/

k delete ns financial-mlops-pytorch; \
k create  ns financial-mlops-pytorch; \
k apply -k k8s/manifests/production/; \
k apply -k k8s/overlays/dev; \
argo submit --from workflowtemplate/financial-data-pipeline-template -n financial-mlops-pytorch

argo submit --from workflowtemplate/financial-training-pipeline-template -n financial-mlops-pytorch


kubectl -n argocd get secret argocd-initial-admin-secret   -o jsonpath="{.data.password}" | base64 -d && echo
argocd login 192.168.1.85:30080
argocd app delete seldon-deployments --cascade=true

mc alias set minio http://192.168.1.85:30900 minioadmin minioadmin123
mc ls minio/financial-models/v1/
mc ls minio/financial-models/v2/


```
Docs: https://docs.seldon.ai/seldon-core-2

### MODEL_VARIANT
```
# Train baseline model
MODEL_VARIANT=baseline python src/train_pytorch_model.py

# Train enhanced model for A/B testing
MODEL_VARIANT=enhanced python src/train_pytorch_model.py

# Train lightweight model for edge deployment
MODEL_VARIANT=lightweight python src/train_pytorch_model.py
```

## run locally
```bash
cd ~/REPOS/financial-mlops-pytorch

MODEL_VARIANT=baseline \
SCALER_DIR=/mnt/shared-artifacts/scalers \
MLFLOW_TRACKING_URI=http://192.168.1.85:30800 \
MLFLOW_S3_ENDPOINT_URL=http://192.168.1.85:30900 \
PROCESSED_DATA_DIR=/mnt/shared-data/processed \
RAW_DATA_DIR=/mnt/shared-data/raw \
MODEL_SAVE_DIR=/mnt/shared-artifacts/models \
python3 src/train_pytorch_model.py
```
### this is a test
```bash
(.venv) user@U850:~/REPOS/k3s-homelab$ mlflow experiments search                   
Experiment Id    Name                                 Artifact Location                               
---------------  -----------------------------------  ------------------------------------------------
0                Default                              s3://mlflow-artifacts/0                         
13               MyTestExperiment-1750452520          s3://mlflow-artifacts/13                        
16               iris_demo                            s3://mlflow-artifacts/16                        
19               Churn_Prediction_Experiment          s3://mlflow-artifacts/19                        
20               Churn_Prediction_XGBoost             s3://mlflow-artifacts/20                        
21               flower-classifier                    s3://mlflow-artifacts/21                        
22               financial-mlops-pytorch              s3://mlflow-artifacts/22                        
25               pytorch_stock_predictor_training     s3://mlflow-artifacts/25                        
27               financial-mlops-pytorch-v2           s3://financial-models/financial-mlops-pytorch-v2
28               financial-mlops-pytorch-baseline     s3://mlflow-artifacts/28                        
29               financial-mlops-pytorch-enhanced     s3://mlflow-artifacts/29                        
30               financial-mlops-pytorch-lightweight  s3://mlflow-artifacts/30   
```

### The model that was created
MLflow stores artifacts for each experiment in a path like:

```bash
s3://mlflow-artifacts/<experiment_id>/<run_id>/artifacts/
```

You should see directories named with your MLflow run IDs (e.g., bd289d1bba51421182470d343b6f5e10/).
```
$ mc ls minio/mlflow-artifacts/28
[2025-07-01 10:32:42 EDT]     0B 7772275aced94cfd9e2a22d6c05d11a9/
[2025-07-01 10:32:42 EDT]     0B bd289d1bba51421182470d343b6f5e10/
[2025-07-01 10:32:42 EDT]     0B models/

$ mc ls minio/mlflow-artifacts/28/models -r
[2025-07-01 10:20:55 EDT] 1.1KiB STANDARD m-0738680078ac430bafe56f8ff7f42e5b/artifacts/MLmodel
[2025-07-01 10:20:55 EDT] 1.3KiB STANDARD m-0738680078ac430bafe56f8ff7f42e5b/artifacts/code/models.py
[2025-07-01 10:20:55 EDT]   185B STANDARD m-0738680078ac430bafe56f8ff7f42e5b/artifacts/conda.yaml
[2025-07-01 10:20:55 EDT] 236KiB STANDARD m-0738680078ac430bafe56f8ff7f42e5b/artifacts/data/model.pth
[2025-07-01 10:20:55 EDT]    28B STANDARD m-0738680078ac430bafe56f8ff7f42e5b/artifacts/data/pickle_module_info.txt
[2025-07-01 10:20:55 EDT] 7.1KiB STANDARD m-0738680078ac430bafe56f8ff7f42e5b/artifacts/input_example.json
[2025-07-01 10:20:55 EDT]   122B STANDARD m-0738680078ac430bafe56f8ff7f42e5b/artifacts/python_env.yaml
[2025-07-01 10:20:55 EDT]    72B STANDARD m-0738680078ac430bafe56f8ff7f42e5b/artifacts/requirements.txt
[2025-07-01 10:20:55 EDT]  10KiB STANDARD m-0738680078ac430bafe56f8ff7f42e5b/artifacts/serving_input_example.json
[2025-07-01 08:02:54 EDT] 1.1KiB STANDARD m-504e5a0d73a64ac59f009fb6dda7220b/artifacts/MLmodel
[2025-07-01 08:02:54 EDT] 1.3KiB STANDARD m-504e5a0d73a64ac59f009fb6dda7220b/artifacts/code/models.py
[2025-07-01 08:02:54 EDT]   183B STANDARD m-504e5a0d73a64ac59f009fb6dda7220b/artifacts/conda.yaml
[2025-07-01 08:02:54 EDT] 236KiB STANDARD m-504e5a0d73a64ac59f009fb6dda7220b/artifacts/data/model.pth
[2025-07-01 08:02:54 EDT]    28B STANDARD m-504e5a0d73a64ac59f009fb6dda7220b/artifacts/data/pickle_module_info.txt
[2025-07-01 08:02:54 EDT] 7.1KiB STANDARD m-504e5a0d73a64ac59f009fb6dda7220b/artifacts/input_example.json
[2025-07-01 08:02:54 EDT]   112B STANDARD m-504e5a0d73a64ac59f009fb6dda7220b/artifacts/python_env.yaml
[2025-07-01 08:02:54 EDT]    72B STANDARD m-504e5a0d73a64ac59f009fb6dda7220b/artifacts/requirements.txt
[2025-07-01 08:02:54 EDT]  10KiB STANDARD m-504e5a0d73a64ac59f009fb6dda7220b/artifacts/serving_input_example.json
```

✅ Completed:

- Baseline model trained and registered in MLflow
- Multiple experiment variants created (baseline, enhanced, lightweight)
- Models stored in S3/MinIO
- Training pipeline working in Argo Workflows

🎯 Next Step: Train the Enhanced Model
```
argo submit --from workflowtemplate/financial-training-pipeline-template \
  -p model-variant=enhanced \
  -n financial-mlops-pytorch
```

then
```bash
# Check if models are loading
kubectl get models -n financial-mlops-pytorch

# Check experiment status  
kubectl get experiments -n financial-mlops-pytorch

# Test A/B endpoint
curl -H "Host: financial-predictor.local" http://your-k3s-ip/predict
```

## Submit workflows
```bash
argo submit --from workflowtemplate/financial-training-pipeline-template \
  -p model-variant=baseline \
  -n financial-mlops-pytorch

argo submit --from workflowtemplate/financial-training-pipeline-template \
  -p model-variant=enhanced \
  -n financial-mlops-pytorch

argo submit --from workflowtemplate/financial-training-pipeline-template \
  -p model-variant=lightweight \
  -n financial-mlops-pytorch  

```
-----

## Check Seldon Core v2 Status

```
kubectl get pods -n seldon-system
kubectl get models -n seldon-mesh
kubectl get pipelines -n seldon-mesh
kubectl get experiments -n seldon-mesh
```
## Check Seldon runtime logs:
```bash
kubectl logs -n seldon-system deployment/seldon-scheduler
kubectl logs -n seldon-system deployment/seldon-dataplane
```

## Quick Start: Deploying the Financial MLOps Pipeline

This section provides a quick guide to deploying and interacting with the `financial-mlops-pytorch` MLOps pipeline on a Kubernetes cluster. We assume you have a Kubernetes environment (like k3s) with Argo Workflows, MLflow, and JupyterHub already provisioned.

**Prerequisites:**

  * `kubectl` configured to access your Kubernetes cluster.
  * `helm` installed (for managing Kubernetes applications).
  * Access to a container registry (e.g., Docker Hub) where you can push images.
  * The `financial-mlops-pytorch` repository cloned locally.

### 1\. Build and Push the Application Docker Image

The core ML pipeline components (data ingestion, feature engineering, model training) run within a Docker container.

1.  **Navigate to the project root:**

    ```bash
    cd financial-mlops-pytorch
    ```

2.  **Build the Docker image:**
    This command builds the `financial-predictor` image for `linux/amd64` (a common architecture for Kubernetes nodes) and tags it with `latest` and a unique version based on your Git commit.

    ```bash
    # Ensure you are at the root of the financial-mlops-pytorch repository
    docker buildx build --platform linux/amd64 -t jtayl22/financial-predictor:latest . --push
    ```

    *Replace `jtayl22` with your Docker Hub username or preferred registry prefix.*

### 2\. Prepare Kubernetes Resources with Kustomize

Our Kubernetes manifests are managed using Kustomize, allowing for environment-specific overlays.

1.  **Build the Kubernetes manifests for your environment:**
    This command generates the combined YAML for deploying your application components (Argo WorkflowTemplates, SeldonDeployments, etc.).
    ```bash
    # Assuming you're still in the financial-mlops-pytorch root
    kubectl kustomize kubernetes/base > deploy/manifests.yaml
    # For a development environment, you might use:
    # kubectl kustomize kubernetes/overlays/development > deploy/manifests.yaml
    ```

### 3\. Deploy to Kubernetes

Apply the generated manifests to your Kubernetes cluster.

```bash
kubectl apply -f deploy/manifests.yaml
```

This will deploy:

  * Necessary `WorkflowTemplates` for data pipelines and model training.
  * `PersistentVolumeClaims` (e.g., `shared-data-pvc`, `shared-artifacts-pvc`) if not already present.
  * `SeldonDeployments` for model serving, if configured in your Kustomize overlay.

### 4\. Trigger an ML Pipeline Run (Argo Workflows)

You can trigger a data pipeline or model training workflow using Argo Workflows.

1.  **View available WorkflowTemplates:**

    ```bash
    argo template list -n financial-mlops-pytorch
    ```

2.  **Submit a workflow:**
    For example, to run the data ingestion pipeline:

    ```bash
    argo submit --from workflowtemplate/data-ingestion-template -n financial-mlops-pytorch
    ```

    To train a model:

    ```bash
    argo submit --from workflowtemplate/model-training-template -n financial-mlops-pytorch
    ```

3.  **Monitor the workflow:**

    ```bash
    argo list -n financial-mlops-pytorch -w # Watch ongoing workflows
    argo logs <workflow-name> -n financial-mlops-pytorch # View logs for a specific workflow
    ```

### 5\. Interact with JupyterHub (Optional)

For interactive development, data exploration, and debugging, you can use the pre-configured JupyterHub environment.

1.  **Access JupyterHub:** Navigate to your JupyterHub URL (provided by your infrastructure administrator).
2.  **Develop:** Within JupyterHub, you can `git clone` this repository and interact with the data and MLflow server, which are automatically mounted and configured.

For detailed instructions on using JupyterHub, including managing dependencies and best practices, refer to:
**[docs/jupyterhub.md](https://www.google.com/search?q=docs/jupyterhub.md)**

-----