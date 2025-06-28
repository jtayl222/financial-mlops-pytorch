# financial-mlops-pytorch

```
# b3175a7bc9d7473aaa3ea61316de1889

docker buildx build --platform linux/amd64 -t jtayl22/financial-predictor:latest . --push

# Build and push the Jupyter image
docker buildx build --platform linux/amd64 -t jtayl22/financial-predictor-jupyter:latest -f jupyter/Dockerfile . --push


kustomize build k8s/applications/financial-mlops-pytorch/overlays/dev | kubectl apply -f -

argo submit --from workflowtemplate/financial-data-pipeline-template -n financial-mlops-pytorch --watch
```

-----



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