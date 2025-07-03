# Financial MLOps PyTorch

A production-ready MLOps pipeline for financial market prediction using PyTorch, featuring automated model training, A/B testing, and deployment on Kubernetes.

## Architecture

This platform implements a complete MLOps workflow with:

- **Data Pipeline**: Automated data ingestion and feature engineering using Argo Workflows
- **Model Training**: PyTorch-based financial prediction models with MLflow tracking
- **Model Serving**: Seldon Core v2 for scalable model deployment and A/B testing
- **Infrastructure**: Kubernetes-native deployment with GitOps practices

## Features

- ✅ **Multiple Model Variants**: Baseline, enhanced, and lightweight models for different use cases
- ✅ **A/B Testing**: Traffic splitting between model variants with Seldon Experiments
- ✅ **Automated Pipelines**: Argo Workflows for data processing and model training
- ✅ **Model Registry**: MLflow integration for experiment tracking and model versioning
- ✅ **Container-native**: Full containerization with Docker and Kubernetes deployment
- ✅ **GitOps Ready**: Kustomize-based configuration management

## Quick Start

### Prerequisites

- Kubernetes cluster with Argo Workflows, MLflow, and Seldon Core v2 installed
- Docker registry access for container images
- `kubectl` and `argo` CLI tools configured

### 1. Build and Push Images

```bash
# Build main application image
docker build -t <REGISTRY>/<USERNAME>/financial-predictor:latest . --push

# Build Jupyter development image
docker build -t <REGISTRY>/<USERNAME>/financial-predictor-jupyter:latest -f jupyter/Dockerfile . --push
```

### 2. Deploy Infrastructure Requirements

Ensure your infrastructure provides the required secrets and namespaces as documented in [`INFRASTRUCTURE-REQUIREMENTS.md`](./INFRASTRUCTURE-REQUIREMENTS.md).

### 3. Deploy Application

```bash
# Deploy base resources
kubectl apply -k k8s/base

# Verify deployment
kubectl get pods -n financial-mlops-pytorch
kubectl get models -n financial-ml
```

### 4. Run Training Pipeline

```bash
# Submit data pipeline
argo submit --from workflowtemplate/financial-data-pipeline-template -n financial-mlops-pytorch

# Train baseline model
argo submit --from workflowtemplate/financial-training-pipeline-template \
  -p model-variant=baseline \
  -n financial-mlops-pytorch

# Train enhanced model for A/B testing
argo submit --from workflowtemplate/financial-training-pipeline-template \
  -p model-variant=enhanced \
  -n financial-mlops-pytorch
```

### 5. Verify Model Deployment

```bash
# Check model status
kubectl get models,experiments -n financial-ml

# Test model endpoint
curl -H "Host: financial-predictor.local" http://<CLUSTER_IP>/predict
```

## Model Variants

The platform supports multiple model configurations:

- **baseline**: Standard LSTM model for reliable predictions
- **enhanced**: Advanced model with additional features for A/B testing
- **lightweight**: Optimized model for edge deployment scenarios

## Documentation

- **[Operations Guide](./OPERATIONS.md)**: Detailed operational procedures and troubleshooting
- **[Infrastructure Requirements](./INFRASTRUCTURE-REQUIREMENTS.md)**: Platform requirements and secret configuration
- **[Networking Guide](./NETWORKING.md)**: Network architecture and deployment options (Istio, NodePort, LoadBalancer)
- **[Seldon Core Documentation](https://docs.seldon.ai/seldon-core-2)**: Official Seldon Core v2 documentation

## Development

### Local Development

Set up local development environment:

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment variables
export MLFLOW_TRACKING_URI=http://<MLFLOW_HOST>:<PORT>
export MLFLOW_S3_ENDPOINT_URL=http://<MINIO_HOST>:<PORT>

# Train model locally
MODEL_VARIANT=baseline python src/train_pytorch_model.py
```

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.