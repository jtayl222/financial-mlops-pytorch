# Financial MLOps PyTorch

**Enterprise-grade MLOps infrastructure demonstrating production-ready capabilities for machine learning operations at scale.**

This platform showcases comprehensive MLOps infrastructure engineering with multi-namespace Kubernetes architecture, GitOps automation, and advanced model serving capabilities. While implementing financial market prediction as a domain example, the primary focus is on demonstrating enterprise-grade platform engineering that can support any ML workload.

## Infrastructure-First Architecture

This platform demonstrates enterprise-grade MLOps infrastructure with production-ready capabilities:

### **Core Infrastructure Components**
- **Multi-Namespace Kubernetes**: Separation of training (`financial-mlops-pytorch`) and serving (`financial-mlops-pytorch`) environments
- **GitOps Automation**: ArgoCD-based deployment with Kustomize configuration management
- **Comprehensive Monitoring**: Prometheus/Grafana stack with business and technical metrics
- **Security Controls**: RBAC, network policies, and secure secret management
- **Advanced Model Serving**: Seldon Core v2 with A/B testing and traffic management

### **MLOps Workflow Implementation**
- **Data Pipeline**: Automated data ingestion and feature engineering using Argo Workflows
- **Model Training**: PyTorch-based models with MLflow experiment tracking and registry
- **Model Serving**: Production-ready deployment with health checks and monitoring
- **A/B Testing**: Traffic splitting and performance comparison between model variants

## Enterprise-Ready Features

### **Production Infrastructure**
- ✅ **Multi-Namespace Architecture**: Separate training and serving environments with strict isolation
- ✅ **GitOps Automation**: ArgoCD-based deployment with automated sync and rollback capabilities
- ✅ **Comprehensive Monitoring**: Prometheus metrics, Grafana dashboards, and alerting rules
- ✅ **Security by Design**: RBAC, network policies, and encrypted secret management
- ✅ **High Availability**: Load balancing, health checks, and auto-scaling configurations

### **Advanced MLOps Capabilities**
- ✅ **A/B Testing Framework**: Seldon Experiments with traffic splitting and performance comparison
- ✅ **Model Registry**: MLflow integration for experiment tracking, model versioning, and lineage
- ✅ **Automated Pipelines**: Argo Workflows for data processing, training, and deployment
- ✅ **Multi-Model Serving**: Support for baseline, enhanced, and lightweight model variants
- ✅ **Business Impact Measurement**: ROI tracking and business metrics integration

### **Operational Excellence**
- ✅ **Infrastructure as Code**: Complete Kubernetes manifests with Kustomize overlays
- ✅ **Comprehensive Documentation**: Runbooks, troubleshooting guides, and operational procedures
- ✅ **Testing Framework**: Unit, integration, and user acceptance testing
- ✅ **Stakeholder Communication**: Multi-perspective critique and assessment framework

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

```bash
# Apply infrastructure-delivered secret packages
kubectl apply -k k8s/manifests/financial-mlops-pytorch/production
kubectl apply -k k8s/manifests/financial-mlops-pytorch/production
```

### 3. Deploy Application

```bash
# Deploy base resources
kubectl apply -k k8s/base

# Verify deployment
kubectl get pods -n financial-mlops-pytorch
kubectl get models -n financial-mlops-pytorch
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
kubectl get models,experiments -n financial-mlops-pytorch

# Test model endpoint
curl -H "Host: financial-predictor.local" http://<CLUSTER_IP>/predict
```

## Infrastructure Demonstration

### **Model Variants for A/B Testing**

The platform demonstrates infrastructure capabilities using multiple model configurations:

- **baseline**: Standard LSTM model (52.7% accuracy) - demonstrates reliable deployment patterns
- **enhanced**: Advanced model with additional features - showcases A/B testing infrastructure
- **lightweight**: Optimized model for edge deployment - demonstrates multi-environment support

**Note**: The focus is on infrastructure reliability and operational excellence rather than model accuracy. The platform demonstrates how robust infrastructure can support any ML workload effectively.

## Documentation

### **Infrastructure and Operations**
- **[Operations Guide](./OPERATIONS.md)**: Detailed operational procedures and troubleshooting
- **[Infrastructure Requirements](./INFRASTRUCTURE-REQUIREMENTS.md)**: Platform requirements and secret configuration  
- **[Networking Guide](./NETWORKING.md)**: Network architecture and deployment options (Istio, NodePort, LoadBalancer)
- **[Testing Framework](./TESTING.md)**: Comprehensive testing procedures and validation
- **[Lessons Learned](./LESSONS-LEARNED.md)**: Technical insights and enterprise MLOps patterns

### **External Resources**
- **[Seldon Core Documentation](https://docs.seldon.ai/seldon-core-2)**: Official Seldon Core v2 documentation
- **[ArgoCD Best Practices](https://argoproj.github.io/argo-cd/best_practices/)**: GitOps deployment patterns
- **[Prometheus Monitoring](https://prometheus.io/docs/introduction/overview/)**: Metrics and monitoring setup

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