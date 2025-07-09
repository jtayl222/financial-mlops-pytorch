# Financial MLOps with PyTorch and Seldon Core

This repository showcases a complete, enterprise-grade MLOps workflow for a financial machine learning application. It demonstrates advanced A/B testing, multi-model deployment, and robust monitoring using PyTorch and Seldon Core on a Kubernetes-native platform.

This project is intended as a portfolio piece to demonstrate a deep, end-to-end understanding of MLOps principles in a practical, production-like setting.

***

## Portfolio Context: A Two-Part System

This project is the **Application Layer** of a larger, two-part MLOps ecosystem. It is designed to run on a separate, foundational **Platform Layer**.

### 1. The MLOps Platform (Infrastructure)

The underlying infrastructure is a complete, production-ready MLOps platform built from the ground up on Kubernetes (K3s). It provides all the core services needed for a robust MLOps environment.

*   **Repository:** [**github.com/jtayl222/ml-platform**](https://github.com/jtayl222/ml-platform)
*   **Features:** Experiment tracking (MLflow), model serving (Seldon Core), pipeline orchestration (Argo Workflows), GitOps (Argo CD), and comprehensive monitoring (Prometheus, Grafana).

### 2. The Financial ML Application (This Repository)

This repository contains the machine learning application that **runs on top of the platform**. It includes the ML code, training pipelines, and deployment manifests to solve a specific business problem—financial market prediction.

*   **Repository:** [**github.com/jtayl222/financial-mlops-pytorch**](https://github.com/jtayl222/financial-mlops-pytorch)
*   **Features:** Demonstrates how a real ML application leverages the underlying platform to achieve automated training, versioning, deployment, and advanced A/B testing in a simulated production environment.

Together, these two repositories demonstrate the crucial separation of concerns between the MLOps platform and the ML application, a key principle in building scalable and maintainable machine learning systems.

***

## The Machine Learning System

### Data Source and Feature Engineering

The model is trained on historical stock market data sourced dynamically from **Yahoo Finance** using the `yfinance` library.

1.  **Data Ingestion:** The `src/data_ingestion.py` script downloads daily price data (Open, High, Low, Close, Volume) for a predefined list of stock tickers (e.g., AAPL, MSFT, GOOG).
2.  **Feature Engineering:** The `src/feature_engineering_pytorch.py` script creates a rich feature set from the raw data, including:
    *   **Lagged Features:** Past values of price and volume.
    *   **Technical Indicators:** Simple Moving Averages (SMA) and Relative Strength Index (RSI).
    *   **Daily Returns:** Percentage change in closing price.
3.  **Target Variable:** The modeling task is a **binary classification** problem: predict whether the closing price will go **up (1)** or **down (0)** on the next day.
4.  **Data Splitting:** The data is split chronologically into training, validation, and test sets to prevent lookahead bias.

### Python Implementation and Model Architecture

The Python implementation in the `src/` directory is solid, modular, and follows good software engineering practices, separating concerns for data ingestion, feature engineering, model definition, and training.

*   **Model:** The core model, defined in `src/models.py`, is a **Long Short-Term Memory (LSTM)** network built with PyTorch. This architecture is well-suited for learning patterns in sequential time-series data.
*   **Training:** The `src/train_pytorch_model.py` script handles the end-to-end training loop, including:
    *   Class weighting to handle imbalanced data.
    *   MLflow integration for logging parameters, metrics, and model artifacts.
    *   Comprehensive evaluation on a hold-out test set.
    *   Exporting the final model to the ONNX format for high-performance, interoperable serving.
*   **Serving:** The `src/Model.py` script defines the serving class for Seldon Core, which loads the trained model artifacts and exposes a `predict` method.

### Future Roadmap: Towards a Real-World System

This project provides a strong foundation. To build this out into a truly production-grade financial prediction system, the following steps would be next:

1.  **Incorporate More Complex Data Sources:**
    *   Integrate alternative data like SEC filings (EDGAR), market news sentiment (using NLP), or macroeconomic indicators.
    *   Move from daily data to intraday (minute-level) data for more granular predictions, requiring a more robust data ingestion pipeline (e.g., using Kafka).

2.  **Evolve the Model Architecture:**
    *   **Attention Mechanisms:** Enhance the LSTM with attention layers to allow the model to focus on more influential parts of the time-series.
    *   **Graph Neural Networks (GNNs):** Model the relationships between different stocks as a graph to capture market-wide dependencies.
    *   **Ensemble Models:** Combine predictions from multiple models (e.g., LSTM, Gradient Boosting Trees) to improve robustness.

3.  **Implement Advanced Feature Engineering:**
    *   **Feature Store:** Integrate a feature store like Feast to provide consistent, versioned features for both training and real-time inference.
    *   **Automated Feature Engineering:** Use libraries like `featuretools` to automatically discover complex patterns and create new features.

4.  **Systematic Hyperparameter Tuning:**
    *   Integrate an automated hyperparameter optimization framework like **Optuna** or **Ray Tune**, orchestrated by Argo Workflows, to systematically find the best model configurations.

5.  **Enhance Model Explainability and Trust:**
    *   Incorporate model explainability tools like **SHAP** or **Captum** to understand *why* the model is making certain predictions. This is critical for building trust and for regulatory compliance in finance.

***

## Quick Start

### Prerequisites

- A running instance of the [ml-platform](https://github.com/jtayl222/ml-platform) or a similar Kubernetes environment with Argo Workflows, MLflow, and Seldon Core v2 installed.
- Docker registry access for container images.
- `kubectl` and `argo` CLI tools configured.

### Demo Instructions

For a full walkthrough of the A/B testing demo, see [**DEMO_INSTRUCTIONS.md**](./DEMO_INSTRUCTIONS.md).

### Manual Setup

#### 1. Build and Push Images
```bash
# Build main application image
docker build -t <REGISTRY>/<USERNAME>/financial-predictor:latest . --push

# Build Jupyter development image
docker build -t <REGISTRY>/<USERNAME>/financial-predictor-jupyter:latest -f jupyter/Dockerfile . --push
```

#### 2. Deploy Application
```bash
# Deploy base resources
kubectl apply -k k8s/base
```

#### 3. Run Training Pipeline
```bash
# Submit data pipeline
argo submit --from workflowtemplate/financial-data-pipeline-template -n financial-mlops-pytorch

# Train baseline model
argo submit --from workflowtemplate/financial-training-pipeline-template -p model-variant=baseline -n financial-mlops-pytorch

# Train enhanced model for A/B testing
argo submit --from workflowtemplate/financial-training-pipeline-template -p model-variant=enhanced -n financial-mlops-pytorch
```

***

## Documentation

- [**Environment Strategy: Development vs. Production**](./docs/environments.md): Details on how dev and prod environments are managed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
