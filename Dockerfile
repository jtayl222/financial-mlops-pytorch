FROM python:3.9-slim-buster

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY src ./src

# Create directories for data and models. These are the *default* locations
# if environment variables aren't set, but will be overridden by Kubernetes.
# It's still good practice to ensure they exist within the container.
RUN mkdir -p data/raw data/processed models artifacts/scalers

# Define MLflow tracking URI as an environment variable to be set by K8s
ENV MLFLOW_TRACKING_URI="http://mlflow-service.mlflow.svc.cluster.local:5000"
# Other default environment variables can be set here or overridden by K8s
ENV RAW_DATA_OUTPUT_DIR="data/raw"
ENV PROCESSED_DATA_OUTPUT_DIR="data/processed"
ENV SCALER_OUTPUT_DIR="artifacts/scalers"
ENV MODEL_SAVE_DIR="models"

# Set python path if your imports need it (e.g., `from src.models import ...`)
ENV PYTHONPATH=/app

# CMD ["python", "--version"] # Default command can be a no-op or removed, as Argo will provide it