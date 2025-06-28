# jupyter/Dockerfile
FROM jupyter/datascience-notebook:latest

WORKDIR /home/jovyan

# Copy your project's main requirements.txt from the parent directory
# This assumes requirements.txt is at the root of financial-mlops-pytorch
COPY requirements.txt .

# Install project-specific dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
	pip install --upgrade pip setuptools && \
	apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Optional: Install k8s client if you want to interact with K8s from notebooks
# RUN pip install kubernetes

# Set environment variables that Jupyter notebooks will inherit
# E.g., for connecting to MLflow Tracking Server
ENV MLFLOW_TRACKING_URI="http://mlflow-service.mlflow.svc.cluster.local:5000"

# You might want to copy any shared utility scripts or custom configs
# COPY .jupyter/jupyter_notebook_config.py /home/jovyan/.jupyter/

# Ensure the user has permissions (already handled by jupyter/datascience-notebook base image)
USER root
RUN chown -R jovyan:users /home/jovyan
USER jovyan