# jupyter/Dockerfile
FROM jupyter/datascience-notebook:latest

WORKDIR /home/jovyan

# Copy your project's main requirements.txt from the parent directory
# This assumes requirements.txt is at the root of seldon-system
COPY requirements.txt .

# --- IMPORTANT Dockerfile Fixes Below ---

# Ensure apt-get commands run as root, before switching to jovyan for pip
USER root
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install project-specific Python dependencies as jovyan (after apt updates)
# Switch back to jovyan for pip installs, as jupyter base images expect this
USER jovyan
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --upgrade pip setuptools

# The chown and final USER jovyan are handled effectively by the base image
# If you want to explicitly ensure permissions for newly copied files, you can add:
# USER root
# RUN chown -R jovyan:users /home/jovyan
# USER jovyan

# Optional: Install k8s client if you want to interact with K8s from notebooks
# RUN pip install kubernetes

# Set environment variables that Jupyter notebooks will inherit
# E.g., for connecting to MLflow Tracking Server
ENV MLFLOW_TRACKING_URI="http://mlflow.mlflow.svc.cluster.local:5000"

# You might want to copy any shared utility scripts or custom configs
# COPY .jupyter/jupyter_notebook_config.py /home/jovyan/.jupyter/
