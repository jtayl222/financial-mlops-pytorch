# Using JupyterHub with the Financial MLOps Project

This document guides developers on how to leverage JupyterHub for interactive development, experimentation, and analysis within the `financial-mlops-pytorch` project. JupyterHub provides a pre-configured, scalable environment that integrates seamlessly with your project's data, artifacts, and MLflow tracking server.

## 1\. Introduction to JupyterHub

JupyterHub is a multi-user environment for Jupyter notebooks. It is deployed as a shared service on our Kubernetes cluster and serves as your primary interactive workspace for data science and machine learning tasks. It removes the need for local environment setup, providing a consistent and powerful environment for all project contributors.

## 2\. Capabilities for Developers

With JupyterHub, you can perform the following key activities for the `financial-mlops-pytorch` project:

  * **Interactive Development & Prototyping:**
      * Quickly write, test, and debug Python code for data ingestion, feature engineering (`src/data_ingestion.py`, `src/feature_engineering_pytorch.py`), and model training (`src/train_pytorch_model.py`).
      * Rapidly iterate on new model architectures (`src/models.py`) or explore different hyperparameter settings.
      * Experiment with new data sources or feature transformations before integrating them into the automated pipelines.
  * **Data Exploration & Analysis:**
      * Load and visualize raw data, processed features, and model outputs directly from the shared Persistent Volume Claims (PVCs) mounted in your notebook environment (e.g., `/mnt/shared-data`).
      * Perform ad-hoc queries, generate summary statistics, and create visualizations to gain insights into the financial datasets.
      * Conduct data quality checks and identify potential issues.
  * **Model Evaluation & Debugging:**
      * Load trained models from the MLflow Model Registry and perform detailed evaluation using new or held-out data.
      * Analyze model predictions, errors, and biases using interactive plots and metrics.
      * Utilize debugging tools to step through model inference logic.
  * **Experiment Tracking with MLflow:**
      * Seamlessly log your interactive experiments (parameters, metrics, artifacts, models) directly to the central MLflow Tracking Server. The `MLFLOW_TRACKING_URI` environment variable is pre-configured.
      * Compare results from different interactive runs or against automated pipeline runs.
  * **Ad-hoc Script Execution:**
      * Run specific functions or parts of your `src/` scripts to test isolated components or generate specific data samples for analysis.
  * **Collaboration:**
      * Share notebooks (via Git version control) with team members working on the same Hub instance.

## 3\. Advantages of Using JupyterHub

  * **Pre-configured Environment:** No need to install Python, PyTorch, pandas, yfinance, or other project dependencies locally. The environment is ready to use.
  * **Consistent Environment:** All developers work in the same environment, reducing "it works on my machine" issues.
  * **Access to Shared Resources:** Direct access to shared data, ML artifacts (scalers, preprocessors), and the MLflow Tracking Server eliminates complex data transfer steps.
  * **Scalability:** Leverages Kubernetes resources, allowing your interactive sessions to scale with your needs (within defined resource limits).
  * **Collaboration:** Centralized platform facilitates teamwork and knowledge sharing.

## 4\. Disadvantages & Considerations

While powerful, it's important to be aware of JupyterHub's considerations:

  * **Not for Production Code:** Jupyter notebooks are excellent for exploration and prototyping, but the definitive code for your automated pipelines (`src/` directory) should reside in `.py` files and be integrated into your Argo Workflows.
  * **Resource Consumption:** Interactive sessions can consume significant CPU, memory, and storage. Be mindful of your resource usage and terminate your server when not actively working.
  * **Notebook Version Control:** `.ipynb` files can be challenging to review in Git due to embedded output. Consider using tools like `nbstripout` to remove output before committing, or focus on moving finalized code to `.py` modules.
  * **Limited Orchestration:** Jupyter notebooks are for interactive work. For complex, multi-step, scheduled, or event-driven tasks, use Argo Workflows.
  * **Security:** Always be mindful of data access and sensitive information within notebooks.

## 5\. Getting Started

Follow these steps to begin using JupyterHub for your project:

1.  **Access JupyterHub:**

      * Navigate to the JupyterHub URL provided by the platform team (e.g., `http://your-k3s-ip:30888/` or your configured ingress URL).
      * Log in using your provided credentials (e.g., your Kubernetes/LDAP/GitHub credentials, as configured).

2.  **Launch Your Server:**

      * Once logged in, you'll be prompted to "Start My Server". Select the default profile, which is pre-configured with the `jtayl22/financial-predictor-jupyter:latest` image.
      * Wait for your server to provision and start (this might take a minute or two for the first launch as the image is pulled).

3.  **Clone the Project Repository:**

      * Once your Jupyter environment loads, open a new **Terminal** (File \> New \> Terminal or the `+` icon on the Launcher).
      * Navigate to your desired working directory (e.g., `cd work`).
      * Clone the `financial-mlops-pytorch` repository:
        ```bash
        git clone https://github.com/your-username/financial-mlops-pytorch.git
        ```
      * `cd financial-mlops-pytorch` to enter the project directory.

4.  **Verify Environment & Access:**

      * **MLflow:**
        ```bash
        echo $MLFLOW_TRACKING_URI
        # Expected output: http://mlflow-service.mlflow.svc.cluster.local:5000
        ```
        You can also run a small Python snippet in a notebook cell:
        ```python
        import mlflow
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        ```
      * **Shared Data:**
        ```bash
        ls -l /mnt/shared-data/processed/
        ls -l /mnt/shared-artifacts/scalers/
        ```
        You should see the contents of your PVCs, assuming the Argo data pipeline has run at least once.
      * **Project Code:**
        To import modules from your `src` directory, you might need to add it to your Python path:
        ```python
        import sys
        import os
        # Assumes you are in the root of the financial-mlops-pytorch repo
        sys.path.append(os.path.abspath('src'))

        # Now you can import your modules
        from data_ingestion import download_stock_data
        print("Successfully imported download_stock_data function.")
        ```

5.  **Start Developing\!**

      * Open existing notebooks (`.ipynb` files) or create new ones.
      * Start importing functions from your `src` directory and experimenting.

## 6\. Best Practices for Developers

  * **Move Code to `src/`:** When you've developed and validated a piece of code in a notebook, refactor it into clean, modular Python functions/classes in the `src/` directory. Your `src/` directory should be the source of truth for your automated pipelines.
  * **Version Control Notebooks (Carefully):** Commit your notebooks to Git. However, consider configuring `nbstripout` (or similar tools) to remove cell output before committing to avoid large diffs and merge conflicts.
  * **Be Resource Aware:**
      * Stop your Jupyter server when you are done working to free up cluster resources.
      * Avoid running extremely long-running or resource-intensive tasks directly in notebooks if they belong in an automated workflow.
      * Clean up large, temporary files you create in your user's home directory.
  * **Leverage MLflow:** Use `mlflow.start_run()`, `mlflow.log_param()`, `mlflow.log_metric()`, and `mlflow.pytorch.log_model()` for *all* your experiments, whether interactive or automated. This ensures a centralized record of your work.
  * **Understand Shared Storage:** Remember that `/mnt/shared-data` and `/mnt/shared-artifacts` are shared PVCs. Be careful when writing to these directories, especially if multiple users or automated pipelines might be writing to the same locations. Coordinate with your team on data partitioning strategies.
  * **Don't Install System-Wide Libraries:** Avoid `!pip install <package>` directly in notebook cells without `--user`. Your user image is pre-configured; if you need a new dependency, request it from the platform team for inclusion in the next image build.

## 7\. Managing Python Dependencies

Your JupyterHub environment is built using a custom Docker image (`jtayl22/financial-predictor-jupyter:latest`) which includes a comprehensive set of pre-installed Python libraries defined in the project's `requirements.txt`. This approach ensures a consistent and reproducible environment for everyone.

  * **For Project Dependencies:** If you identify a new core library needed for the `financial-mlops-pytorch` project (e.g., a new data source connector, a widely used ML utility), please open an issue in the `k3s-homelab` (infrastructure) repository. The platform team will review the request, add it to the `requirements.txt` used for the JupyterHub user image, rebuild the image on a dedicated Linux runner, and push the updated version. This ensures that the base development environment remains consistent and performant for all users.
  * **For Personal Experimentation:** For temporary or highly specialized libraries needed for a short-term experiment, you can install them locally within your JupyterHub user environment using `pip install --user <package_name>`. Be aware that these installations are specific to your user's home directory and will persist across sessions unless your home directory is recreated. These changes will not be visible to other users or automated pipelines.

## 8\. Troubleshooting

  * **Jupyter Server Fails to Start:** Check the server logs (often visible in the JupyterHub UI itself or via `kubectl logs -f <your-user-pod-name> -n jupyterhub`). Contact the platform team with the error message.
  * **Missing Python Libraries:** If a library is not found (`ModuleNotFoundError`), ensure it's listed in the `requirements.txt` used for the JupyterHub user image. If it is, contact the platform team for a potential image rebuild.
  * **Cannot Access Shared Data:** Double-check the `mountPath` (e.g., `/mnt/shared-data`) and ensure the shared PVCs (`shared-data-pvc`, `shared-artifacts-pvc`) exist and are `Bound` in the `financial-mlops-pytorch` namespace (`kubectl get pvc -n financial-mlops-pytorch`).

-----





You're spot on. We need to update `docs/jupyterhub.md` to reflect the multi-architecture build strategy, and a Quick Start in the `README.md` is essential for new users.

-----




