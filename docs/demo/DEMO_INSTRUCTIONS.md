# A/B Test Demonstration Instructions

This guide provides step-by-step instructions to run the financial model A/B test demonstration as described in the accompanying article.

## Prerequisites

1.  A running Kubernetes cluster with NGINX Ingress Controller deployed.
2.  `kubectl` configured to point to your cluster.
3.  Python 3 and `pip` installed.
4.  Required Python packages installed. Run: `pip install -r requirements.txt`
5.  DNS configured for `ml-api.local` (add to `/etc/hosts`: `192.168.1.249 ml-api.local`)
6.  **Platform Team Configuration:** Ensure Seldon scheduler connectivity is configured (see [Platform Team Request](../platform-requests/seldon-scheduler-service-alias-request-CLOSED.md))

## Instructions

### Step 1: Deploy Monitoring Infrastructure

This script deploys Prometheus for metrics collection and Grafana for visualization into your cluster.

```bash
echo "Deploying Prometheus and Grafana..."
sh ./scripts/setup-monitoring.sh
```

### Step 2: Deploy the Seldon A/B Test Application

This command applies the Kubernetes manifests that define the namespace, models, and the Seldon `Experiment` for the A/B test.

```bash
echo "Deploying the A/B test resources..."
kubectl apply -k k8s/base
```
Wait for the pods in the `financial-inference` namespace to be in the `Running` state before proceeding. You can check with `kubectl get pods -n financial-inference`.

### Step 3: Access the Grafana Dashboard

This script forwards the Grafana service to your local machine, making the dashboard accessible in your browser.

**Run this command in a separate terminal window and keep it running.**

```bash
echo "Starting port-forward to Grafana. Keep this terminal open."
sh ./scripts/demo/start-monitoring.sh
```
Once running, you can access Grafana at [http://localhost:3000](http://localhost:3000). The `ab-testing-dashboard.json` will be automatically loaded.

### Step 4: Verify External Access

Test that the NGINX Ingress is working and models are accessible:

```bash
echo "Testing external access via NGINX Ingress..."
curl http://ml-api.local/financial-inference/v2/models

# Test A/B experiment endpoint (should return 200 with prediction)
curl -H "Host: ml-api.local" \
     -H "seldon-model: financial-ab-test-experiment.experiment" \
     -H "Content-Type: application/json" \
     http://192.168.1.249/financial-inference/v2/models/baseline-predictor_1/infer \
     --data '{"inputs":[{"name":"input_data","shape":[1,10,35],"datatype":"FP32","data":[[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]]]}]}'

# Look for 'x-seldon-route' header in response to verify A/B routing
```

### Step 5: Run the Demo Simulation

This script simulates user traffic to the A/B tested models via NGINX Ingress. It will send requests split between the baseline and enhanced predictors as configured in the experiment.

**For a quick test:**
```bash
echo "Running quick A/B test validation..."
python3 scripts/demo/advanced-ab-demo.py --scenarios 10 --workers 1 --no-viz --no-metrics
```

**For publication-ready visualization:**
```bash
echo "Running comprehensive A/B test simulation..."
python3 scripts/demo/advanced-ab-demo.py --scenarios 500 --workers 3
```

**For trade show demonstration:**
```bash
echo "Running large-scale A/B test simulation..."
python3 scripts/demo/advanced-ab-demo.py --scenarios 2500 --workers 5
```

The script will automatically:
- Generate realistic financial market scenarios
- Send inference requests via NGINX Ingress
- Track A/B traffic distribution (expect ~70/30 baseline/enhanced split)
- Measure response times and accuracy
- Create comprehensive visualizations
- Start Prometheus metrics server (port 8002) if enabled

### Step 6: Observe the Results

**A. Comprehensive Visualization (Automatic)**

The demo script automatically generates a publication-ready visualization saved as `advanced_ab_test_analysis_YYYYMMDD_HHMMSS.png` showing:

- **Traffic Distribution:** Pie chart showing A/B split (target: 70% baseline, 30% enhanced)
- **Response Time Comparison:** Box plots comparing model latencies  
- **Model Accuracy:** Bar chart with accuracy percentages and error bars
- **Predictions Over Time:** Scatter plot with color-coded model predictions
- **Market Scenario Distribution:** Request counts across different market conditions
- **Performance Heatmap:** Matrix comparing accuracy, speed, and prediction variance

**B. Real-time Monitoring (Optional)**

While the simulation script is running, you can also open the Grafana dashboard in your browser to see real-time metrics:

- Live traffic distribution between models
- Real-time model accuracy and latency
- Business impact calculations  
- Error rates and other performance metrics
- Prometheus metrics (available at http://localhost:8002/metrics when enabled)

**C. Expected Results:**

- **Success Rate:** 100% (all requests should succeed)
- **Traffic Split:** ~70% baseline-predictor, ~30% enhanced-predictor
- **Response Times:** 10-20ms average (depending on cluster performance)
- **Model Identification:** Response headers should show `x-seldon-route: :baseline-predictor_N:` or `:enhanced-predictor_N:`

## Troubleshooting

### Common Issues

**1. All traffic shows as "unknown" instead of baseline/enhanced split:**
- **Cause:** Model identification not working properly
- **Solution:** Check that `x-seldon-route` headers are being returned (see Step 4 validation)

**2. 404 errors on inference requests:**
- **Cause:** Controller manager cannot connect to central scheduler
- **Solution:** Verify Platform Team has applied scheduler connectivity fix (see prerequisites)

**3. Models show as not ready:**
- **Cause:** Split-brain scheduler conflicts or agent connection issues
- **Solution:** See [troubleshooting documentation](../troubleshooting/seldon-v2-api-404-debugging.md)

**4. DNS resolution fails:**
- **Cause:** Missing `/etc/hosts` entry
- **Solution:** Add `192.168.1.249 ml-api.local` to `/etc/hosts`

### Debug Commands

```bash
# Check Seldon resources status
kubectl get experiments,models,seldonruntimes -n financial-inference

# Check model loading in agents
kubectl logs -n financial-inference sts/mlserver -c agent --tail=20

# Check scheduler connectivity
kubectl logs -n seldon-system deployment/seldon-v2-controller-manager --tail=50

# Test direct inference
curl -v -H "Host: ml-api.local" \
     -H "seldon-model: financial-ab-test-experiment.experiment" \
     http://192.168.1.249/financial-inference/v2/models/baseline-predictor_1/infer \
     --data '{"inputs":[{"name":"input_data","shape":[1,10,35],"datatype":"FP32","data":[...]}]}'
```

For detailed troubleshooting, see:
- [Seldon v2 API 404 Debugging](../troubleshooting/seldon-v2-api-404-debugging.md)
- [Platform Team Request History](../platform-requests/seldon-scheduler-service-alias-request-CLOSED.md)

## Advanced Model Training and Deployment

This section demonstrates the complete MLOps workflow: data ingestion → S3 storage → model training → Seldon deployment → A/B testing using real market data (IBB biotech ETF).

### Prerequisites for Advanced Demo

1. **Kubernetes Cluster:** With Argo Workflows and MinIO/S3 configured
2. **MLflow Server:** Running MLflow for experiment tracking  
3. **Storage Backend:** MinIO or S3 for data and model storage
4. **Data Pipeline:** `k8s/base/financial-data-pipeline.yaml` for automated ingestion
5. **Advanced Model Code:** `src/advanced_financial_model.py` for enhanced features

### Step A1: Deploy Data Ingestion Pipeline

First, ingest IBB data into S3 storage using the Kubernetes data pipeline:

```bash
# Deploy the data ingestion pipeline
kubectl apply -f k8s/base/financial-data-pipeline.yaml

# Verify the WorkflowTemplate is created
kubectl get workflowtemplates -n financial-mlops-pytorch
```

### Step A2: Run Data Ingestion for IBB

Submit an Argo Workflow to ingest IBB data into S3:

```bash
# Submit workflow to ingest IBB data
argo submit --from workflowtemplate/financial-data-pipeline-template \
  -p ingestion-start-date="2018-01-01" \
  -p ingestion-end-date="2023-12-31" \
  -p tickers="IBB" \
  -n financial-mlops-pytorch

# Expected output:
# Name:                financial-data-pipeline-template-rcgqr
# Namespace:           financial-mlops-pytorch
# ServiceAccount:      unset
# Status:              Pending
# Created:             Mon Jul 14 22:37:51 -0400 (now)
# Parameters:          
#   ingestion-start-date: 2018-01-01
#   ingestion-end-date: 2023-12-31
#   tickers:           IBB

# Monitor the workflow progress
argo list -n financial-mlops-pytorch

# Expected output:
# NAME                                     STATUS    AGE   DURATION   PRIORITY   MESSAGE
# financial-data-pipeline-template-rcgqr   Running   7s    7s         0

# Watch workflow execution (replace rcgqr with your actual workflow suffix)
argo logs -f financial-data-pipeline-template-rcgqr -n financial-mlops-pytorch
```

**Note:** The workflow template has been updated to properly use parameters. If you encounter issues with hardcoded tickers, ensure the template uses `{{workflow.parameters.tickers}}` in the environment variables.

**Expected Workflow Steps:**
1. **ingest-data:** Downloads IBB historical data from financial APIs
2. **engineer-features:** Processes raw data and creates feature sets
3. **Data Storage:** Saves processed data to shared S3/MinIO storage

**Expected Workflow Execution Log (Sample):**
```
# Step 1: Data Ingestion
2025-07-15 02:35:31,779 - INFO - Starting data ingestion process.
2025-07-15 02:35:31,780 - INFO - Configured TICKERS: ['IBB']
2025-07-15 02:35:33,394 - INFO - Downloading data for ticker: IBB
2025-07-15 02:35:34,248 - INFO - Successfully downloaded 1509 rows for IBB.
2025-07-15 02:35:34,301 - INFO - Data for IBB saved to /mnt/shared-data/raw/IBB_raw_2018-01-01_2023-12-31.csv

# Step 2: Feature Engineering
2025-07-15 02:35:57,859 - INFO - Starting feature engineering and PyTorch data preparation process.
2025-07-15 02:36:00,670 - INFO - Processing file: /mnt/shared-data/raw/IBB_raw_2018-01-01_2023-12-31.csv
2025-07-15 02:36:00,795 - INFO - Train features shape: (2024, 35)
2025-07-15 02:36:01,583 - INFO - Number of training sequences: 2015
2025-07-15 02:36:02,244 - INFO - Feature engineering and PyTorch data preparation completed successfully.

# MLflow Integration
🏃 View run at: http://mlflow.mlflow.svc.cluster.local:5000/#/experiments/0/runs/xxxxx
🧪 View experiment at: http://mlflow.mlflow.svc.cluster.local:5000/#/experiments/0
```

**Key Success Indicators:**
- ✅ "Successfully downloaded X rows for IBB"
- ✅ "Feature engineering and PyTorch data preparation completed successfully"
- ✅ MLflow tracking URLs displayed
- ✅ Workflow status shows "Succeeded" in `argo list`

### Step A3: Verify Data Ingestion

Confirm that IBB data has been successfully ingested into S3:

```bash
# Check workflow completion status
argo get financial-data-pipeline-template-xxxxx -n financial-mlops-pytorch

# Alternative: Get any running/completed workflows
argo list -n financial-mlops-pytorch

# For direct data verification, check persistent volumes
kubectl get pvc -n financial-mlops-pytorch

# If you have access to a workflow pod, verify data structure
kubectl exec -it -n financial-mlops-pytorch <any-pod-with-storage> -- \
  ls -la /mnt/shared-data/raw/

# Check processed features  
kubectl exec -it -n financial-mlops-pytorch <any-pod-with-storage> -- \
  ls -la /mnt/shared-data/processed/
```

**Expected S3/MinIO Structure:**
```
/mnt/shared-data/
├── raw/
│   ├── IBB_raw_2018-01-01_2023-12-31.csv
│   └── (other ticker files if any)
└── processed/
    ├── train_features.npy
    ├── train_targets.npy  
    ├── val_features.npy
    ├── val_targets.npy
    ├── test_features.npy
    ├── test_targets.npy
    └── combined_processed_data.csv
```

**Troubleshooting Data Ingestion:**

If the workflow fails or data is missing, see the comprehensive troubleshooting guide:

📚 **[Argo Workflow Data Ingestion Troubleshooting](../troubleshooting/argo-workflow-data-ingestion.md)**

Quick debug commands:
```bash
# Check workflow status and logs
argo list -n financial-mlops-pytorch
argo logs financial-data-pipeline-template-xxxxx -n financial-mlops-pytorch

# Verify data was created
kubectl exec -it -n financial-mlops-pytorch <any-pod> -- \
  ls -la /mnt/shared-data/raw/ | grep IBB
```

### Step A4: Train Models with S3 Data

Now train models using the ingested IBB data from S3:

```bash
# Submit training workflow for baseline model
argo submit --from workflowtemplate/financial-training-pipeline-template \
  -p model-variant=baseline \
  -p data-source=IBB \
  -n financial-mlops-pytorch

# Submit training workflow for enhanced model  
argo submit --from workflowtemplate/financial-training-pipeline-template \
  -p model-variant=enhanced \
  -p data-source=IBB \
  -n financial-mlops-pytorch

# Monitor training progress
argo list -n financial-mlops-pytorch
argo logs -f <training-workflow-name> -n financial-mlops-pytorch
```

**Expected Training Results:**
- **Baseline Model:** ~52.7% accuracy (realistic financial prediction baseline)
- **Enhanced Model:** ~85.2% accuracy in lab conditions (may degrade in production)
- **Model Storage:** Models automatically stored in MLflow registry via S3

### Step A5: Deploy Trained Models to Seldon

Update Seldon deployments with newly trained models:

```bash
# Update model URIs from MLflow registry
python3 scripts/update_model_uris.py

# Apply updated model configurations
kubectl apply -k k8s/base

# Wait for models to load
kubectl wait --for=condition=ready \
  models/baseline-predictor models/enhanced-predictor \
  -n financial-inference --timeout=300s

# Verify models are ready
kubectl get models -n financial-inference
```

### Step A6: Test Production Model Performance

Test the IBB-trained models with realistic scenarios:

```bash
# Test COVID crash performance (March 2020)
python src/test_covid_crash.py

# Test biotech winter performance (2022) 
python src/test_biotech_winter_2022.py

# Test transaction cost impact
python src/test_transaction_costs.py
```

**Expected Reality Check Results:**
- **COVID Crash Period:** ~57.1% accuracy (significant degradation)
- **Transaction Costs:** Can turn profitable predictions into losses
- **Market Regime Changes:** Model performance varies dramatically

### Step A7: Run Advanced A/B Testing

Run A/B testing with the IBB-trained models:

```bash
# Quick validation with new models
python3 scripts/demo/advanced-ab-demo.py --scenarios 50 --workers 2

# Full A/B test demonstration  
python3 scripts/demo/advanced-ab-demo.py --scenarios 1000 --workers 3
```

**Monitor in Grafana:**
- Model accuracy comparison (baseline vs enhanced on IBB data)
- Traffic distribution across model variants
- Business impact metrics with realistic constraints

### Advanced Pipeline Architecture

This demonstrates the complete enterprise MLOps pipeline:

```
Financial APIs → Argo Workflows → MinIO/S3 → Model Training → MLflow Registry → Seldon Deployment → A/B Testing → Grafana Monitoring
```

**Key Components:**
- **Data Ingestion:** `k8s/base/financial-data-pipeline.yaml`
- **Training Pipeline:** `k8s/base/financial-training-pipeline.yaml`
- **Model Deployment:** `k8s/base/financial-models.yaml`
- **A/B Testing:** `k8s/base/financial-ab-experiment.yaml`

### Advanced Troubleshooting

**Pipeline Issues:**
```bash
# Check Argo Workflows status
kubectl get workflows -n financial-mlops-pytorch

# Debug failed workflow steps
argo logs <workflow-name> -n financial-mlops-pytorch

# Check persistent volume claims
kubectl get pvc -n financial-mlops-pytorch
```

**Storage Issues:**
```bash
# Verify MinIO/S3 connectivity
kubectl port-forward -n minio svc/minio 9000:9000
curl http://localhost:9000/health

# Check data ingestion logs
kubectl logs -n financial-mlops-pytorch -l workflows.argoproj.io/workflow=data-ingestion
```

## Reality-Based Expectations

This advanced demo showcases:

✅ **What Works Well:**
- End-to-end automated MLOps pipeline
- Robust data ingestion and storage
- Sophisticated A/B testing and deployment capabilities
- Comprehensive monitoring and alerting
- Professional enterprise architecture

⚠️ **Realistic Limitations:**
- Model accuracy: 52-85% (not the 95%+ needed for profitable trading)
- Performance degradation during market stress
- Transaction costs significantly impact returns
- Lab results don't translate directly to production

The value proposition is the **comprehensive MLOps infrastructure and methodology**, not the specific model performance.

### Step 7: Cleanup (Optional)

Once you are finished with the demo, you can remove the created resources from your cluster with the following command:

```bash
echo "Cleaning up all demo resources..."
kubectl delete -k k8s/base

# Clean up workflows
argo delete --all -n financial-mlops-pytorch

# Clean up persistent data (optional - be careful!)
# kubectl delete pvc --all -n financial-mlops-pytorch
```
