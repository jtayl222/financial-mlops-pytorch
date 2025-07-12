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

### Step 7: Cleanup (Optional)

Once you are finished with the demo, you can remove the created resources from your cluster with the following command:

```bash
echo "Cleaning up all demo resources..."
kubectl delete -k k8s/base
# You may also want to run a cleanup script for monitoring if available.
```
