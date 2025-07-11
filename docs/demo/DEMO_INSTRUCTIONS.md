# A/B Test Demonstration Instructions

This guide provides step-by-step instructions to run the financial model A/B test demonstration as described in the accompanying article.

## Prerequisites

1.  A running Kubernetes cluster.
2.  `kubectl` configured to point to your cluster.
3.  Python 3 and `pip` installed.
4.  Required Python packages installed. Run: `pip install -r requirements.txt`

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
sh ./scripts/start-monitoring.sh
```
Once running, you can access Grafana at [http://localhost:3000](http://localhost:3000). The `ab-testing-dashboard.json` will be automatically loaded.

### Step 4: Run the Demo Simulation

This script simulates user traffic to the A/B tested models. It will send 2,500 requests, split between the baseline and enhanced predictors as configured in the experiment.

```bash
echo "Running the A/B test simulation..."
python3 scripts/advanced-ab-demo.py --scenarios 2500 --workers 5
```

### Step 5: Observe the Results

While the simulation script is running, open the Grafana dashboard in your browser. You will see the panels populate in real-time, showing:

-   Traffic distribution between the models.
-   Real-time model accuracy and latency.
-   Business impact calculations.
-   Error rates and other performance metrics.

### Step 6: Cleanup (Optional)

Once you are finished with the demo, you can remove the created resources from your cluster with the following command:

```bash
echo "Cleaning up all demo resources..."
kubectl delete -k k8s/base
# You may also want to run a cleanup script for monitoring if available.
```
