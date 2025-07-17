# A/B Test Demonstration Instructions

This guide provides step-by-step instructions to run the financial model A/B test demonstration using the Scoped Operator Pattern (v2.9.1).

## Prerequisites

1.  A running Kubernetes cluster with:
    - **Seldon Core v2.9.1** installed (see platform team for deployment)
    - **NGINX Ingress Controller** deployed
    - **MetalLB** or similar LoadBalancer configuration
2.  `kubectl` configured to point to your cluster
3.  Python 3 and `pip` installed
4.  Required Python packages installed: `pip install -r requirements.txt`
5.  **Architecture:** Scoped Operator Pattern (v2.9.1) - uses local scheduler in each namespace

## Instructions

### Step 1: Deploy Monitoring Infrastructure

This script deploys Prometheus for metrics collection and Grafana for visualization into your cluster.

```bash
echo "Deploying Prometheus and Grafana..."
sh ./scripts/setup-monitoring.sh
```

### Step 2: Deploy the Seldon A/B Test Application

This command applies the Kubernetes manifests that define the namespace, models, and the Seldon `Experiment` for the A/B test using the Scoped Operator Pattern.

```bash
echo "Creating namespace and adding secrets..."
kubectl create namespace financial-inference
kubectl apply -k k8s/manifests/financial-inference/production/

echo "Deploying the A/B test resources..."
kubectl apply -k k8s/base
```

Wait for all resources to be ready. You can check with:
```bash
kubectl get models,servers,experiments -n financial-inference
```

Expected output:
```
NAME                                       READY   DESIRED REPLICAS   AVAILABLE REPLICAS   AGE
model.mlops.seldon.io/baseline-predictor   True                       1                    2m
model.mlops.seldon.io/enhanced-predictor   True                       1                    2m

NAME                              READY   REPLICAS   LOADED MODELS   AGE
server.mlops.seldon.io/mlserver   True    1          2               2m

NAME                                                      EXPERIMENT READY   MESSAGE   AGE
experiment.mlops.seldon.io/financial-ab-test-experiment   True                         2m
```

### Step 3: Access the Grafana Dashboard

This script forwards the Grafana service to your local machine, making the dashboard accessible in your browser.

**Run this command in a separate terminal window and keep it running.**

```bash
echo "Starting port-forward to Grafana. Keep this terminal open."
sh ./scripts/demo/start-monitoring.sh
```
Once running, you can access Grafana at [http://localhost:3000](http://localhost:3000). The `ab-testing-dashboard.json` will be automatically loaded.

### Step 4: Run Architecture Validation

Validate that the Scoped Operator Pattern is working correctly:

```bash
echo "Validating Scoped Operator Pattern deployment..."
./scripts/validate-seldon-architecture.sh
```

Expected output should show:
- ✅ Local scheduler running in financial-inference namespace
- ✅ Both models ready and loaded
- ✅ A/B test experiment operational
- ✅ Overall Status: HEALTHY

### Step 5: Verify Model Access

Test that the models are accessible via port-forwarding:

```bash
# Start port-forwarding to Seldon mesh (run in separate terminal)
kubectl port-forward -n financial-inference svc/seldon-mesh 8082:80 &

# Test individual models with correct shape (1, 10, 35)
echo "Testing both models..."
python3 scripts/demo/test-model-inference.py

# Expected response: HTTP 200 with prediction output
```

### Step 6: Run the Demo Simulation

This script simulates user traffic to the A/B tested models via port-forwarding. It will send requests to both models to demonstrate the system.

**For a quick test:**
```bash
echo "Running quick model validation..."
python3 scripts/demo/test-model-inference.py
```

**For comprehensive demonstration:**
```bash
echo "Running comprehensive A/B test simulation..."
python3 scripts/demo/advanced-ab-demo.py --scenarios 100 --workers 2
```

The script will automatically:
- Generate realistic financial market scenarios
- Send inference requests via port-forwarding
- Track model performance and response times
- Create comprehensive visualizations
- Demonstrate the Scoped Operator Pattern in action

### Step 7: Observe the Results

**A. Model Performance Validation**

The validation should show:
- ✅ **Success Rate:** 100% (all requests should succeed)
- ✅ **Response Times:** 10-50ms average (depending on cluster performance)
- ✅ **Model Outputs:** Both models returning predictions (different values expected)
- ✅ **Scoped Operator Pattern:** Local scheduler managing both models in namespace

**B. Architecture Verification**

The validation script provides comprehensive checks:
- ✅ **Local Scheduler:** Running in financial-inference namespace (not centralized)
- ✅ **Model Status:** Both baseline and enhanced models ready
- ✅ **Server Status:** MLServer with 2 loaded models
- ✅ **Network Connectivity:** All components communicating within namespace
- ✅ **Overall Status:** HEALTHY deployment

**C. Expected System Behavior**

With the Scoped Operator Pattern (v2.9.1):
- **Simplified Architecture:** No cross-namespace scheduler dependencies
- **Faster Deployment:** All components start in ~2 minutes
- **Better Isolation:** Namespace-scoped operations reduce complexity
- **Industry Standard:** Following v2.9.1 best practices

## Troubleshooting

### Common Issues

**1. Models show as not ready:**
- **Cause:** Local scheduler or agent connection issues
- **Solution:** Check local scheduler logs: `kubectl logs -n financial-inference sts/seldon-scheduler`

**2. Port-forwarding connection refused:**
- **Cause:** seldon-mesh service not ready
- **Solution:** Wait for all pods to be ready: `kubectl get pods -n financial-inference`

**3. Model input shape errors:**
- **Cause:** Incorrect input format
- **Solution:** Use shape `[1, 10, 35]` with input name `input-0`

**4. Validation script fails:**
- **Cause:** Components still starting up
- **Solution:** Wait 2-3 minutes after deployment for all components to be ready

### Debug Commands

```bash
# Check Seldon resources status (Scoped Operator Pattern)
kubectl get models,servers,experiments -n financial-inference

# Check local scheduler logs
kubectl logs -n financial-inference sts/seldon-scheduler --tail=20

# Check model loading in agents
kubectl logs -n financial-inference sts/mlserver -c agent --tail=20

# Check all pods status
kubectl get pods -n financial-inference

# Run architecture validation
./scripts/validate-seldon-architecture.sh

# Test direct model inference
python3 scripts/demo/test-model-inference.py

# Test specific model
python3 scripts/demo/test-model-inference.py --models baseline-predictor
```

For detailed troubleshooting, see:
- [Architecture Decision Document](../architecture-decisions/seldon-production-architecture-decision.md)
- [Validation Script](../../scripts/validate-seldon-architecture.sh)

### Step 8: Cleanup (Optional)

Once you are finished with the demo, you can remove the created resources from your cluster with the following command:

```bash
echo "Cleaning up all demo resources..."
./scripts/delete-financial-inference-namespace.sh
# This will remove the entire namespace and all resources
```
