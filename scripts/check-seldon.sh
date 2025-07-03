#!/bin/bash
# filepath: /home/user/REPOS/financial-mlops-pytorch/scripts/check-seldon.sh

set -e

echo "=== ARGO WORKFLOWS ==="
argo -n financial-mlops-pytorch list | head -10
echo

echo "=== SELDON SYSTEM PODS ==="
kubectl get pods -n seldon-system
echo

echo "=== SELDON MESH RESOURCES ==="
kubectl get models,pipelines,experiments -n seldon-mesh -o wide
echo

echo "=== SELDON CONTROLLER LOGS (Last 20 lines) ==="
kubectl logs -n seldon-system deployment/seldon-v2-controller-manager --tail=20
echo

echo "=== SELDON SCHEDULER LOGS (Last 20 lines) ==="
kubectl logs -n seldon-system seldon-scheduler-0 --tail=20
echo

echo "=== MODEL STATUS DETAILS ==="
for model in baseline-predictor enhanced-predictor; do
    echo "--- $model ---"
    kubectl get model $model -n seldon-mesh -o jsonpath='{.status}' | jq . 2>/dev/null || echo "No status or jq not available"
    echo
done

echo "=== PIPELINE STATUS DETAILS ==="
kubectl get pipeline financial-ab-test-pipeline -n seldon-mesh -o jsonpath='{.status}' | jq . 2>/dev/null || echo "No status or jq not available"
echo

echo "=== EXPERIMENT STATUS DETAILS ==="
kubectl get experiment financial-ab-test-experiment -n seldon-mesh -o jsonpath='{.status}' | jq . 2>/dev/null || echo "No status or jq not available"
echo

echo "=== SELDON MESH PODS (Should show model servers) ==="
kubectl get pods -n seldon-mesh
echo

echo "=== SELDON MESH SERVICES ==="
kubectl get svc -n seldon-mesh
echo