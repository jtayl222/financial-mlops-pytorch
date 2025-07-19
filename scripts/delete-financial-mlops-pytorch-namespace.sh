# Handle stuck resources with finalizers (see troubleshooting docs)
kubectl patch model baseline-predictor -n financial-mlops-pytorch --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true
kubectl patch model enhanced-predictor -n financial-mlops-pytorch --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true
kubectl patch experiment financial-ab-test-experiment -n financial-mlops-pytorch --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true
kubectl patch server mlserver -n financial-mlops-pytorch --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true
kubectl patch  seldonruntimes financial-mlops-pytorch-runtime -n financial-mlops-pytorch --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true

kubectl patch  namespace financial-mlops-pytorch --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true

kubectl delete model baseline-predictor -n financial-mlops-pytorch
kubectl delete model enhanced-predictor -n financial-mlops-pytorch
kubectl delete experiment financial-ab-test-experiment -n financial-mlops-pytorch
kubectl delete server mlserver -n financial-mlops-pytorch
kubectl delete seldonruntimes financial-mlops-pytorch-runtime -n financial-mlops-pytorch
kubectl delete namespace financial-mlops-pytorch
