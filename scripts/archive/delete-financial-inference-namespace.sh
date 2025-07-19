# Handle stuck resources with finalizers (see troubleshooting docs)
kubectl patch model baseline-predictor -n financial-inference --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true
kubectl patch model enhanced-predictor -n financial-inference --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true
kubectl patch experiment financial-ab-test-experiment -n financial-inference --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true
kubectl patch server mlserver -n financial-inference --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true
kubectl patch  seldonruntimes financial-inference-runtime -n financial-inference --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true

kubectl patch  namespace financial-inference --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true

kubectl delete model baseline-predictor -n financial-inference
kubectl delete model enhanced-predictor -n financial-inference
kubectl delete experiment financial-ab-test-experiment -n financial-inference
kubectl delete server mlserver -n financial-inference
kubectl delete seldonruntimes financial-inference-runtime -n financial-inference
kubectl delete namespace financial-inference