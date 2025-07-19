# Handle stuck resources with finalizers (see troubleshooting docs)
kubectl patch model baseline-predictor -n seldon-system --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true
kubectl patch model enhanced-predictor -n seldon-system --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true
kubectl patch experiment financial-ab-test-experiment -n seldon-system --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true
kubectl patch server mlserver -n seldon-system --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true
kubectl patch  seldonruntimes seldon-system-runtime -n seldon-system --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true

kubectl patch  namespace seldon-system --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true

kubectl delete model baseline-predictor -n seldon-system
kubectl delete model enhanced-predictor -n seldon-system
kubectl delete experiment financial-ab-test-experiment -n seldon-system
kubectl delete server mlserver -n seldon-system
kubectl delete seldonruntimes seldon-system-runtime -n seldon-system
kubectl delete namespace seldon-system
