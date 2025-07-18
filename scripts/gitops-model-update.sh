#!/bin/bash
# gitops-model-update.sh
# Automated model deployment via GitOps after training completion

set -e

MODEL_VARIANT="${1:-enhanced}"
COMMIT_MESSAGE="${2:-update: model URIs from latest training}"
MLFLOW_ENDPOINT="${MLFLOW_ENDPOINT:-http://192.168.1.203:5000}"

echo "🚀 Starting GitOps model deployment for variant: $MODEL_VARIANT"

# Step 1: Update model URIs
echo "📝 Updating model URIs from MLflow..."
python3 scripts/update_model_uris.py \
  --mlflow-endpoint "$MLFLOW_ENDPOINT" \
  --model-variant "$MODEL_VARIANT"

# Step 2: Check for changes
if git diff --quiet k8s/base/financial-predictor-ab-test.yaml; then
  echo "ℹ️  No changes detected in model URIs"
  exit 0
fi

# Step 3: Show changes
echo "📋 Model URI changes detected:"
git diff k8s/base/financial-predictor-ab-test.yaml

# Step 4: Commit changes
echo "💾 Committing changes to git..."
git add k8s/base/financial-predictor-ab-test.yaml
git commit -m "$COMMIT_MESSAGE

🤖 Automated model deployment via GitOps
- Model variant: $MODEL_VARIANT  
- MLflow endpoint: $MLFLOW_ENDPOINT
- Updated URIs for Seldon deployment

Co-Authored-By: GitOps-Bot <noreply@company.com>"

# Step 5: Push to trigger Argo CD sync
echo "🔄 Pushing to remote repository..."
git push origin "$(git branch --show-current)"

# Step 6: Monitor Argo CD sync status
echo "⏳ Waiting for Argo CD to sync changes..."
echo "   Monitor at: http://192.168.1.85:30080"

# Wait for sync to complete
for i in {1..30}; do
  SYNC_STATUS=$(kubectl get application financial-mlops-infrastructure -n argocd -o jsonpath='{.status.sync.status}' 2>/dev/null || echo "Unknown")
  
  case "$SYNC_STATUS" in
    "Synced")
      echo "✅ Argo CD sync completed successfully!"
      break
      ;;
    "OutOfSync")
      echo "   🔄 Sync in progress... ($i/30)"
      ;;
    *)
      echo "   ⏳ Sync status: $SYNC_STATUS ($i/30)"
      ;;
  esac
  
  if [ $i -eq 30 ]; then
    echo "⚠️  Timeout waiting for sync completion"
    echo "   Check Argo CD UI: http://192.168.1.85:30080"
    exit 1
  fi
  
  sleep 10
done

# Step 7: Verify model deployment
echo "🔍 Verifying model deployment..."
kubectl get models -n financial-mlops-pytorch

# Step 8: Check model readiness
BASELINE_READY=$(kubectl get model baseline-predictor -n financial-mlops-pytorch -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "Unknown")
ENHANCED_READY=$(kubectl get model enhanced-predictor -n financial-mlops-pytorch -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "Unknown")

echo ""
echo "📊 Model Status Summary:"
echo "   Baseline: $BASELINE_READY"
echo "   Enhanced: $ENHANCED_READY"

if [ "$BASELINE_READY" = "True" ] && [ "$ENHANCED_READY" = "True" ]; then
  echo "🎉 All models deployed successfully!"
  echo ""
  echo "🔗 Next steps:"
  echo "   - Verify A/B experiment: kubectl get experiments -n financial-mlops-pytorch"
  echo "   - Test model endpoints via Seldon mesh"
  echo "   - Monitor model performance in MLflow"
else
  echo "⚠️  Some models may still be loading. Check:"
  echo "   kubectl describe models -n financial-mlops-pytorch"
fi

echo ""
echo "✅ GitOps model deployment completed!"