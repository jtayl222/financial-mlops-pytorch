#!/bin/bash
# update-model-uris.sh
# Updates Seldon model deployment YAML files with latest MLflow model URIs

set -e

# Configuration
MLFLOW_ENDPOINT="${MLFLOW_ENDPOINT:-http://mlflow.mlflow.svc.cluster.local:5000}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-financial-forecasting}"
YAML_FILE="${YAML_FILE:-k8s/base/financial-predictor-ab-test.yaml}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --experiment)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --model-variant)
      MODEL_VARIANT="$2"
      shift 2
      ;;
    --yaml-file)
      YAML_FILE="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --experiment NAME     MLflow experiment name (default: financial-forecasting)"
      echo "  --model-variant NAME  Model variant to update (baseline, enhanced, lightweight)"
      echo "  --yaml-file PATH      YAML file to update (default: k8s/base/financial-predictor-ab-test.yaml)"
      echo "  --help               Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

echo "🔍 Fetching latest model URIs from MLflow..."
echo "   Experiment: $EXPERIMENT_NAME"
echo "   MLflow endpoint: $MLFLOW_ENDPOINT"

# Get experiment ID
EXPERIMENT_ID=$(curl -s "$MLFLOW_ENDPOINT/api/2.0/mlflow/experiments/get-by-name?experiment_name=$EXPERIMENT_NAME" | jq -r '.experiment.experiment_id // empty')

if [ -z "$EXPERIMENT_ID" ]; then
  echo "❌ Experiment '$EXPERIMENT_NAME' not found"
  exit 1
fi

echo "   Experiment ID: $EXPERIMENT_ID"

# Get latest runs for each model variant
echo "🔍 Getting latest runs..."

get_latest_model_uri() {
  local variant=$1
  local tag_filter="tags.model_variant='$variant'"
  
  # Get latest run for this variant
  local run_data=$(curl -s "$MLFLOW_ENDPOINT/api/2.0/mlflow/runs/search" \
    -H "Content-Type: application/json" \
    -d "{
      \"experiment_ids\": [\"$EXPERIMENT_ID\"],
      \"filter\": \"$tag_filter\",
      \"order_by\": [\"start_time DESC\"],
      \"max_results\": 1
    }")
  
  local run_id=$(echo "$run_data" | jq -r '.runs[0].info.run_id // empty')
  
  if [ -z "$run_id" ]; then
    echo "❌ No runs found for variant '$variant'"
    return 1
  fi
  
  # Construct model URI
  local model_uri="s3://mlflow-artifacts/$EXPERIMENT_ID/models/m-$run_id/artifacts/"
  echo "$model_uri"
}

# Update specific variant or all variants
if [ -n "$MODEL_VARIANT" ]; then
  echo "📝 Updating model variant: $MODEL_VARIANT"
  
  MODEL_URI=$(get_latest_model_uri "$MODEL_VARIANT")
  if [ $? -eq 0 ]; then
    echo "   Latest URI: $MODEL_URI"
    
    # Update YAML file
    sed -i.bak "s|storageUri: s3://mlflow-artifacts/.*/artifacts/|storageUri: $MODEL_URI|g" "$YAML_FILE"
    echo "✅ Updated $YAML_FILE"
  fi
else
  echo "📝 Updating all model variants..."
  
  # Update baseline model
  BASELINE_URI=$(get_latest_model_uri "baseline")
  if [ $? -eq 0 ]; then
    echo "   Baseline URI: $BASELINE_URI"
    # Update baseline-predictor storageUri
    sed -i.bak "/name: baseline-predictor/,/storageUri:/ s|storageUri: s3://mlflow-artifacts/.*/artifacts/|storageUri: $BASELINE_URI|" "$YAML_FILE"
  fi
  
  # Update enhanced model
  ENHANCED_URI=$(get_latest_model_uri "enhanced")
  if [ $? -eq 0 ]; then
    echo "   Enhanced URI: $ENHANCED_URI"
    # Update enhanced-predictor storageUri
    sed -i.bak "/name: enhanced-predictor/,/storageUri:/ s|storageUri: s3://mlflow-artifacts/.*/artifacts/|storageUri: $ENHANCED_URI|" "$YAML_FILE"
  fi
  
  echo "✅ Updated $YAML_FILE with latest model URIs"
fi

# Show diff
if [ -f "$YAML_FILE.bak" ]; then
  echo ""
  echo "📋 Changes made:"
  diff "$YAML_FILE.bak" "$YAML_FILE" || true
  rm "$YAML_FILE.bak"
fi

echo ""
echo "🎯 Next steps:"
echo "   1. Review changes: git diff $YAML_FILE"
echo "   2. Apply to cluster: kubectl apply -f $YAML_FILE"
echo "   3. Commit changes: git add $YAML_FILE && git commit -m 'update: model URIs from MLflow'"