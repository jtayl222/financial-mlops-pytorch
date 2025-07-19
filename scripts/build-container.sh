#!/bin/bash

# Container Build Script
# Usage: ./scripts/build-container.sh [quick|full] [tag]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
BUILD_TYPE="${1:-quick}"
IMAGE_TAG="${2:-latest}"
NAMESPACE="seldon-system"

echo "🐳 Container Build Script"
echo "========================"
echo "Build Type: $BUILD_TYPE"
echo "Image Tag: $IMAGE_TAG"
echo "Namespace: $NAMESPACE"
echo ""

# Apply the workflow template
echo "📋 Applying Docker build pipeline template..."
kubectl apply -f "$ROOT_DIR/k8s/base/docker-build-pipeline.yaml"

# Submit the appropriate workflow
if [[ "$BUILD_TYPE" == "quick" ]]; then
    echo "⚡ Starting quick build..."
    WORKFLOW_NAME=$(argo submit \
        --from workflowtemplate/docker-build-pipeline-template \
        --entrypoint quick-build \
        --parameter image-tag="$IMAGE_TAG" \
        -n "$NAMESPACE" \
        --output name)
else
    echo "🏗️ Starting full build..."
    WORKFLOW_NAME=$(argo submit \
        --from workflowtemplate/docker-build-pipeline-template \
        --parameter image-tag="$IMAGE_TAG" \
        -n "$NAMESPACE" \
        --output name)
fi

echo "✅ Workflow submitted: $WORKFLOW_NAME"
echo ""

# Monitor the workflow
echo "👀 Monitoring workflow progress..."
echo "Commands:"
echo "  argo get $WORKFLOW_NAME -n $NAMESPACE"
echo "  argo logs $WORKFLOW_NAME -n $NAMESPACE -f"
echo ""

# Optional: Auto-follow logs
read -p "Follow logs automatically? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📊 Following workflow logs..."
    argo logs "$WORKFLOW_NAME" -n "$NAMESPACE" -f
fi

echo ""
echo "🎉 Build process initiated!"
echo "Image will be available at: harbor.test/library/financial-predictor:$IMAGE_TAG"