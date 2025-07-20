#!/usr/bin/env python3
"""
Promote latest trained models to Production stage in MLflow Model Registry
"""
import os
import mlflow
from mlflow.tracking import MlflowClient

def promote_model_to_production(model_name, run_id=None):
    """Promote the latest version of a model to Production stage"""
    client = MlflowClient()
    
    try:
        # Get the latest version of the model
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
        if not latest_versions:
            print(f"❌ No versions found for model {model_name}")
            return False
            
        latest_version = latest_versions[0]
        version_number = latest_version.version
        
        print(f"📦 Found model {model_name} version {version_number}")
        print(f"   Run ID: {latest_version.run_id}")
        print(f"   Source: {latest_version.source}")
        
        # Transition to Production
        client.transition_model_version_stage(
            name=model_name,
            version=version_number,
            stage="Production",
            archive_existing_versions=True  # Archive old Production versions
        )
        
        print(f"✅ Promoted {model_name} v{version_number} to Production")
        return True
        
    except Exception as e:
        print(f"❌ Failed to promote {model_name}: {e}")
        return False

def main():
    # Configure MLflow
    mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://192.168.1.203:5000')
    mlflow.set_tracking_uri(mlflow_uri)
    
    print(f"🔗 Connecting to MLflow: {mlflow_uri}")
    
    # Promote both models
    models_to_promote = [
        "FinancialDirectionPredictor_Baseline",
        "FinancialDirectionPredictor_Advanced"
    ]
    
    success_count = 0
    for model_name in models_to_promote:
        if promote_model_to_production(model_name):
            success_count += 1
        print()
    
    print(f"🎯 Summary: {success_count}/{len(models_to_promote)} models promoted to Production")
    
    if success_count == len(models_to_promote):
        print("✅ All models successfully promoted!")
        print("💡 You can now use these Model Registry URIs:")
        for model_name in models_to_promote:
            print(f"   models:/{model_name}/Production")
    else:
        print("⚠️  Some models failed to promote. Check MLflow UI for details.")

if __name__ == "__main__":
    main()