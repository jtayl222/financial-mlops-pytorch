#!/usr/bin/env python3
"""
Update Seldon model deployment YAML files with latest MLflow model URIs
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
from urllib.parse import urljoin
from urllib.request import urlopen, Request
from urllib.error import URLError


def get_mlflow_data(endpoint, path, data=None):
    """Make request to MLflow API"""
    url = urljoin(endpoint, path)
    
    headers = {'Content-Type': 'application/json'}
    
    # Add basic auth if credentials are available
    username = os.environ.get('MLFLOW_TRACKING_USERNAME')
    password = os.environ.get('MLFLOW_TRACKING_PASSWORD')
    if username and password:
        credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
        headers['Authorization'] = f'Basic {credentials}'
    
    if data:
        req = Request(url, 
                     data=json.dumps(data).encode('utf-8'),
                     headers=headers)
    else:
        req = Request(url, headers=headers)
    
    try:
        with urlopen(req) as response:
            return json.loads(response.read().decode('utf-8'))
    except URLError as e:
        print(f"❌ Error accessing MLflow API: {e}")
        return None


def get_experiment_id(mlflow_endpoint, experiment_name):
    """Get experiment ID by name"""
    # For now, use hardcoded experiment ID 29 since we know it contains the latest models
    # In production, you'd want to implement experiment lookup via ajax-api
    return "29"


def get_latest_model_uri(mlflow_endpoint, experiment_id, model_variant):
    """Get latest model URI for a specific variant"""
    search_data = {
        "experiment_ids": [experiment_id],
        "filter": f"params.model_variant = '{model_variant}'",
        "order_by": ["attributes.start_time DESC"],
        "max_results": 1
    }
    
    data = get_mlflow_data(mlflow_endpoint, "ajax-api/2.0/mlflow/runs/search", search_data)
    
    if not data or 'runs' not in data or len(data['runs']) == 0:
        print(f"❌ No runs found for variant '{model_variant}'")
        return None
    
    run_id = data['runs'][0]['info']['run_id']
    model_uri = f"s3://mlflow-artifacts/{experiment_id}/models/m-{run_id}/artifacts/"
    
    return model_uri


def update_yaml_file(yaml_file, model_updates):
    """Update YAML file with new model URIs"""
    try:
        with open(yaml_file, 'r') as f:
            content = f.read()
        
        # Create backup
        backup_file = yaml_file + '.bak'
        with open(backup_file, 'w') as f:
            f.write(content)
        
        original_content = content
        
        # Update each model variant
        for model_name, new_uri in model_updates.items():
            # Pattern to match storageUri within a specific model block
            pattern = rf'(name: {model_name}.*?storageUri: )s3://mlflow-artifacts/[^/]+/[^/]+/[^/]+/artifacts/'
            replacement = rf'\1{new_uri}'
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        # Write updated content
        with open(yaml_file, 'w') as f:
            f.write(content)
        
        # Show diff
        if content != original_content:
            print("\n📋 Changes made:")
            try:
                result = subprocess.run(['diff', backup_file, yaml_file], 
                                      capture_output=True, text=True)
                if result.stdout:
                    print(result.stdout)
            except FileNotFoundError:
                print("   (diff command not available)")
        
        # Clean up backup
        os.remove(backup_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating YAML file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Update Seldon model URIs from MLflow')
    parser.add_argument('--experiment', default='financial-forecasting',
                       help='MLflow experiment name')
    parser.add_argument('--model-variant', 
                       help='Specific model variant to update (baseline, enhanced, lightweight)')
    parser.add_argument('--yaml-file', default='k8s/base/financial-predictor-ab-test.yaml',
                       help='YAML file to update')
    parser.add_argument('--mlflow-endpoint', 
                       default=os.environ.get('MLFLOW_TRACKING_URI', 'http://192.168.1.203:5000'),
                       help='MLflow endpoint URL')
    parser.add_argument('--experiment-id',
                       help='MLflow experiment ID (overrides experiment name lookup)')
    
    args = parser.parse_args()
    
    print("🔍 Fetching latest model URIs from MLflow...")
    print(f"   Experiment: {args.experiment}")
    print(f"   MLflow endpoint: {args.mlflow_endpoint}")
    
    # Get experiment ID
    if args.experiment_id:
        experiment_id = args.experiment_id
    else:
        experiment_id = get_experiment_id(args.mlflow_endpoint, args.experiment)
        if not experiment_id:
            print(f"❌ Experiment '{args.experiment}' not found")
            sys.exit(1)
    
    print(f"   Experiment ID: {experiment_id}")
    
    # Determine which variants to update
    variants_to_update = [args.model_variant] if args.model_variant else ['baseline', 'enhanced']
    
    print("🔍 Getting latest runs...")
    
    # Get model URIs
    model_updates = {}
    for variant in variants_to_update:
        model_uri = get_latest_model_uri(args.mlflow_endpoint, experiment_id, variant)
        if model_uri:
            model_name = f"{variant}-predictor"
            model_updates[model_name] = model_uri
            print(f"   {variant.title()} URI: {model_uri}")
    
    if not model_updates:
        print("❌ No model URIs found to update")
        sys.exit(1)
    
    # Update YAML file
    print(f"📝 Updating {args.yaml_file}...")
    if update_yaml_file(args.yaml_file, model_updates):
        print(f"✅ Updated {args.yaml_file} with latest model URIs")
        
        print("\n🎯 Next steps:")
        print(f"   1. Review changes: git diff {args.yaml_file}")
        print(f"   2. Apply to cluster: kubectl apply -f {args.yaml_file}")
        print(f"   3. Commit changes: git add {args.yaml_file} && git commit -m 'update: model URIs from MLflow'")
    else:
        print("❌ Failed to update YAML file")
        sys.exit(1)


if __name__ == '__main__':
    main()