#!/usr/bin/env python3
"""
Test Model Inference Script
Tests both baseline and enhanced models with correct input format
"""

import requests
import json
import numpy as np
import sys
import argparse

def test_model_inference(endpoint_url="http://localhost:8082", models=None):
    """Test model inference with correct input shape"""
    
    if models is None:
        models = ['baseline-predictor', 'enhanced-predictor']
    
    # Generate test data with correct shape (1, 10, 350) - matching processed data
    test_data = np.random.rand(1, 10, 350).astype(np.float32)
    payload = {
        'inputs': [{
            'name': 'input-0',
            'shape': [1, 10, 350],
            'datatype': 'FP32',
            'data': test_data.tolist()
        }]
    }
    
    print(f"Testing models at {endpoint_url}")
    print(f"Input shape: {test_data.shape}")
    print("-" * 50)
    
    results = {}
    for model in models:
        try:
            response = requests.post(
                f'{endpoint_url}/v2/models/{model}/infer',
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result['outputs'][0]['data'][0]
                results[model] = {
                    'status': 'success',
                    'prediction': prediction,
                    'response_time': response.elapsed.total_seconds()
                }
                print(f"✅ {model}: {prediction:.4f} ({response.elapsed.total_seconds():.3f}s)")
            else:
                results[model] = {
                    'status': 'error',
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
                print(f"❌ {model}: Error {response.status_code}")
                print(f"   Response: {response.text}")
        
        except requests.exceptions.RequestException as e:
            results[model] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"❌ {model}: Connection error - {e}")
    
    # Summary
    print("-" * 50)
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    total_count = len(results)
    
    if success_count == total_count:
        print(f"🎉 All {total_count} models tested successfully!")
        return True
    else:
        print(f"⚠️  {success_count}/{total_count} models working")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test model inference endpoints')
    parser.add_argument('--endpoint', default='http://localhost:8082', 
                       help='Base URL for model endpoint (default: http://localhost:8082)')
    parser.add_argument('--models', nargs='+', 
                       default=['baseline-predictor', 'enhanced-predictor'],
                       help='Models to test (default: baseline-predictor enhanced-predictor)')
    
    args = parser.parse_args()
    
    success = test_model_inference(args.endpoint, args.models)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()