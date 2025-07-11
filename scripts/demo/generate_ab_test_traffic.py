#!/usr/bin/env python3
"""
Generate live traffic to Seldon A/B test experiment.
Sends real requests and generates actual Prometheus metrics.
"""

import json
import time
import random
import requests
import numpy as np
from datetime import datetime
import argparse
import sys

def generate_sample_data(batch_size=1):
    """Generate realistic financial time series data for prediction.
    Model expects shape [-1, 10, 35] = [batch, sequence_length, features]
    """
    # Create realistic financial features for the expected shape
    data = []
    for _ in range(batch_size):
        # Generate sequence of 10 time steps with 35 features each
        sequence_data = []
        for step in range(10):
            # Generate 35 financial features per time step
            # Features: price, volume, technical indicators, ratios, etc.
            step_features = [
                np.random.normal(100, 5),      # price
                np.random.normal(1.5, 0.3),   # volume (normalized)
                np.random.normal(0, 0.1),     # return
                np.random.normal(0.5, 0.2),   # volatility
                np.random.normal(0, 0.05),    # price change
            ]
            
            # Add 30 more technical indicator features
            for _ in range(30):
                step_features.append(np.random.normal(0, 1))
            
            sequence_data.append(step_features)
        
        data.append(sequence_data)
    
    return data

def send_prediction_request(endpoint, data, headers=None):
    """Send prediction request to Seldon endpoint."""
    # Flatten the sequence data for the model
    flattened_data = []
    for batch in data:
        # Flatten each sequence: 10 timesteps * 35 features = 350 features
        flat_sequence = []
        for timestep in batch:
            flat_sequence.extend(timestep)
        flattened_data.append(flat_sequence)
    
    payload = {
        "inputs": [
            {
                "name": "input-0",
                "shape": [len(flattened_data), len(flattened_data[0])],
                "datatype": "FP32",
                "data": flattened_data
            }
        ]
    }
    
    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers=headers or {"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        return response.json(), response.status_code, None
    except requests.exceptions.RequestException as e:
        return None, getattr(e.response, 'status_code', 0), str(e)

def main():
    parser = argparse.ArgumentParser(description='Generate A/B test traffic')
    parser.add_argument('--endpoint', required=True, help='Seldon experiment endpoint URL')
    parser.add_argument('--requests', type=int, default=100, help='Number of requests to send')
    parser.add_argument('--rate', type=int, default=5, help='Requests per second')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for each request')
    parser.add_argument('--duration', type=int, help='Run for specified seconds (overrides --requests)')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting traffic generation to {args.endpoint}")
    print(f"📊 Configuration: {args.requests} requests, {args.rate} req/s, batch size {args.batch_size}")
    
    # Statistics tracking
    stats = {
        'total_requests': 0,
        'successful_requests': 0,
        'failed_requests': 0,
        'baseline_predictions': 0,
        'enhanced_predictions': 0,
        'response_times': [],
        'errors': []
    }
    
    start_time = time.time()
    interval = 1.0 / args.rate
    
    try:
        request_count = 0
        while True:
            # Check exit conditions
            if args.duration:
                if time.time() - start_time >= args.duration:
                    break
            else:
                if request_count >= args.requests:
                    break
            
            # Generate sample data
            data = generate_sample_data(args.batch_size)
            
            # Send request
            request_start = time.time()
            result, status_code, error = send_prediction_request(args.endpoint, data)
            response_time = time.time() - request_start
            
            stats['total_requests'] += 1
            stats['response_times'].append(response_time)
            
            if result and status_code == 200:
                stats['successful_requests'] += 1
                
                # Try to detect which model was used based on response
                # This is heuristic - different models might have different response patterns
                model_info = result.get('model_name', '')
                if 'baseline' in model_info.lower():
                    stats['baseline_predictions'] += 1
                elif 'enhanced' in model_info.lower():
                    stats['enhanced_predictions'] += 1
                
                print(f"✅ Request {request_count + 1}: {status_code} ({response_time:.3f}s) - {model_info}")
            else:
                stats['failed_requests'] += 1
                stats['errors'].append(f"{status_code}: {error}")
                print(f"❌ Request {request_count + 1}: {status_code} ({response_time:.3f}s) - {error}")
            
            request_count += 1
            
            # Rate limiting
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Traffic generation interrupted by user")
    
    # Print final statistics
    duration = time.time() - start_time
    print(f"\n📈 Traffic Generation Complete")
    print(f"⏱️  Duration: {duration:.1f} seconds")
    print(f"📊 Total Requests: {stats['total_requests']}")
    print(f"✅ Successful: {stats['successful_requests']} ({stats['successful_requests']/stats['total_requests']*100:.1f}%)")
    print(f"❌ Failed: {stats['failed_requests']} ({stats['failed_requests']/stats['total_requests']*100:.1f}%)")
    
    if stats['response_times']:
        avg_response = np.mean(stats['response_times'])
        p95_response = np.percentile(stats['response_times'], 95)
        print(f"⚡ Avg Response Time: {avg_response:.3f}s")
        print(f"📊 95th Percentile: {p95_response:.3f}s")
    
    if stats['baseline_predictions'] or stats['enhanced_predictions']:
        total_predictions = stats['baseline_predictions'] + stats['enhanced_predictions']
        baseline_pct = stats['baseline_predictions'] / total_predictions * 100
        enhanced_pct = stats['enhanced_predictions'] / total_predictions * 100
        print(f"🎯 A/B Split - Baseline: {stats['baseline_predictions']} ({baseline_pct:.1f}%), Enhanced: {stats['enhanced_predictions']} ({enhanced_pct:.1f}%)")
    
    if stats['errors']:
        print(f"⚠️  Sample Errors:")
        for error in stats['errors'][:5]:  # Show first 5 errors
            print(f"   {error}")
    
    return 0 if stats['failed_requests'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())