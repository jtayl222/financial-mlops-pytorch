#!/usr/bin/env python3
"""
A/B Testing Demonstration for Financial MLOps Platform

This script demonstrates traffic splitting between baseline and enhanced models
using Seldon Core v2 experiments for model comparison and performance analysis.
"""

import json
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict
import argparse

class ABTestingDemo:
    def __init__(self, seldon_endpoint="http://ml-api.local/financial-inference", experiment_name="financial-ab-test-experiment"):
        self.seldon_endpoint = seldon_endpoint
        self.experiment_name = experiment_name
        self.results = defaultdict(list)
        
    def generate_sample_features(self, n_samples=100):
        """Generate realistic financial features for testing"""
        np.random.seed(42)  # For reproducible results
        
        # Simulate stock price data
        timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='D')
        
        # Base price with trend and volatility
        base_price = 100
        trend = np.cumsum(np.random.normal(0.001, 0.02, n_samples))
        prices = base_price * (1 + trend)
        
        # Technical indicators
        sma_5 = pd.Series(prices).rolling(5, min_periods=1).mean()
        sma_10 = pd.Series(prices).rolling(10, min_periods=1).mean()
        sma_20 = pd.Series(prices).rolling(20, min_periods=1).mean()
        
        # RSI calculation (simplified)
        price_changes = np.diff(prices, prepend=prices[0])
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)
        avg_gain = pd.Series(gains).rolling(14, min_periods=1).mean()
        avg_loss = pd.Series(losses).rolling(14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Volume (random but realistic)
        volume = np.random.lognormal(mean=15, sigma=0.5, size=n_samples)
        
        features = []
        for i in range(n_samples):
            # Create sequence of 10 time steps (as expected by models)
            sequence_length = 10
            start_idx = max(0, i - sequence_length + 1)
            
            feature_vector = []
            for j in range(start_idx, i + 1):
                if j < len(prices):
                    # Normalize features (simple min-max scaling)
                    normalized_price = (prices[j] - np.min(prices)) / (np.max(prices) - np.min(prices))
                    normalized_volume = (volume[j] - np.min(volume)) / (np.max(volume) - np.min(volume))
                    normalized_sma5 = (sma_5.iloc[j] - np.min(sma_5)) / (np.max(sma_5) - np.min(sma_5))
                    normalized_sma10 = (sma_10.iloc[j] - np.min(sma_10)) / (np.max(sma_10) - np.min(sma_10))
                    normalized_sma20 = (sma_20.iloc[j] - np.min(sma_20)) / (np.max(sma_20) - np.min(sma_20))
                    normalized_rsi = rsi.iloc[j] / 100.0
                    
                    # Add more features to reach 35 total (as expected by models)
                    daily_features = [
                        normalized_price, normalized_volume, normalized_sma5, normalized_sma10, normalized_sma20, normalized_rsi,
                        # Additional features (price ratios, momentum indicators, etc.)
                        normalized_price / normalized_sma5, normalized_price / normalized_sma10, normalized_price / normalized_sma20,
                        normalized_sma5 / normalized_sma10, normalized_sma10 / normalized_sma20,
                        # Volatility proxies
                        abs(normalized_price - normalized_sma5), abs(normalized_price - normalized_sma10),
                        # Momentum indicators
                        normalized_price - np.roll(prices, 1)[j] if j > 0 else 0,
                        normalized_price - np.roll(prices, 5)[j] if j >= 5 else 0,
                        # More technical indicators (simplified)
                        *[np.random.normal(0.5, 0.1) for _ in range(20)]  # Placeholder features
                    ]
                    feature_vector.extend(daily_features[:35])  # Ensure exactly 35 features
            
            # Pad or trim to exactly sequence_length * 35 features
            target_length = sequence_length * 35
            if len(feature_vector) < target_length:
                # Pad with the last known values
                padding_needed = target_length - len(feature_vector)
                if feature_vector:
                    last_features = feature_vector[-35:] if len(feature_vector) >= 35 else feature_vector
                    padding = (last_features * (padding_needed // len(last_features) + 1))[:padding_needed]
                    feature_vector.extend(padding)
                else:
                    feature_vector = [0.5] * target_length
            elif len(feature_vector) > target_length:
                feature_vector = feature_vector[:target_length]
            
            features.append({
                'timestamp': timestamps[i].isoformat(),
                'features': feature_vector,
                'expected_price': prices[i]
            })
        
        return features
    
    def send_prediction_request(self, features, experiment=True):
        """Send prediction request to Seldon mesh"""
        # Convert features to the format expected by MLflow models
        input_data = {
            "inputs": [
                {
                    "name": "input_data",
                    "shape": [1, 10, 35],  # batch_size=1, sequence_length=10, features=35
                    "datatype": "FP32",
                    "data": np.array(features['features']).reshape(1, 10, 35).tolist()
                }
            ]
        }
        
        try:
            if experiment:
                # Use experiment endpoint for A/B testing
                url = f"{self.seldon_endpoint}/v2/models/{self.experiment_name}/infer"
            else:
                # Direct model endpoint
                url = f"{self.seldon_endpoint}/v2/models/baseline-predictor/infer"
            
            response = requests.post(
                url,
                json=input_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'prediction': result.get('outputs', [{}])[0].get('data', [0])[0],
                    'model_used': response.headers.get('X-Model-Name', 'unknown'),
                    'response_time': response.elapsed.total_seconds(),
                    'timestamp': features['timestamp']
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}",
                    'response_time': response.elapsed.total_seconds(),
                    'timestamp': features['timestamp']
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': 0,
                'timestamp': features['timestamp']
            }
    
    def run_ab_test(self, n_requests=100, delay=0.1):
        """Run A/B test with multiple prediction requests"""
        print(f"🚀 Starting A/B Testing Demonstration")
        print(f"   Target: {self.seldon_endpoint}")
        print(f"   Experiment: {self.experiment_name}")
        print(f"   Requests: {n_requests}")
        print(f"   Expected traffic split: 70% baseline, 30% enhanced")
        print()
        
        # Generate test data
        features_list = self.generate_sample_features(n_requests)
        
        results = []
        model_counts = defaultdict(int)
        success_count = 0
        
        print("📊 Sending prediction requests...")
        for i, features in enumerate(features_list):
            result = self.send_prediction_request(features, experiment=True)
            results.append(result)
            
            if result['success']:
                success_count += 1
                model_used = result.get('model_used', 'unknown')
                model_counts[model_used] += 1
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"   Completed {i + 1}/{n_requests} requests ({success_count} successful)")
            
            time.sleep(delay)
        
        print(f"\n✅ A/B Test Complete!")
        print(f"   Total requests: {len(results)}")
        print(f"   Successful requests: {success_count}")
        print(f"   Success rate: {success_count/len(results)*100:.1f}%")
        
        if model_counts:
            print(f"\n📈 Traffic Distribution:")
            total_successful = sum(model_counts.values())
            for model, count in model_counts.items():
                percentage = count / total_successful * 100
                print(f"   {model}: {count} requests ({percentage:.1f}%)")
        
        return results
    
    def analyze_results(self, results):
        """Analyze A/B test results and generate insights"""
        df = pd.DataFrame(results)
        
        if df.empty or df['success'].sum() == 0:
            print("❌ No successful predictions to analyze")
            return
        
        successful_df = df[df['success'] == True]
        
        print(f"\n📊 Performance Analysis:")
        print(f"   Average response time: {successful_df['response_time'].mean():.3f}s")
        print(f"   Min response time: {successful_df['response_time'].min():.3f}s")
        print(f"   Max response time: {successful_df['response_time'].max():.3f}s")
        
        if 'model_used' in successful_df.columns:
            model_performance = successful_df.groupby('model_used')['response_time'].agg(['mean', 'std', 'count'])
            print(f"\n🏆 Model Performance Comparison:")
            for model, stats in model_performance.iterrows():
                print(f"   {model}:")
                print(f"     Average response time: {stats['mean']:.3f}s (±{stats['std']:.3f}s)")
                print(f"     Request count: {stats['count']}")
        
        return successful_df
    
    def generate_visualizations(self, results_df):
        """Generate visualizations for the A/B test results"""
        if results_df.empty:
            print("❌ No data to visualize")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Financial MLOps A/B Testing Results', fontsize=16, fontweight='bold')
        
        # 1. Traffic Distribution Pie Chart
        if 'model_used' in results_df.columns:
            model_counts = results_df['model_used'].value_counts()
            axes[0, 0].pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('Traffic Distribution Between Models')
        
        # 2. Response Time Distribution
        axes[0, 1].hist(results_df['response_time'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Response Time Distribution')
        axes[0, 1].set_xlabel('Response Time (seconds)')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Response Time by Model (if multiple models)
        if 'model_used' in results_df.columns and len(results_df['model_used'].unique()) > 1:
            for model in results_df['model_used'].unique():
                model_data = results_df[results_df['model_used'] == model]['response_time']
                axes[1, 0].hist(model_data, alpha=0.7, label=model, bins=15)
            axes[1, 0].set_title('Response Time by Model')
            axes[1, 0].set_xlabel('Response Time (seconds)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
        
        # 4. Predictions Over Time
        if 'prediction' in results_df.columns:
            axes[1, 1].plot(range(len(results_df)), results_df['prediction'], 'o-', alpha=0.7)
            axes[1, 1].set_title('Model Predictions Over Time')
            axes[1, 1].set_xlabel('Request Number')
            axes[1, 1].set_ylabel('Predicted Value')
        
        plt.tight_layout()
        
        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ab_test_results_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📈 Visualization saved as: {filename}")
        
        return filename

def main():
    parser = argparse.ArgumentParser(description='Financial MLOps A/B Testing Demonstration')
    parser.add_argument('--endpoint', default='http://ml-api.local/financial-inference', help='Seldon mesh endpoint (via NGINX ingress)')
    parser.add_argument('--experiment', default='financial-ab-test-experiment', help='Experiment name')
    parser.add_argument('--requests', type=int, default=50, help='Number of test requests')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay between requests (seconds)')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = ABTestingDemo(args.endpoint, args.experiment)
    
    # Run A/B test
    results = demo.run_ab_test(args.requests, args.delay)
    
    # Analyze results
    results_df = demo.analyze_results(results)
    
    # Generate visualizations (if requested and data available)
    if not args.no_viz and results_df is not None and not results_df.empty:
        demo.generate_visualizations(results_df)
    
    print(f"\n🎯 A/B Testing Demonstration Complete!")
    print(f"   This demonstrates how Seldon Core v2 enables sophisticated")
    print(f"   model experimentation with traffic splitting and performance monitoring.")

if __name__ == "__main__":
    main()