#!/usr/bin/env python3
"""
Advanced A/B Testing Demonstration for Financial MLOps Platform

This script demonstrates sophisticated model comparison, performance analysis,
and business impact measurement using Seldon Core v2 experiments.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Prometheus metrics integration
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("⚠️  Prometheus client not available - metrics will be simulated")

@dataclass
class ModelMetrics:
    """Container for model performance metrics"""
    name: str
    request_count: int = 0
    total_response_time: float = 0.0
    predictions: List[float] = None
    errors: int = 0
    accuracy_score: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    
    def __post_init__(self):
        if self.predictions is None:
            self.predictions = []
    
    @property
    def avg_response_time(self) -> float:
        return self.total_response_time / max(self.request_count, 1)
    
    @property
    def success_rate(self) -> float:
        return (self.request_count - self.errors) / max(self.request_count, 1) * 100

class AdvancedABTester:
    def __init__(self, seldon_endpoint: str, experiment_name: str = "financial-ab-test-experiment", 
                 enable_metrics: bool = True, metrics_port: int = 8002):
        self.seldon_endpoint = seldon_endpoint
        self.experiment_name = experiment_name
        self.results = []
        self.model_metrics = {
            'baseline-predictor': ModelMetrics('baseline-predictor'),
            'enhanced-predictor': ModelMetrics('enhanced-predictor')
        }
        self.lock = threading.Lock()
        
        # Initialize Prometheus metrics if available
        self.enable_metrics = enable_metrics and PROMETHEUS_AVAILABLE
        if self.enable_metrics:
            self.setup_prometheus_metrics(metrics_port)
    
    def setup_prometheus_metrics(self, port: int):
        """Setup Prometheus metrics collection"""
        try:
            # Start HTTP server for metrics
            start_http_server(port)
            print(f"📊 Prometheus metrics server started on port {port}")
            
            # Define metrics
            self.request_counter = Counter(
                'ab_test_requests_total',
                'Total requests processed',
                ['model_name', 'experiment', 'status']
            )
            
            self.response_time_histogram = Histogram(
                'ab_test_response_time_seconds',
                'Response time distribution',
                ['model_name', 'experiment'],
                buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
            )
            
            self.accuracy_gauge = Gauge(
                'ab_test_model_accuracy',
                'Model accuracy percentage',
                ['model_name', 'experiment']
            )
            
            self.traffic_distribution_gauge = Gauge(
                'ab_test_traffic_percentage',
                'Traffic distribution percentage',
                ['model_name', 'experiment']
            )
            
            self.prediction_value_histogram = Histogram(
                'ab_test_prediction_value',
                'Distribution of prediction values',
                ['model_name', 'experiment'],
                buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            )
            
            self.business_impact_gauge = Gauge(
                'ab_test_business_impact',
                'Business impact metrics',
                ['model_name', 'experiment', 'metric_type']
            )
            
            print("✅ Prometheus metrics initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to setup Prometheus metrics: {e}")
            self.enable_metrics = False
    
    def update_prometheus_metrics(self, result: Dict):
        """Update Prometheus metrics with result data"""
        if not self.enable_metrics:
            return
            
        try:
            model_name = result.get('model_used', 'unknown')
            status = 'success' if result.get('success', False) else 'error'
            
            # Update counters
            self.request_counter.labels(
                model_name=model_name,
                experiment=self.experiment_name,
                status=status
            ).inc()
            
            if result.get('success', False):
                # Update response time
                self.response_time_histogram.labels(
                    model_name=model_name,
                    experiment=self.experiment_name
                ).observe(result.get('response_time', 0))
                
                # Update prediction value
                if 'prediction' in result:
                    self.prediction_value_histogram.labels(
                        model_name=model_name,
                        experiment=self.experiment_name
                    ).observe(result['prediction'])
                    
        except Exception as e:
            print(f"❌ Error updating Prometheus metrics: {e}")
    
    def update_aggregate_metrics(self, analysis: Dict):
        """Update aggregate metrics in Prometheus"""
        if not self.enable_metrics:
            return
            
        try:
            total_requests = analysis.get('successful_requests', 0)
            
            for model_name, model_data in analysis.get('models', {}).items():
                # Update accuracy gauge
                self.accuracy_gauge.labels(
                    model_name=model_name,
                    experiment=self.experiment_name
                ).set(model_data.get('avg_accuracy', 0))
                
                # Update traffic distribution
                self.traffic_distribution_gauge.labels(
                    model_name=model_name,
                    experiment=self.experiment_name
                ).set(model_data.get('traffic_percentage', 0))
                
        except Exception as e:
            print(f"❌ Error updating aggregate metrics: {e}")
    
    def update_business_impact_metrics(self, business_impact: Dict):
        """Update business impact metrics in Prometheus"""
        if not self.enable_metrics:
            return
            
        try:
            # Simulate business impact for both models
            for model_name in ['baseline-predictor', 'enhanced-predictor']:
                revenue_impact = business_impact.get('potential_revenue_lift', 0)
                if model_name == 'baseline-predictor':
                    revenue_impact = 0  # Baseline is the reference
                
                self.business_impact_gauge.labels(
                    model_name=model_name,
                    experiment=self.experiment_name,
                    metric_type='revenue_impact'
                ).set(revenue_impact)
                
                latency_cost = business_impact.get('latency_difference', 0) * 100
                if model_name == 'baseline-predictor':
                    latency_cost = 0  # Baseline is the reference
                
                self.business_impact_gauge.labels(
                    model_name=model_name,
                    experiment=self.experiment_name,
                    metric_type='latency_cost'
                ).set(latency_cost)
                
                net_value = revenue_impact - latency_cost
                self.business_impact_gauge.labels(
                    model_name=model_name,
                    experiment=self.experiment_name,
                    metric_type='net_business_value'
                ).set(net_value)
                
        except Exception as e:
            print(f"❌ Error updating business impact metrics: {e}")
        
    def generate_realistic_market_data(self, n_samples: int = 200) -> List[Dict]:
        """Generate realistic financial market scenarios for testing"""
        np.random.seed(42)
        
        scenarios = [
            {"name": "Bull Market", "trend": 0.003, "volatility": 0.015, "weight": 0.3},
            {"name": "Bear Market", "trend": -0.002, "volatility": 0.025, "weight": 0.2},
            {"name": "Sideways Market", "trend": 0.0001, "volatility": 0.012, "weight": 0.3},
            {"name": "High Volatility", "trend": 0.001, "volatility": 0.040, "weight": 0.2}
        ]
        
        market_data = []
        current_price = 100.0
        
        for i in range(n_samples):
            # Choose market scenario
            scenario = np.random.choice(scenarios, p=[s["weight"] for s in scenarios])
            
            # Generate price movement
            return_pct = np.random.normal(scenario["trend"], scenario["volatility"])
            current_price *= (1 + return_pct)
            
            # Generate volume with realistic patterns
            base_volume = 1000000
            volume = np.random.lognormal(np.log(base_volume), 0.3)
            
            # Technical indicators
            lookback_prices = [current_price * (1 + np.random.normal(0, 0.01)) for _ in range(20)]
            sma_5 = np.mean(lookback_prices[-5:])
            sma_10 = np.mean(lookback_prices[-10:])
            sma_20 = np.mean(lookback_prices)
            
            # RSI calculation (simplified)
            price_changes = np.diff(lookback_prices)
            gains = np.where(price_changes > 0, price_changes, 0)
            losses = np.where(price_changes < 0, -price_changes, 0)
            rsi = 50 + np.random.normal(0, 15)  # Simplified RSI
            rsi = np.clip(rsi, 0, 100)
            
            # Create feature vector (35 features as expected by models)
            features = [
                current_price / 100.0,  # Normalized price
                volume / base_volume,   # Normalized volume
                sma_5 / current_price,  # SMA ratios
                sma_10 / current_price,
                sma_20 / current_price,
                rsi / 100.0,           # Normalized RSI
                return_pct + 0.5,      # Shifted returns
                abs(return_pct) * 10,  # Volatility proxy
            ]
            
            # Add more technical features to reach 35
            features.extend([
                # Moving average crossovers
                1 if sma_5 > sma_10 else 0,
                1 if sma_10 > sma_20 else 0,
                # Price momentum
                (current_price - sma_5) / sma_5,
                (current_price - sma_10) / sma_10,
                (current_price - sma_20) / sma_20,
                # Volume indicators
                volume / (base_volume * 2),
                # Market regime indicators
                1 if scenario["name"] == "Bull Market" else 0,
                1 if scenario["name"] == "Bear Market" else 0,
                1 if scenario["name"] == "High Volatility" else 0,
                # Additional features (padding to 35)
                *[np.random.normal(0.5, 0.1) for _ in range(16)]
            ])
            
            # Ensure exactly 35 features
            features = features[:35]
            while len(features) < 35:
                features.append(0.5)
            
            # Create sequence data (10 timesteps)
            sequence_data = []
            for t in range(10):
                # Add some temporal variation
                temporal_features = [f + np.random.normal(0, 0.02) for f in features]
                sequence_data.extend(temporal_features)
            
            market_data.append({
                'timestamp': datetime.now() + timedelta(minutes=i),
                'scenario': scenario["name"],
                'current_price': current_price,
                'expected_direction': 1 if return_pct > 0 else 0,
                'expected_magnitude': abs(return_pct),
                'features': sequence_data,
                'metadata': {
                    'volume': volume,
                    'rsi': rsi,
                    'sma_5': sma_5,
                    'sma_10': sma_10,
                    'sma_20': sma_20
                }
            })
        
        return market_data
    
    def send_prediction_request(self, market_data: Dict, use_experiment: bool = True) -> Dict:
        """Send prediction request with detailed error handling"""
        try:
            # Prepare input data
            input_data = {
                "inputs": [
                    {
                        "name": "input_data",
                        "shape": [1, 10, 35],
                        "datatype": "FP32",
                        "data": np.array(market_data['features']).reshape(1, 10, 35).tolist()
                    }
                ]
            }
            
            # Choose endpoint
            if use_experiment:
                url = f"{self.seldon_endpoint}/v2/models/{self.experiment_name}/infer"
                headers = {
                    "Content-Type": "application/json",
                    "Host": "financial-predictor.local"  # Required for Istio routing
                }
            else:
                url = f"{self.seldon_endpoint}/v2/models/baseline-predictor/infer"
                headers = {"Content-Type": "application/json"}
            
            start_time = time.time()
            response = requests.post(url, json=input_data, headers=headers, timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('outputs', [{}])[0].get('data', [0])[0]
                model_used = response.headers.get('X-Model-Name', 'unknown')
                
                # Calculate prediction accuracy (simulated)
                expected = market_data['expected_direction']
                predicted_direction = 1 if prediction > 0.5 else 0
                accuracy = 1 if predicted_direction == expected else 0
                
                return {
                    'success': True,
                    'prediction': prediction,
                    'model_used': model_used,
                    'response_time': response_time,
                    'accuracy': accuracy,
                    'timestamp': market_data['timestamp'],
                    'scenario': market_data['scenario'],
                    'expected_price': market_data['current_price'],
                    'metadata': market_data['metadata']
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text[:200]}",
                    'response_time': response_time,
                    'timestamp': market_data['timestamp'],
                    'scenario': market_data['scenario']
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': 0,
                'timestamp': market_data['timestamp'],
                'scenario': market_data['scenario']
            }
    
    def run_concurrent_test(self, market_data: List[Dict], max_workers: int = 5) -> List[Dict]:
        """Run A/B test with concurrent requests for realistic load"""
        print(f"🚀 Starting Advanced A/B Testing with {len(market_data)} scenarios")
        print(f"   Concurrent workers: {max_workers}")
        print(f"   Target endpoint: {self.seldon_endpoint}")
        print(f"   Experiment: {self.experiment_name}")
        print()
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all requests
            future_to_data = {
                executor.submit(self.send_prediction_request, data): data 
                for data in market_data
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_data):
                result = future.result()
                results.append(result)
                completed += 1
                
                # Update Prometheus metrics for each result
                self.update_prometheus_metrics(result)
                
                if completed % 20 == 0:
                    success_count = sum(1 for r in results if r.get('success', False))
                    print(f"   Progress: {completed}/{len(market_data)} ({success_count} successful)")
        
        return results
    
    def analyze_performance(self, results: List[Dict]) -> Dict:
        """Comprehensive performance analysis"""
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            print("❌ No successful predictions to analyze")
            return {}
        
        # Group by model
        model_results = defaultdict(list)
        for result in successful_results:
            model_name = result.get('model_used', 'unknown')
            model_results[model_name].append(result)
        
        analysis = {
            'total_requests': len(results),
            'successful_requests': len(successful_results),
            'success_rate': len(successful_results) / len(results) * 100,
            'models': {}
        }
        
        print(f"📊 Performance Analysis Results:")
        print(f"   Total requests: {analysis['total_requests']}")
        print(f"   Successful requests: {analysis['successful_requests']}")
        print(f"   Overall success rate: {analysis['success_rate']:.1f}%")
        print()
        
        for model_name, model_data in model_results.items():
            response_times = [r['response_time'] for r in model_data]
            accuracies = [r.get('accuracy', 0) for r in model_data]
            predictions = [r['prediction'] for r in model_data]
            
            model_analysis = {
                'request_count': len(model_data),
                'traffic_percentage': len(model_data) / len(successful_results) * 100,
                'avg_response_time': np.mean(response_times),
                'p95_response_time': np.percentile(response_times, 95),
                'p99_response_time': np.percentile(response_times, 99),
                'avg_accuracy': np.mean(accuracies) * 100,
                'avg_prediction': np.mean(predictions),
                'prediction_std': np.std(predictions),
                'scenarios': defaultdict(int)
            }
            
            # Scenario breakdown
            for result in model_data:
                model_analysis['scenarios'][result['scenario']] += 1
            
            analysis['models'][model_name] = model_analysis
            
            print(f"🏆 {model_name} Performance:")
            print(f"   Requests: {model_analysis['request_count']} ({model_analysis['traffic_percentage']:.1f}% of traffic)")
            print(f"   Avg Response Time: {model_analysis['avg_response_time']:.3f}s")
            print(f"   P95 Response Time: {model_analysis['p95_response_time']:.3f}s")
            print(f"   Avg Accuracy: {model_analysis['avg_accuracy']:.1f}%")
            print(f"   Prediction Range: {model_analysis['avg_prediction']:.3f} ± {model_analysis['prediction_std']:.3f}")
            print()
        
        # Update aggregate Prometheus metrics
        self.update_aggregate_metrics(analysis)
        
        return analysis
    
    def generate_business_impact_report(self, results: List[Dict], analysis: Dict) -> Dict:
        """Generate business impact analysis"""
        print("💼 Business Impact Analysis:")
        
        # Simulated business metrics
        baseline_accuracy = analysis['models'].get('baseline-predictor', {}).get('avg_accuracy', 50)
        enhanced_accuracy = analysis['models'].get('enhanced-predictor', {}).get('avg_accuracy', 50)
        
        # Calculate potential revenue impact
        accuracy_improvement = enhanced_accuracy - baseline_accuracy
        potential_revenue_lift = accuracy_improvement * 0.02  # 2% revenue per 1% accuracy improvement
        
        baseline_latency = analysis['models'].get('baseline-predictor', {}).get('avg_response_time', 1.0)
        enhanced_latency = analysis['models'].get('enhanced-predictor', {}).get('avg_response_time', 1.0)
        
        business_impact = {
            'accuracy_improvement': accuracy_improvement,
            'potential_revenue_lift': potential_revenue_lift,
            'latency_difference': enhanced_latency - baseline_latency,
            'recommendation': ''
        }
        
        # Generate recommendation
        if accuracy_improvement > 2 and abs(business_impact['latency_difference']) < 0.1:
            business_impact['recommendation'] = "✅ RECOMMEND: Deploy enhanced model - significant accuracy improvement with minimal latency impact"
        elif accuracy_improvement > 5:
            business_impact['recommendation'] = "✅ STRONG RECOMMEND: Deploy enhanced model - substantial accuracy improvement"
        elif business_impact['latency_difference'] > 0.2:
            business_impact['recommendation'] = "⚠️  CAUTION: Enhanced model slower - evaluate if accuracy gain justifies latency cost"
        else:
            business_impact['recommendation'] = "📊 CONTINUE TESTING: Marginal differences - need more data for decision"
        
        print(f"   Accuracy Improvement: {accuracy_improvement:.1f} percentage points")
        print(f"   Potential Revenue Lift: {potential_revenue_lift:.1f}%")
        print(f"   Latency Impact: {business_impact['latency_difference']:.3f}s")
        print(f"   Recommendation: {business_impact['recommendation']}")
        print()
        
        # Update business impact Prometheus metrics
        self.update_business_impact_metrics(business_impact)
        
        return business_impact
    
    def create_comprehensive_visualizations(self, results: List[Dict], analysis: Dict) -> str:
        """Create comprehensive A/B testing visualizations"""
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            print("❌ No data to visualize")
            return ""
        
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # Create a 3x3 grid of subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Traffic Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        model_counts = {}
        for result in successful_results:
            model = result.get('model_used', 'unknown')
            model_counts[model] = model_counts.get(model, 0) + 1
        
        if model_counts:
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            wedges, texts, autotexts = ax1.pie(model_counts.values(), labels=model_counts.keys(), 
                                              autopct='%1.1f%%', startangle=90, colors=colors)
            ax1.set_title('Traffic Distribution', fontsize=14, fontweight='bold')
        
        # 2. Response Time Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        model_response_times = defaultdict(list)
        for result in successful_results:
            model_response_times[result.get('model_used', 'unknown')].append(result['response_time'])
        
        if len(model_response_times) > 1:
            ax2.boxplot(model_response_times.values(), labels=model_response_times.keys())
            ax2.set_title('Response Time Distribution', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Response Time (seconds)')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Accuracy by Model
        ax3 = fig.add_subplot(gs[0, 2])
        model_accuracies = defaultdict(list)
        for result in successful_results:
            if 'accuracy' in result:
                model_accuracies[result.get('model_used', 'unknown')].append(result['accuracy'] * 100)
        
        if model_accuracies:
            models = list(model_accuracies.keys())
            accuracy_means = [np.mean(model_accuracies[model]) for model in models]
            accuracy_stds = [np.std(model_accuracies[model]) for model in models]
            
            bars = ax3.bar(models, accuracy_means, yerr=accuracy_stds, capsize=5, 
                          color=['#FF6B6B', '#4ECDC4'])
            ax3.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Accuracy (%)')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, mean in zip(bars, accuracy_means):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{mean:.1f}%', ha='center', va='bottom')
        
        # 4. Predictions Over Time
        ax4 = fig.add_subplot(gs[1, :])
        timestamps = [r['timestamp'] for r in successful_results]
        predictions = [r['prediction'] for r in successful_results]
        models = [r.get('model_used', 'unknown') for r in successful_results]
        
        # Group by model for different colors
        model_data = defaultdict(lambda: {'timestamps': [], 'predictions': []})
        for ts, pred, model in zip(timestamps, predictions, models):
            model_data[model]['timestamps'].append(ts)
            model_data[model]['predictions'].append(pred)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for i, (model, data) in enumerate(model_data.items()):
            ax4.scatter(data['timestamps'], data['predictions'], 
                       label=model, alpha=0.7, s=30, color=colors[i % len(colors)])
        
        ax4.set_title('Model Predictions Over Time', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Prediction Value')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Scenario Performance
        ax5 = fig.add_subplot(gs[2, 0])
        scenario_counts = defaultdict(int)
        for result in successful_results:
            scenario_counts[result['scenario']] += 1
        
        if scenario_counts:
            ax5.bar(scenario_counts.keys(), scenario_counts.values(), 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            ax5.set_title('Requests by Market Scenario', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Number of Requests')
            ax5.tick_params(axis='x', rotation=45)
        
        # 6. Model Performance Heatmap
        ax6 = fig.add_subplot(gs[2, 1:])
        
        # Create performance matrix
        metrics = ['Accuracy', 'Speed', 'Prediction Variance']
        models_list = list(analysis.get('models', {}).keys())
        
        if models_list:
            performance_matrix = []
            for model in models_list:
                model_data = analysis['models'][model]
                # Normalize metrics for comparison
                accuracy_norm = model_data.get('avg_accuracy', 50) / 100
                speed_norm = 1 / (model_data.get('avg_response_time', 1) + 0.1)  # Invert for better=higher
                variance_norm = 1 / (model_data.get('prediction_std', 0.1) + 0.1)  # Invert for better=higher
                
                performance_matrix.append([accuracy_norm, speed_norm, variance_norm])
            
            im = ax6.imshow(performance_matrix, cmap='RdYlGn', aspect='auto')
            ax6.set_xticks(range(len(metrics)))
            ax6.set_xticklabels(metrics)
            ax6.set_yticks(range(len(models_list)))
            ax6.set_yticklabels(models_list)
            ax6.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold')
            
            # Add text annotations
            for i in range(len(models_list)):
                for j in range(len(metrics)):
                    text = ax6.text(j, i, f'{performance_matrix[i][j]:.2f}',
                                   ha="center", va="center", color="black", fontweight='bold')
            
            plt.colorbar(im, ax=ax6, label='Performance Score (Higher = Better)')
        
        # Overall title
        fig.suptitle('Financial MLOps A/B Testing Comprehensive Analysis', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"advanced_ab_test_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📈 Comprehensive analysis saved as: {filename}")
        
        return filename

def main():
    parser = argparse.ArgumentParser(description='Advanced Financial MLOps A/B Testing')
    parser.add_argument('--endpoint', default='http://ml-api.local/financial-inference', 
                       help='Seldon mesh endpoint (via NGINX ingress)')
    parser.add_argument('--experiment', default='financial-ab-test-experiment', 
                       help='Experiment name')
    parser.add_argument('--scenarios', type=int, default=100, 
                       help='Number of market scenarios to test')
    parser.add_argument('--workers', type=int, default=3, 
                       help='Concurrent workers')
    parser.add_argument('--no-viz', action='store_true', 
                       help='Skip visualization generation')
    parser.add_argument('--metrics-port', type=int, default=8002,
                       help='Port for Prometheus metrics server')
    parser.add_argument('--no-metrics', action='store_true',
                       help='Disable Prometheus metrics collection')
    
    args = parser.parse_args()
    
    print("🎯 Advanced Financial MLOps A/B Testing Demonstration")
    print("="*60)
    
    # Initialize tester
    tester = AdvancedABTester(args.endpoint, args.experiment, 
                             enable_metrics=not args.no_metrics,
                             metrics_port=args.metrics_port)
    
    # Generate realistic market scenarios
    print("📊 Generating realistic market scenarios...")
    market_data = tester.generate_realistic_market_data(args.scenarios)
    print(f"   Created {len(market_data)} diverse market scenarios")
    
    # Run concurrent A/B test
    results = tester.run_concurrent_test(market_data, args.workers)
    
    # Analyze performance
    analysis = tester.analyze_performance(results)
    
    # Generate business impact report
    if analysis:
        business_impact = tester.generate_business_impact_report(results, analysis)
        
        # Create visualizations
        if not args.no_viz:
            tester.create_comprehensive_visualizations(results, analysis)
    
    print("🎉 Advanced A/B Testing Complete!")
    print("\n💡 Key Insights:")
    print("   • This demonstrates enterprise-grade A/B testing capabilities")
    print("   • Real-time performance monitoring and business impact analysis")
    print("   • Automated decision making for model deployment")
    print("   • Comprehensive visualization for stakeholder reporting")

if __name__ == "__main__":
    main()