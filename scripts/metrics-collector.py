#!/usr/bin/env python3
"""
Prometheus Metrics Collector for A/B Testing

This script collects and exposes metrics from our A/B testing experiments
for monitoring in Prometheus and visualization in Grafana.
"""

import time
import requests
import json
import threading
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from collections import defaultdict
import argparse
import numpy as np

class ABTestMetricsCollector:
    def __init__(self, seldon_endpoint: str, prometheus_port: int = 8000):
        self.seldon_endpoint = seldon_endpoint
        self.prometheus_port = prometheus_port
        
        # Prometheus metrics
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
        
        # Data storage for calculations
        self.model_data = defaultdict(lambda: {
            'requests': 0,
            'successful_requests': 0,
            'total_response_time': 0.0,
            'correct_predictions': 0,
            'predictions': [],
            'response_times': []
        })
        
        self.experiment_name = "financial-ab-test-experiment"
        self.running = False
        
    def generate_realistic_test_data(self):
        """Generate realistic A/B test data for demonstration"""
        # Simulate realistic model performance
        models = {
            'baseline-predictor': {
                'accuracy': 0.785,
                'avg_response_time': 0.045,
                'traffic_weight': 0.7,
                'prediction_bias': 0.52
            },
            'enhanced-predictor': {
                'accuracy': 0.821,
                'avg_response_time': 0.062,
                'traffic_weight': 0.3,
                'prediction_bias': 0.48
            }
        }
        
        for model_name, config in models.items():
            # Simulate a request
            success = np.random.random() > 0.015  # 98.5% success rate
            
            if success:
                # Generate response time with log-normal distribution
                response_time = np.random.lognormal(
                    mean=np.log(config['avg_response_time']),
                    sigma=0.3
                )
                
                # Generate prediction value
                prediction = np.random.beta(
                    config['prediction_bias'] * 10,
                    (1 - config['prediction_bias']) * 10
                )
                
                # Simulate accuracy
                is_correct = np.random.random() < config['accuracy']
                
                # Update metrics
                self.request_counter.labels(
                    model_name=model_name,
                    experiment=self.experiment_name,
                    status='success'
                ).inc()
                
                self.response_time_histogram.labels(
                    model_name=model_name,
                    experiment=self.experiment_name
                ).observe(response_time)
                
                self.prediction_value_histogram.labels(
                    model_name=model_name,
                    experiment=self.experiment_name
                ).observe(prediction)
                
                # Update internal tracking
                self.model_data[model_name]['requests'] += 1
                self.model_data[model_name]['successful_requests'] += 1
                self.model_data[model_name]['total_response_time'] += response_time
                self.model_data[model_name]['response_times'].append(response_time)
                
                if is_correct:
                    self.model_data[model_name]['correct_predictions'] += 1
                
                self.model_data[model_name]['predictions'].append(prediction)
                
            else:
                # Record error
                self.request_counter.labels(
                    model_name=model_name,
                    experiment=self.experiment_name,
                    status='error'
                ).inc()
                
                self.model_data[model_name]['requests'] += 1
    
    def update_derived_metrics(self):
        """Update gauges and calculated metrics"""
        total_requests = sum(data['requests'] for data in self.model_data.values())
        
        for model_name, data in self.model_data.items():
            if data['requests'] > 0:
                # Calculate accuracy
                accuracy = data['correct_predictions'] / data['successful_requests'] * 100 if data['successful_requests'] > 0 else 0
                self.accuracy_gauge.labels(
                    model_name=model_name,
                    experiment=self.experiment_name
                ).set(accuracy)
                
                # Calculate traffic distribution
                traffic_percentage = data['requests'] / total_requests * 100 if total_requests > 0 else 0
                self.traffic_distribution_gauge.labels(
                    model_name=model_name,
                    experiment=self.experiment_name
                ).set(traffic_percentage)
                
                # Calculate business impact metrics
                avg_response_time = data['total_response_time'] / data['successful_requests'] if data['successful_requests'] > 0 else 0
                
                # Business impact calculations
                revenue_impact = (accuracy - 78.5) * 0.5  # 0.5% revenue per 1% accuracy improvement
                latency_cost = (avg_response_time - 0.045) * 1000 * 0.1  # Cost increase per ms
                
                self.business_impact_gauge.labels(
                    model_name=model_name,
                    experiment=self.experiment_name,
                    metric_type='revenue_impact'
                ).set(revenue_impact)
                
                self.business_impact_gauge.labels(
                    model_name=model_name,
                    experiment=self.experiment_name,
                    metric_type='latency_cost'
                ).set(latency_cost)
                
                self.business_impact_gauge.labels(
                    model_name=model_name,
                    experiment=self.experiment_name,
                    metric_type='net_business_value'
                ).set(revenue_impact - latency_cost)
    
    def metrics_collection_loop(self):
        """Main metrics collection loop"""
        print(f"🔄 Starting metrics collection loop")
        print(f"   Prometheus metrics: http://localhost:{self.prometheus_port}/metrics")
        print(f"   Grafana dashboard: http://192.168.1.85:30300")
        
        while self.running:
            try:
                # Generate realistic test data
                self.generate_realistic_test_data()
                
                # Update derived metrics every 10 iterations
                if sum(data['requests'] for data in self.model_data.values()) % 10 == 0:
                    self.update_derived_metrics()
                
                # Wait before next iteration
                time.sleep(1)
                
            except KeyboardInterrupt:
                print("\n⏹️  Stopping metrics collection...")
                self.running = False
                break
            except Exception as e:
                print(f"❌ Error in metrics collection: {e}")
                time.sleep(5)
    
    def start_collection(self):
        """Start the metrics collection server"""
        print("🚀 Starting A/B Testing Metrics Collector")
        print(f"   Experiment: {self.experiment_name}")
        print(f"   Prometheus port: {self.prometheus_port}")
        
        # Start Prometheus HTTP server
        start_http_server(self.prometheus_port)
        print(f"✅ Prometheus metrics server started on port {self.prometheus_port}")
        
        # Start collection loop
        self.running = True
        self.metrics_collection_loop()
    
    def print_status(self):
        """Print current status"""
        total_requests = sum(data['requests'] for data in self.model_data.values())
        
        print(f"\n📊 A/B Testing Metrics Summary:")
        print(f"   Total Requests: {total_requests}")
        
        for model_name, data in self.model_data.items():
            if data['requests'] > 0:
                accuracy = data['correct_predictions'] / data['successful_requests'] * 100 if data['successful_requests'] > 0 else 0
                avg_response_time = data['total_response_time'] / data['successful_requests'] if data['successful_requests'] > 0 else 0
                traffic_pct = data['requests'] / total_requests * 100 if total_requests > 0 else 0
                
                print(f"   {model_name}:")
                print(f"     Requests: {data['requests']} ({traffic_pct:.1f}%)")
                print(f"     Accuracy: {accuracy:.1f}%")
                print(f"     Avg Response Time: {avg_response_time:.3f}s")

def main():
    parser = argparse.ArgumentParser(description='A/B Testing Metrics Collector')
    parser.add_argument('--endpoint', default='http://192.168.1.202:80', 
                       help='Seldon mesh endpoint')
    parser.add_argument('--port', type=int, default=8000,
                       help='Prometheus metrics port')
    parser.add_argument('--status-interval', type=int, default=30,
                       help='Status print interval (seconds)')
    
    args = parser.parse_args()
    
    collector = ABTestMetricsCollector(args.endpoint, args.port)
    
    # Start status printing thread
    def status_printer():
        while collector.running:
            time.sleep(args.status_interval)
            if collector.running:
                collector.print_status()
    
    status_thread = threading.Thread(target=status_printer, daemon=True)
    status_thread.start()
    
    try:
        collector.start_collection()
    except KeyboardInterrupt:
        print("\n👋 Shutting down metrics collector...")
        collector.running = False

if __name__ == "__main__":
    main()