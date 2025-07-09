#!/usr/bin/env python3
"""
Quick test of Prometheus metrics collection
"""
import time
from prometheus_client import Counter, Histogram, Gauge, start_http_server, generate_latest

# Initialize metrics
request_counter = Counter('ab_test_requests_total', 'Total requests', ['model_name', 'status'])
response_time_histogram = Histogram('ab_test_response_time_seconds', 'Response time', ['model_name'])
accuracy_gauge = Gauge('ab_test_model_accuracy', 'Model accuracy', ['model_name'])

def generate_sample_metrics():
    # Generate sample metrics data
    for i in range(10):
        # Baseline model
        request_counter.labels(model_name='baseline-predictor', status='success').inc()
        response_time_histogram.labels(model_name='baseline-predictor').observe(0.045)
        accuracy_gauge.labels(model_name='baseline-predictor').set(78.5)
        
        # Enhanced model
        request_counter.labels(model_name='enhanced-predictor', status='success').inc()
        response_time_histogram.labels(model_name='enhanced-predictor').observe(0.062)
        accuracy_gauge.labels(model_name='enhanced-predictor').set(82.1)
        
        time.sleep(0.1)

if __name__ == "__main__":
    print("🔄 Generating sample metrics...")
    generate_sample_metrics()
    
    print("📊 Sample metrics generated successfully!")
    print("Metrics output:")
    print(generate_latest().decode('utf-8'))