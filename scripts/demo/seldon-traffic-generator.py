#!/usr/bin/env python3
"""
Seldon Traffic Generator - Send Real Requests to financial-ab-test-experiment
Generates actual traffic to create real Prometheus metrics for dashboard
"""

import os
import sys
import asyncio
import aiohttp
import json
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class TrafficConfig:
    """Configuration for traffic generation"""
    seldon_endpoint: str
    experiment_name: str = "financial-ab-test-experiment"
    namespace: str = "financial-inference"
    requests_per_minute: int = 120  # 2 requests per second
    duration_minutes: int = 10
    concurrent_workers: int = 5
    model_timeout: float = 5.0  # 5 second timeout per request

class FinancialDataGenerator:
    """Generates realistic financial market data for model requests"""
    
    def __init__(self):
        self.base_price = 100.0
        self.volatility = 0.02
        self.trend = 0.001
        
    def generate_market_features(self) -> Dict:
        """Generate realistic financial features"""
        
        # Base price with random walk
        price_change = np.random.normal(self.trend, self.volatility)
        self.base_price *= (1 + price_change)
        
        # Technical indicators
        features = {
            'price': round(self.base_price, 2),
            'volume': int(np.random.exponential(1000000)),
            'volatility': round(np.random.beta(2, 5) * 0.1, 4),
            'rsi': round(np.random.uniform(20, 80), 2),
            'macd': round(np.random.normal(0, 1), 4),
            'moving_avg_5': round(self.base_price * (1 + np.random.normal(0, 0.01)), 2),
            'moving_avg_20': round(self.base_price * (1 + np.random.normal(0, 0.02)), 2),
            'bollinger_upper': round(self.base_price * 1.02, 2),
            'bollinger_lower': round(self.base_price * 0.98, 2),
            'momentum': round(np.random.normal(0, 0.5), 4),
            'stoch_k': round(np.random.uniform(0, 100), 2),
            'stoch_d': round(np.random.uniform(0, 100), 2),
        }
        
        # Market context
        market_conditions = ['bull', 'bear', 'sideways', 'volatile']
        features['market_condition'] = random.choice(market_conditions)
        features['trading_session'] = 'market_hours' if 9 <= datetime.now().hour <= 16 else 'after_hours'
        features['day_of_week'] = datetime.now().strftime('%A').lower()
        
        return features

class SeldonTrafficGenerator:
    """Generates real traffic to Seldon Core v2 experiment"""
    
    def __init__(self, config: TrafficConfig):
        self.config = config
        self.data_generator = FinancialDataGenerator()
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'model_responses': {},
            'start_time': None,
            'end_time': None
        }
        
    async def discover_seldon_endpoint(self) -> str:
        """Try to discover the actual Seldon experiment endpoint"""
        
        # Common Seldon endpoint patterns - try HTTP first to avoid SSL issues
        potential_endpoints = [
            f"http://192.168.1.85:30080/seldon/{self.config.namespace}/{self.config.experiment_name}/api/v1.0/predictions",
            f"http://192.168.1.85:8080/seldon/{self.config.namespace}/{self.config.experiment_name}/api/v1.0/predictions", 
            f"http://192.168.1.85:30080/seldon/{self.config.namespace}/{self.config.experiment_name}/v2/models/infer",
            f"http://seldon-gateway.{self.config.namespace}.svc.cluster.local:8080/seldon/{self.config.namespace}/{self.config.experiment_name}/api/v1.0/predictions",
            # Try existing advanced-ab-demo endpoint pattern
            f"http://192.168.1.85:30080",
            self.config.seldon_endpoint  # Use configured endpoint as fallback
        ]
        
        logger.info("🔍 Discovering Seldon experiment endpoint...")
        
        for endpoint in potential_endpoints:
            try:
                logger.info(f"   Testing: {endpoint}")
                timeout = aiohttp.ClientTimeout(total=5.0)
                # Create SSL context that bypasses certificate verification for testing
                ssl_context = False  # Disable SSL verification completely
                connector = aiohttp.TCPConnector(ssl=ssl_context)
                async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                    # Try a simple health check or metadata request
                    test_payload = {
                        "inputs": [{
                            "name": "financial_features",
                            "datatype": "FP32",
                            "shape": [1, 12],
                            "data": [100.0, 1000000, 0.02, 50.0, 0.0, 99.0, 98.0, 102.0, 98.0, 0.0, 50.0, 50.0]
                        }]
                    }
                    
                    async with session.post(endpoint, json=test_payload) as response:
                        if response.status in [200, 400, 422]:  # 400/422 might be format issues but endpoint exists
                            logger.info(f"✅ Found accessible endpoint: {endpoint}")
                            return endpoint
                        else:
                            logger.debug(f"   HTTP {response.status}")
                            
            except Exception as e:
                logger.debug(f"   Failed: {e}")
                continue
        
        logger.warning(f"⚠️ No accessible endpoints found, using configured: {self.config.seldon_endpoint}")
        return self.config.seldon_endpoint
    
    async def send_prediction_request(self, session: aiohttp.ClientSession, endpoint: str, request_id: int) -> Dict:
        """Send a single prediction request to Seldon"""
        
        # Generate realistic financial data
        features = self.data_generator.generate_market_features()
        
        # Convert to Seldon v2 format
        request_payload = {
            "model_name": self.config.experiment_name,
            "inputs": [{
                "name": "financial_features",
                "datatype": "FP32", 
                "shape": [1, len(features)],
                "data": list(features.values())[:12]  # Take first 12 numeric features
            }],
            "parameters": {
                "experiment_name": self.config.experiment_name,
                "request_id": str(request_id),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        start_time = time.time()
        
        try:
            async with session.post(endpoint, json=request_payload) as response:
                duration = time.time() - start_time
                response_data = await response.text()
                
                result = {
                    'request_id': request_id,
                    'status_code': response.status,
                    'duration': duration,
                    'response_size': len(response_data),
                    'timestamp': datetime.now(),
                    'features': features,
                    'success': response.status == 200
                }
                
                # Try to parse JSON response
                try:
                    if response_data:
                        json_response = json.loads(response_data)
                        result['response_data'] = json_response
                        
                        # Extract model name from response headers or body
                        model_name = response.headers.get('x-model-name', 'unknown')
                        if 'model_name' in json_response:
                            model_name = json_response['model_name']
                        result['model_name'] = model_name
                        
                except json.JSONDecodeError:
                    result['response_data'] = response_data[:200]  # First 200 chars
                
                return result
                
        except asyncio.TimeoutError:
            return {
                'request_id': request_id,
                'status_code': 408,  # Request Timeout
                'duration': time.time() - start_time,
                'error': 'timeout',
                'timestamp': datetime.now(),
                'success': False
            }
        except Exception as e:
            return {
                'request_id': request_id,
                'status_code': 500,
                'duration': time.time() - start_time,
                'error': str(e),
                'timestamp': datetime.now(),
                'success': False
            }
    
    async def worker(self, worker_id: int, session: aiohttp.ClientSession, endpoint: str, request_queue: asyncio.Queue):
        """Worker function to process requests"""
        
        logger.info(f"🏃 Worker {worker_id} started")
        
        while True:
            try:
                request_id = await asyncio.wait_for(request_queue.get(), timeout=1.0)
                
                # Send request
                result = await self.send_prediction_request(session, endpoint, request_id)
                
                # Update statistics
                self.stats['total_requests'] += 1
                
                if result['success']:
                    self.stats['successful_requests'] += 1
                    self.stats['response_times'].append(result['duration'])
                    
                    # Track model responses
                    model_name = result.get('model_name', 'unknown')
                    if model_name not in self.stats['model_responses']:
                        self.stats['model_responses'][model_name] = 0
                    self.stats['model_responses'][model_name] += 1
                    
                    if self.stats['total_requests'] % 20 == 0:  # Log every 20 requests
                        avg_response_time = np.mean(self.stats['response_times'][-20:]) * 1000
                        logger.info(f"📊 Requests: {self.stats['total_requests']}, "
                                  f"Success: {self.stats['successful_requests']}, "
                                  f"Avg RT: {avg_response_time:.1f}ms, "
                                  f"Models: {self.stats['model_responses']}")
                else:
                    self.stats['failed_requests'] += 1
                    if result.get('status_code') != 408:  # Don't log timeouts as errors
                        logger.warning(f"❌ Request {request_id} failed: {result.get('error', 'HTTP ' + str(result.get('status_code')))}")
                
                request_queue.task_done()
                
            except asyncio.TimeoutError:
                # No more requests in queue
                break
            except Exception as e:
                logger.error(f"❌ Worker {worker_id} error: {e}")
                break
        
        logger.info(f"🏁 Worker {worker_id} finished")
    
    async def generate_traffic(self) -> Dict:
        """Generate traffic to the Seldon experiment"""
        
        logger.info(f"🚀 Starting traffic generation for {self.config.experiment_name}")
        logger.info(f"📊 Config: {self.config.requests_per_minute} req/min for {self.config.duration_minutes} min with {self.config.concurrent_workers} workers")
        
        # Discover endpoint
        endpoint = await self.discover_seldon_endpoint()
        
        # Calculate timing
        total_requests = self.config.requests_per_minute * self.config.duration_minutes
        interval_seconds = 60.0 / self.config.requests_per_minute
        
        logger.info(f"🎯 Target: {total_requests} requests over {self.config.duration_minutes} minutes")
        logger.info(f"⏱️ Request interval: {interval_seconds:.2f} seconds")
        logger.info(f"🌐 Endpoint: {endpoint}")
        
        # Create request queue
        request_queue = asyncio.Queue()
        
        # Add all requests to queue
        for i in range(total_requests):
            await request_queue.put(i + 1)
        
        # Create session with appropriate timeout and SSL bypass
        timeout = aiohttp.ClientTimeout(total=self.config.model_timeout)
        ssl_context = False  # Disable SSL verification
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        
        self.stats['start_time'] = datetime.now()
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            # Start workers
            workers = []
            for worker_id in range(self.config.concurrent_workers):
                worker = asyncio.create_task(self.worker(worker_id, session, endpoint, request_queue))
                workers.append(worker)
            
            # Wait for all requests to be processed or timeout
            try:
                await asyncio.wait_for(request_queue.join(), timeout=self.config.duration_minutes * 60 + 60)
            except asyncio.TimeoutError:
                logger.warning("⏰ Traffic generation timed out")
            
            # Cancel workers
            for worker in workers:
                worker.cancel()
            
            # Wait for workers to finish
            await asyncio.gather(*workers, return_exceptions=True)
        
        self.stats['end_time'] = datetime.now()
        
        return self.stats
    
    def print_summary(self, stats: Dict):
        """Print traffic generation summary"""
        
        duration = (stats['end_time'] - stats['start_time']).total_seconds()
        success_rate = (stats['successful_requests'] / stats['total_requests'] * 100) if stats['total_requests'] > 0 else 0
        
        logger.info("\n" + "=" * 50)
        logger.info("📊 TRAFFIC GENERATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"⏱️ Duration: {duration:.1f} seconds")
        logger.info(f"📨 Total Requests: {stats['total_requests']}")
        logger.info(f"✅ Successful: {stats['successful_requests']} ({success_rate:.1f}%)")
        logger.info(f"❌ Failed: {stats['failed_requests']}")
        
        if stats['response_times']:
            avg_rt = np.mean(stats['response_times']) * 1000
            p95_rt = np.percentile(stats['response_times'], 95) * 1000
            logger.info(f"⚡ Avg Response Time: {avg_rt:.1f}ms")
            logger.info(f"📈 P95 Response Time: {p95_rt:.1f}ms")
        
        if stats['model_responses']:
            logger.info(f"🤖 Model Responses:")
            for model, count in stats['model_responses'].items():
                percentage = (count / stats['successful_requests'] * 100) if stats['successful_requests'] > 0 else 0
                logger.info(f"   {model}: {count} ({percentage:.1f}%)")
        
        actual_rps = stats['total_requests'] / duration if duration > 0 else 0
        logger.info(f"📊 Actual Rate: {actual_rps:.1f} requests/second")
        logger.info("=" * 50)

async def main():
    """Main traffic generation function"""
    
    # Configuration
    config = TrafficConfig(
        seldon_endpoint=os.getenv('SELDON_ENDPOINT', 'http://192.168.1.85:30080/seldon/financial-inference/financial-ab-test-experiment/api/v1.0/predictions'),
        experiment_name=os.getenv('SELDON_EXPERIMENT_NAME', 'financial-ab-test-experiment'),
        namespace=os.getenv('SELDON_NAMESPACE', 'financial-inference'),
        requests_per_minute=int(os.getenv('TRAFFIC_RATE', '120')),  # 2 requests per second
        duration_minutes=int(os.getenv('TRAFFIC_DURATION', '10')),   # 10 minutes
        concurrent_workers=int(os.getenv('TRAFFIC_WORKERS', '5')),   # 5 concurrent workers
        model_timeout=float(os.getenv('MODEL_TIMEOUT', '5.0'))      # 5 second timeout
    )
    
    logger.info("🚀 Seldon Traffic Generator Starting")
    logger.info(f"🎯 Target Experiment: {config.experiment_name}")
    logger.info(f"🏗️ Namespace: {config.namespace}")
    
    # Create traffic generator
    generator = SeldonTrafficGenerator(config)
    
    try:
        # Generate traffic
        stats = await generator.generate_traffic()
        
        # Print summary
        generator.print_summary(stats)
        
        logger.info("✅ Traffic generation completed successfully!")
        logger.info("📊 Check Prometheus metrics and re-run dashboard to see real data")
        
    except KeyboardInterrupt:
        logger.info("🛑 Traffic generation interrupted by user")
    except Exception as e:
        logger.error(f"❌ Traffic generation failed: {e}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())