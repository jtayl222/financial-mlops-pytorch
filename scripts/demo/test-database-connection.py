#!/usr/bin/env python3
"""
Test connections for live dashboards using MLflow API and Prometheus
"""

import os
import asyncio
import aiohttp
import mlflow
from mlflow.client import MlflowClient
from dotenv import load_dotenv
from datetime import datetime

async def test_connections():
    """Test MLflow API and Prometheus connections"""
    
    # Load environment
    load_dotenv('.env.live-dashboards')
    
    print("🧪 Testing Live Dashboard Connections")
    print("=" * 40)
    
    # Test MLflow API
    print(f"\n📊 [{datetime.now().strftime('%H:%M:%S')}] Testing MLflow API...")
    try:
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://192.168.1.203:5000')
        print(f"🔗 Using MLflow tracking URI: {tracking_uri}")
        
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        
        # Test basic connection
        experiments = client.search_experiments()
        print(f"✅ MLflow API: Connected successfully")
        print(f"   📈 Found {len(experiments)} total experiments")
        
        # Test A/B test experiments
        ab_experiments = []
        for exp in experiments:
            exp_name = exp.name
            if 'ab' in exp_name.lower() or 'test' in exp_name.lower() or 'financial' in exp_name.lower():
                ab_experiments.append(exp_name)
                
        print(f"   🧪 Found {len(ab_experiments)} A/B testing experiments:")
        for exp_name in ab_experiments[:5]:  # Show first 5
            print(f"     - {exp_name}")
        if len(ab_experiments) > 5:
            print(f"     ... and {len(ab_experiments) - 5} more")
            
        # Test runs in A/B experiments
        total_runs = 0
        for exp in experiments:
            if 'ab' in exp.name.lower() or 'test' in exp.name.lower() or 'financial' in exp.name.lower():
                runs = client.search_runs([exp.experiment_id])
                total_runs += len(runs)
                
        print(f"   🏃 Found {total_runs} total runs in A/B experiments")
        
    except Exception as e:
        print(f"❌ MLflow API: {e}")
        import traceback
        print(f"   📍 Error details: {traceback.format_exc()}")
    
    # Test Prometheus
    print(f"\n📈 [{datetime.now().strftime('%H:%M:%S')}] Testing Prometheus...")
    try:
        prometheus_url = os.getenv('PROMETHEUS_URL', 'http://192.168.1.85:30090')
        print(f"🔗 Using Prometheus URL: {prometheus_url}")
        
        # Add timeout for Prometheus requests
        timeout = aiohttp.ClientTimeout(total=10.0)  # 10 second timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{prometheus_url}/api/v1/query", params={'query': 'up'}) as response:
                if response.status == 200:
                    data = await response.json()
                    targets = len(data['data']['result'])
                    print(f"✅ Prometheus: Connected successfully")
                    print(f"   🎯 Found {targets} active targets")
                else:
                    print(f"❌ Prometheus: HTTP {response.status}")
                    
    except asyncio.TimeoutError:
        print(f"❌ Prometheus: Connection timeout (10s)")
    except Exception as e:
        print(f"❌ Prometheus: {e}")
        import traceback
        print(f"   📍 Error details: {traceback.format_exc()}")
    
    print("\n" + "=" * 40)
    print(f"🏁 [{datetime.now().strftime('%H:%M:%S')}] Connection test completed")

if __name__ == "__main__":
    asyncio.run(test_connections())
