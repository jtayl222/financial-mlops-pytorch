#!/usr/bin/env python3
"""
Multi-Armed Bandit Demo for Financial Models
Demonstrates dynamic traffic allocation based on performance
"""

import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class BanditArm:
    """Represents a model in the multi-armed bandit"""
    name: str
    total_reward: float = 0.0
    total_pulls: int = 0
    confidence_interval: float = 0.0
    
    @property
    def average_reward(self) -> float:
        return self.total_reward / max(self.total_pulls, 1)
    
    @property
    def upper_confidence_bound(self) -> float:
        if self.total_pulls == 0:
            return float('inf')
        return self.average_reward + self.confidence_interval

class MultiArmedBanditController:
    """Thompson Sampling Multi-Armed Bandit for model selection"""
    
    def __init__(self, model_names: List[str], exploration_rate: float = 0.1):
        self.models = {name: BanditArm(name) for name in model_names}
        self.exploration_rate = exploration_rate
        self.total_pulls = 0
        self.selection_history = []
        self.reward_history = []
        
    def select_model(self) -> str:
        """Select model using Upper Confidence Bound algorithm"""
        self.total_pulls += 1
        
        # Exploration phase: try all models at least once
        for model_name, arm in self.models.items():
            if arm.total_pulls == 0:
                self.selection_history.append(model_name)
                return model_name
        
        # Exploitation phase: select model with highest UCB
        for arm in self.models.values():
            confidence_bonus = np.sqrt(2 * np.log(self.total_pulls) / arm.total_pulls)
            arm.confidence_interval = confidence_bonus
        
        selected_model = max(self.models.values(), key=lambda x: x.upper_confidence_bound).name
        self.selection_history.append(selected_model)
        return selected_model
    
    def update_reward(self, model_name: str, reward: float):
        """Update model performance with new reward"""
        if model_name in self.models:
            arm = self.models[model_name]
            arm.total_reward += reward
            arm.total_pulls += 1
            self.reward_history.append(reward)
    
    def get_statistics(self) -> Dict:
        """Get current bandit statistics"""
        return {
            'total_pulls': self.total_pulls,
            'model_statistics': {
                name: {
                    'average_reward': arm.average_reward,
                    'total_pulls': arm.total_pulls,
                    'selection_percentage': arm.total_pulls / max(self.total_pulls, 1) * 100,
                    'total_reward': arm.total_reward,
                    'ucb_value': arm.upper_confidence_bound
                }
                for name, arm in self.models.items()
            }
        }

class FinancialMABDemo:
    """Demo class for multi-armed bandit experiment"""
    
    def __init__(self, endpoint: str, models: List[str]):
        self.endpoint = endpoint
        self.models = models
        self.bandit = MultiArmedBanditController(models)
        self.results = []
        self.running = False
        
    def generate_market_scenario(self) -> Dict:
        """Generate realistic market scenario for testing"""
        # Simulate different market conditions
        market_conditions = ['bull', 'bear', 'sideways', 'volatile']
        condition = np.random.choice(market_conditions)
        
        # Generate features based on market condition
        if condition == 'bull':
            base_return = np.random.normal(0.02, 0.01)
            volatility = np.random.normal(0.15, 0.05)
        elif condition == 'bear':
            base_return = np.random.normal(-0.02, 0.01)
            volatility = np.random.normal(0.20, 0.05)
        elif condition == 'volatile':
            base_return = np.random.normal(0.001, 0.005)
            volatility = np.random.normal(0.35, 0.10)
        else:  # sideways
            base_return = np.random.normal(0.001, 0.005)
            volatility = np.random.normal(0.12, 0.03)
        
        # Generate 35 features (matching our model input)
        features = []
        
        # Price and volume features
        features.extend([
            100 + np.random.normal(0, 5),  # price
            np.random.lognormal(15, 0.5),  # volume
            base_return,  # returns
            volatility,  # volatility
        ])
        
        # Technical indicators
        features.extend([
            np.random.normal(0.5, 0.1),  # RSI
            np.random.normal(0, 0.02),   # MACD
            np.random.normal(1, 0.05),   # SMA ratio
            np.random.normal(1, 0.05),   # Another SMA ratio
        ])
        
        # Fill remaining features
        while len(features) < 35:
            features.append(np.random.normal(0.5, 0.1))
        
        # Create sequence data (10 timesteps)
        sequence_data = []
        for _ in range(10):
            timestep_features = [f + np.random.normal(0, 0.01) for f in features]
            sequence_data.extend(timestep_features)
        
        return {
            'features': sequence_data,
            'market_condition': condition,
            'expected_return': base_return,
            'volatility': volatility,
            'timestamp': datetime.now()
        }
    
    def calculate_business_reward(self, prediction: float, actual_return: float, 
                                model_name: str, response_time: float) -> float:
        """Calculate business reward for the prediction"""
        # Accuracy component
        predicted_direction = 1 if prediction > 0.5 else -1
        actual_direction = 1 if actual_return > 0 else -1
        accuracy_reward = 1.0 if predicted_direction == actual_direction else 0.0
        
        # Confidence component (higher confidence = higher reward if correct)
        confidence = abs(prediction - 0.5) * 2  # Scale to 0-1
        confidence_reward = accuracy_reward * confidence
        
        # Latency penalty (faster models get bonus)
        latency_penalty = max(0, (response_time - 0.050) * 10)  # Penalty for >50ms
        
        # Model-specific bonuses (simulate different model capabilities)
        model_bonuses = {
            'baseline-predictor': 0.0,
            'enhanced-predictor': 0.1,
            'transformer-predictor': 0.15,
            'ensemble-predictor': 0.2
        }
        
        model_bonus = model_bonuses.get(model_name, 0.0)
        
        # Final reward calculation
        final_reward = (accuracy_reward + confidence_reward + model_bonus) - latency_penalty
        
        return max(0, final_reward)  # Ensure non-negative reward
    
    def send_prediction_request(self, model_name: str, scenario: Dict) -> Dict:
        """Send prediction request to selected model"""
        try:
            # Prepare request data
            input_data = {
                "inputs": [{
                    "name": "input_data",
                    "shape": [1, 10, 35],
                    "datatype": "FP32",
                    "data": np.array(scenario['features']).reshape(1, 10, 35).tolist()
                }]
            }
            
            # Send request
            url = f"{self.endpoint}/v2/models/{model_name}/infer"
            headers = {
                "Content-Type": "application/json",
                "Host": "financial-predictor.local"
            }
            
            start_time = time.time()
            response = requests.post(url, json=input_data, headers=headers, timeout=5)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('outputs', [{}])[0].get('data', [0.5])[0]
                
                # Calculate reward
                reward = self.calculate_business_reward(
                    prediction, 
                    scenario['expected_return'],
                    model_name,
                    response_time
                )
                
                return {
                    'success': True,
                    'model_name': model_name,
                    'prediction': prediction,
                    'response_time': response_time,
                    'reward': reward,
                    'scenario': scenario
                }
            else:
                return {
                    'success': False,
                    'model_name': model_name,
                    'error': f"HTTP {response.status_code}",
                    'reward': 0.0,
                    'scenario': scenario
                }
                
        except Exception as e:
            return {
                'success': False,
                'model_name': model_name,
                'error': str(e),
                'reward': 0.0,
                'scenario': scenario
            }
    
    def run_experiment(self, duration_minutes: int = 10, requests_per_minute: int = 20):
        """Run the multi-armed bandit experiment"""
        print(f"🎯 Starting Multi-Armed Bandit Experiment")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   Rate: {requests_per_minute} requests/minute")
        print(f"   Models: {', '.join(self.models)}")
        print()
        
        self.running = True
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        request_count = 0
        while time.time() < end_time and self.running:
            # Generate market scenario
            scenario = self.generate_market_scenario()
            
            # Select model using bandit algorithm
            selected_model = self.bandit.select_model()
            
            # Send prediction request
            result = self.send_prediction_request(selected_model, scenario)
            
            # Update bandit with reward
            self.bandit.update_reward(selected_model, result['reward'])
            
            # Store result
            self.results.append(result)
            request_count += 1
            
            # Print progress
            if request_count % 10 == 0:
                stats = self.bandit.get_statistics()
                print(f"   Request {request_count}: Selected {selected_model}, "
                      f"Reward: {result['reward']:.3f}")
                
                # Show current model selection percentages
                for model_name, model_stats in stats['model_statistics'].items():
                    print(f"      {model_name}: {model_stats['selection_percentage']:.1f}% "
                          f"(avg reward: {model_stats['average_reward']:.3f})")
                print()
            
            # Wait before next request
            time.sleep(60 / requests_per_minute)
        
        self.running = False
        print(f"🎯 Experiment completed! Total requests: {request_count}")
        return self.results
    
    def generate_report(self) -> Dict:
        """Generate comprehensive experiment report"""
        stats = self.bandit.get_statistics()
        
        # Calculate additional metrics
        successful_requests = [r for r in self.results if r['success']]
        total_reward = sum(r['reward'] for r in successful_requests)
        average_reward = total_reward / len(successful_requests) if successful_requests else 0
        
        # Model performance analysis
        model_performance = {}
        for model_name in self.models:
            model_results = [r for r in successful_requests if r['model_name'] == model_name]
            if model_results:
                model_performance[model_name] = {
                    'total_requests': len(model_results),
                    'average_reward': np.mean([r['reward'] for r in model_results]),
                    'average_response_time': np.mean([r['response_time'] for r in model_results]),
                    'success_rate': len(model_results) / len([r for r in self.results if r['model_name'] == model_name]) * 100
                }
        
        # Convergence analysis
        convergence_data = []
        cumulative_rewards = defaultdict(float)
        cumulative_counts = defaultdict(int)
        
        for i, result in enumerate(self.results):
            if result['success']:
                model_name = result['model_name']
                cumulative_rewards[model_name] += result['reward']
                cumulative_counts[model_name] += 1
                
                convergence_data.append({
                    'iteration': i,
                    'model': model_name,
                    'cumulative_average_reward': cumulative_rewards[model_name] / cumulative_counts[model_name]
                })
        
        report = {
            'experiment_summary': {
                'total_requests': len(self.results),
                'successful_requests': len(successful_requests),
                'total_reward': total_reward,
                'average_reward': average_reward,
                'experiment_duration': len(self.results) * 3  # Approximate duration
            },
            'bandit_statistics': stats,
            'model_performance': model_performance,
            'convergence_data': convergence_data
        }
        
        return report
    
    def create_visualization(self, report: Dict):
        """Create visualizations of the experiment results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model Selection Distribution
        stats = report['bandit_statistics']['model_statistics']
        models = list(stats.keys())
        selections = [stats[model]['selection_percentage'] for model in models]
        
        ax1.pie(selections, labels=models, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Model Selection Distribution')
        
        # 2. Average Reward by Model
        avg_rewards = [stats[model]['average_reward'] for model in models]
        bars = ax2.bar(models, avg_rewards, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.set_title('Average Reward by Model')
        ax2.set_ylabel('Average Reward')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, reward in zip(bars, avg_rewards):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{reward:.3f}', ha='center', va='bottom')
        
        # 3. Convergence Analysis
        convergence = report['convergence_data']
        for model in models:
            model_data = [d for d in convergence if d['model'] == model]
            if model_data:
                iterations = [d['iteration'] for d in model_data]
                rewards = [d['cumulative_average_reward'] for d in model_data]
                ax3.plot(iterations, rewards, label=model, marker='o', markersize=2)
        
        ax3.set_title('Reward Convergence Over Time')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Cumulative Average Reward')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Metrics Comparison
        performance = report['model_performance']
        if performance:
            metrics = ['average_reward', 'success_rate', 'average_response_time']
            metric_labels = ['Avg Reward', 'Success Rate (%)', 'Avg Response Time (s)']
            
            x = np.arange(len(models))
            width = 0.25
            
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                if metric == 'success_rate':
                    values = [performance[model][metric] for model in models if model in performance]
                elif metric == 'average_response_time':
                    values = [performance[model][metric] * 1000 for model in models if model in performance]  # Convert to ms
                else:
                    values = [performance[model][metric] for model in models if model in performance]
                
                ax4.bar(x + i*width, values, width, label=label)
            
            ax4.set_title('Model Performance Comparison')
            ax4.set_xlabel('Models')
            ax4.set_xticks(x + width)
            ax4.set_xticklabels(models, rotation=45)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mab_experiment_results_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved as: {filename}")
        
        return filename

def main():
    parser = argparse.ArgumentParser(description='Multi-Armed Bandit Demo for Financial Models')
    parser.add_argument('--endpoint', default='http://192.168.1.202:80', 
                       help='Seldon mesh endpoint')
    parser.add_argument('--models', type=int, default=4,
                       help='Number of models to test (2-4)')
    parser.add_argument('--duration', type=int, default=10,
                       help='Experiment duration in minutes')
    parser.add_argument('--rate', type=int, default=12,
                       help='Requests per minute')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Define available models
    all_models = ['baseline-predictor', 'enhanced-predictor', 'transformer-predictor', 'ensemble-predictor']
    selected_models = all_models[:args.models]
    
    print("🎯 Multi-Armed Bandit Experiment for Financial Models")
    print("=" * 60)
    
    # Initialize demo
    demo = FinancialMABDemo(args.endpoint, selected_models)
    
    # Run experiment
    results = demo.run_experiment(args.duration, args.rate)
    
    # Generate report
    report = demo.generate_report()
    
    # Print summary
    print("\n📊 Experiment Summary:")
    print(f"   Total Requests: {report['experiment_summary']['total_requests']}")
    print(f"   Successful Requests: {report['experiment_summary']['successful_requests']}")
    print(f"   Total Reward: {report['experiment_summary']['total_reward']:.3f}")
    print(f"   Average Reward: {report['experiment_summary']['average_reward']:.3f}")
    
    print("\n🏆 Final Model Rankings:")
    model_stats = report['bandit_statistics']['model_statistics']
    sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['average_reward'], reverse=True)
    
    for i, (model_name, stats) in enumerate(sorted_models, 1):
        print(f"   {i}. {model_name}: {stats['average_reward']:.3f} avg reward "
              f"({stats['selection_percentage']:.1f}% selection)")
    
    print(f"\n🎉 Multi-Armed Bandit Experiment Complete!")
    print(f"   Best performing model: {sorted_models[0][0]}")
    print(f"   Optimal traffic allocation achieved through exploration-exploitation balance")
    
    # Create visualization
    if not args.no_viz:
        demo.create_visualization(report)

if __name__ == "__main__":
    main()