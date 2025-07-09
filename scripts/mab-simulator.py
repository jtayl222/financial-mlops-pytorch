#!/usr/bin/env python3
"""
Multi-Armed Bandit Simulator for Financial Models
Demonstrates advanced model selection with Thompson Sampling
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import argparse

@dataclass
class ModelArm:
    """Represents a model in the multi-armed bandit"""
    name: str
    true_performance: float  # Hidden true performance
    alpha: float = 1.0  # Beta distribution parameter
    beta: float = 1.0   # Beta distribution parameter
    total_reward: float = 0.0
    total_pulls: int = 0
    
    @property
    def estimated_performance(self) -> float:
        """Current estimated performance"""
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """95% confidence interval"""
        mean = self.estimated_performance
        variance = (self.alpha * self.beta) / ((self.alpha + self.beta)**2 * (self.alpha + self.beta + 1))
        std = np.sqrt(variance)
        return (mean - 1.96 * std, mean + 1.96 * std)
    
    def sample_performance(self) -> float:
        """Sample from current belief distribution"""
        return np.random.beta(self.alpha, self.beta)
    
    def update(self, reward: float):
        """Update belief based on observed reward"""
        self.total_reward += reward
        self.total_pulls += 1
        
        # Update Beta distribution parameters
        self.alpha += reward
        self.beta += (1 - reward)

class FinancialMABSimulator:
    """Multi-Armed Bandit simulator for financial model selection"""
    
    def __init__(self, model_configs: Dict[str, float]):
        """
        Initialize with model configurations
        model_configs: {model_name: true_performance_rate}
        """
        self.models = {
            name: ModelArm(name, performance) 
            for name, performance in model_configs.items()
        }
        self.selection_history = []
        self.reward_history = []
        self.regret_history = []
        self.optimal_arm = max(model_configs.keys(), key=lambda k: model_configs[k])
        self.optimal_performance = max(model_configs.values())
        self.cumulative_regret = 0.0
        
    def thompson_sampling_select(self) -> str:
        """Select model using Thompson Sampling"""
        samples = {
            name: arm.sample_performance() 
            for name, arm in self.models.items()
        }
        selected = max(samples.keys(), key=lambda k: samples[k])
        self.selection_history.append(selected)
        return selected
    
    def ucb_select(self, exploration_param: float = 2.0) -> str:
        """Select model using Upper Confidence Bound"""
        total_pulls = sum(arm.total_pulls for arm in self.models.values())
        
        if total_pulls == 0:
            # Random selection for first pull
            selected = np.random.choice(list(self.models.keys()))
        else:
            ucb_values = {}
            for name, arm in self.models.items():
                if arm.total_pulls == 0:
                    ucb_values[name] = float('inf')
                else:
                    confidence_bonus = np.sqrt(
                        exploration_param * np.log(total_pulls) / arm.total_pulls
                    )
                    ucb_values[name] = arm.estimated_performance + confidence_bonus
            
            selected = max(ucb_values.keys(), key=lambda k: ucb_values[k])
        
        self.selection_history.append(selected)
        return selected
    
    def simulate_market_reward(self, model_name: str, market_scenario: str) -> float:
        """Simulate realistic market-based reward"""
        base_performance = self.models[model_name].true_performance
        
        # Market condition modifiers
        market_modifiers = {
            'bull': {'baseline-predictor': 0.0, 'enhanced-predictor': 0.05, 
                    'transformer-predictor': 0.08, 'ensemble-predictor': 0.10},
            'bear': {'baseline-predictor': 0.02, 'enhanced-predictor': 0.0, 
                    'transformer-predictor': 0.03, 'ensemble-predictor': 0.05},
            'sideways': {'baseline-predictor': 0.05, 'enhanced-predictor': 0.02, 
                        'transformer-predictor': 0.0, 'ensemble-predictor': 0.03},
            'volatile': {'baseline-predictor': -0.05, 'enhanced-predictor': -0.02, 
                        'transformer-predictor': 0.02, 'ensemble-predictor': 0.08}
        }
        
        modifier = market_modifiers.get(market_scenario, {}).get(model_name, 0.0)
        adjusted_performance = base_performance + modifier
        
        # Add noise and ensure valid probability
        noise = np.random.normal(0, 0.05)
        final_performance = np.clip(adjusted_performance + noise, 0.0, 1.0)
        
        # Generate binary reward based on performance
        reward = 1.0 if np.random.random() < final_performance else 0.0
        
        return reward
    
    def run_experiment(self, n_rounds: int = 1000, algorithm: str = 'thompson', 
                      market_scenarios: List[str] = None) -> Dict:
        """Run the multi-armed bandit experiment"""
        
        if market_scenarios is None:
            market_scenarios = ['bull', 'bear', 'sideways', 'volatile']
        
        print(f"🎯 Starting Multi-Armed Bandit Experiment")
        print(f"   Algorithm: {algorithm}")
        print(f"   Rounds: {n_rounds}")
        print(f"   Models: {list(self.models.keys())}")
        print(f"   Market scenarios: {market_scenarios}")
        print()
        
        results = []
        
        for round_num in range(n_rounds):
            # Select market scenario
            market_scenario = np.random.choice(market_scenarios)
            
            # Select model based on algorithm
            if algorithm == 'thompson':
                selected_model = self.thompson_sampling_select()
            elif algorithm == 'ucb':
                selected_model = self.ucb_select()
            else:
                selected_model = np.random.choice(list(self.models.keys()))
            
            # Simulate reward
            reward = self.simulate_market_reward(selected_model, market_scenario)
            
            # Update model
            self.models[selected_model].update(reward)
            
            # Calculate regret
            optimal_reward = self.simulate_market_reward(self.optimal_arm, market_scenario)
            regret = optimal_reward - reward
            self.cumulative_regret += regret
            
            # Store results
            result = {
                'round': round_num,
                'selected_model': selected_model,
                'market_scenario': market_scenario,
                'reward': reward,
                'regret': regret,
                'cumulative_regret': self.cumulative_regret
            }
            results.append(result)
            self.reward_history.append(reward)
            self.regret_history.append(regret)
            
            # Print progress
            if (round_num + 1) % 100 == 0:
                self._print_progress(round_num + 1, n_rounds)
        
        print(f"\n🎯 Experiment completed!")
        return results
    
    def _print_progress(self, current_round: int, total_rounds: int):
        """Print experiment progress"""
        progress = current_round / total_rounds * 100
        
        # Calculate current selection percentages
        recent_selections = self.selection_history[-100:]  # Last 100 selections
        selection_counts = defaultdict(int)
        for selection in recent_selections:
            selection_counts[selection] += 1
        
        print(f"   Progress: {current_round}/{total_rounds} ({progress:.1f}%)")
        print(f"   Recent selections (last 100):")
        for model_name, count in selection_counts.items():
            percentage = count / len(recent_selections) * 100
            estimated_perf = self.models[model_name].estimated_performance
            print(f"      {model_name}: {percentage:.1f}% (est. perf: {estimated_perf:.3f})")
        print(f"   Cumulative regret: {self.cumulative_regret:.2f}")
        print()
    
    def generate_report(self) -> Dict:
        """Generate comprehensive experiment report"""
        
        # Calculate final statistics
        total_selections = len(self.selection_history)
        selection_counts = defaultdict(int)
        for selection in self.selection_history:
            selection_counts[selection] += 1
        
        # Model performance analysis
        model_stats = {}
        for name, arm in self.models.items():
            model_stats[name] = {
                'true_performance': arm.true_performance,
                'estimated_performance': arm.estimated_performance,
                'confidence_interval': arm.confidence_interval,
                'total_pulls': arm.total_pulls,
                'selection_percentage': selection_counts[name] / total_selections * 100,
                'total_reward': arm.total_reward,
                'average_reward': arm.total_reward / max(arm.total_pulls, 1)
            }
        
        # Performance metrics
        total_reward = sum(self.reward_history)
        average_reward = total_reward / len(self.reward_history)
        
        # Regret analysis
        final_regret = self.cumulative_regret
        average_regret = final_regret / len(self.regret_history)
        
        # Convergence analysis
        convergence_window = 100
        convergence_data = []
        for i in range(convergence_window, len(self.selection_history)):
            window_selections = self.selection_history[i-convergence_window:i]
            optimal_selections = sum(1 for s in window_selections if s == self.optimal_arm)
            convergence_data.append({
                'round': i,
                'optimal_selection_rate': optimal_selections / convergence_window
            })
        
        report = {
            'experiment_summary': {
                'total_rounds': len(self.selection_history),
                'total_reward': total_reward,
                'average_reward': average_reward,
                'cumulative_regret': final_regret,
                'average_regret': average_regret,
                'optimal_arm': self.optimal_arm,
                'optimal_performance': self.optimal_performance
            },
            'model_statistics': model_stats,
            'convergence_data': convergence_data,
            'final_selection_distribution': dict(selection_counts)
        }
        
        return report
    
    def create_comprehensive_visualization(self, report: Dict) -> str:
        """Create comprehensive visualization of results"""
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Color scheme for models
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#F7B801']
        model_colors = {name: colors[i] for i, name in enumerate(self.models.keys())}
        
        # 1. Selection Distribution Over Time
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Calculate rolling selection percentages
        window_size = 100
        rounds = list(range(window_size, len(self.selection_history)))
        
        for model_name in self.models.keys():
            rolling_percentages = []
            for i in range(window_size, len(self.selection_history)):
                window_selections = self.selection_history[i-window_size:i]
                percentage = sum(1 for s in window_selections if s == model_name) / window_size * 100
                rolling_percentages.append(percentage)
            
            ax1.plot(rounds, rolling_percentages, label=model_name, 
                    color=model_colors[model_name], linewidth=2)
        
        ax1.set_title('Model Selection Distribution Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Selection Percentage (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative Regret
        ax2 = fig.add_subplot(gs[0, 2:])
        
        cumulative_regret = np.cumsum(self.regret_history)
        ax2.plot(cumulative_regret, color='red', linewidth=2, label='Cumulative Regret')
        ax2.fill_between(range(len(cumulative_regret)), cumulative_regret, alpha=0.3, color='red')
        
        ax2.set_title('Cumulative Regret Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Cumulative Regret')
        ax2.grid(True, alpha=0.3)
        
        # 3. Model Performance Comparison
        ax3 = fig.add_subplot(gs[1, :2])
        
        models = list(self.models.keys())
        true_perf = [self.models[m].true_performance for m in models]
        estimated_perf = [self.models[m].estimated_performance for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, true_perf, width, label='True Performance', 
                       alpha=0.8, color=[model_colors[m] for m in models])
        bars2 = ax3.bar(x + width/2, estimated_perf, width, label='Estimated Performance', 
                       alpha=0.6, color=[model_colors[m] for m in models])
        
        ax3.set_title('True vs Estimated Performance', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Performance')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Confidence Intervals
        ax4 = fig.add_subplot(gs[1, 2:])
        
        for i, (model_name, arm) in enumerate(self.models.items()):
            ci_lower, ci_upper = arm.confidence_interval
            estimated = arm.estimated_performance
            true_perf = arm.true_performance
            
            # Plot confidence interval
            ax4.errorbar(i, estimated, yerr=[[estimated - ci_lower], [ci_upper - estimated]], 
                        fmt='o', color=model_colors[model_name], capsize=5, capthick=2, 
                        markersize=8, label=f'{model_name} (est)')
            
            # Plot true performance
            ax4.scatter(i, true_perf, color=model_colors[model_name], marker='x', 
                       s=100, linewidth=3, label=f'{model_name} (true)')
        
        ax4.set_title('Performance Estimates with Confidence Intervals', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Performance')
        ax4.set_xticks(range(len(models)))
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # 5. Reward Distribution
        ax5 = fig.add_subplot(gs[2, :2])
        
        # Calculate rewards by model
        model_rewards = defaultdict(list)
        for i, (selection, reward) in enumerate(zip(self.selection_history, self.reward_history)):
            model_rewards[selection].append(reward)
        
        # Create box plot
        reward_data = [model_rewards[model] for model in models]
        bp = ax5.boxplot(reward_data, labels=models, patch_artist=True)
        
        for patch, model in zip(bp['boxes'], models):
            patch.set_facecolor(model_colors[model])
            patch.set_alpha(0.7)
        
        ax5.set_title('Reward Distribution by Model', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Models')
        ax5.set_ylabel('Reward')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # 6. Selection Frequency
        ax6 = fig.add_subplot(gs[2, 2:])
        
        selection_counts = report['final_selection_distribution']
        models_sorted = sorted(selection_counts.keys(), key=lambda x: selection_counts[x], reverse=True)
        counts = [selection_counts[model] for model in models_sorted]
        
        wedges, texts, autotexts = ax6.pie(counts, labels=models_sorted, autopct='%1.1f%%',
                                          colors=[model_colors[m] for m in models_sorted],
                                          startangle=90)
        
        ax6.set_title('Final Selection Distribution', fontsize=14, fontweight='bold')
        
        # 7. Convergence to Optimal
        ax7 = fig.add_subplot(gs[3, :2])
        
        convergence_data = report['convergence_data']
        if convergence_data:
            rounds = [d['round'] for d in convergence_data]
            optimal_rates = [d['optimal_selection_rate'] for d in convergence_data]
            
            ax7.plot(rounds, optimal_rates, color='green', linewidth=2, label='Optimal Selection Rate')
            ax7.axhline(y=1.0, color='red', linestyle='--', label='Perfect Selection')
            ax7.fill_between(rounds, optimal_rates, alpha=0.3, color='green')
        
        ax7.set_title('Convergence to Optimal Model', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Round')
        ax7.set_ylabel('Optimal Selection Rate')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Summary Statistics
        ax8 = fig.add_subplot(gs[3, 2:])
        ax8.axis('off')
        
        # Create summary table
        summary_text = f"""
Multi-Armed Bandit Experiment Results

Total Rounds: {report['experiment_summary']['total_rounds']:,}
Total Reward: {report['experiment_summary']['total_reward']:.1f}
Average Reward: {report['experiment_summary']['average_reward']:.3f}
Cumulative Regret: {report['experiment_summary']['cumulative_regret']:.2f}

Best Model: {report['experiment_summary']['optimal_arm']}
True Performance: {report['experiment_summary']['optimal_performance']:.3f}

Final Selection Distribution:
"""
        
        for model, count in sorted(selection_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / sum(selection_counts.values()) * 100
            summary_text += f"  {model}: {percentage:.1f}% ({count:,} selections)\n"
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
        
        # Overall title
        fig.suptitle('Multi-Armed Bandit Experiment: Thompson Sampling for Model Selection', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mab_experiment_results_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Comprehensive visualization saved as: {filename}")
        
        return filename

def main():
    parser = argparse.ArgumentParser(description='Multi-Armed Bandit Simulator for Financial Models')
    parser.add_argument('--rounds', type=int, default=1000,
                       help='Number of experiment rounds')
    parser.add_argument('--algorithm', choices=['thompson', 'ucb', 'random'], 
                       default='thompson', help='Selection algorithm')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Define model configurations (true performance rates)
    model_configs = {
        'baseline-predictor': 0.65,      # 65% success rate
        'enhanced-predictor': 0.72,      # 72% success rate  
        'transformer-predictor': 0.78,   # 78% success rate
        'ensemble-predictor': 0.82       # 82% success rate (best)
    }
    
    print("🎯 Multi-Armed Bandit Simulator for Financial Models")
    print("=" * 60)
    print("Demonstrating Thompson Sampling for optimal model selection")
    print()
    
    # Initialize simulator
    simulator = FinancialMABSimulator(model_configs)
    
    # Run experiment
    results = simulator.run_experiment(args.rounds, args.algorithm)
    
    # Generate report
    report = simulator.generate_report()
    
    # Print results
    print("\n📊 Final Results:")
    print(f"   Total Reward: {report['experiment_summary']['total_reward']:.1f}")
    print(f"   Average Reward: {report['experiment_summary']['average_reward']:.3f}")
    print(f"   Cumulative Regret: {report['experiment_summary']['cumulative_regret']:.2f}")
    print(f"   Optimal Model: {report['experiment_summary']['optimal_arm']}")
    
    print("\n🏆 Model Performance:")
    for model_name, stats in report['model_statistics'].items():
        print(f"   {model_name}:")
        print(f"     True Performance: {stats['true_performance']:.3f}")
        print(f"     Estimated Performance: {stats['estimated_performance']:.3f}")
        print(f"     Selection Rate: {stats['selection_percentage']:.1f}%")
        print(f"     Total Pulls: {stats['total_pulls']}")
        print()
    
    # Create visualization
    if not args.no_viz:
        simulator.create_comprehensive_visualization(report)
    
    print("🎉 Multi-Armed Bandit Simulation Complete!")
    print("This demonstrates advanced model selection optimization in production MLOps")

if __name__ == "__main__":
    main()