#!/usr/bin/env python3
"""
Simulated A/B Testing Demonstration for Article/Blog Content

This creates a realistic demonstration of A/B testing results for 
documentation and article purposes, showing the types of insights
that would be generated in a production MLOps environment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from typing import Dict, List
import argparse

@dataclass
class SimulatedModelMetrics:
    name: str
    traffic_percentage: float
    avg_response_time: float
    p95_response_time: float
    accuracy: float
    prediction_confidence: float
    error_rate: float

class ABTestingSimulator:
    def __init__(self):
        # Realistic model performance characteristics
        self.baseline_model = SimulatedModelMetrics(
            name="baseline-predictor",
            traffic_percentage=70.0,
            avg_response_time=0.045,
            p95_response_time=0.089,
            accuracy=78.5,
            prediction_confidence=0.72,
            error_rate=1.2
        )
        
        self.enhanced_model = SimulatedModelMetrics(
            name="enhanced-predictor", 
            traffic_percentage=30.0,
            avg_response_time=0.062,
            p95_response_time=0.118,
            accuracy=82.1,
            prediction_confidence=0.79,
            error_rate=0.8
        )
    
    def simulate_traffic_distribution(self, total_requests: int) -> Dict:
        """Simulate realistic traffic distribution with some variance"""
        # Add realistic variance to traffic splitting
        baseline_requests = int(total_requests * (self.baseline_model.traffic_percentage / 100))
        enhanced_requests = total_requests - baseline_requests
        
        # Add some random variance (±5%)
        variance = int(total_requests * 0.05)
        baseline_requests += np.random.randint(-variance, variance)
        enhanced_requests = total_requests - baseline_requests
        
        return {
            'baseline-predictor': max(0, baseline_requests),
            'enhanced-predictor': max(0, enhanced_requests)
        }
    
    def generate_response_times(self, model: SimulatedModelMetrics, n_requests: int) -> List[float]:
        """Generate realistic response time distribution"""
        # Use log-normal distribution for realistic response times
        base_times = np.random.lognormal(
            mean=np.log(model.avg_response_time), 
            sigma=0.3, 
            size=n_requests
        )
        
        # Add some outliers (network issues, etc.)
        outlier_count = int(n_requests * 0.02)  # 2% outliers
        outlier_indices = np.random.choice(n_requests, outlier_count, replace=False)
        base_times[outlier_indices] *= np.random.uniform(3, 8, outlier_count)
        
        return base_times.tolist()
    
    def generate_predictions(self, model: SimulatedModelMetrics, n_requests: int) -> List[Dict]:
        """Generate realistic prediction results"""
        predictions = []
        
        for i in range(n_requests):
            # Generate prediction value (0-1 probability)
            pred_value = np.random.beta(
                model.prediction_confidence * 10, 
                (1 - model.prediction_confidence) * 10
            )
            
            # Generate actual outcome (for accuracy calculation)
            actual_outcome = np.random.choice([0, 1], p=[0.48, 0.52])  # Slightly bullish market
            
            # Determine if prediction is correct
            predicted_outcome = 1 if pred_value > 0.5 else 0
            is_correct = predicted_outcome == actual_outcome
            
            # Simulate market scenario
            scenarios = ['Bull Market', 'Bear Market', 'Sideways', 'High Volatility']
            scenario = np.random.choice(scenarios, p=[0.35, 0.2, 0.3, 0.15])
            
            predictions.append({
                'prediction': pred_value,
                'actual': actual_outcome,
                'correct': is_correct,
                'scenario': scenario,
                'confidence': model.prediction_confidence + np.random.normal(0, 0.05)
            })
        
        return predictions
    
    def simulate_ab_test(self, total_requests: int = 1000) -> Dict:
        """Simulate a complete A/B test"""
        print(f"🎯 Simulating A/B Test with {total_requests:,} requests")
        print("="*50)
        
        # Distribute traffic
        traffic_distribution = self.simulate_traffic_distribution(total_requests)
        
        results = {
            'metadata': {
                'total_requests': total_requests,
                'test_duration': '2 hours 15 minutes',
                'start_time': datetime.now() - timedelta(hours=2, minutes=15),
                'end_time': datetime.now(),
                'experiment_name': 'financial-ab-test-experiment'
            },
            'models': {}
        }
        
        for model in [self.baseline_model, self.enhanced_model]:
            n_requests = traffic_distribution[model.name]
            
            # Generate response times
            response_times = self.generate_response_times(model, n_requests)
            
            # Generate predictions
            predictions = self.generate_predictions(model, n_requests)
            
            # Calculate metrics
            successful_requests = int(n_requests * (1 - model.error_rate / 100))
            accuracy = np.mean([p['correct'] for p in predictions]) * 100
            
            model_results = {
                'name': model.name,
                'requests': n_requests,
                'successful_requests': successful_requests,
                'traffic_percentage': n_requests / total_requests * 100,
                'response_times': {
                    'mean': np.mean(response_times),
                    'median': np.median(response_times),
                    'p95': np.percentile(response_times, 95),
                    'p99': np.percentile(response_times, 99),
                    'min': np.min(response_times),
                    'max': np.max(response_times)
                },
                'accuracy': accuracy,
                'error_rate': model.error_rate,
                'predictions': {
                    'mean': np.mean([p['prediction'] for p in predictions]),
                    'std': np.std([p['prediction'] for p in predictions]),
                    'confidence': np.mean([p['confidence'] for p in predictions])
                },
                'scenarios': {
                    scenario: sum(1 for p in predictions if p['scenario'] == scenario)
                    for scenario in ['Bull Market', 'Bear Market', 'Sideways', 'High Volatility']
                },
                'raw_data': {
                    'response_times': response_times[:100],  # Sample for visualization
                    'predictions': [p['prediction'] for p in predictions[:100]]
                }
            }
            
            results['models'][model.name] = model_results
            
            print(f"📊 {model.name}:")
            print(f"   Requests: {n_requests:,} ({model_results['traffic_percentage']:.1f}% of traffic)")
            print(f"   Success Rate: {successful_requests/n_requests*100:.1f}%")
            print(f"   Avg Response Time: {model_results['response_times']['mean']:.3f}s")
            print(f"   P95 Response Time: {model_results['response_times']['p95']:.3f}s")
            print(f"   Accuracy: {accuracy:.1f}%")
            print()
        
        return results
    
    def calculate_business_impact(self, results: Dict) -> Dict:
        """Calculate business impact and recommendations"""
        baseline = results['models']['baseline-predictor']
        enhanced = results['models']['enhanced-predictor']
        
        # Performance differences
        accuracy_improvement = enhanced['accuracy'] - baseline['accuracy']
        latency_impact = enhanced['response_times']['mean'] - baseline['response_times']['mean']
        error_rate_improvement = baseline['error_rate'] - enhanced['error_rate']
        
        # Business calculations
        # Assuming 1% accuracy improvement = 0.5% revenue lift in trading
        revenue_impact = accuracy_improvement * 0.5
        
        # Cost impact (higher latency = higher infrastructure cost)
        cost_impact = latency_impact * 100  # 100ms = 1% cost increase (simplified)
        
        # Risk reduction (lower error rate = reduced risk)
        risk_reduction = error_rate_improvement * 10  # Error rate improvement multiplier
        
        business_impact = {
            'accuracy_improvement': accuracy_improvement,
            'latency_impact_ms': latency_impact * 1000,
            'error_rate_improvement': error_rate_improvement,
            'estimated_revenue_lift': revenue_impact,
            'estimated_cost_impact': cost_impact,
            'risk_reduction_score': risk_reduction,
            'net_business_value': revenue_impact - cost_impact + risk_reduction
        }
        
        # Generate recommendation
        if business_impact['net_business_value'] > 2:
            recommendation = "✅ STRONG RECOMMEND: Deploy enhanced model"
            confidence = "High"
        elif business_impact['net_business_value'] > 0.5:
            recommendation = "✅ RECOMMEND: Deploy enhanced model with monitoring"
            confidence = "Medium"
        elif business_impact['accuracy_improvement'] > 2:
            recommendation = "⚠️  CONDITIONAL: Consider enhanced model if latency acceptable"
            confidence = "Medium"
        else:
            recommendation = "❌ NOT RECOMMENDED: Continue with baseline model"
            confidence = "High"
        
        business_impact['recommendation'] = recommendation
        business_impact['confidence'] = confidence
        
        return business_impact
    
    def create_comprehensive_dashboard(self, results: Dict, business_impact: Dict) -> str:
        """Create a comprehensive A/B testing dashboard"""
        # Set up the figure with a professional style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        fig.patch.set_facecolor('white')
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
        
        # Color scheme
        colors = {
            'baseline-predictor': '#FF6B6B',
            'enhanced-predictor': '#4ECDC4'
        }
        
        # 1. Traffic Distribution (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        model_names = list(results['models'].keys())
        traffic_counts = [results['models'][model]['requests'] for model in model_names]
        model_labels = ['Baseline Model', 'Enhanced Model']
        
        wedges, texts, autotexts = ax1.pie(traffic_counts, labels=model_labels, autopct='%1.1f%%',
                                          colors=[colors[name] for name in model_names],
                                          startangle=90, textprops={'fontsize': 10})
        ax1.set_title('Traffic Distribution', fontsize=12, fontweight='bold', pad=20)
        
        # 2. Response Time Comparison (Top Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        models = list(results['models'].keys())
        response_data = []
        labels = []
        
        for model in models:
            response_data.append(results['models'][model]['raw_data']['response_times'])
            labels.append('Baseline' if 'baseline' in model else 'Enhanced')
        
        bp = ax2.boxplot(response_data, labels=labels, patch_artist=True)
        for patch, model in zip(bp['boxes'], models):
            patch.set_facecolor(colors[model])
            patch.set_alpha(0.7)
        
        ax2.set_title('Response Time Distribution', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Response Time (seconds)', fontsize=10)
        
        # 3. Accuracy Comparison (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        model_labels = ['Baseline', 'Enhanced']
        accuracies = [results['models']['baseline-predictor']['accuracy'],
                     results['models']['enhanced-predictor']['accuracy']]
        
        bars = ax3.bar(model_labels, accuracies, 
                      color=[colors['baseline-predictor'], colors['enhanced-predictor']],
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax3.set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Accuracy (%)', fontsize=10)
        ax3.set_ylim(0, 100)
        
        # 4. Business Impact Summary (Top Far Right)
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.axis('off')
        
        impact_text = f"""
Business Impact Analysis

Accuracy Improvement: {business_impact['accuracy_improvement']:+.1f}%
Latency Impact: {business_impact['latency_impact_ms']:+.1f}ms
Revenue Lift: {business_impact['estimated_revenue_lift']:+.1f}%
Risk Reduction: {business_impact['risk_reduction_score']:+.1f}%

Net Business Value: {business_impact['net_business_value']:+.1f}%

{business_impact['recommendation']}
Confidence: {business_impact['confidence']}
        """
        
        ax4.text(0.05, 0.95, impact_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='lightblue', alpha=0.3))
        
        # 5. Performance Metrics Heatmap (Second Row, Spans 2 columns)
        ax5 = fig.add_subplot(gs[1, :2])
        
        metrics_data = []
        metric_names = ['Accuracy', 'Speed (inv)', 'Reliability', 'Confidence']
        
        for model in models:
            model_data = results['models'][model]
            accuracy_norm = model_data['accuracy'] / 100
            speed_norm = 1 / (model_data['response_times']['mean'] * 10)  # Inverted and scaled
            reliability_norm = (100 - model_data['error_rate']) / 100
            confidence_norm = model_data['predictions']['confidence']
            
            metrics_data.append([accuracy_norm, speed_norm, reliability_norm, confidence_norm])
        
        im = ax5.imshow(metrics_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax5.set_xticks(range(len(metric_names)))
        ax5.set_xticklabels(metric_names, fontsize=10)
        ax5.set_yticks(range(len(models)))
        ax5.set_yticklabels(['Baseline', 'Enhanced'], fontsize=10)
        ax5.set_title('Performance Metrics Heatmap', fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(metric_names)):
                text = ax5.text(j, i, f'{metrics_data[i][j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax5, label='Performance Score')
        
        # 6. Scenario Performance (Second Row, Right Side)
        ax6 = fig.add_subplot(gs[1, 2:])
        
        scenarios = ['Bull Market', 'Bear Market', 'Sideways', 'High Volatility']
        baseline_scenario_counts = [results['models']['baseline-predictor']['scenarios'][s] for s in scenarios]
        enhanced_scenario_counts = [results['models']['enhanced-predictor']['scenarios'][s] for s in scenarios]
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        ax6.bar(x - width/2, baseline_scenario_counts, width, label='Baseline', 
               color=colors['baseline-predictor'], alpha=0.8)
        ax6.bar(x + width/2, enhanced_scenario_counts, width, label='Enhanced',
               color=colors['enhanced-predictor'], alpha=0.8)
        
        ax6.set_xlabel('Market Scenarios', fontsize=10)
        ax6.set_ylabel('Number of Requests', fontsize=10)
        ax6.set_title('Performance by Market Scenario', fontsize=12, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(scenarios, rotation=45, ha='right')
        ax6.legend()
        
        # 7. Time Series of Predictions (Third Row, Full Width)
        ax7 = fig.add_subplot(gs[2, :])
        
        # Simulate time series data
        time_points = pd.date_range(start=results['metadata']['start_time'], 
                                   end=results['metadata']['end_time'], periods=100)
        
        baseline_predictions = results['models']['baseline-predictor']['raw_data']['predictions']
        enhanced_predictions = results['models']['enhanced-predictor']['raw_data']['predictions']
        
        ax7.plot(time_points[:len(baseline_predictions)], baseline_predictions, 
                'o-', label='Baseline Model', color=colors['baseline-predictor'], alpha=0.7, markersize=3)
        ax7.plot(time_points[:len(enhanced_predictions)], enhanced_predictions,
                's-', label='Enhanced Model', color=colors['enhanced-predictor'], alpha=0.7, markersize=3)
        
        ax7.set_xlabel('Time', fontsize=10)
        ax7.set_ylabel('Prediction Probability', fontsize=10)
        ax7.set_title('Model Predictions Over Time', fontsize=12, fontweight='bold')
        ax7.legend()
        ax7.tick_params(axis='x', rotation=45)
        
        # 8. Statistical Summary (Bottom Row)
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        
        # Create summary table
        summary_data = []
        for model_name in models:
            model = results['models'][model_name]
            display_name = 'Baseline Model' if 'baseline' in model_name else 'Enhanced Model'
            summary_data.append([
                display_name,
                f"{model['requests']:,}",
                f"{model['traffic_percentage']:.1f}%",
                f"{model['response_times']['mean']:.3f}s",
                f"{model['response_times']['p95']:.3f}s",
                f"{model['accuracy']:.1f}%",
                f"{model['error_rate']:.1f}%"
            ])
        
        columns = ['Model', 'Requests', 'Traffic %', 'Avg Latency', 'P95 Latency', 'Accuracy', 'Error Rate']
        
        table = ax8.table(cellText=summary_data, colLabels=columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2)
        
        # Style the table
        table.auto_set_column_width(col=list(range(len(columns))))
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code the rows
        for i in range(1, len(summary_data) + 1):
            model_name = list(models)[i-1]
            for j in range(len(columns)):
                table[(i, j)].set_facecolor(colors[model_name])
                table[(i, j)].set_alpha(0.3)
        
        ax8.set_title('Statistical Summary', fontsize=12, fontweight='bold', pad=20)
        
        # Overall title
        fig.suptitle('Financial MLOps A/B Testing Dashboard\nComprehensive Performance Analysis', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Add timestamp
        timestamp_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        fig.text(0.99, 0.01, timestamp_text, ha='right', va='bottom', fontsize=8, alpha=0.7)
        
        # Save the dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"financial_ab_testing_dashboard_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Dashboard saved as: {filename}")
        
        return filename
    
    def generate_executive_summary(self, results: Dict, business_impact: Dict) -> str:
        """Generate an executive summary for stakeholders"""
        summary = f"""
        
📋 EXECUTIVE SUMMARY: A/B Testing Results
{'='*50}

🎯 TEST OVERVIEW:
   Duration: {results['metadata']['test_duration']}
   Total Requests: {results['metadata']['total_requests']:,}
   Models Tested: Baseline vs Enhanced Financial Predictor

📊 KEY FINDINGS:
   • Enhanced model achieved {business_impact['accuracy_improvement']:+.1f}% accuracy improvement
   • Latency impact: {business_impact['latency_impact_ms']:+.1f}ms ({business_impact['latency_impact_ms']/results['models']['baseline-predictor']['response_times']['mean']/10:+.1f}%)
   • Error rate reduced by {business_impact['error_rate_improvement']:.1f} percentage points
   • Estimated revenue lift: {business_impact['estimated_revenue_lift']:+.1f}%

💼 BUSINESS IMPACT:
   Net Business Value: {business_impact['net_business_value']:+.1f}%
   Risk Reduction Score: {business_impact['risk_reduction_score']:+.1f}%
   
🎯 RECOMMENDATION:
   {business_impact['recommendation']}
   Confidence Level: {business_impact['confidence']}

📈 NEXT STEPS:
   1. {"Deploy enhanced model to full production traffic" if business_impact['net_business_value'] > 1 else "Extended A/B testing recommended"}
   2. Monitor key performance indicators for 2 weeks
   3. Measure actual business impact vs predictions
   4. Document lessons learned for future experiments
        """
        
        print(summary)
        return summary

def main():
    parser = argparse.ArgumentParser(description='Simulated A/B Testing Demo for Articles')
    parser.add_argument('--requests', type=int, default=1000, help='Total requests to simulate')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization generation')
    
    args = parser.parse_args()
    
    print("🎭 Financial MLOps A/B Testing Simulation")
    print("   (Realistic demonstration for article/documentation)")
    print("="*60)
    
    # Initialize simulator
    simulator = ABTestingSimulator()
    
    # Run simulation
    results = simulator.simulate_ab_test(args.requests)
    
    # Calculate business impact
    business_impact = simulator.calculate_business_impact(results)
    
    print("💼 Business Impact Analysis:")
    print(f"   Accuracy Improvement: {business_impact['accuracy_improvement']:+.1f} percentage points")
    print(f"   Latency Impact: {business_impact['latency_impact_ms']:+.1f}ms")
    print(f"   Estimated Revenue Lift: {business_impact['estimated_revenue_lift']:+.1f}%")
    print(f"   Net Business Value: {business_impact['net_business_value']:+.1f}%")
    print(f"   Recommendation: {business_impact['recommendation']}")
    print()
    
    # Generate dashboard
    if not args.no_viz:
        simulator.create_comprehensive_dashboard(results, business_impact)
    
    # Generate executive summary
    simulator.generate_executive_summary(results, business_impact)
    
    print("🎉 Simulation Complete!")
    print("   Perfect for demonstrating A/B testing concepts in articles/presentations")

if __name__ == "__main__":
    main()