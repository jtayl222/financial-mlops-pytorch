#!/usr/bin/env python3
"""
Contextual Routing Demo for Financial Models
Demonstrates intelligent model selection based on market conditions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import argparse
import requests
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MarketCondition:
    """Market condition data structure"""
    timestamp: datetime
    volatility: float
    trend: float
    volume: float
    sentiment: float
    vix: float
    condition_type: str
    confidence: float

@dataclass
class RoutingDecision:
    """Routing decision data structure"""
    timestamp: datetime
    market_condition: MarketCondition
    selected_model: str
    confidence: float
    routing_reason: str
    alternative_models: List[str]
    performance_expected: float

class ContextualRouter:
    """Intelligent model router based on market conditions"""
    
    def __init__(self):
        # Model capabilities for different market conditions
        self.model_capabilities = {
            'robust-predictor': {
                'volatility_performance': {'high': 0.85, 'medium': 0.75, 'low': 0.65},
                'trend_performance': {'bull': 0.70, 'bear': 0.80, 'sideways': 0.85},
                'volume_performance': {'high': 0.80, 'medium': 0.75, 'low': 0.70},
                'description': 'Optimized for high volatility and uncertain conditions',
                'strengths': ['High volatility', 'Market stress', 'Rapid changes'],
                'weaknesses': ['Low volatility', 'Trending markets']
            },
            'aggressive-predictor': {
                'volatility_performance': {'high': 0.60, 'medium': 0.75, 'low': 0.85},
                'trend_performance': {'bull': 0.90, 'bear': 0.65, 'sideways': 0.70},
                'volume_performance': {'high': 0.85, 'medium': 0.80, 'low': 0.75},
                'description': 'Optimized for bull markets and trending conditions',
                'strengths': ['Bull markets', 'Strong trends', 'High volume'],
                'weaknesses': ['Bear markets', 'High volatility']
            },
            'conservative-predictor': {
                'volatility_performance': {'high': 0.70, 'medium': 0.80, 'low': 0.75},
                'trend_performance': {'bull': 0.65, 'bear': 0.85, 'sideways': 0.80},
                'volume_performance': {'high': 0.75, 'medium': 0.80, 'low': 0.85},
                'description': 'Optimized for bear markets and risk-averse scenarios',
                'strengths': ['Bear markets', 'Risk management', 'Stable conditions'],
                'weaknesses': ['Bull markets', 'High growth scenarios']
            },
            'baseline-predictor': {
                'volatility_performance': {'high': 0.65, 'medium': 0.75, 'low': 0.80},
                'trend_performance': {'bull': 0.75, 'bear': 0.75, 'sideways': 0.85},
                'volume_performance': {'high': 0.70, 'medium': 0.75, 'low': 0.80},
                'description': 'Balanced model for general market conditions',
                'strengths': ['Sideways markets', 'Balanced conditions', 'Stability'],
                'weaknesses': ['Extreme conditions', 'Specialized scenarios']
            }
        }
        
        # Routing thresholds
        self.routing_thresholds = {
            'volatility': {'high': 0.30, 'medium': 0.20},
            'trend': {'bull': 0.02, 'bear': -0.02},
            'volume': {'high': 1.5, 'medium': 1.0},  # Multiplier of average
            'vix': {'high': 25, 'medium': 20},
            'sentiment': {'positive': 0.6, 'negative': 0.4}
        }
        
        # Performance tracking
        self.routing_history = []
        self.performance_history = []
        self.model_usage_stats = defaultdict(int)
        
        print("🧠 Contextual Router Initialized")
        print(f"   📊 Models available: {len(self.model_capabilities)}")
        print(f"   🎯 Routing strategies: Market condition-based")
    
    def analyze_market_condition(self, features: Dict[str, float]) -> MarketCondition:
        """Analyze current market conditions from features"""
        
        # Extract key market indicators
        volatility = features.get('volatility', 0.15)
        returns = features.get('returns', 0.001)
        volume = features.get('volume', 1000000)
        vix = features.get('vix', 20)
        sentiment = features.get('news_sentiment', 0.5)
        
        # Calculate trend (simplified momentum indicator)
        trend = returns * 10  # Scale for easier interpretation
        
        # Normalize volume (assume average volume of 1M)
        volume_normalized = volume / 1000000
        
        # Determine market condition type
        condition_type = self._classify_market_condition(volatility, trend, volume_normalized, vix, sentiment)
        
        # Calculate confidence in condition classification
        confidence = self._calculate_condition_confidence(volatility, trend, vix, sentiment)
        
        return MarketCondition(
            timestamp=datetime.now(),
            volatility=volatility,
            trend=trend,
            volume=volume_normalized,
            sentiment=sentiment,
            vix=vix,
            condition_type=condition_type,
            confidence=confidence
        )
    
    def _classify_market_condition(self, volatility: float, trend: float, volume: float, 
                                 vix: float, sentiment: float) -> str:
        """Classify market condition based on indicators"""
        
        # High volatility takes precedence
        if volatility > self.routing_thresholds['volatility']['high'] or vix > self.routing_thresholds['vix']['high']:
            return 'high_volatility'
        
        # Strong trends
        if trend > self.routing_thresholds['trend']['bull']:
            return 'bull_market'
        elif trend < self.routing_thresholds['trend']['bear']:
            return 'bear_market'
        
        # Medium volatility with mixed signals
        if volatility > self.routing_thresholds['volatility']['medium']:
            return 'mixed_volatility'
        
        # Default to sideways/stable
        return 'sideways'
    
    def _calculate_condition_confidence(self, volatility: float, trend: float, 
                                      vix: float, sentiment: float) -> float:
        """Calculate confidence in market condition classification"""
        
        confidence_factors = []
        
        # Volatility confidence
        if volatility > 0.4 or volatility < 0.1:
            confidence_factors.append(0.9)  # Very clear signal
        elif volatility > 0.3 or volatility < 0.15:
            confidence_factors.append(0.7)  # Clear signal
        else:
            confidence_factors.append(0.5)  # Moderate signal
        
        # Trend confidence
        if abs(trend) > 0.05:
            confidence_factors.append(0.9)  # Strong trend
        elif abs(trend) > 0.02:
            confidence_factors.append(0.7)  # Moderate trend
        else:
            confidence_factors.append(0.5)  # Weak trend
        
        # VIX confidence
        if vix > 30 or vix < 15:
            confidence_factors.append(0.8)  # Clear fear/complacency
        else:
            confidence_factors.append(0.6)  # Moderate signal
        
        # Sentiment confidence
        if sentiment > 0.7 or sentiment < 0.3:
            confidence_factors.append(0.8)  # Strong sentiment
        else:
            confidence_factors.append(0.6)  # Moderate sentiment
        
        return np.mean(confidence_factors)
    
    def route_request(self, market_condition: MarketCondition) -> RoutingDecision:
        """Route request to optimal model based on market condition"""
        
        # Calculate performance scores for each model
        model_scores = {}
        
        for model_name, capabilities in self.model_capabilities.items():
            score = self._calculate_model_score(model_name, market_condition)
            model_scores[model_name] = score
        
        # Select best model
        best_model = max(model_scores.keys(), key=lambda k: model_scores[k])
        best_score = model_scores[best_model]
        
        # Get alternative models (sorted by score)
        alternatives = sorted(
            [m for m in model_scores.keys() if m != best_model],
            key=lambda k: model_scores[k],
            reverse=True
        )
        
        # Generate routing reason
        routing_reason = self._generate_routing_reason(best_model, market_condition)
        
        # Calculate confidence in routing decision
        confidence = self._calculate_routing_confidence(model_scores, best_model)
        
        decision = RoutingDecision(
            timestamp=datetime.now(),
            market_condition=market_condition,
            selected_model=best_model,
            confidence=confidence,
            routing_reason=routing_reason,
            alternative_models=alternatives,
            performance_expected=best_score
        )
        
        # Track routing decision
        self.routing_history.append(decision)
        self.model_usage_stats[best_model] += 1
        
        return decision
    
    def _calculate_model_score(self, model_name: str, condition: MarketCondition) -> float:
        """Calculate model performance score for given market condition"""
        
        capabilities = self.model_capabilities[model_name]
        
        # Volatility score
        if condition.volatility > self.routing_thresholds['volatility']['high']:
            vol_score = capabilities['volatility_performance']['high']
        elif condition.volatility > self.routing_thresholds['volatility']['medium']:
            vol_score = capabilities['volatility_performance']['medium']
        else:
            vol_score = capabilities['volatility_performance']['low']
        
        # Trend score
        if condition.trend > self.routing_thresholds['trend']['bull']:
            trend_score = capabilities['trend_performance']['bull']
        elif condition.trend < self.routing_thresholds['trend']['bear']:
            trend_score = capabilities['trend_performance']['bear']
        else:
            trend_score = capabilities['trend_performance']['sideways']
        
        # Volume score
        if condition.volume > self.routing_thresholds['volume']['high']:
            volume_score = capabilities['volume_performance']['high']
        elif condition.volume > self.routing_thresholds['volume']['medium']:
            volume_score = capabilities['volume_performance']['medium']
        else:
            volume_score = capabilities['volume_performance']['low']
        
        # Weighted combination
        weights = {'volatility': 0.4, 'trend': 0.35, 'volume': 0.25}
        
        total_score = (
            weights['volatility'] * vol_score +
            weights['trend'] * trend_score +
            weights['volume'] * volume_score
        )
        
        # Apply confidence multiplier
        total_score *= condition.confidence
        
        return total_score
    
    def _generate_routing_reason(self, model_name: str, condition: MarketCondition) -> str:
        """Generate human-readable routing reason"""
        
        model_info = self.model_capabilities[model_name]
        
        # Primary reason based on condition type
        if condition.condition_type == 'high_volatility':
            primary_reason = f"High volatility ({condition.volatility:.3f}) detected"
        elif condition.condition_type == 'bull_market':
            primary_reason = f"Bull market trend ({condition.trend:.3f}) identified"
        elif condition.condition_type == 'bear_market':
            primary_reason = f"Bear market trend ({condition.trend:.3f}) identified"
        else:
            primary_reason = f"Sideways market ({condition.condition_type}) detected"
        
        # Model strength
        model_strength = model_info['description']
        
        return f"{primary_reason}. {model_strength}."
    
    def _calculate_routing_confidence(self, model_scores: Dict[str, float], 
                                    best_model: str) -> float:
        """Calculate confidence in routing decision"""
        
        scores = list(model_scores.values())
        best_score = model_scores[best_model]
        
        # If best score is significantly higher than others, high confidence
        second_best = sorted(scores, reverse=True)[1]
        score_gap = best_score - second_best
        
        if score_gap > 0.15:
            return 0.9  # High confidence
        elif score_gap > 0.08:
            return 0.7  # Medium confidence
        else:
            return 0.5  # Low confidence (close scores)
    
    def generate_market_scenarios(self, n_scenarios: int = 100) -> List[Dict[str, float]]:
        """Generate diverse market scenarios for testing"""
        
        scenarios = []
        
        # Scenario types with different probabilities
        scenario_types = [
            ('bull_market', 0.3),
            ('bear_market', 0.2),
            ('high_volatility', 0.2),
            ('sideways', 0.3)
        ]
        
        for i in range(n_scenarios):
            # Select scenario type
            scenario_type = np.random.choice(
                [s[0] for s in scenario_types],
                p=[s[1] for s in scenario_types]
            )
            
            # Generate scenario based on type
            if scenario_type == 'bull_market':
                scenario = {
                    'volatility': np.random.normal(0.15, 0.05),
                    'returns': np.random.normal(0.025, 0.01),
                    'volume': np.random.lognormal(15.5, 0.3),
                    'vix': np.random.normal(18, 3),
                    'news_sentiment': np.random.normal(0.65, 0.1),
                    'rsi': np.random.normal(65, 10)
                }
            elif scenario_type == 'bear_market':
                scenario = {
                    'volatility': np.random.normal(0.25, 0.05),
                    'returns': np.random.normal(-0.02, 0.01),
                    'volume': np.random.lognormal(15.3, 0.4),
                    'vix': np.random.normal(28, 5),
                    'news_sentiment': np.random.normal(0.35, 0.1),
                    'rsi': np.random.normal(35, 10)
                }
            elif scenario_type == 'high_volatility':
                scenario = {
                    'volatility': np.random.normal(0.4, 0.1),
                    'returns': np.random.normal(0.002, 0.02),
                    'volume': np.random.lognormal(15.7, 0.5),
                    'vix': np.random.normal(35, 8),
                    'news_sentiment': np.random.normal(0.45, 0.15),
                    'rsi': np.random.normal(50, 15)
                }
            else:  # sideways
                scenario = {
                    'volatility': np.random.normal(0.12, 0.03),
                    'returns': np.random.normal(0.001, 0.005),
                    'volume': np.random.lognormal(15.0, 0.2),
                    'vix': np.random.normal(16, 2),
                    'news_sentiment': np.random.normal(0.50, 0.08),
                    'rsi': np.random.normal(50, 8)
                }
            
            # Ensure realistic bounds
            scenario['volatility'] = np.clip(scenario['volatility'], 0.05, 0.8)
            scenario['volume'] = max(scenario['volume'], 100000)
            scenario['vix'] = np.clip(scenario['vix'], 10, 80)
            scenario['news_sentiment'] = np.clip(scenario['news_sentiment'], 0, 1)
            scenario['rsi'] = np.clip(scenario['rsi'], 0, 100)
            
            # Add metadata
            scenario['scenario_type'] = scenario_type
            scenario['timestamp'] = datetime.now() + timedelta(minutes=i)
            
            scenarios.append(scenario)
        
        return scenarios
    
    def run_routing_simulation(self, scenarios: List[Dict[str, float]]) -> Dict:
        """Run contextual routing simulation across multiple scenarios"""
        
        print("🎯 Running Contextual Routing Simulation")
        print(f"   📊 Scenarios: {len(scenarios)}")
        print(f"   🧠 Models: {list(self.model_capabilities.keys())}")
        print()
        
        results = {
            'scenarios_processed': 0,
            'routing_decisions': [],
            'model_usage': defaultdict(int),
            'performance_by_condition': defaultdict(list),
            'routing_accuracy': defaultdict(list),
            'confidence_distribution': []
        }
        
        for i, scenario in enumerate(scenarios):
            # Analyze market condition
            condition = self.analyze_market_condition(scenario)
            
            # Make routing decision
            decision = self.route_request(condition)
            
            # Store results
            results['scenarios_processed'] += 1
            results['routing_decisions'].append(decision)
            results['model_usage'][decision.selected_model] += 1
            results['performance_by_condition'][condition.condition_type].append(decision.performance_expected)
            results['confidence_distribution'].append(decision.confidence)
            
            # Print progress
            if (i + 1) % 20 == 0:
                print(f"   Progress: {i+1}/{len(scenarios)} scenarios processed")
                
                # Show recent routing decisions
                recent_decisions = results['routing_decisions'][-10:]
                model_counts = defaultdict(int)
                for d in recent_decisions:
                    model_counts[d.selected_model] += 1
                
                print(f"   Recent routing (last 10): {dict(model_counts)}")
                print()
        
        # Calculate summary statistics
        results['summary'] = self._calculate_simulation_summary(results)
        
        print(f"✅ Simulation complete! Processed {results['scenarios_processed']} scenarios")
        
        return results
    
    def _calculate_simulation_summary(self, results: Dict) -> Dict:
        """Calculate summary statistics for simulation"""
        
        total_scenarios = results['scenarios_processed']
        
        # Model usage percentages
        model_usage_pct = {
            model: (count / total_scenarios) * 100
            for model, count in results['model_usage'].items()
        }
        
        # Average performance by condition
        avg_performance_by_condition = {
            condition: np.mean(performances)
            for condition, performances in results['performance_by_condition'].items()
        }
        
        # Confidence statistics
        confidence_stats = {
            'mean': np.mean(results['confidence_distribution']),
            'std': np.std(results['confidence_distribution']),
            'min': np.min(results['confidence_distribution']),
            'max': np.max(results['confidence_distribution'])
        }
        
        # Routing efficiency (how often optimal model was selected)
        routing_efficiency = self._calculate_routing_efficiency(results['routing_decisions'])
        
        return {
            'total_scenarios': total_scenarios,
            'model_usage_percentage': model_usage_pct,
            'avg_performance_by_condition': avg_performance_by_condition,
            'confidence_statistics': confidence_stats,
            'routing_efficiency': routing_efficiency,
            'unique_conditions_encountered': len(results['performance_by_condition'])
        }
    
    def _calculate_routing_efficiency(self, decisions: List[RoutingDecision]) -> float:
        """Calculate how efficiently the router selected optimal models"""
        
        # This is a simplified efficiency calculation
        # In practice, you'd compare against ground truth performance
        
        high_confidence_decisions = [d for d in decisions if d.confidence > 0.7]
        efficiency = len(high_confidence_decisions) / len(decisions) * 100
        
        return efficiency
    
    def create_routing_visualization(self, results: Dict) -> str:
        """Create comprehensive visualization of routing results"""
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Color scheme
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#F7B801']
        model_colors = {model: colors[i] for i, model in enumerate(self.model_capabilities.keys())}
        
        # 1. Model Usage Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        
        model_usage = results['summary']['model_usage_percentage']
        models = list(model_usage.keys())
        usage_pct = list(model_usage.values())
        
        wedges, texts, autotexts = ax1.pie(usage_pct, labels=models, autopct='%1.1f%%',
                                          colors=[model_colors[m] for m in models],
                                          startangle=90)
        ax1.set_title('Model Usage Distribution', fontsize=14, fontweight='bold')
        
        # 2. Performance by Market Condition
        ax2 = fig.add_subplot(gs[0, 1:])
        
        conditions = list(results['performance_by_condition'].keys())
        condition_performances = [results['performance_by_condition'][c] for c in conditions]
        
        bp = ax2.boxplot(condition_performances, labels=conditions, patch_artist=True)
        
        for patch, condition in zip(bp['boxes'], conditions):
            patch.set_facecolor(colors[hash(condition) % len(colors)])
            patch.set_alpha(0.7)
        
        ax2.set_title('Model Performance by Market Condition', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Market Condition')
        ax2.set_ylabel('Expected Performance')
        ax2.grid(True, alpha=0.3)
        
        # 3. Routing Confidence Distribution
        ax3 = fig.add_subplot(gs[1, 0])
        
        confidence_dist = results['confidence_distribution']
        ax3.hist(confidence_dist, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(np.mean(confidence_dist), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidence_dist):.3f}')
        ax3.set_title('Routing Confidence Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Confidence Score')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Routing Timeline
        ax4 = fig.add_subplot(gs[1, 1:])
        
        # Sample routing decisions over time
        sample_decisions = results['routing_decisions'][::max(1, len(results['routing_decisions'])//50)]
        
        timestamps = [d.timestamp for d in sample_decisions]
        models = [d.selected_model for d in sample_decisions]
        
        # Create timeline plot
        model_to_y = {model: i for i, model in enumerate(self.model_capabilities.keys())}
        y_values = [model_to_y[model] for model in models]
        
        ax4.scatter(timestamps, y_values, c=[model_colors[m] for m in models], 
                   s=60, alpha=0.7)
        ax4.set_yticks(list(model_to_y.values()))
        ax4.set_yticklabels(list(model_to_y.keys()))
        ax4.set_title('Model Selection Timeline', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Selected Model')
        ax4.grid(True, alpha=0.3)
        
        # 5. Market Condition Analysis
        ax5 = fig.add_subplot(gs[2, :2])
        
        # Analyze market conditions from decisions
        condition_counts = defaultdict(int)
        for decision in results['routing_decisions']:
            condition_counts[decision.market_condition.condition_type] += 1
        
        conditions = list(condition_counts.keys())
        counts = list(condition_counts.values())
        
        bars = ax5.bar(conditions, counts, color=colors[:len(conditions)], alpha=0.7)
        ax5.set_title('Market Condition Distribution', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Market Condition Type')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 6. Routing Efficiency Metrics
        ax6 = fig.add_subplot(gs[2, 2])
        
        efficiency_metrics = {
            'Routing Efficiency': results['summary']['routing_efficiency'],
            'Avg Confidence': results['summary']['confidence_statistics']['mean'] * 100,
            'Condition Coverage': (results['summary']['unique_conditions_encountered'] / 6) * 100
        }
        
        metrics = list(efficiency_metrics.keys())
        values = list(efficiency_metrics.values())
        
        bars = ax6.bar(metrics, values, color=['green', 'blue', 'orange'], alpha=0.7)
        ax6.set_title('Routing Efficiency Metrics', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Percentage (%)')
        ax6.set_ylim(0, 100)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 7. Model Capabilities Heatmap
        ax7 = fig.add_subplot(gs[3, :2])
        
        # Create heatmap of model capabilities
        capabilities_matrix = []
        capability_labels = []
        
        for model_name, capabilities in self.model_capabilities.items():
            row = []
            if not capability_labels:  # First iteration
                for perf_type, perf_dict in capabilities.items():
                    if isinstance(perf_dict, dict) and 'high' in perf_dict:
                        for condition in ['high', 'medium', 'low']:
                            capability_labels.append(f"{perf_type}_{condition}")
                            row.append(perf_dict[condition])
                    elif isinstance(perf_dict, dict) and 'bull' in perf_dict:
                        for condition in ['bull', 'bear', 'sideways']:
                            capability_labels.append(f"{perf_type}_{condition}")
                            row.append(perf_dict[condition])
            else:
                for perf_type, perf_dict in capabilities.items():
                    if isinstance(perf_dict, dict) and 'high' in perf_dict:
                        for condition in ['high', 'medium', 'low']:
                            row.append(perf_dict[condition])
                    elif isinstance(perf_dict, dict) and 'bull' in perf_dict:
                        for condition in ['bull', 'bear', 'sideways']:
                            row.append(perf_dict[condition])
            
            capabilities_matrix.append(row)
        
        if capabilities_matrix:
            heatmap = ax7.imshow(capabilities_matrix, cmap='RdYlGn', aspect='auto')
            ax7.set_xticks(range(len(capability_labels)))
            ax7.set_xticklabels(capability_labels, rotation=45, ha='right')
            ax7.set_yticks(range(len(self.model_capabilities)))
            ax7.set_yticklabels(list(self.model_capabilities.keys()))
            ax7.set_title('Model Capabilities Heatmap', fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(heatmap, ax=ax7, shrink=0.6)
            cbar.set_label('Performance Score')
        
        # 8. Summary Statistics
        ax8 = fig.add_subplot(gs[3, 2])
        ax8.axis('off')
        
        summary = results['summary']
        
        summary_text = f"""
Contextual Routing Results

Scenarios Processed: {summary['total_scenarios']:,}
Routing Efficiency: {summary['routing_efficiency']:.1f}%
Avg Confidence: {summary['confidence_statistics']['mean']:.3f}

Model Usage:
• Robust: {summary['model_usage_percentage'].get('robust-predictor', 0):.1f}%
• Aggressive: {summary['model_usage_percentage'].get('aggressive-predictor', 0):.1f}%
• Conservative: {summary['model_usage_percentage'].get('conservative-predictor', 0):.1f}%
• Baseline: {summary['model_usage_percentage'].get('baseline-predictor', 0):.1f}%

Performance Insights:
• High confidence decisions: {len([d for d in results['routing_decisions'] if d.confidence > 0.7])}/{summary['total_scenarios']}
• Conditions encountered: {summary['unique_conditions_encountered']}
• Avg performance: {np.mean([d.performance_expected for d in results['routing_decisions']]):.3f}

System demonstrates intelligent model 
selection based on market conditions,
optimizing performance through context-
aware routing decisions.
        """
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
        
        # Overall title
        fig.suptitle('Contextual Routing Analysis: AI-Driven Model Selection', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"contextual_routing_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Contextual routing visualization saved as: {filename}")
        
        return filename
    
    def simulate_live_routing(self, duration_minutes: int = 5, 
                            requests_per_minute: int = 20) -> Dict:
        """Simulate live routing with real-time market conditions"""
        
        print(f"🎯 Starting Live Routing Simulation")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   Rate: {requests_per_minute} requests/minute")
        print()
        
        live_results = {
            'start_time': datetime.now(),
            'routing_decisions': [],
            'market_conditions': [],
            'performance_tracking': []
        }
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        request_count = 0
        
        while datetime.now() < end_time:
            # Generate realistic market scenario
            scenarios = self.generate_market_scenarios(1)
            scenario = scenarios[0]
            
            # Analyze market condition
            condition = self.analyze_market_condition(scenario)
            
            # Make routing decision
            decision = self.route_request(condition)
            
            # Store results
            live_results['routing_decisions'].append(decision)
            live_results['market_conditions'].append(condition)
            
            # Simulate performance feedback (in real system, this would be actual model performance)
            simulated_performance = decision.performance_expected + np.random.normal(0, 0.05)
            live_results['performance_tracking'].append({
                'timestamp': datetime.now(),
                'expected_performance': decision.performance_expected,
                'actual_performance': simulated_performance,
                'model': decision.selected_model,
                'condition': condition.condition_type
            })
            
            request_count += 1
            
            # Print progress
            if request_count % 10 == 0:
                print(f"   ⏰ {request_count} requests processed")
                print(f"   📊 Latest: {decision.selected_model} for {condition.condition_type}")
                print(f"   🎯 Confidence: {decision.confidence:.3f}")
                print()
            
            # Wait before next request
            time.sleep(60 / requests_per_minute)
        
        live_results['end_time'] = datetime.now()
        live_results['total_requests'] = request_count
        
        print(f"✅ Live simulation complete! {request_count} requests processed")
        
        return live_results

def main():
    parser = argparse.ArgumentParser(description='Contextual Routing Demo for Financial Models')
    parser.add_argument('--mode', choices=['simulation', 'live'], default='simulation',
                       help='Run simulation or live demo')
    parser.add_argument('--scenarios', type=int, default=100,
                       help='Number of scenarios for simulation')
    parser.add_argument('--duration', type=int, default=5,
                       help='Duration in minutes for live demo')
    parser.add_argument('--rate', type=int, default=20,
                       help='Requests per minute for live demo')
    parser.add_argument('--market-conditions', type=str,
                       help='Focus on specific market conditions')
    parser.add_argument('--output-json', type=str,
                       help='Save results to JSON file')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    print("🧠 Contextual Routing Demo for Financial Models")
    print("=" * 60)
    print("Demonstrating AI-driven model selection based on market conditions")
    print()
    
    # Initialize contextual router
    router = ContextualRouter()
    
    if args.mode == 'simulation':
        # Run simulation mode
        print("🎯 Running Simulation Mode")
        scenarios = router.generate_market_scenarios(args.scenarios)
        
        # Filter scenarios if specific conditions requested
        if args.market_conditions:
            scenarios = [s for s in scenarios if s.get('scenario_type') == args.market_conditions]
            print(f"   📊 Filtered to {len(scenarios)} {args.market_conditions} scenarios")
        
        results = router.run_routing_simulation(scenarios)
        
        # Display results
        print(f"\n📊 Simulation Results:")
        print(f"   Total Scenarios: {results['summary']['total_scenarios']}")
        print(f"   Routing Efficiency: {results['summary']['routing_efficiency']:.1f}%")
        print(f"   Average Confidence: {results['summary']['confidence_statistics']['mean']:.3f}")
        
        print(f"\n🎯 Model Usage:")
        for model, pct in results['summary']['model_usage_percentage'].items():
            print(f"   {model}: {pct:.1f}%")
        
        print(f"\n🏆 Performance by Condition:")
        for condition, perf in results['summary']['avg_performance_by_condition'].items():
            print(f"   {condition}: {perf:.3f}")
        
        # Create visualization
        if not args.no_viz:
            print(f"\n📊 Creating routing visualization...")
            viz_filename = router.create_routing_visualization(results)
        
        # Save results
        if args.output_json:
            # Convert to JSON-serializable format
            json_results = json.loads(json.dumps(results, default=str))
            with open(args.output_json, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"\n💾 Results saved to: {args.output_json}")
    
    else:
        # Run live demo mode
        print("🎯 Running Live Demo Mode")
        live_results = router.simulate_live_routing(args.duration, args.rate)
        
        # Display live results
        print(f"\n📊 Live Demo Results:")
        print(f"   Duration: {(live_results['end_time'] - live_results['start_time']).total_seconds():.1f} seconds")
        print(f"   Total Requests: {live_results['total_requests']}")
        print(f"   Avg Routing Confidence: {np.mean([d.confidence for d in live_results['routing_decisions']]):.3f}")
        
        # Model usage in live demo
        live_usage = defaultdict(int)
        for decision in live_results['routing_decisions']:
            live_usage[decision.selected_model] += 1
        
        print(f"\n🎯 Live Model Usage:")
        for model, count in live_usage.items():
            pct = (count / live_results['total_requests']) * 100
            print(f"   {model}: {count} requests ({pct:.1f}%)")
        
        # Performance tracking
        if live_results['performance_tracking']:
            expected_perf = np.mean([p['expected_performance'] for p in live_results['performance_tracking']])
            actual_perf = np.mean([p['actual_performance'] for p in live_results['performance_tracking']])
            print(f"\n📈 Performance Tracking:")
            print(f"   Expected Performance: {expected_perf:.3f}")
            print(f"   Actual Performance: {actual_perf:.3f}")
            print(f"   Performance Gap: {abs(expected_perf - actual_perf):.3f}")
    
    print(f"\n🎉 Contextual Routing Demo Complete!")
    print("This demonstrates intelligent model selection that adapts to market conditions")
    print("AI-driven routing optimizes performance through context-aware decisions")

if __name__ == "__main__":
    main()