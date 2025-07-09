#!/usr/bin/env python3
"""
Explainable AI Demo for Financial Models
Simplified version focusing on clear explainability concepts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import argparse
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FinancialExplainabilityDemo:
    """Simplified explainable AI demo for financial predictions"""
    
    def __init__(self):
        # Financial features
        self.feature_names = [
            'price', 'volume', 'returns', 'volatility', 'rsi', 'macd',
            'sma_5', 'sma_20', 'bb_upper', 'bb_lower', 'atr', 'momentum',
            'williams_r', 'stoch_k', 'stoch_d', 'adx', 'cci', 'roc',
            'news_sentiment', 'social_sentiment', 'market_fear_greed',
            'vix', 'dollar_index', 'bond_yield', 'commodity_index'
        ]
        
        # Feature categories for analysis
        self.feature_categories = {
            'Price & Volume': ['price', 'volume', 'returns', 'volatility'],
            'Technical Indicators': ['rsi', 'macd', 'sma_5', 'sma_20', 'bb_upper', 'bb_lower'],
            'Momentum': ['atr', 'momentum', 'williams_r', 'stoch_k', 'stoch_d'],
            'Market Structure': ['adx', 'cci', 'roc'],
            'Alternative Data': ['news_sentiment', 'social_sentiment', 'market_fear_greed'],
            'Macro Factors': ['vix', 'dollar_index', 'bond_yield', 'commodity_index']
        }
        
        # Risk factor thresholds
        self.risk_thresholds = {
            'volatility': {'high': 0.3, 'medium': 0.2},
            'rsi': {'overbought': 70, 'oversold': 30},
            'vix': {'high': 25, 'medium': 20},
            'news_sentiment': {'negative': 0.3, 'positive': 0.7}
        }
    
    def generate_market_scenario(self, scenario_type: str = None) -> Dict:
        """Generate realistic market scenario"""
        if scenario_type is None:
            scenario_type = np.random.choice(['bull', 'bear', 'volatile', 'sideways'])
        
        # Base scenario parameters
        scenarios = {
            'bull': {
                'base_return': 0.015,
                'volatility': 0.15,
                'rsi': 65,
                'vix': 18,
                'news_sentiment': 0.65,
                'description': 'Bull market with positive momentum'
            },
            'bear': {
                'base_return': -0.020,
                'volatility': 0.25,
                'rsi': 35,
                'vix': 28,
                'news_sentiment': 0.35,
                'description': 'Bear market with negative sentiment'
            },
            'volatile': {
                'base_return': 0.002,
                'volatility': 0.40,
                'rsi': 50,
                'vix': 35,
                'news_sentiment': 0.45,
                'description': 'High volatility with uncertain direction'
            },
            'sideways': {
                'base_return': 0.001,
                'volatility': 0.12,
                'rsi': 50,
                'vix': 15,
                'news_sentiment': 0.50,
                'description': 'Sideways market with low volatility'
            }
        }
        
        scenario = scenarios[scenario_type]
        
        # Generate features
        features = {}
        
        # Price and volume
        features['price'] = 100 + np.random.normal(0, 5)
        features['volume'] = max(0, np.random.lognormal(15, 0.3))
        features['returns'] = scenario['base_return'] + np.random.normal(0, 0.005)
        features['volatility'] = scenario['volatility'] + np.random.normal(0, 0.02)
        
        # Technical indicators
        features['rsi'] = np.clip(scenario['rsi'] + np.random.normal(0, 8), 0, 100)
        features['macd'] = np.random.normal(0, 0.5)
        features['sma_5'] = 1 + np.random.normal(0, 0.02)
        features['sma_20'] = 1 + np.random.normal(0, 0.03)
        features['bb_upper'] = features['price'] * 1.02
        features['bb_lower'] = features['price'] * 0.98
        features['atr'] = features['volatility'] * features['price'] * 0.02
        features['momentum'] = features['returns'] * 10
        features['williams_r'] = np.clip(np.random.normal(-50, 20), -100, 0)
        features['stoch_k'] = np.clip(features['rsi'] + np.random.normal(0, 10), 0, 100)
        features['stoch_d'] = features['stoch_k'] * 0.9
        features['adx'] = np.clip(np.random.normal(25, 10), 0, 100)
        features['cci'] = np.random.normal(0, 50)
        features['roc'] = features['returns'] * 100
        
        # Alternative data
        features['news_sentiment'] = np.clip(scenario['news_sentiment'] + np.random.normal(0, 0.1), 0, 1)
        features['social_sentiment'] = np.clip(features['news_sentiment'] + np.random.normal(0, 0.05), 0, 1)
        features['market_fear_greed'] = np.clip(100 - scenario['vix'] * 2 + np.random.normal(0, 5), 0, 100)
        
        # Macro factors
        features['vix'] = max(10, scenario['vix'] + np.random.normal(0, 3))
        features['dollar_index'] = 95 + np.random.normal(0, 2)
        features['bond_yield'] = max(0, 0.025 + np.random.normal(0, 0.005))
        features['commodity_index'] = 100 + np.random.normal(0, 3)
        
        return {
            'type': scenario_type,
            'description': scenario['description'],
            'features': features,
            'scenario_params': scenario
        }
    
    def simulate_model_prediction(self, features: Dict) -> Dict:
        """Simulate model prediction with realistic behavior"""
        
        # Calculate base prediction using feature weights
        feature_weights = {
            'returns': 0.25,
            'rsi': 0.15,
            'volatility': -0.10,
            'news_sentiment': 0.20,
            'vix': -0.15,
            'momentum': 0.18,
            'volume': 0.05,
            'macd': 0.12
        }
        
        # Calculate weighted prediction
        prediction_score = 0.5  # Start neutral
        
        for feature, weight in feature_weights.items():
            if feature in features:
                if feature == 'rsi':
                    # RSI: above 50 is bullish, below 50 is bearish
                    normalized_value = (features[feature] - 50) / 50
                elif feature == 'vix':
                    # VIX: higher values are bearish
                    normalized_value = -(features[feature] - 20) / 20
                elif feature == 'volatility':
                    # Volatility: higher values are typically bearish
                    normalized_value = -features[feature] / 0.3
                elif feature in ['news_sentiment', 'social_sentiment']:
                    # Sentiment: 0.5 is neutral, >0.5 is bullish
                    normalized_value = (features[feature] - 0.5) * 2
                elif feature == 'returns':
                    # Returns: scale to reasonable range
                    normalized_value = features[feature] / 0.05
                elif feature == 'momentum':
                    # Momentum: scale to reasonable range
                    normalized_value = features[feature] / 0.5
                elif feature == 'volume':
                    # Volume: normalize to impact
                    normalized_value = (features[feature] - 1000000) / 1000000
                elif feature == 'macd':
                    # MACD: scale to reasonable range
                    normalized_value = features[feature] / 1.0
                else:
                    normalized_value = features[feature]
                
                # Clip to reasonable range
                normalized_value = np.clip(normalized_value, -1, 1)
                prediction_score += weight * normalized_value
        
        # Ensure prediction is between 0 and 1
        prediction_prob = np.clip(prediction_score, 0, 1)
        
        # Add some noise
        prediction_prob += np.random.normal(0, 0.02)
        prediction_prob = np.clip(prediction_prob, 0, 1)
        
        # Calculate confidence based on feature consistency
        confidence = self._calculate_confidence(features, prediction_prob)
        
        return {
            'probability': prediction_prob,
            'direction': 'Up' if prediction_prob > 0.5 else 'Down',
            'confidence': confidence,
            'class_probabilities': {
                'Down': 1 - prediction_prob,
                'Up': prediction_prob
            }
        }
    
    def _calculate_confidence(self, features: Dict, prediction: float) -> float:
        """Calculate prediction confidence based on feature consistency"""
        
        # Factors that increase confidence
        confidence_factors = []
        
        # Strong RSI signals
        if features['rsi'] > 70 or features['rsi'] < 30:
            confidence_factors.append(0.2)
        
        # Low volatility increases confidence
        if features['volatility'] < 0.15:
            confidence_factors.append(0.15)
        
        # Consistent sentiment and price action
        if (features['news_sentiment'] > 0.6 and prediction > 0.6) or \
           (features['news_sentiment'] < 0.4 and prediction < 0.4):
            confidence_factors.append(0.25)
        
        # Strong momentum
        if abs(features['momentum']) > 0.3:
            confidence_factors.append(0.15)
        
        # Base confidence
        base_confidence = 0.5
        
        # Add confidence factors
        total_confidence = base_confidence + sum(confidence_factors)
        
        # Reduce confidence for extreme predictions without strong signals
        if prediction > 0.8 or prediction < 0.2:
            total_confidence *= 0.8
        
        return np.clip(total_confidence, 0.1, 0.95)
    
    def generate_shap_like_explanations(self, features: Dict, prediction: Dict) -> Dict:
        """Generate SHAP-like feature importance explanations"""
        
        # Calculate feature impacts based on prediction logic
        feature_impacts = {}
        
        # Price & Volume impacts
        feature_impacts['returns'] = features['returns'] * 0.25 / 0.05
        feature_impacts['volume'] = (features['volume'] - 1000000) / 1000000 * 0.05
        feature_impacts['price'] = 0.02 * np.random.normal(0, 1)  # Price level less important
        feature_impacts['volatility'] = -features['volatility'] / 0.3 * 0.10
        
        # Technical indicators
        feature_impacts['rsi'] = (features['rsi'] - 50) / 50 * 0.15
        feature_impacts['macd'] = features['macd'] / 1.0 * 0.12
        feature_impacts['momentum'] = features['momentum'] / 0.5 * 0.18
        feature_impacts['sma_5'] = (features['sma_5'] - 1) * 0.05
        feature_impacts['sma_20'] = (features['sma_20'] - 1) * 0.03
        
        # Alternative data
        feature_impacts['news_sentiment'] = (features['news_sentiment'] - 0.5) * 2 * 0.20
        feature_impacts['social_sentiment'] = (features['social_sentiment'] - 0.5) * 2 * 0.08
        feature_impacts['market_fear_greed'] = (features['market_fear_greed'] - 50) / 50 * 0.05
        
        # Macro factors
        feature_impacts['vix'] = -(features['vix'] - 20) / 20 * 0.15
        feature_impacts['dollar_index'] = (features['dollar_index'] - 95) / 5 * 0.03
        feature_impacts['bond_yield'] = (features['bond_yield'] - 0.025) / 0.025 * 0.08
        
        # Add remaining features with small random impacts
        for feature in self.feature_names:
            if feature not in feature_impacts:
                feature_impacts[feature] = np.random.normal(0, 0.02)
        
        # Add some noise to make it realistic
        for feature in feature_impacts:
            feature_impacts[feature] += np.random.normal(0, 0.01)
        
        return feature_impacts
    
    def identify_risk_factors(self, features: Dict, feature_impacts: Dict) -> List[Dict]:
        """Identify key risk factors in the prediction"""
        
        risk_factors = []
        
        # Check volatility risk
        if features['volatility'] > self.risk_thresholds['volatility']['high']:
            risk_factors.append({
                'factor': 'High Volatility',
                'value': features['volatility'],
                'threshold': self.risk_thresholds['volatility']['high'],
                'impact': feature_impacts['volatility'],
                'risk_level': 'High',
                'explanation': f"Volatility of {features['volatility']:.3f} exceeds high risk threshold"
            })
        
        # Check RSI extremes
        if features['rsi'] > self.risk_thresholds['rsi']['overbought']:
            risk_factors.append({
                'factor': 'RSI Overbought',
                'value': features['rsi'],
                'threshold': self.risk_thresholds['rsi']['overbought'],
                'impact': feature_impacts['rsi'],
                'risk_level': 'Medium',
                'explanation': f"RSI of {features['rsi']:.1f} indicates overbought conditions"
            })
        elif features['rsi'] < self.risk_thresholds['rsi']['oversold']:
            risk_factors.append({
                'factor': 'RSI Oversold',
                'value': features['rsi'],
                'threshold': self.risk_thresholds['rsi']['oversold'],
                'impact': feature_impacts['rsi'],
                'risk_level': 'Medium',
                'explanation': f"RSI of {features['rsi']:.1f} indicates oversold conditions"
            })
        
        # Check VIX levels
        if features['vix'] > self.risk_thresholds['vix']['high']:
            risk_factors.append({
                'factor': 'High Fear (VIX)',
                'value': features['vix'],
                'threshold': self.risk_thresholds['vix']['high'],
                'impact': feature_impacts['vix'],
                'risk_level': 'High',
                'explanation': f"VIX of {features['vix']:.1f} indicates high market fear"
            })
        
        # Check negative sentiment
        if features['news_sentiment'] < self.risk_thresholds['news_sentiment']['negative']:
            risk_factors.append({
                'factor': 'Negative Sentiment',
                'value': features['news_sentiment'],
                'threshold': self.risk_thresholds['news_sentiment']['negative'],
                'impact': feature_impacts['news_sentiment'],
                'risk_level': 'Medium',
                'explanation': f"News sentiment of {features['news_sentiment']:.2f} is notably negative"
            })
        
        # Check high impact features
        for feature, impact in feature_impacts.items():
            if abs(impact) > 0.1:  # High impact threshold
                risk_factors.append({
                    'factor': f'High Impact: {feature}',
                    'value': features.get(feature, 0),
                    'threshold': 0.1,
                    'impact': impact,
                    'risk_level': 'High' if abs(impact) > 0.15 else 'Medium',
                    'explanation': f"{feature} has significant impact on prediction"
                })
        
        # Sort by impact magnitude
        risk_factors.sort(key=lambda x: abs(x['impact']), reverse=True)
        
        return risk_factors[:10]  # Return top 10 risk factors
    
    def generate_counterfactuals(self, features: Dict, prediction: Dict) -> List[Dict]:
        """Generate counterfactual explanations"""
        
        counterfactuals = []
        
        # Scenario 1: Improve sentiment
        if features['news_sentiment'] < 0.7:
            new_features = features.copy()
            new_features['news_sentiment'] = 0.75
            new_features['social_sentiment'] = 0.72
            new_prediction = self.simulate_model_prediction(new_features)
            
            counterfactuals.append({
                'scenario': 'Improved Market Sentiment',
                'changes': {
                    'news_sentiment': {'from': features['news_sentiment'], 'to': 0.75},
                    'social_sentiment': {'from': features['social_sentiment'], 'to': 0.72}
                },
                'prediction_change': new_prediction['probability'] - prediction['probability'],
                'new_prediction': new_prediction['probability'],
                'explanation': 'Positive sentiment shift would increase bullish probability'
            })
        
        # Scenario 2: Reduce volatility
        if features['volatility'] > 0.15:
            new_features = features.copy()
            new_features['volatility'] = 0.12
            new_features['vix'] = 16
            new_prediction = self.simulate_model_prediction(new_features)
            
            counterfactuals.append({
                'scenario': 'Reduced Market Volatility',
                'changes': {
                    'volatility': {'from': features['volatility'], 'to': 0.12},
                    'vix': {'from': features['vix'], 'to': 16}
                },
                'prediction_change': new_prediction['probability'] - prediction['probability'],
                'new_prediction': new_prediction['probability'],
                'explanation': 'Lower volatility would increase prediction confidence'
            })
        
        # Scenario 3: Normalize RSI
        if features['rsi'] > 70 or features['rsi'] < 30:
            new_features = features.copy()
            new_features['rsi'] = 55
            new_prediction = self.simulate_model_prediction(new_features)
            
            counterfactuals.append({
                'scenario': 'Normalized RSI',
                'changes': {
                    'rsi': {'from': features['rsi'], 'to': 55}
                },
                'prediction_change': new_prediction['probability'] - prediction['probability'],
                'new_prediction': new_prediction['probability'],
                'explanation': 'RSI normalization would change momentum signals'
            })
        
        # Scenario 4: Positive momentum
        if features['momentum'] < 0.2:
            new_features = features.copy()
            new_features['momentum'] = 0.3
            new_features['returns'] = 0.015
            new_prediction = self.simulate_model_prediction(new_features)
            
            counterfactuals.append({
                'scenario': 'Positive Momentum',
                'changes': {
                    'momentum': {'from': features['momentum'], 'to': 0.3},
                    'returns': {'from': features['returns'], 'to': 0.015}
                },
                'prediction_change': new_prediction['probability'] - prediction['probability'],
                'new_prediction': new_prediction['probability'],
                'explanation': 'Positive momentum would increase bullish signals'
            })
        
        return counterfactuals
    
    def create_comprehensive_explanation(self, scenario: Dict) -> Dict:
        """Create comprehensive explanation for a market scenario"""
        
        features = scenario['features']
        
        # Generate prediction
        prediction = self.simulate_model_prediction(features)
        
        # Generate SHAP-like explanations
        feature_impacts = self.generate_shap_like_explanations(features, prediction)
        
        # Identify risk factors
        risk_factors = self.identify_risk_factors(features, feature_impacts)
        
        # Generate counterfactuals
        counterfactuals = self.generate_counterfactuals(features, prediction)
        
        # Calculate category importance
        category_importance = {}
        for category, category_features in self.feature_categories.items():
            category_impacts = [feature_impacts.get(f, 0) for f in category_features if f in feature_impacts]
            if category_impacts:
                category_importance[category] = {
                    'total_impact': sum(category_impacts),
                    'average_impact': np.mean(category_impacts),
                    'max_impact': max(category_impacts, key=abs),
                    'feature_count': len(category_impacts)
                }
        
        # Top features by absolute impact
        top_features = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        
        return {
            'scenario': scenario,
            'prediction': prediction,
            'feature_impacts': feature_impacts,
            'top_features': top_features,
            'category_importance': category_importance,
            'risk_factors': risk_factors,
            'counterfactuals': counterfactuals,
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'explanation_type': 'comprehensive',
                'features_analyzed': len(features),
                'risk_factors_identified': len(risk_factors)
            }
        }
    
    def create_visualization(self, explanation: Dict) -> str:
        """Create comprehensive visualization of the explanation"""
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Explainable AI Analysis: Financial Model Prediction', fontsize=16, fontweight='bold')
        
        # 1. Prediction Summary
        ax = axes[0, 0]
        prediction = explanation['prediction']
        labels = ['Down', 'Up']
        sizes = [1 - prediction['probability'], prediction['probability']]
        colors = ['red' if prediction['probability'] < 0.5 else 'lightcoral', 
                 'green' if prediction['probability'] > 0.5 else 'lightgreen']
        
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title(f'Prediction: {prediction["direction"]}\nConfidence: {prediction["confidence"]:.1%}')
        
        # 2. Top Feature Impacts
        ax = axes[0, 1]
        top_features = explanation['top_features'][:8]
        feature_names = [f[0] for f in top_features]
        impacts = [f[1] for f in top_features]
        colors = ['red' if x < 0 else 'green' for x in impacts]
        
        bars = ax.barh(range(len(feature_names)), impacts, color=colors, alpha=0.7)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Feature Impact')
        ax.set_title('Top Feature Contributions')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # 3. Category Importance
        ax = axes[0, 2]
        categories = list(explanation['category_importance'].keys())
        importances = [explanation['category_importance'][cat]['total_impact'] for cat in categories]
        colors = plt.cm.Set3(range(len(categories)))
        
        bars = ax.bar(range(len(categories)), importances, color=colors, alpha=0.8)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_ylabel('Total Impact')
        ax.set_title('Category Importance')
        ax.grid(True, alpha=0.3)
        
        # 4. Risk Factors
        ax = axes[1, 0]
        risk_factors = explanation['risk_factors'][:6]
        if risk_factors:
            risk_names = [rf['factor'] for rf in risk_factors]
            risk_impacts = [abs(rf['impact']) for rf in risk_factors]
            risk_levels = [rf['risk_level'] for rf in risk_factors]
            
            color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'yellow'}
            colors = [color_map.get(level, 'gray') for level in risk_levels]
            
            bars = ax.bar(range(len(risk_names)), risk_impacts, color=colors, alpha=0.7)
            ax.set_xticks(range(len(risk_names)))
            ax.set_xticklabels(risk_names, rotation=45, ha='right')
            ax.set_ylabel('Risk Impact')
            ax.set_title('Key Risk Factors')
            ax.grid(True, alpha=0.3)
        
        # 5. Counterfactual Analysis
        ax = axes[1, 1]
        counterfactuals = explanation['counterfactuals']
        if counterfactuals:
            cf_names = [cf['scenario'] for cf in counterfactuals]
            cf_changes = [cf['prediction_change'] for cf in counterfactuals]
            colors = ['green' if x > 0 else 'red' for x in cf_changes]
            
            bars = ax.barh(range(len(cf_names)), cf_changes, color=colors, alpha=0.7)
            ax.set_yticks(range(len(cf_names)))
            ax.set_yticklabels(cf_names)
            ax.set_xlabel('Prediction Change')
            ax.set_title('Counterfactual Scenarios')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
        
        # 6. Feature Distribution
        ax = axes[1, 2]
        scenario_type = explanation['scenario']['type']
        key_features = ['volatility', 'rsi', 'vix', 'news_sentiment']
        feature_values = [explanation['scenario']['features'][f] for f in key_features]
        
        bars = ax.bar(range(len(key_features)), feature_values, 
                     color=['red', 'blue', 'orange', 'green'], alpha=0.7)
        ax.set_xticks(range(len(key_features)))
        ax.set_xticklabels(key_features, rotation=45)
        ax.set_ylabel('Feature Value')
        ax.set_title(f'Key Features ({scenario_type.title()} Market)')
        ax.grid(True, alpha=0.3)
        
        # 7. Market Scenario Summary
        ax = axes[2, 0]
        ax.axis('off')
        
        scenario_info = explanation['scenario']
        summary_text = f"""
Market Scenario: {scenario_info['type'].title()}
Description: {scenario_info['description']}

Key Metrics:
• Volatility: {scenario_info['features']['volatility']:.3f}
• RSI: {scenario_info['features']['rsi']:.1f}
• VIX: {scenario_info['features']['vix']:.1f}
• Sentiment: {scenario_info['features']['news_sentiment']:.2f}

Prediction: {prediction['direction']} ({prediction['probability']:.1%})
Confidence: {prediction['confidence']:.1%}
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.3))
        
        # 8. Regulatory Compliance
        ax = axes[2, 1]
        ax.axis('off')
        
        compliance_text = f"""
Regulatory Compliance Summary

Explainability Methods:
✓ Feature Impact Analysis
✓ Risk Factor Identification  
✓ Counterfactual Explanations
✓ Category-wise Attribution

Model Transparency:
• Features Analyzed: {len(explanation['feature_impacts'])}
• Risk Factors: {len(explanation['risk_factors'])}
• Alternative Scenarios: {len(explanation['counterfactuals'])}

Audit Trail:
• Timestamp: {explanation['metadata']['analysis_timestamp'][:16]}
• Analysis Type: {explanation['metadata']['explanation_type']}
• Confidence Level: {prediction['confidence']:.1%}
        """
        
        ax.text(0.05, 0.95, compliance_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.3))
        
        # 9. Business Impact
        ax = axes[2, 2]
        ax.axis('off')
        
        # Calculate business metrics
        expected_return = prediction['probability'] * 0.02 + (1 - prediction['probability']) * (-0.015)
        risk_adjusted_return = expected_return / (explanation['scenario']['features']['volatility'] + 0.01)
        
        business_text = f"""
Business Impact Analysis

Expected Return: {expected_return:.3f}
Risk-Adjusted Return: {risk_adjusted_return:.3f}
Volatility: {explanation['scenario']['features']['volatility']:.3f}

Trading Recommendation:
• Position Size: {prediction['confidence'] * 100:.0f}%
• Stop Loss: {5 * explanation['scenario']['features']['volatility']:.1f}%
• Time Horizon: {('Short' if explanation['scenario']['features']['volatility'] > 0.25 else 'Medium')}

Key Risks:
{chr(10).join([f"• {rf['factor']}" for rf in explanation['risk_factors'][:3]])}
        """
        
        ax.text(0.05, 0.95, business_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.3))
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"explainable_ai_demo_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filename

def main():
    parser = argparse.ArgumentParser(description='Explainable AI Demo for Financial Models')
    parser.add_argument('--scenario', choices=['bull', 'bear', 'volatile', 'sideways'],
                       help='Market scenario type')
    parser.add_argument('--output-json', type=str,
                       help='Save explanation results to JSON file')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    print("🧠 Explainable AI Demo for Financial Models")
    print("=" * 60)
    print("Demonstrating comprehensive model explainability for regulatory compliance")
    print()
    
    # Initialize demo
    demo = FinancialExplainabilityDemo()
    
    # Generate scenario
    print("📊 Generating market scenario...")
    scenario = demo.generate_market_scenario(args.scenario)
    print(f"   📈 Market Type: {scenario['type'].title()}")
    print(f"   📝 Description: {scenario['description']}")
    
    # Generate comprehensive explanation
    print("\n🔍 Generating comprehensive explanation...")
    explanation = demo.create_comprehensive_explanation(scenario)
    
    # Display results
    prediction = explanation['prediction']
    print(f"\n🎯 Prediction Results:")
    print(f"   Direction: {prediction['direction']}")
    print(f"   Probability: {prediction['probability']:.3f}")
    print(f"   Confidence: {prediction['confidence']:.1%}")
    
    print(f"\n🔍 Top Feature Impacts:")
    for i, (feature, impact) in enumerate(explanation['top_features'][:5], 1):
        direction = "↑" if impact > 0 else "↓"
        print(f"   {i}. {feature}: {impact:+.3f} {direction}")
    
    print(f"\n⚠️  Key Risk Factors:")
    for risk in explanation['risk_factors'][:3]:
        print(f"   • {risk['factor']}: {risk['risk_level']} ({risk['impact']:+.3f})")
    
    print(f"\n🔄 Counterfactual Scenarios:")
    for cf in explanation['counterfactuals'][:3]:
        direction = "↑" if cf['prediction_change'] > 0 else "↓"
        print(f"   • {cf['scenario']}: {cf['prediction_change']:+.3f} {direction}")
    
    # Save JSON output
    if args.output_json:
        # Convert numpy types to native Python types for JSON serialization
        json_explanation = json.loads(json.dumps(explanation, default=str))
        with open(args.output_json, 'w') as f:
            json.dump(json_explanation, f, indent=2)
        print(f"\n💾 Results saved to: {args.output_json}")
    
    # Create visualization
    if not args.no_viz:
        print("\n📊 Creating explainability visualization...")
        viz_filename = demo.create_visualization(explanation)
        print(f"   📊 Visualization saved as: {viz_filename}")
    
    print("\n🎉 Explainable AI Demo Complete!")
    print("This demonstrates enterprise-grade model explainability for financial applications")
    print("Includes regulatory compliance, risk analysis, and business impact assessment")

if __name__ == "__main__":
    main()