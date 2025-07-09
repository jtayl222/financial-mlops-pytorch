#!/usr/bin/env python3
"""
Advanced Drift Detection System for Financial Models
Demonstrates comprehensive model monitoring with Alibi Detect and Evidently
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import argparse
import warnings
from typing import Dict, List, Tuple, Optional, Any
import pickle
import os
from pathlib import Path

# Drift detection imports
try:
    from alibi_detect.drift import TabularDrift, KSDrift, MMDDrift
except ImportError:
    # Fallback for different versions
    try:
        from alibi_detect.drift.tabular import TabularDrift
        from alibi_detect.drift.ks import KSDrift  
        from alibi_detect.drift.mmd import MMDDrift
    except ImportError:
        print("⚠️ Alibi Detect not available, using statistical fallbacks")
        TabularDrift = None
        KSDrift = None
        MMDDrift = None

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.metrics import *
    EVIDENTLY_AVAILABLE = True
except ImportError:
    print("⚠️ Evidently not available, using basic statistical tests")
    EVIDENTLY_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class FinancialDriftDetector:
    """Comprehensive drift detection system for financial models"""
    
    def __init__(self, reference_data: Optional[pd.DataFrame] = None):
        """Initialize the drift detector with reference data"""
        
        # Financial feature definitions
        self.feature_names = [
            'price', 'volume', 'returns', 'volatility', 'rsi', 'macd',
            'sma_5', 'sma_20', 'bb_upper', 'bb_lower', 'atr', 'momentum',
            'williams_r', 'stoch_k', 'stoch_d', 'adx', 'cci', 'roc',
            'news_sentiment', 'social_sentiment', 'market_fear_greed',
            'vix', 'dollar_index', 'bond_yield', 'commodity_index'
        ]
        
        # Feature categories for organized monitoring
        self.feature_categories = {
            'Price & Volume': ['price', 'volume', 'returns', 'volatility'],
            'Technical Indicators': ['rsi', 'macd', 'sma_5', 'sma_20', 'bb_upper', 'bb_lower'],
            'Momentum': ['atr', 'momentum', 'williams_r', 'stoch_k', 'stoch_d'],
            'Market Structure': ['adx', 'cci', 'roc'],
            'Alternative Data': ['news_sentiment', 'social_sentiment', 'market_fear_greed'],
            'Macro Factors': ['vix', 'dollar_index', 'bond_yield', 'commodity_index']
        }
        
        # Drift detection thresholds
        self.drift_thresholds = {
            'data_drift': 0.05,      # p-value threshold for statistical tests
            'target_drift': 0.10,     # threshold for target distribution change
            'feature_drift': 0.05,    # per-feature drift threshold
            'concept_drift': 0.15,    # threshold for concept drift detection
            'performance_drift': 0.10  # threshold for performance degradation
        }
        
        # Initialize reference data
        if reference_data is not None:
            self.reference_data = reference_data
        else:
            self.reference_data = self._generate_reference_data()
        
        # Initialize drift detectors
        self.drift_detectors = {}
        self._initialize_drift_detectors()
        
        # Monitoring state
        self.monitoring_history = []
        self.alert_history = []
        self.model_performance_history = []
        
        print("🔍 Drift Detection System Initialized")
        print(f"   📊 Reference samples: {len(self.reference_data)}")
        print(f"   🎯 Features monitored: {len(self.feature_names)}")
        print(f"   🚨 Drift detectors: {len(self.drift_detectors)}")
    
    def _generate_reference_data(self, n_samples: int = 2000) -> pd.DataFrame:
        """Generate comprehensive reference data for training period"""
        np.random.seed(42)
        
        data = []
        
        # Generate realistic financial time series
        for i in range(n_samples):
            # Market regime (affects all features)
            market_regime = np.random.choice(['bull', 'bear', 'neutral'], p=[0.3, 0.2, 0.5])
            
            # Base parameters by regime
            if market_regime == 'bull':
                base_return = 0.01
                base_volatility = 0.15
                base_rsi = 60
                base_sentiment = 0.6
                base_vix = 18
            elif market_regime == 'bear':
                base_return = -0.01
                base_volatility = 0.25
                base_rsi = 40
                base_sentiment = 0.4
                base_vix = 28
            else:  # neutral
                base_return = 0.002
                base_volatility = 0.18
                base_rsi = 50
                base_sentiment = 0.5
                base_vix = 20
            
            # Generate correlated features
            sample = {}
            
            # Price and volume
            sample['price'] = 100 + np.random.normal(0, 10)
            sample['volume'] = max(0, np.random.lognormal(15, 0.5))
            sample['returns'] = base_return + np.random.normal(0, 0.01)
            sample['volatility'] = max(0.05, base_volatility + np.random.normal(0, 0.03))
            
            # Technical indicators
            sample['rsi'] = np.clip(base_rsi + np.random.normal(0, 15), 0, 100)
            sample['macd'] = np.random.normal(0, 0.5)
            sample['sma_5'] = 1 + np.random.normal(0, 0.02)
            sample['sma_20'] = 1 + np.random.normal(0, 0.03)
            sample['bb_upper'] = sample['price'] * (1 + sample['volatility'] * 0.5)
            sample['bb_lower'] = sample['price'] * (1 - sample['volatility'] * 0.5)
            sample['atr'] = sample['volatility'] * sample['price'] * 0.02
            sample['momentum'] = sample['returns'] * 10
            sample['williams_r'] = np.clip(np.random.normal(-50, 20), -100, 0)
            sample['stoch_k'] = np.clip(sample['rsi'] + np.random.normal(0, 10), 0, 100)
            sample['stoch_d'] = sample['stoch_k'] * 0.9
            sample['adx'] = np.clip(np.random.normal(25, 10), 0, 100)
            sample['cci'] = np.random.normal(0, 50)
            sample['roc'] = sample['returns'] * 100
            
            # Alternative data
            sample['news_sentiment'] = np.clip(base_sentiment + np.random.normal(0, 0.15), 0, 1)
            sample['social_sentiment'] = np.clip(sample['news_sentiment'] + np.random.normal(0, 0.08), 0, 1)
            sample['market_fear_greed'] = np.clip(100 - base_vix * 2.5 + np.random.normal(0, 10), 0, 100)
            
            # Macro factors
            sample['vix'] = max(10, base_vix + np.random.normal(0, 3))
            sample['dollar_index'] = 95 + np.random.normal(0, 3)
            sample['bond_yield'] = max(0.01, 0.025 + np.random.normal(0, 0.008))
            sample['commodity_index'] = 100 + np.random.normal(0, 5)
            
            # Add timestamp
            sample['timestamp'] = datetime.now() - timedelta(days=n_samples-i)
            sample['market_regime'] = market_regime
            
            data.append(sample)
        
        df = pd.DataFrame(data)
        return df
    
    def _initialize_drift_detectors(self):
        """Initialize various drift detection algorithms"""
        
        # Prepare reference data for detectors
        reference_features = self.reference_data[self.feature_names].values
        
        # Store reference statistics for fallback methods
        self.reference_stats = {}
        for i, feature in enumerate(self.feature_names):
            feature_data = reference_features[:, i]
            self.reference_stats[feature] = {
                'mean': np.mean(feature_data),
                'std': np.std(feature_data),
                'min': np.min(feature_data),
                'max': np.max(feature_data),
                'q25': np.percentile(feature_data, 25),
                'q75': np.percentile(feature_data, 75),
                'data': feature_data  # Store for KS test
            }
        
        # 1. Tabular Drift Detector (comprehensive)
        print("   🔧 Initializing Tabular Drift Detector...")
        if TabularDrift is not None:
            try:
                self.drift_detectors['tabular'] = TabularDrift(
                    x_ref=reference_features,
                    p_val=self.drift_thresholds['data_drift'],
                    x_ref_preprocessed=False
                )
            except Exception as e:
                print(f"   ⚠️  Tabular drift detector init issue: {e}")
                self.drift_detectors['tabular'] = None
        else:
            print("   ⚠️  Using statistical fallback for tabular drift")
            self.drift_detectors['tabular'] = None
        
        # 2. Kolmogorov-Smirnov Drift Detector (univariate)
        print("   🔧 Initializing KS Drift Detectors...")
        self.drift_detectors['ks'] = {}
        for i, feature in enumerate(self.feature_names):
            if KSDrift is not None:
                try:
                    self.drift_detectors['ks'][feature] = KSDrift(
                        x_ref=reference_features[:, i:i+1],
                        p_val=self.drift_thresholds['feature_drift']
                    )
                except Exception as e:
                    print(f"   ⚠️  KS drift detector for {feature}: {e}")
                    self.drift_detectors['ks'][feature] = None
            else:
                self.drift_detectors['ks'][feature] = None
        
        # 3. MMD Drift Detector (multivariate)
        print("   🔧 Initializing MMD Drift Detector...")
        if MMDDrift is not None:
            try:
                # Use subset of features for MMD to avoid computational issues
                key_features = ['returns', 'volatility', 'rsi', 'news_sentiment', 'vix']
                key_indices = [self.feature_names.index(f) for f in key_features if f in self.feature_names]
                
                self.drift_detectors['mmd'] = MMDDrift(
                    x_ref=reference_features[:, key_indices],
                    p_val=self.drift_thresholds['data_drift'],
                    n_permutations=100
                )
                self.mmd_feature_indices = key_indices
            except Exception as e:
                print(f"   ⚠️  MMD drift detector init issue: {e}")
                self.drift_detectors['mmd'] = None
        else:
            print("   ⚠️  Using statistical fallback for MMD drift")
            self.drift_detectors['mmd'] = None
        
        print("   ✅ Drift detectors initialized")
    
    def generate_drifted_data(self, drift_type: str = 'gradual', 
                            drift_magnitude: float = 0.3,
                            n_samples: int = 500) -> pd.DataFrame:
        """Generate synthetic data with various types of drift"""
        
        np.random.seed(42)  # For reproducibility
        
        # Start with reference data patterns
        base_data = self.reference_data.iloc[-100:].copy()  # Recent reference period
        
        drifted_data = []
        
        for i in range(n_samples):
            # Progressive drift factor
            if drift_type == 'gradual':
                drift_factor = (i / n_samples) * drift_magnitude
            elif drift_type == 'sudden':
                drift_factor = drift_magnitude if i > n_samples // 2 else 0
            elif drift_type == 'seasonal':
                drift_factor = drift_magnitude * np.sin(2 * np.pi * i / (n_samples / 4))
            else:  # 'no_drift'
                drift_factor = 0
            
            # Sample from recent reference data
            base_sample = base_data.sample(1).iloc[0].to_dict()
            
            # Apply drift based on type
            sample = base_sample.copy()
            
            if drift_type in ['gradual', 'sudden']:
                # Shift distributions
                sample['returns'] += drift_factor * 0.01
                sample['volatility'] += drift_factor * 0.1
                sample['rsi'] += drift_factor * 20
                sample['news_sentiment'] += drift_factor * 0.2
                sample['vix'] += drift_factor * 10
                
                # Correlated changes
                sample['momentum'] = sample['returns'] * 10
                sample['market_fear_greed'] = max(0, min(100, 100 - sample['vix'] * 2.5))
                
            elif drift_type == 'seasonal':
                # Seasonal patterns
                sample['returns'] += drift_factor * 0.005
                sample['volatility'] += abs(drift_factor) * 0.05
                sample['rsi'] += drift_factor * 10
                sample['news_sentiment'] += drift_factor * 0.1
            
            # Add noise
            for feature in self.feature_names:
                if feature in sample:
                    if feature == 'rsi':
                        sample[feature] = np.clip(sample[feature], 0, 100)
                    elif feature in ['news_sentiment', 'social_sentiment']:
                        sample[feature] = np.clip(sample[feature], 0, 1)
                    elif feature == 'volatility':
                        sample[feature] = max(0.05, sample[feature])
                    elif feature == 'vix':
                        sample[feature] = max(10, sample[feature])
                    
                    # Add some random noise
                    noise_scale = 0.02 if drift_type == 'no_drift' else 0.05
                    sample[feature] += np.random.normal(0, noise_scale * abs(sample[feature]))
            
            # Update timestamp
            sample['timestamp'] = datetime.now() + timedelta(hours=i)
            sample['drift_type'] = drift_type
            sample['drift_magnitude'] = drift_factor
            
            drifted_data.append(sample)
        
        return pd.DataFrame(drifted_data)
    
    def _statistical_ks_test(self, feature: str, current_data: np.ndarray) -> Dict[str, Any]:
        """Fallback KS test using scipy"""
        from scipy.stats import ks_2samp
        
        reference_data = self.reference_stats[feature]['data']
        ks_stat, p_value = ks_2samp(reference_data, current_data)
        
        return {
            'is_drift': int(p_value < self.drift_thresholds['feature_drift']),
            'p_value': float(p_value),
            'threshold': self.drift_thresholds['feature_drift'],
            'distance': float(ks_stat)
        }
    
    def _statistical_drift_score(self, current_features: np.ndarray) -> Dict[str, Any]:
        """Calculate overall drift score using statistical measures"""
        
        # Calculate feature-wise deviations
        deviations = []
        
        # Handle both full feature set and subset
        n_features = current_features.shape[1]
        feature_subset = self.feature_names[:n_features]
        
        for i, feature in enumerate(feature_subset):
            current_feature = current_features[:, i]
            ref_stats = self.reference_stats[feature]
            
            # Calculate normalized deviation from reference statistics
            mean_dev = abs(np.mean(current_feature) - ref_stats['mean']) / max(ref_stats['std'], 1e-8)
            std_dev = abs(np.std(current_feature) - ref_stats['std']) / max(ref_stats['std'], 1e-8)
            
            # Range check
            current_min, current_max = np.min(current_feature), np.max(current_feature)
            range_violation = 0
            if current_min < ref_stats['min'] or current_max > ref_stats['max']:
                range_violation = min(
                    abs(current_min - ref_stats['min']) / max(ref_stats['std'], 1e-8),
                    abs(current_max - ref_stats['max']) / max(ref_stats['std'], 1e-8)
                )
            
            total_deviation = mean_dev + std_dev + range_violation
            deviations.append(total_deviation)
        
        # Calculate overall drift score
        avg_deviation = np.mean(deviations)
        max_deviation = np.max(deviations)
        
        # Convert to p-value like score (higher deviation = lower p-value)
        simulated_p_value = max(0.001, 1.0 / (1.0 + avg_deviation * 10))
        
        return {
            'is_drift': int(simulated_p_value < self.drift_thresholds['data_drift']),
            'p_value': float(simulated_p_value),
            'threshold': self.drift_thresholds['data_drift'],
            'distance': float(avg_deviation),
            'max_feature_deviation': float(max_deviation)
        }

    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive drift detection analysis"""
        
        print("🔍 Running comprehensive drift detection...")
        
        # Prepare current data
        current_features = current_data[self.feature_names].values
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'samples_analyzed': len(current_data),
            'drift_detected': False,
            'drift_alerts': [],
            'drift_scores': {},
            'feature_drift': {},
            'category_drift': {},
            'recommendations': []
        }
        
        # 1. Tabular Drift Detection
        if self.drift_detectors['tabular'] is not None:
            print("   📊 Running tabular drift detection...")
            try:
                tabular_result = self.drift_detectors['tabular'].predict(current_features)
                
                results['drift_scores']['tabular'] = {
                    'is_drift': int(tabular_result['data']['is_drift']),
                    'p_value': float(tabular_result['data']['p_val']),
                    'threshold': float(tabular_result['data']['threshold']),
                    'distance': float(tabular_result['data']['distance'])
                }
                
                if tabular_result['data']['is_drift']:
                    results['drift_detected'] = True
                    results['drift_alerts'].append({
                        'type': 'tabular_drift',
                        'severity': 'high',
                        'message': f"Tabular drift detected (p-value: {tabular_result['data']['p_val']:.4f})"
                    })
                
            except Exception as e:
                print(f"   ⚠️  Tabular drift detection error: {e}")
                results['drift_scores']['tabular'] = {'error': str(e)}
        else:
            # Use statistical fallback
            print("   📊 Running statistical tabular drift detection...")
            statistical_result = self._statistical_drift_score(current_features)
            results['drift_scores']['tabular'] = statistical_result
            
            if statistical_result['is_drift']:
                results['drift_detected'] = True
                results['drift_alerts'].append({
                    'type': 'tabular_drift',
                    'severity': 'high',
                    'message': f"Statistical drift detected (score: {statistical_result['distance']:.4f})"
                })
        
        # 2. Feature-wise KS Drift Detection
        print("   📈 Running feature-wise drift detection...")
        feature_drift_count = 0
        
        for feature in self.feature_names:
            feature_idx = self.feature_names.index(feature)
            current_feature_data = current_features[:, feature_idx]
            
            if (feature in self.drift_detectors['ks'] and 
                self.drift_detectors['ks'][feature] is not None):
                
                try:
                    ks_result = self.drift_detectors['ks'][feature].predict(
                        current_features[:, feature_idx:feature_idx+1]
                    )
                    
                    results['feature_drift'][feature] = {
                        'is_drift': int(ks_result['data']['is_drift']),
                        'p_value': float(ks_result['data']['p_val']),
                        'threshold': float(ks_result['data']['threshold']),
                        'distance': float(ks_result['data']['distance'])
                    }
                    
                    if ks_result['data']['is_drift']:
                        feature_drift_count += 1
                        results['drift_alerts'].append({
                            'type': 'feature_drift',
                            'severity': 'medium',
                            'feature': feature,
                            'message': f"Feature drift detected in {feature} (p-value: {ks_result['data']['p_val']:.4f})"
                        })
                        
                except Exception as e:
                    results['feature_drift'][feature] = {'error': str(e)}
            else:
                # Use statistical fallback
                try:
                    ks_result = self._statistical_ks_test(feature, current_feature_data)
                    results['feature_drift'][feature] = ks_result
                    
                    if ks_result['is_drift']:
                        feature_drift_count += 1
                        results['drift_alerts'].append({
                            'type': 'feature_drift',
                            'severity': 'medium',
                            'feature': feature,
                            'message': f"Feature drift detected in {feature} (p-value: {ks_result['p_value']:.4f})"
                        })
                        
                except Exception as e:
                    results['feature_drift'][feature] = {'error': str(e)}
        
        # 3. MMD Drift Detection
        if self.drift_detectors['mmd'] is not None:
            print("   🎯 Running MMD drift detection...")
            try:
                mmd_result = self.drift_detectors['mmd'].predict(
                    current_features[:, self.mmd_feature_indices]
                )
                
                results['drift_scores']['mmd'] = {
                    'is_drift': int(mmd_result['data']['is_drift']),
                    'p_value': float(mmd_result['data']['p_val']),
                    'threshold': float(mmd_result['data']['threshold']),
                    'distance': float(mmd_result['data']['distance'])
                }
                
                if mmd_result['data']['is_drift']:
                    results['drift_detected'] = True
                    results['drift_alerts'].append({
                        'type': 'mmd_drift',
                        'severity': 'high',
                        'message': f"Multivariate drift detected (p-value: {mmd_result['data']['p_val']:.4f})"
                    })
                    
            except Exception as e:
                print(f"   ⚠️  MMD drift detection error: {e}")
                results['drift_scores']['mmd'] = {'error': str(e)}
        else:
            # Use simplified multivariate analysis
            print("   🎯 Running statistical multivariate drift detection...")
            key_features = ['returns', 'volatility', 'rsi', 'news_sentiment', 'vix']
            key_indices = [self.feature_names.index(f) for f in key_features if f in self.feature_names]
            
            if key_indices:
                multivariate_result = self._statistical_drift_score(current_features[:, key_indices])
                results['drift_scores']['mmd'] = multivariate_result
                
                if multivariate_result['is_drift']:
                    results['drift_detected'] = True
                    results['drift_alerts'].append({
                        'type': 'mmd_drift',
                        'severity': 'high',
                        'message': f"Multivariate drift detected (score: {multivariate_result['distance']:.4f})"
                    })
        
        # 4. Category-wise Drift Analysis
        print("   🏷️  Analyzing category-wise drift...")
        for category, category_features in self.feature_categories.items():
            category_drift_features = []
            category_p_values = []
            
            for feature in category_features:
                if (feature in results['feature_drift'] and 
                    'is_drift' in results['feature_drift'][feature]):
                    
                    if results['feature_drift'][feature]['is_drift']:
                        category_drift_features.append(feature)
                    
                    category_p_values.append(results['feature_drift'][feature]['p_value'])
            
            if category_p_values:
                results['category_drift'][category] = {
                    'drift_feature_count': len(category_drift_features),
                    'total_features': len(category_features),
                    'drift_percentage': len(category_drift_features) / len(category_features) * 100,
                    'min_p_value': min(category_p_values),
                    'avg_p_value': np.mean(category_p_values),
                    'drifted_features': category_drift_features
                }
        
        # 5. Generate Recommendations
        self._generate_drift_recommendations(results)
        
        # 6. Update monitoring history
        self.monitoring_history.append(results)
        
        # 7. Check for alerts
        if results['drift_detected']:
            self._trigger_drift_alerts(results)
        
        print(f"   ✅ Drift detection complete. Drift detected: {results['drift_detected']}")
        print(f"   📊 Feature drift count: {feature_drift_count}/{len(self.feature_names)}")
        
        return results
    
    def _generate_drift_recommendations(self, results: Dict[str, Any]):
        """Generate actionable recommendations based on drift detection"""
        
        recommendations = []
        
        # High-level drift recommendations
        if results['drift_detected']:
            recommendations.append({
                'priority': 'high',
                'category': 'model_retraining',
                'action': 'Initiate model retraining workflow',
                'reason': 'Significant drift detected in input data distribution'
            })
            
            recommendations.append({
                'priority': 'high',
                'category': 'monitoring',
                'action': 'Increase monitoring frequency to hourly',
                'reason': 'Active drift period requires closer monitoring'
            })
        
        # Feature-specific recommendations
        high_drift_features = []
        for feature, drift_info in results['feature_drift'].items():
            if 'is_drift' in drift_info and drift_info['is_drift']:
                if drift_info['p_value'] < 0.01:  # Very significant drift
                    high_drift_features.append(feature)
        
        if high_drift_features:
            recommendations.append({
                'priority': 'medium',
                'category': 'feature_engineering',
                'action': f'Review feature engineering for: {", ".join(high_drift_features)}',
                'reason': 'High drift detected in critical features'
            })
        
        # Category-specific recommendations
        for category, category_info in results['category_drift'].items():
            if category_info['drift_percentage'] > 50:  # More than half features drifted
                recommendations.append({
                    'priority': 'medium',
                    'category': 'data_quality',
                    'action': f'Investigate data quality issues in {category}',
                    'reason': f'{category_info["drift_percentage"]:.1f}% of {category} features show drift'
                })
        
        # Performance recommendations
        if len(results['drift_alerts']) > 5:
            recommendations.append({
                'priority': 'low',
                'category': 'system_performance',
                'action': 'Consider implementing adaptive thresholds',
                'reason': 'Multiple drift alerts may indicate overly sensitive thresholds'
            })
        
        results['recommendations'] = recommendations
    
    def _trigger_drift_alerts(self, results: Dict[str, Any]):
        """Trigger appropriate alerts based on drift detection"""
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'alert_type': 'drift_detected',
            'severity': 'high' if results['drift_detected'] else 'medium',
            'message': f"Drift detected in {results['samples_analyzed']} samples",
            'drift_alerts': results['drift_alerts'],
            'recommendations': results['recommendations']
        }
        
        self.alert_history.append(alert)
        
        # In production, this would trigger:
        # - Slack/email notifications
        # - Webhook calls to retraining pipelines
        # - Dashboard alerts
        # - Automated scaling of monitoring resources
        
        print(f"   🚨 DRIFT ALERT: {alert['message']}")
        
        # Simulate automatic actions
        if len(results['drift_alerts']) > 3:
            print("   🤖 AUTO-ACTION: Triggering model retraining pipeline...")
            print("   📊 AUTO-ACTION: Scaling monitoring resources...")
            print("   📧 AUTO-ACTION: Notifying ML engineering team...")
    
    def create_drift_dashboard(self, drift_results: Dict[str, Any]) -> str:
        """Create comprehensive drift monitoring dashboard"""
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Color scheme
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#F7B801', '#DDA0DD']
        
        # 1. Drift Detection Summary
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Count drift types
        drift_counts = {}
        for alert in drift_results['drift_alerts']:
            drift_type = alert['type']
            drift_counts[drift_type] = drift_counts.get(drift_type, 0) + 1
        
        if drift_counts:
            labels = list(drift_counts.keys())
            sizes = list(drift_counts.values())
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors[:len(labels)])
            ax1.set_title('Drift Types Detected')
        else:
            ax1.text(0.5, 0.5, 'No Drift\nDetected', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=14, fontweight='bold')
            ax1.set_title('Drift Detection Status')
        
        # 2. Feature Drift Overview
        ax2 = fig.add_subplot(gs[0, 1:3])
        
        feature_drift_data = drift_results['feature_drift']
        if feature_drift_data:
            features = list(feature_drift_data.keys())
            p_values = []
            drift_status = []
            
            for feature in features:
                if 'p_value' in feature_drift_data[feature]:
                    p_values.append(feature_drift_data[feature]['p_value'])
                    drift_status.append(feature_drift_data[feature]['is_drift'])
                else:
                    p_values.append(1.0)
                    drift_status.append(0)
            
            # Create bar chart
            bars = ax2.bar(range(len(features)), p_values, 
                          color=['red' if drift else 'green' for drift in drift_status],
                          alpha=0.7)
            
            ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Drift Threshold')
            ax2.set_xticks(range(len(features)))
            ax2.set_xticklabels(features, rotation=45, ha='right')
            ax2.set_ylabel('P-Value')
            ax2.set_title('Feature-wise Drift Detection (Lower = More Drift)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Category Drift Analysis
        ax3 = fig.add_subplot(gs[0, 3])
        
        category_drift = drift_results['category_drift']
        if category_drift:
            categories = list(category_drift.keys())
            drift_percentages = [category_drift[cat]['drift_percentage'] for cat in categories]
            
            bars = ax3.barh(range(len(categories)), drift_percentages, 
                           color=colors[:len(categories)], alpha=0.7)
            ax3.set_yticks(range(len(categories)))
            ax3.set_yticklabels(categories)
            ax3.set_xlabel('Drift Percentage (%)')
            ax3.set_title('Category-wise Drift')
            ax3.grid(True, alpha=0.3)
        
        # 4. Drift Scores Timeline
        ax4 = fig.add_subplot(gs[1, :2])
        
        if len(self.monitoring_history) > 1:
            timestamps = [datetime.fromisoformat(h['timestamp']) for h in self.monitoring_history]
            tabular_scores = []
            mmd_scores = []
            
            for h in self.monitoring_history:
                if 'tabular' in h['drift_scores'] and 'p_value' in h['drift_scores']['tabular']:
                    tabular_scores.append(1 - h['drift_scores']['tabular']['p_value'])
                else:
                    tabular_scores.append(0)
                
                if 'mmd' in h['drift_scores'] and 'p_value' in h['drift_scores']['mmd']:
                    mmd_scores.append(1 - h['drift_scores']['mmd']['p_value'])
                else:
                    mmd_scores.append(0)
            
            ax4.plot(timestamps, tabular_scores, label='Tabular Drift Score', marker='o')
            ax4.plot(timestamps, mmd_scores, label='MMD Drift Score', marker='s')
            ax4.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Alert Threshold')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Drift Score (1 - p_value)')
            ax4.set_title('Drift Scores Over Time')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient\nHistory', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Drift Scores Timeline')
        
        # 5. Alert History
        ax5 = fig.add_subplot(gs[1, 2:])
        
        if self.alert_history:
            # Count alerts by severity
            alert_severities = [alert['severity'] for alert in self.alert_history]
            severity_counts = {}
            for severity in alert_severities:
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            severities = list(severity_counts.keys())
            counts = list(severity_counts.values())
            
            bars = ax5.bar(severities, counts, color=['red', 'orange', 'yellow'][:len(severities)], alpha=0.7)
            ax5.set_xlabel('Alert Severity')
            ax5.set_ylabel('Count')
            ax5.set_title('Alert History by Severity')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No Alerts\nTriggered', ha='center', va='center',
                    transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Alert History')
        
        # 6. Recommendations Summary
        ax6 = fig.add_subplot(gs[2, :2])
        ax6.axis('off')
        
        recommendations = drift_results['recommendations']
        if recommendations:
            rec_text = "🔧 Drift Detection Recommendations:\n\n"
            for i, rec in enumerate(recommendations[:6], 1):
                priority_icon = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
                rec_text += f"{priority_icon} {rec['action']}\n"
                rec_text += f"   Category: {rec['category']}\n"
                rec_text += f"   Reason: {rec['reason']}\n\n"
        else:
            rec_text = "✅ No specific recommendations at this time.\nSystem operating within normal parameters."
        
        ax6.text(0.05, 0.95, rec_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
        
        # 7. System Health Metrics
        ax7 = fig.add_subplot(gs[2, 2:])
        
        # Calculate system health metrics
        drift_percentage = len([a for a in drift_results['drift_alerts'] if a['type'] == 'feature_drift']) / len(self.feature_names) * 100
        system_health = max(0, 100 - drift_percentage)
        
        # Create health gauge
        theta = np.linspace(0, np.pi, 100)
        health_color = 'green' if system_health > 80 else 'orange' if system_health > 60 else 'red'
        
        ax7.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=3)
        ax7.fill_between(np.cos(theta), 0, np.sin(theta), alpha=0.3, color=health_color)
        
        # Add health percentage
        ax7.text(0, 0.2, f'{system_health:.1f}%', ha='center', va='center',
                fontsize=24, fontweight='bold', color=health_color)
        ax7.text(0, -0.1, 'System Health', ha='center', va='center',
                fontsize=12, fontweight='bold')
        
        ax7.set_xlim(-1.2, 1.2)
        ax7.set_ylim(-0.2, 1.2)
        ax7.set_aspect('equal')
        ax7.axis('off')
        ax7.set_title('Model Health Score')
        
        # 8. Detailed Statistics
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        
        # Create detailed statistics table
        stats_text = f"""
Drift Detection Analysis Summary

Analysis Details:
• Timestamp: {drift_results['timestamp'][:19]}
• Samples Analyzed: {drift_results['samples_analyzed']:,}
• Total Features: {len(self.feature_names)}
• Drift Detected: {'Yes' if drift_results['drift_detected'] else 'No'}

Detection Results:
• Alert Count: {len(drift_results['drift_alerts'])}
• Feature Drift Count: {len([f for f, d in drift_results['feature_drift'].items() if d.get('is_drift', False)])}
• Categories Affected: {len([c for c, d in drift_results['category_drift'].items() if d['drift_percentage'] > 25])}
• Recommendations: {len(drift_results['recommendations'])}

Drift Scores:
• Tabular Drift p-value: {drift_results['drift_scores'].get('tabular', {}).get('p_value', 'N/A')}
• MMD Drift p-value: {drift_results['drift_scores'].get('mmd', {}).get('p_value', 'N/A')}

System Status:
• Health Score: {system_health:.1f}%
• Monitoring Status: {'Active Alert' if drift_results['drift_detected'] else 'Normal'}
• Next Action: {'Investigate drift sources' if drift_results['drift_detected'] else 'Continue monitoring'}

This dashboard provides comprehensive drift monitoring for production ML systems.
Regular monitoring helps maintain model performance and reliability.
        """
        
        ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.3))
        
        # Overall title
        fig.suptitle('Advanced Drift Detection Dashboard: Model Monitoring & Alerting', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Save the dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drift_detection_dashboard_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Drift detection dashboard saved as: {filename}")
        
        return filename
    
    def run_drift_simulation(self, drift_type: str = 'gradual', 
                           drift_magnitude: float = 0.3) -> Dict[str, Any]:
        """Run complete drift detection simulation"""
        
        print(f"🎯 Running drift simulation: {drift_type} drift")
        print(f"   📊 Drift magnitude: {drift_magnitude}")
        print(f"   📈 Reference data: {len(self.reference_data)} samples")
        
        # Generate drifted data
        drifted_data = self.generate_drifted_data(drift_type, drift_magnitude)
        
        # Detect drift
        drift_results = self.detect_drift(drifted_data)
        
        # Create dashboard
        dashboard_filename = self.create_drift_dashboard(drift_results)
        
        # Compile complete results
        simulation_results = {
            'simulation_config': {
                'drift_type': drift_type,
                'drift_magnitude': drift_magnitude,
                'samples_generated': len(drifted_data),
                'reference_samples': len(self.reference_data)
            },
            'drift_detection_results': drift_results,
            'dashboard_filename': dashboard_filename,
            'summary': {
                'drift_detected': drift_results['drift_detected'],
                'alert_count': len(drift_results['drift_alerts']),
                'recommendations_count': len(drift_results['recommendations']),
                'health_score': max(0, 100 - (len(drift_results['drift_alerts']) / len(self.feature_names) * 100))
            }
        }
        
        return simulation_results

def main():
    parser = argparse.ArgumentParser(description='Advanced Drift Detection System')
    parser.add_argument('--drift-type', choices=['gradual', 'sudden', 'seasonal', 'no_drift'],
                       default='gradual', help='Type of drift to simulate')
    parser.add_argument('--drift-magnitude', type=float, default=0.3,
                       help='Magnitude of drift (0.0-1.0)')
    parser.add_argument('--samples', type=int, default=500,
                       help='Number of samples to generate')
    parser.add_argument('--output-json', type=str,
                       help='Save results to JSON file')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    print("🔍 Advanced Drift Detection System")
    print("=" * 60)
    print("Demonstrating comprehensive model drift monitoring for production ML systems")
    print()
    
    # Initialize drift detector
    print("🔧 Initializing drift detection system...")
    detector = FinancialDriftDetector()
    
    # Run simulation
    print(f"\n🎯 Running {args.drift_type} drift simulation...")
    results = detector.run_drift_simulation(
        drift_type=args.drift_type,
        drift_magnitude=args.drift_magnitude
    )
    
    # Display results
    print(f"\n📊 Simulation Results:")
    print(f"   Drift Type: {results['simulation_config']['drift_type']}")
    print(f"   Drift Detected: {results['summary']['drift_detected']}")
    print(f"   Alert Count: {results['summary']['alert_count']}")
    print(f"   Health Score: {results['summary']['health_score']:.1f}%")
    print(f"   Recommendations: {results['summary']['recommendations_count']}")
    
    # Show key alerts
    if results['drift_detection_results']['drift_alerts']:
        print(f"\n🚨 Key Drift Alerts:")
        for alert in results['drift_detection_results']['drift_alerts'][:5]:
            print(f"   • {alert['type']}: {alert['message']}")
    
    # Show recommendations
    if results['drift_detection_results']['recommendations']:
        print(f"\n💡 Key Recommendations:")
        for rec in results['drift_detection_results']['recommendations'][:3]:
            print(f"   • {rec['action']} ({rec['priority']} priority)")
    
    # Save results
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {args.output_json}")
    
    print(f"\n🎉 Drift Detection Analysis Complete!")
    print("This demonstrates enterprise-grade model monitoring for production ML systems")
    print("Includes automated alerts, recommendations, and comprehensive drift analysis")

if __name__ == "__main__":
    main()