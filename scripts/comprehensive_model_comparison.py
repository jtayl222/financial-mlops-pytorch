#!/usr/bin/env python3
"""
Comprehensive Model Comparison and Analysis
Compares baseline vs enhanced models and analyzes results
"""

import os
import sys
import torch
import numpy as np
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import StockPredictor
from train_enhanced_model import EnhancedFinancialLSTM, EnhancedDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_baseline_data():
    """Load baseline test data"""
    data_dir = "/Users/user/REPOS/financial-mlops-pytorch/data/processed"
    
    test_sequences = np.load(os.path.join(data_dir, 'test_sequences.npy'))
    test_targets = np.load(os.path.join(data_dir, 'test_sequence_targets.npy'))
    
    with open(os.path.join(data_dir, 'shape_contract_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    return test_sequences, test_targets, metadata

def load_enhanced_data():
    """Load enhanced test data"""
    enhanced_dir = "/Users/user/REPOS/financial-mlops-pytorch/data/processed/enhanced"
    
    test_sequences = np.load(os.path.join(enhanced_dir, 'test_sequences.npy'))
    test_targets = np.load(os.path.join(enhanced_dir, 'test_targets.npy'))
    
    with open(os.path.join(enhanced_dir, 'enhanced_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    return test_sequences, test_targets, metadata

def evaluate_baseline_model(test_sequences, test_targets, metadata):
    """Evaluate baseline model with untrained weights"""
    logger.info("Evaluating baseline model...")
    
    # Initialize baseline model
    model = StockPredictor(
        input_size=metadata['n_features'],
        hidden_size=32,
        num_layers=1,
        num_classes=1,
        dropout_prob=0.1
    )
    
    model.eval()
    
    # Convert to tensor dataset
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(test_sequences, dtype=torch.float32),
        torch.tensor(test_targets, dtype=torch.float32)
    )
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x).squeeze()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            
            all_predictions.extend(predictions.numpy())
            all_targets.extend(batch_y.numpy())
    
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, zero_division=0)
    recall = recall_score(all_targets, all_predictions, zero_division=0)
    f1 = f1_score(all_targets, all_predictions, zero_division=0)
    cm = confusion_matrix(all_targets, all_predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'targets': all_targets
    }

def evaluate_enhanced_model(test_sequences, test_targets, metadata):
    """Evaluate enhanced model"""
    logger.info("Evaluating enhanced model...")
    
    model_path = "best_enhanced_model_v2.pth"
    if not os.path.exists(model_path):
        logger.error(f"Enhanced model not found: {model_path}")
        return None
    
    # Initialize enhanced model
    model = EnhancedFinancialLSTM(
        input_size=metadata['n_features'],
        hidden_size=128,
        num_layers=3,
        dropout_prob=0.3
    )
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Convert to tensor dataset
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(test_sequences, dtype=torch.float32),
        torch.tensor(test_targets, dtype=torch.float32)
    )
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x).squeeze()
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            
            all_predictions.extend(predictions.numpy())
            all_targets.extend(batch_y.numpy())
            all_probabilities.extend(probabilities.numpy())
    
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, zero_division=0)
    recall = recall_score(all_targets, all_predictions, zero_division=0)
    f1 = f1_score(all_targets, all_predictions, zero_division=0)
    cm = confusion_matrix(all_targets, all_predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities
    }

def analyze_data_differences():
    """Analyze differences between baseline and enhanced datasets"""
    logger.info("Analyzing data differences...")
    
    # Load both datasets
    baseline_seq, baseline_targets, baseline_meta = load_baseline_data()
    enhanced_seq, enhanced_targets, enhanced_meta = load_enhanced_data()
    
    analysis = {
        'baseline': {
            'sequences_shape': baseline_seq.shape,
            'targets_shape': baseline_targets.shape,
            'target_distribution': {
                'positive': float(np.sum(baseline_targets)),
                'negative': float(len(baseline_targets) - np.sum(baseline_targets)),
                'positive_rate': float(np.mean(baseline_targets))
            },
            'features': baseline_meta['n_features'],
            'sequence_length': baseline_meta['sequence_length']
        },
        'enhanced': {
            'sequences_shape': enhanced_seq.shape,
            'targets_shape': enhanced_targets.shape,
            'target_distribution': {
                'positive': float(np.sum(enhanced_targets)),
                'negative': float(len(enhanced_targets) - np.sum(enhanced_targets)),
                'positive_rate': float(np.mean(enhanced_targets))
            },
            'features': enhanced_meta['n_features'],
            'sequence_length': enhanced_meta['sequence_length']
        }
    }
    
    return analysis

def comprehensive_comparison():
    """Run comprehensive model comparison"""
    
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE MODEL COMPARISON")
    logger.info("=" * 80)
    
    # Analyze data differences
    data_analysis = analyze_data_differences()
    
    logger.info("\n📊 DATA ANALYSIS:")
    logger.info(f"Baseline dataset: {data_analysis['baseline']['sequences_shape']}")
    logger.info(f"Enhanced dataset: {data_analysis['enhanced']['sequences_shape']}")
    logger.info(f"Baseline positive rate: {data_analysis['baseline']['target_distribution']['positive_rate']:.3f}")
    logger.info(f"Enhanced positive rate: {data_analysis['enhanced']['target_distribution']['positive_rate']:.3f}")
    
    # Load data
    baseline_seq, baseline_targets, baseline_meta = load_baseline_data()
    enhanced_seq, enhanced_targets, enhanced_meta = load_enhanced_data()
    
    # Evaluate models
    baseline_results = evaluate_baseline_model(baseline_seq, baseline_targets, baseline_meta)
    enhanced_results = evaluate_enhanced_model(enhanced_seq, enhanced_targets, enhanced_meta)
    
    if enhanced_results is None:
        return
    
    # Comparison
    logger.info("\n" + "=" * 80)
    logger.info("MODEL PERFORMANCE COMPARISON")
    logger.info("=" * 80)
    
    print(f"\n{'Metric':<15} {'Baseline':<12} {'Enhanced':<12} {'Difference':<15}")
    print("-" * 60)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        baseline_val = baseline_results[metric]
        enhanced_val = enhanced_results[metric]
        diff = enhanced_val - baseline_val
        
        print(f"{metric.title():<15} {baseline_val:.4f}      {enhanced_val:.4f}      {diff:+.4f}")
    
    # Confusion matrices
    logger.info("\n📊 CONFUSION MATRICES:")
    
    logger.info(f"\nBaseline (Untrained):")
    cm_b = baseline_results['confusion_matrix']
    logger.info(f"              Predicted")
    logger.info(f"           Down    Up")
    logger.info(f"Actual Down  {cm_b[0][0]:<4}  {cm_b[0][1]:<4}")
    logger.info(f"       Up    {cm_b[1][0]:<4}  {cm_b[1][1]:<4}")
    
    logger.info(f"\nEnhanced (Trained):")
    cm_e = enhanced_results['confusion_matrix']
    logger.info(f"              Predicted")
    logger.info(f"           Down    Up")
    logger.info(f"Actual Down  {cm_e[0][0]:<4}  {cm_e[0][1]:<4}")
    logger.info(f"       Up    {cm_e[1][0]:<4}  {cm_e[1][1]:<4}")
    
    # Analysis
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS")
    logger.info("=" * 80)
    
    # Check for majority class prediction
    baseline_up_rate = np.mean(baseline_results['predictions'])
    enhanced_up_rate = np.mean(enhanced_results['predictions'])
    
    logger.info(f"\n🔍 PREDICTION PATTERNS:")
    logger.info(f"Baseline predicts 'Up': {baseline_up_rate:.3f}")
    logger.info(f"Enhanced predicts 'Up': {enhanced_up_rate:.3f}")
    logger.info(f"Actual 'Up' rate: {np.mean(enhanced_results['targets']):.3f}")
    
    if enhanced_up_rate > 0.9:
        logger.info("⚠️  Enhanced model predicts mostly 'Up' - potential overfitting to majority class")
    elif enhanced_up_rate < 0.1:
        logger.info("⚠️  Enhanced model predicts mostly 'Down' - potential overfitting to minority class")
    
    # Probability analysis
    if 'probabilities' in enhanced_results:
        probs = np.array(enhanced_results['probabilities'])
        logger.info(f"\n📈 PROBABILITY ANALYSIS:")
        logger.info(f"Mean probability: {np.mean(probs):.3f}")
        logger.info(f"Std probability: {np.std(probs):.3f}")
        logger.info(f"Min probability: {np.min(probs):.3f}")
        logger.info(f"Max probability: {np.max(probs):.3f}")
        
        if np.std(probs) < 0.1:
            logger.info("⚠️  Low probability variance - model may not be learning meaningful patterns")
    
    # Key insights
    logger.info(f"\n💡 KEY INSIGHTS:")
    
    if enhanced_results['accuracy'] < baseline_results['accuracy']:
        logger.info("❌ Enhanced model performs worse than baseline")
        logger.info("   Possible causes:")
        logger.info("   - Overfitting to training data")
        logger.info("   - Feature engineering not optimal")
        logger.info("   - Different data distributions")
        logger.info("   - Model complexity too high")
    else:
        logger.info("✅ Enhanced model shows improvement")
    
    if enhanced_results['accuracy'] < 0.55:
        logger.info("📊 Both models near random performance")
        logger.info("   This confirms financial prediction difficulty")
    
    # Save comparison
    comparison = {
        'data_analysis': data_analysis,
        'baseline_results': {k: v for k, v in baseline_results.items() 
                           if k not in ['predictions', 'targets']},
        'enhanced_results': {k: v for k, v in enhanced_results.items() 
                           if k not in ['predictions', 'targets', 'probabilities']},
        'analysis': {
            'baseline_up_rate': baseline_up_rate,
            'enhanced_up_rate': enhanced_up_rate,
            'actual_up_rate': float(np.mean(enhanced_results['targets'])),
            'enhanced_better': enhanced_results['accuracy'] > baseline_results['accuracy']
        }
    }
    
    with open('comprehensive_model_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    logger.info(f"\n📁 Comparison saved to: comprehensive_model_comparison.json")
    
    return comparison

if __name__ == "__main__":
    comparison = comprehensive_comparison()
    
    if comparison:
        enhanced_acc = comparison['enhanced_results']['accuracy']
        baseline_acc = comparison['baseline_results']['accuracy']
        
        print(f"\n🎯 Comprehensive Comparison Complete!")
        print(f"Baseline: {baseline_acc:.1%}")
        print(f"Enhanced: {enhanced_acc:.1%}")
        print(f"Difference: {enhanced_acc - baseline_acc:+.1%}")
        
        if enhanced_acc > baseline_acc:
            print("✅ Enhanced model shows improvement")
        else:
            print("⚠️ Enhanced model needs further optimization")