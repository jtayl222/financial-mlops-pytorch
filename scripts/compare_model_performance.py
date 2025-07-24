#!/usr/bin/env python3
"""
Model Performance Comparison Script
Compares baseline and advanced model performance on the test split
"""

import os
import sys
import torch
import numpy as np
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import StockPredictor
from advanced_financial_model_v2 import AdvancedFinancialLSTM, FinancialTimeSeriesDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_data():
    """Load test dataset following shape contract"""
    data_dir = "/Users/user/REPOS/financial-mlops-pytorch/data/processed"
    
    # Load shape contract
    with open(os.path.join(data_dir, 'shape_contract_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    # Load test sequences and targets
    test_sequences = np.load(os.path.join(data_dir, 'test_sequences.npy'))
    test_targets = np.load(os.path.join(data_dir, 'test_sequence_targets.npy'))
    
    # Create dataset
    test_dataset = FinancialTimeSeriesDataset(test_sequences, test_targets)
    
    logger.info(f"Test dataset loaded: {len(test_dataset)} samples")
    logger.info(f"Input shape per sample: {test_sequences.shape[1:]}")
    
    return test_dataset, metadata

def load_baseline_model(metadata):
    """Load and configure baseline model"""
    logger.info("Loading baseline model...")
    
    model = StockPredictor(
        input_size=metadata['n_features'],
        hidden_size=32,  # baseline config
        num_layers=1,
        num_classes=1,
        dropout_prob=0.1
    )
    
    # Try to load trained weights if available
    model_path = "/Users/user/REPOS/financial-mlops-pytorch/models/best_baseline_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        logger.info("✅ Loaded trained baseline weights")
    else:
        logger.warning("⚠️ Using untrained baseline model")
    
    model.eval()
    return model

def load_advanced_model(metadata):
    """Load and configure advanced model"""
    logger.info("Loading advanced model...")
    
    model = AdvancedFinancialLSTM(
        input_size=metadata['n_features'],
        hidden_size=128,  # advanced config
        num_layers=3,
        dropout_prob=0.3
    )
    
    # Try to load trained weights if available
    model_path = "/Users/user/REPOS/financial-mlops-pytorch/models/best_advanced_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        logger.info("✅ Loaded trained advanced weights")
    else:
        logger.warning("⚠️ Using untrained advanced model")
    
    model.eval()
    return model

def evaluate_model(model, test_dataset, model_name):
    """Evaluate a model on the test dataset"""
    logger.info(f"Evaluating {model_name} model...")
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(test_loader):
            # Forward pass
            outputs = model(features)
            
            # Convert logits to probabilities and predictions
            probabilities = torch.sigmoid(outputs).squeeze()
            predictions = (probabilities > 0.5).float()
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted', zero_division=0)
    recall = recall_score(targets, predictions, average='weighted', zero_division=0)
    f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
    
    # Class-specific metrics
    cm = confusion_matrix(targets, predictions)
    report = classification_report(targets, predictions, target_names=['Down', 'Up'], output_dict=True)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': predictions,
        'targets': targets,
        'probabilities': probabilities,
        'n_samples': len(targets)
    }
    
    return results

def print_model_comparison(baseline_results, advanced_results):
    """Print detailed comparison between models"""
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 80)
    
    # Overall metrics comparison
    print(f"\n📊 OVERALL METRICS:")
    print(f"{'Metric':<15} {'Baseline':<12} {'Advanced':<12} {'Difference':<15}")
    print("-" * 60)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric in metrics:
        baseline_val = baseline_results[metric]
        advanced_val = advanced_results[metric]
        diff = advanced_val - baseline_val
        diff_str = f"{diff:+.4f}"
        
        print(f"{metric.title():<15} {baseline_val:.4f}      {advanced_val:.4f}      {diff_str:<15}")
    
    # Sample size
    print(f"{'Samples':<15} {baseline_results['n_samples']:<12} {advanced_results['n_samples']:<12}")
    
    # Class-wise performance
    print(f"\n📈 CLASS-WISE PERFORMANCE:")
    print(f"\nBaseline Model:")
    for class_name, metrics in baseline_results['classification_report'].items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            print(f"  {class_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
    
    print(f"\nAdvanced Model:")
    for class_name, metrics in advanced_results['classification_report'].items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            print(f"  {class_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
    
    # Confusion matrices
    print(f"\n🎯 CONFUSION MATRICES:")
    print(f"\nBaseline Model:")
    print(f"             Predicted")
    print(f"           Down    Up")
    print(f"Actual Down  {baseline_results['confusion_matrix'][0][0]:<4}  {baseline_results['confusion_matrix'][0][1]:<4}")
    print(f"       Up    {baseline_results['confusion_matrix'][1][0]:<4}  {baseline_results['confusion_matrix'][1][1]:<4}")
    
    print(f"\nAdvanced Model:")
    print(f"             Predicted")
    print(f"           Down    Up")
    print(f"Actual Down  {advanced_results['confusion_matrix'][0][0]:<4}  {advanced_results['confusion_matrix'][0][1]:<4}")
    print(f"       Up    {advanced_results['confusion_matrix'][1][0]:<4}  {advanced_results['confusion_matrix'][1][1]:<4}")

def analyze_prediction_differences(baseline_results, advanced_results):
    """Analyze where models disagree in predictions"""
    baseline_preds = baseline_results['predictions']
    advanced_preds = advanced_results['predictions']
    targets = baseline_results['targets']
    
    # Find disagreements
    disagreements = baseline_preds != advanced_preds
    n_disagreements = np.sum(disagreements)
    disagreement_rate = n_disagreements / len(targets)
    
    print(f"\n🤔 PREDICTION DISAGREEMENTS:")
    print(f"Total disagreements: {n_disagreements}/{len(targets)} ({disagreement_rate:.2%})")
    
    if n_disagreements > 0:
        # Analyze disagreement patterns
        disagreement_indices = np.where(disagreements)[0]
        
        # Where baseline is right and advanced is wrong
        baseline_right_advanced_wrong = (baseline_preds[disagreements] == targets[disagreements]) & (advanced_preds[disagreements] != targets[disagreements])
        n_baseline_wins = np.sum(baseline_right_advanced_wrong)
        
        # Where advanced is right and baseline is wrong  
        advanced_right_baseline_wrong = (advanced_preds[disagreements] == targets[disagreements]) & (baseline_preds[disagreements] != targets[disagreements])
        n_advanced_wins = np.sum(advanced_right_baseline_wrong)
        
        # Where both are wrong
        both_wrong = (baseline_preds[disagreements] != targets[disagreements]) & (advanced_preds[disagreements] != targets[disagreements])
        n_both_wrong = np.sum(both_wrong)
        
        print(f"  Baseline correct, Advanced wrong: {n_baseline_wins}")
        print(f"  Advanced correct, Baseline wrong: {n_advanced_wins}")
        print(f"  Both models wrong: {n_both_wrong}")
        
        if n_advanced_wins > n_baseline_wins:
            print("  ✅ Advanced model wins more disagreements")
        elif n_baseline_wins > n_advanced_wins:
            print("  ✅ Baseline model wins more disagreements")
        else:
            print("  🤝 Models tie in disagreement resolution")

def save_comparison_results(baseline_results, advanced_results):
    """Save comparison results to JSON"""
    comparison = {
        'baseline': {
            'accuracy': baseline_results['accuracy'],
            'precision': baseline_results['precision'],
            'recall': baseline_results['recall'],
            'f1_score': baseline_results['f1_score'],
            'n_samples': baseline_results['n_samples']
        },
        'advanced': {
            'accuracy': advanced_results['accuracy'],
            'precision': advanced_results['precision'],
            'recall': advanced_results['recall'],
            'f1_score': advanced_results['f1_score'],
            'n_samples': advanced_results['n_samples']
        },
        'comparison': {
            'accuracy_diff': advanced_results['accuracy'] - baseline_results['accuracy'],
            'precision_diff': advanced_results['precision'] - baseline_results['precision'],
            'recall_diff': advanced_results['recall'] - baseline_results['recall'],
            'f1_diff': advanced_results['f1_score'] - baseline_results['f1_score']
        }
    }
    
    output_path = "/Users/user/REPOS/financial-mlops-pytorch/model_comparison_results.json"
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Comparison results saved to: {output_path}")

def main():
    """Main comparison function"""
    logger.info("=" * 80)
    logger.info("BASELINE vs ADVANCED MODEL PERFORMANCE COMPARISON")
    logger.info("=" * 80)
    
    # Load test data
    test_dataset, metadata = load_test_data()
    
    # Load models
    baseline_model = load_baseline_model(metadata)
    advanced_model = load_advanced_model(metadata)
    
    # Evaluate both models
    baseline_results = evaluate_model(baseline_model, test_dataset, "Baseline")
    advanced_results = evaluate_model(advanced_model, test_dataset, "Advanced")
    
    # Print comparison
    print_model_comparison(baseline_results, advanced_results)
    
    # Analyze disagreements
    analyze_prediction_differences(baseline_results, advanced_results)
    
    # Save results
    save_comparison_results(baseline_results, advanced_results)
    
    # Final summary
    print(f"\n" + "=" * 80)
    print(f"SUMMARY")
    print(f"=" * 80)
    accuracy_diff = advanced_results['accuracy'] - baseline_results['accuracy']
    f1_diff = advanced_results['f1_score'] - baseline_results['f1_score']
    
    print(f"Baseline accuracy: {baseline_results['accuracy']:.4f}")
    print(f"Advanced accuracy: {advanced_results['accuracy']:.4f}")
    print(f"Accuracy difference: {accuracy_diff:+.4f}")
    print(f"F1 difference: {f1_diff:+.4f}")
    
    if accuracy_diff > 0.01:  # 1% improvement threshold
        print("✅ Advanced model shows meaningful improvement")
    elif accuracy_diff < -0.01:
        print("⚠️ Baseline model outperforms advanced model")
    else:
        print("🤝 Models show similar performance")
    
    print(f"\nBoth models ready for A/B testing with shape contract [10, 205]")

if __name__ == "__main__":
    main()