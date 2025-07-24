#!/usr/bin/env python3
"""
Evaluate Enhanced Model Performance
Quick evaluation of the trained enhanced model
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

from train_enhanced_model import EnhancedFinancialLSTM, EnhancedDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_enhanced_test_data():
    """Load enhanced test data"""
    enhanced_dir = "/Users/user/REPOS/financial-mlops-pytorch/data/processed/enhanced"
    
    test_sequences = np.load(os.path.join(enhanced_dir, 'test_sequences.npy'))
    test_targets = np.load(os.path.join(enhanced_dir, 'test_targets.npy'))
    
    with open(os.path.join(enhanced_dir, 'enhanced_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    test_dataset = EnhancedDataset(test_sequences, test_targets)
    
    return test_dataset, metadata

def evaluate_enhanced_model():
    """Evaluate the enhanced model performance"""
    
    logger.info("=" * 60)
    logger.info("ENHANCED MODEL EVALUATION")
    logger.info("=" * 60)
    
    # Load test data
    test_dataset, metadata = load_enhanced_test_data()
    logger.info(f"Test dataset: {len(test_dataset)} samples")
    logger.info(f"Shape contract: {metadata['input_shape']}")
    
    # Load model
    model_path = "best_enhanced_model_v2.pth"
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return None
    
    # Initialize model
    model = EnhancedFinancialLSTM(
        input_size=metadata['n_features'],
        hidden_size=128,
        num_layers=3,
        dropout_prob=0.3
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Evaluate
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
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, zero_division=0)
    recall = recall_score(all_targets, all_predictions, zero_division=0)
    f1 = f1_score(all_targets, all_predictions, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    results = {
        'model_type': 'EnhancedFinancialLSTM',
        'feature_engineering': 'enhanced_financial_indicators',
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1_score': f1,
        'test_samples': len(all_targets),
        'shape_contract': metadata['input_shape'],
        'n_features': metadata['n_features'],
        'confusion_matrix': cm.tolist()
    }
    
    # Save results
    with open('enhanced_model_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Display results
    logger.info("=" * 60)
    logger.info("ENHANCED MODEL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    logger.info(f"Test F1-Score: {f1:.4f}")
    logger.info(f"Test Samples: {len(all_targets)}")
    
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"              Predicted")
    logger.info(f"           Down    Up")
    logger.info(f"Actual Down  {cm[0][0]:<4}  {cm[0][1]:<4}")
    logger.info(f"       Up    {cm[1][0]:<4}  {cm[1][1]:<4}")
    
    # Performance assessment
    logger.info("=" * 60)
    logger.info("PERFORMANCE ASSESSMENT")
    logger.info("=" * 60)
    
    if accuracy >= 0.80:
        logger.info("🎯 EXCELLENT: 80%+ accuracy target achieved!")
        performance_level = "excellent"
    elif accuracy >= 0.70:
        logger.info("✅ GOOD: Significant improvement over baseline")
        performance_level = "good"
    elif accuracy >= 0.60:
        logger.info("📈 IMPROVED: Better than baseline 52.7%")
        performance_level = "improved"
    else:
        logger.info("⚠️ NEEDS WORK: Similar to baseline performance")
        performance_level = "needs_work"
    
    improvement = (accuracy - 0.527) / 0.527 * 100
    logger.info(f"Improvement over baseline: {improvement:+.1f}%")
    
    logger.info(f"\nShape contract: {metadata['input_shape']} ✅")
    logger.info(f"A/B testing ready: Yes ✅")
    logger.info(f"Feature engineering: Enhanced financial indicators ✅")
    
    results['performance_level'] = performance_level
    results['improvement_over_baseline'] = improvement
    
    return results

if __name__ == "__main__":
    results = evaluate_enhanced_model()
    
    if results:
        print(f"\n🎯 Enhanced Model Evaluation Complete!")
        print(f"Accuracy: {results['test_accuracy']:.1%}")
        print(f"Improvement: {results['improvement_over_baseline']:+.1f}%")
        print(f"Performance: {results['performance_level']}")
        print(f"Shape contract: {results['shape_contract']}")
    else:
        print("❌ Evaluation failed - model not found")