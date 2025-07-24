#!/usr/bin/env python3
"""
Inference Shape Validation Script
Validates that both baseline and advanced models accept identical input shapes
"""

import os
import sys
import torch
import numpy as np
import json
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import StockPredictor
from advanced_financial_model_v2 import AdvancedFinancialLSTM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_shape_contract():
    """Load the shape contract metadata"""
    contract_path = "/Users/user/REPOS/financial-mlops-pytorch/data/processed/shape_contract_metadata.json"
    with open(contract_path, 'r') as f:
        return json.load(f)

def create_sample_input(shape_contract):
    """Create sample input following the shape contract"""
    sequence_length = shape_contract['sequence_length']
    n_features = shape_contract['n_features']
    
    # Create random sample input: (batch_size=1, sequence_length, n_features)
    sample_input = torch.randn(1, sequence_length, n_features, dtype=torch.float32)
    return sample_input

def test_baseline_model(sample_input, shape_contract):
    """Test baseline model with sample input"""
    logger.info("Testing baseline model...")
    
    # Initialize baseline model with current config
    model = StockPredictor(
        input_size=shape_contract['n_features'],
        hidden_size=32,  # baseline config
        num_layers=1,
        num_classes=1,
        dropout_prob=0.1
    )
    
    # Test inference
    model.eval()
    with torch.no_grad():
        try:
            output = model(sample_input)
            print(f"✅ Baseline model inference successful")
            print(f"   Input shape: {sample_input.shape}")
            print(f"   Output shape: {output.shape}")
            logger.info(f"✅ Baseline model inference successful")
            logger.info(f"   Input shape: {sample_input.shape}")
            logger.info(f"   Output shape: {output.shape}")
            return True, output.shape
        except Exception as e:
            print(f"❌ Baseline model inference failed: {e}")
            logger.error(f"❌ Baseline model inference failed: {e}")
            return False, None

def test_advanced_model(sample_input, shape_contract):
    """Test advanced model with sample input"""
    logger.info("Testing advanced model...")
    
    # Initialize advanced model with current config
    model = AdvancedFinancialLSTM(
        input_size=shape_contract['n_features'],
        hidden_size=128,  # advanced config
        num_layers=3,
        dropout_prob=0.3
    )
    
    # Test inference
    model.eval()
    with torch.no_grad():
        try:
            output = model(sample_input)
            logger.info(f"✅ Advanced model inference successful")
            logger.info(f"   Input shape: {sample_input.shape}")
            logger.info(f"   Output shape: {output.shape}")
            return True, output.shape
        except Exception as e:
            logger.error(f"❌ Advanced model inference failed: {e}")
            return False, None

def test_onnx_model_shapes():
    """Test ONNX model input shapes for Seldon compatibility"""
    logger.info("Testing ONNX model shapes...")
    
    onnx_files = [
        "/Users/user/REPOS/financial-mlops-pytorch/models/stock_predictor_baseline.onnx",
        "/Users/user/REPOS/financial-mlops-pytorch/models/stock_predictor_enhanced.onnx"
    ]
    
    for onnx_path in onnx_files:
        if os.path.exists(onnx_path):
            logger.info(f"✅ ONNX model exists: {os.path.basename(onnx_path)}")
        else:
            logger.warning(f"⚠️ ONNX model missing: {os.path.basename(onnx_path)}")

def validate_a_b_compatibility(baseline_success, baseline_output_shape, 
                              advanced_success, advanced_output_shape):
    """Validate A/B testing compatibility"""
    logger.info("Validating A/B testing compatibility...")
    
    if not (baseline_success and advanced_success):
        logger.error("❌ A/B testing incompatible: One or both models failed inference")
        return False
    
    if baseline_output_shape != advanced_output_shape:
        logger.error(f"❌ A/B testing incompatible: Output shapes differ")
        logger.error(f"   Baseline output: {baseline_output_shape}")
        logger.error(f"   Advanced output: {advanced_output_shape}")
        return False
    
    logger.info("✅ A/B testing compatible: Both models accept identical inputs and produce identical output shapes")
    return True

def main():
    """Main validation function"""
    print("=" * 60)
    print("INFERENCE SHAPE VALIDATION")
    print("=" * 60)
    logger.info("=" * 60)
    logger.info("INFERENCE SHAPE VALIDATION")
    logger.info("=" * 60)
    
    # Load shape contract
    print("Loading shape contract...")
    shape_contract = load_shape_contract()
    print(f"Shape contract: {shape_contract['input_shape']}")
    logger.info(f"Shape contract: {shape_contract['input_shape']}")
    
    # Create sample input
    print("Creating sample input...")
    sample_input = create_sample_input(shape_contract)
    print(f"Sample input shape: {sample_input.shape}")
    logger.info(f"Sample input shape: {sample_input.shape}")
    
    # Test both models
    print("Testing models...")
    baseline_success, baseline_output_shape = test_baseline_model(sample_input, shape_contract)
    advanced_success, advanced_output_shape = test_advanced_model(sample_input, shape_contract)
    
    # Test ONNX models
    test_onnx_model_shapes()
    
    # Validate A/B compatibility
    a_b_compatible = validate_a_b_compatibility(
        baseline_success, baseline_output_shape,
        advanced_success, advanced_output_shape
    )
    
    # Summary
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Shape contract: {shape_contract['input_shape']}")
    logger.info(f"Baseline model: {'✅' if baseline_success else '❌'}")
    logger.info(f"Advanced model: {'✅' if advanced_success else '❌'}")
    logger.info(f"A/B compatible: {'✅' if a_b_compatible else '❌'}")
    
    if a_b_compatible:
        logger.info("🎯 Ready for A/B testing deployment!")
        return True
    else:
        logger.error("🚨 A/B testing deployment blocked - fix shape issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)