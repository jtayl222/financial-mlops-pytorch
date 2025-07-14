"""
Device configuration utilities for PyTorch training scripts.
Provides consistent device selection across all training scripts.
"""

import os
import torch
import logging


def get_device():
    """
    Get the appropriate device for training with FORCE_CPU support.
    
    Returns:
        torch.device: The device to use for training (CPU, CUDA, or MPS)
    """
    # Check if CPU is forced via environment variable
    force_cpu = os.environ.get("FORCE_CPU", "0").lower() in ("1", "true", "yes")
    
    if force_cpu:
        device = torch.device("cpu")
        logging.info("FORCE_CPU enabled - using CPU device")
    else:
        # Set MPS fallback for unsupported operations
        if torch.backends.mps.is_available():
            os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else
            "cuda:0" if torch.cuda.is_available() else
            "cpu"
        )
    
    logging.info(f"Using device: {device}")
    if device.type == 'mps':
        logging.info("MPS fallback enabled for unsupported operations")
    
    return device