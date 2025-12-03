import torch
import torch.nn as nn
import numpy as np
import os

try:
    print("--- Dependency Check Success ---")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"NumPy Version: {np.__version__}")
    
    # Check for device capability
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    
    # Run a simple tensor operation
    x = torch.randn(5, 5)
    print(f"Tensor operation result: {x.mean():.4f}")
    
    # Check if we can import local files
    import sam
    import model
    import dataset
    print("Local file imports successful: sam, model, dataset")
    
except Exception as e:
    print(f"!!! CRITICAL FAILURE: {e}")