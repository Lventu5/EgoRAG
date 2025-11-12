"""
Test script for memory monitoring utilities.
Run this to verify memory monitoring works correctly.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.memory_monitor import MemoryMonitor, monitor_memory
    import torch
    import numpy as np
    
    print("="*80)
    print("Memory Monitor Test")
    print("="*80)
    
    # Test 1: Basic memory logging
    print("\n[Test 1] Initial memory state:")
    MemoryMonitor.log_memory("[INITIAL] ")
    
    # Test 2: CPU memory allocation
    print("\n[Test 2] Allocating 500MB on CPU...")
    with monitor_memory("CPU allocation test"):
        big_array = np.zeros((500, 1024, 1024), dtype=np.uint8)  # ~500MB
    
    print("\n[Test 3] After freeing CPU memory:")
    del big_array
    MemoryMonitor.log_memory("[AFTER FREE] ")
    
    # Test 4: GPU memory allocation (if available)
    if torch.cuda.is_available():
        print("\n[Test 4] Allocating 1GB on GPU...")
        with monitor_memory("GPU allocation test"):
            gpu_tensor = torch.zeros((256, 1024, 1024), dtype=torch.float32, device='cuda')  # ~1GB
        
        print("\n[Test 5] After freeing GPU memory:")
        del gpu_tensor
        torch.cuda.empty_cache()
        MemoryMonitor.log_memory("[AFTER GPU FREE] ")
    else:
        print("\n[Test 4-5] Skipped (No GPU available)")
    
    print("\n" + "="*80)
    print("Memory Monitor Test PASSED ✓")
    print("="*80)
    
except ImportError as e:
    print(f"\n❌ ERROR: Missing dependency - {e}")
    print("\nTo fix this, install psutil:")
    print("  pip install psutil")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
