"""
Test script for LLaVA Video integration
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

print("Testing LLaVA Video integration...")

# Test 1: Load VideoEncoder with LLaVA Video
print("\n" + "="*80)
print("Test 1: Loading VideoEncoder with LLaVA Video")
print("="*80)

try:
    from indexing.components.video_encoder import VideoEncoder
    from configuration.config import CONFIG
    
    print(f"Current model_name in config: {CONFIG.indexing.video.model_name}")
    
    if CONFIG.indexing.video.model_name == "llava-video":
        print("Loading VideoEncoder...")
        encoder = VideoEncoder(device="cuda")
        encoder.load_models()
        print("✓ VideoEncoder loaded!")
        
        # Test 2: Encode dummy video
        print("\n" + "="*80)
        print("Test 2: Encoding dummy video frames")
        print("="*80)
        
        # Create test frames (16 frames of 480x640x3)
        test_frames = np.random.randint(0, 255, (16, 480, 640, 3), dtype=np.uint8)
        print(f"Created test video with {len(test_frames)} frames")
        
        print("Encoding video...")
        result = encoder.encode(test_frames)
        print("✓ Encoding successful!")
        print(f"  Video embedding shape: {result['video'].shape}")
        print(f"  Image embedding shape: {result['image'].shape}")
        print(f"  Keyframes shape: {result['keyframes'].shape}")
        
        print("\n" + "="*80)
        print("All tests completed successfully! ✓")
        print("="*80)
    else:
        print(f"⚠ Skipping test - model_name is '{CONFIG.indexing.video.model_name}', not 'llava-video'")
        print("  To test, set CONFIG.indexing.video.model_name = 'llava-video' in config.yaml")
    
except Exception as e:
    print(f"\n❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
