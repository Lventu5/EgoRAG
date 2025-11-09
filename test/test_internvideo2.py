"""
Test script for InternVideo2 integration
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image

print("Testing InternVideo2 integration...")

# Test 1: Load the wrapper directly
print("\n" + "="*80)
print("Test 1: Loading InternVideo2 wrapper directly")
print("="*80)

try:
    from indexing.components.internvideo2_wrapper import load_internvideo2_model
    
    print("Loading model...")
    model = load_internvideo2_model(device="cuda", num_frames=8)
    print("✓ Model loaded successfully!")
    
    # Test 2: Create dummy video frames
    print("\n" + "="*80)
    print("Test 2: Encoding dummy video frames")
    print("="*80)
    
    # Create 8 random frames (224x224x3)
    dummy_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(8)]
    print(f"Created {len(dummy_frames)} dummy frames")
    
    # Encode the frames
    print("Encoding frames...")
    embeddings = model.encode_video(dummy_frames)
    print(f"✓ Encoded! Embedding shape: {embeddings.shape}")
    print(f"  Expected shape: (1, 1408) for InternVideo2-1B")
    
    if embeddings.shape[1] == 1408:
        print("✓ Embedding dimension is correct!")
    else:
        print(f"⚠ Warning: Expected dimension 1408, got {embeddings.shape[1]}")
    
    # Test 3: Test with VideoEncoder
    print("\n" + "="*80)
    print("Test 3: Testing with VideoEncoder class")
    print("="*80)
    
    from indexing.components.video_encoder import VideoEncoder
    from configuration.config import CONFIG
    
    print(f"Current model_name in config: {CONFIG.indexing.video.model_name}")
    
    if CONFIG.indexing.video.model_name == "internvideo2":
        print("Loading VideoEncoder...")
        encoder = VideoEncoder(device="cuda")
        encoder.load_models()
        print("✓ VideoEncoder loaded!")
        
        # Create more frames for a realistic test
        test_frames = np.random.randint(0, 255, (16, 480, 640, 3), dtype=np.uint8)
        print(f"Created test video with {len(test_frames)} frames")
        
        print("Encoding video...")
        result = encoder.encode(test_frames)
        print("✓ Encoding successful!")
        print(f"  Video embedding shape: {result['video'].shape}")
        print(f"  Image embedding shape: {result['image'].shape}")
        print(f"  Keyframes shape: {result['keyframes'].shape}")
    else:
        print(f"⚠ Skipping VideoEncoder test - model_name is '{CONFIG.indexing.video.model_name}', not 'internvideo2'")
        print("  To test, set CONFIG.indexing.video.model_name = 'internvideo2' in config.yaml")
    
    print("\n" + "="*80)
    print("All tests completed successfully! ✓")
    print("="*80)
    
except Exception as e:
    print(f"\n❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
