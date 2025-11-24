"""
Code to encode the videos in a specific directory
"""
import warnings
import os
import gc

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set environment variable to disable warnings
os.environ["HF_HOME"] = os.environ.get("TRANSFORMERS_CACHE", "")
os.environ["TORCHVISION_DISABLE_TORCHCODEC"] = "1"

# CRITICAL: Setup cache BEFORE importing transformers!
from utils.cache_manager import setup_smart_cache, cleanup_smart_cache

# Setup smart cache (only copies large models like LLaVA)
# cache_info = setup_smart_cache(verbose=True)

# Now safe to import (will use fast cache)
from indexing.multimodal_encoder import MultiModalEncoder
from data.video_dataset import VideoDataset
from utils.memory_monitor import MemoryMonitor
import glob
import os
import numpy as np

import torch
from tqdm import tqdm


def encode(video_dir, save_dir, force_reencoding=False, force_video=None, force_audio=None, 
           force_caption=None, force_text=None):
    """
    Encode videos in a directory.
    
    Args:
        video_dir: Directory containing video files
        save_dir: Directory to save encoded pickles
        force_reencoding: If True, re-encode all modalities (default for all force_* params)
        force_video: If True, force re-encode video embeddings. If None, uses force_reencoding.
        force_audio: If True, force re-encode audio embeddings. If None, uses force_reencoding.
        force_caption: If True, force re-encode captions. If None, uses force_reencoding.
        force_text: If True, force re-encode text embeddings. If None, uses force_reencoding.
    """
    video_ids = glob.glob(os.path.join(video_dir, "*.mp4"))
    print(f"Found {len(video_ids)} videos")

    print("="*80)
    MemoryMonitor.log_memory("[INITIAL STATE] ")
    print("="*80)

    for video in tqdm(video_ids[:2]):
        print("-"*50)
        print(f"Encoding video {video}")
        print("-"*50)
        
        base = os.path.splitext(os.path.basename(video))[0]
        pickle_path = f"{save_dir}/{base}_encoded.pkl"
        
        # Determine if we should load existing pickle
        any_force = force_reencoding or force_video or force_audio or force_caption or force_text
        
        # Check if pickle already exists and load it if not forcing complete re-encoding
        if os.path.exists(pickle_path) and not (force_reencoding and not any([force_video is False, force_audio is False, force_caption is False, force_text is False])):
            print(f"Found existing pickle at {pickle_path}, loading and updating...")
            encoder = MultiModalEncoder(pickle_path=pickle_path, max_workers=4)
            video_dataset = encoder.encode_videos(
                force=force_reencoding,
                force_video=force_video,
                force_audio=force_audio,
                force_caption=force_caption,
                force_text=force_text
            )
        else:
            if os.path.exists(pickle_path):
                print(f"Force re-encoding enabled, creating new pickle")
            dataset = VideoDataset([video])
            encoder = MultiModalEncoder(dataset, max_workers=4)
            video_dataset = encoder.encode_videos(
                force=force_reencoding,
                force_video=force_video,
                force_audio=force_audio,
                force_caption=force_caption,
                force_text=force_text
            )

        video_dataset.save_to_pickle(pickle_path)

        # Free memory (CPU RAM, not GPU) to avoid OOM on long batch processing
        MemoryMonitor.log_memory("[BEFORE cleanup] ")
        del encoder
        if 'dataset' in locals():
            del dataset
        del video_dataset
        
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()  # Double gc to ensure everything is freed
        
        MemoryMonitor.log_memory("[AFTER cleanup] ")
        print("="*80)

if __name__ == "__main__":
    video_dir = "../ego4d_data/v2/full_scale"
    save_dir = "../ego4d_data/v2/internvideo_encoded_videos"
    
    # Option 1: Re-encode everything
    # force_reencoding = True
    
    # Option 2: Only update missing embeddings (default)
    force_reencoding = False
    
    # Option 3: Fine-grained control - force specific modalities
    # Set individual modality flags to override force_reencoding
    # Example: Only re-encode captions, keep everything else
    # force_video = False      # Keep existing video embeddings
    # force_audio = False      # Keep existing audio embeddings  
    # force_caption = True     # Re-encode captions
    # force_text = True        # Re-encode text (depends on captions)
    
    # Example: Only re-encode video embeddings
    # force_video = True
    # force_audio = False
    # force_caption = False
    # force_text = False
    
    encode(video_dir, save_dir, force_reencoding=force_reencoding)
    
    # With modality-specific control:
    # encode(video_dir, save_dir, 
    #        force_video=True, force_audio=False, 
    #        force_caption=False, force_text=False)
    
    # Clean up local cache after encoding
    # cleanup_smart_cache(cache_info, verbose=True)
