"""
Code to encode the videos in a specific directory
"""
import warnings
import os
import gc

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set environment variable to disable torchcodec warnings
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


def encode(video_dir, save_dir):
    video_ids = glob.glob(os.path.join(video_dir, "*.mp4"))
    print(f"Found {len(video_ids)} videos")

    print("="*80)
    MemoryMonitor.log_memory("[INITIAL STATE] ")
    print("="*80)

    for video in tqdm(video_ids):
        print("-"*50)
        print(f"Encoding video {video}")
        print("-"*50)
        dataset = VideoDataset([video])
        encoder = MultiModalEncoder(dataset, max_workers=4)
        video_dataset = encoder.encode_videos()

        base = os.path.splitext(os.path.basename(video))[0]
        pickle_path = f"{save_dir}/{base}_encoded.pkl"
        video_dataset.save_to_pickle(pickle_path)

        # Free memory (CPU RAM, not GPU) to avoid OOM on long batch processing
        MemoryMonitor.log_memory("[BEFORE cleanup] ")
        del encoder
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
    
    encode(video_dir, save_dir)
    # Clean up local cache after encoding
    # cleanup_smart_cache(cache_info, verbose=True)
