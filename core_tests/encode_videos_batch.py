import warnings
import os
import gc
import logging
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set environment variable to disable warnings
os.environ["HF_HOME"] = os.environ.get("TRANSFORMERS_CACHE", "")
os.environ["TORCHVISION_DISABLE_TORCHCODEC"] = "1"

# CRITICAL: Setup cache BEFORE importing transformers!
from utils.cache_manager import setup_smart_cache, cleanup_smart_cache
from utils.merge_pickles import merge_pickles
from configuration.config import CONFIG, save_config_snapshot
# Setup smart cache (only copies large models like LLaVA)
# cache_info = setup_smart_cache(verbose=True)

# Now safe to import (will use fast cache)
from indexing.multimodal_encoder import MultiModalEncoder
from data.video_dataset import VideoDataset
import glob
import os
import numpy as np

import torch
from tqdm import tqdm


def encode_batch(video_dir, save_dir, force_reencoding=False, force_video=None, force_audio=None, 
                 force_caption=None, force_text=None):
    """
    Encode videos in batch using multi-GPU parallelization.
    Distributes videos across all available GPUs and processes them in parallel.
    Each video is saved to its own pickle file after encoding.
    
    Args:
        video_dir: Directory containing video files
        save_dir: Directory to save encoded pickles
        force_reencoding: If True, re-encode all modalities (default for all force_* params)
        force_video: If True, force re-encode video embeddings. If None, uses force_reencoding.
        force_audio: If True, force re-encode audio embeddings. If None, uses force_reencoding.
        force_caption: If True, force re-encode captions. If None, uses force_reencoding.
        force_text: If True, force re-encode text embeddings. If None, uses force_reencoding.
    """
    os.makedirs(save_dir, exist_ok=True)

    video_ids = glob.glob(os.path.join(video_dir, "*.mp4"))
    print(f"Found {len(video_ids)} videos")

    config_txt_path = Path(save_dir) / "config_snapshot.txt"
    extra_info = {
        "experiment": Path(save_dir).name,
        "video_dir": video_dir,
        "save_dir": save_dir,
        "force_reencoding": force_reencoding,
        "force_video": force_video,
        "force_audio": force_audio,
        "force_caption": force_caption,
        "force_text": force_text,
        "batch_mode": True,
        "multi_gpu": True,
    }
    save_config_snapshot(CONFIG, config_txt_path, extra_info)
    print(f"Saved config snapshot to {config_txt_path}")

    # Step 1: Collect videos to process
    videos_to_process = []
    video_pickle_paths = {}
    
    for video in video_ids:
        base = os.path.splitext(os.path.basename(video))[0]
        pickle_path = f"{save_dir}/{base}_encoded.pkl"
        video_pickle_paths[video] = pickle_path
        
        # Skip if already encoded and not forcing re-encoding
        if os.path.exists(pickle_path) and not force_reencoding:
            print(f"Skipping {base} (already encoded)")
            continue
        
        videos_to_process.append(video)
    
    if not videos_to_process:
        print("All videos already encoded, nothing to process")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing {len(videos_to_process)} videos using multi-GPU batch mode")
    print(f"{'='*60}\n")
    
    # Step 2: Create dataset with all videos to process
    dataset = VideoDataset(videos_to_process)
    
    # Step 3: Initialize encoder with multi-GPU support
    encoder = MultiModalEncoder(
        video_dataset=dataset,
        max_workers=2,  # Reduce workers per GPU to avoid memory issues
        use_tagging=CONFIG.indexing.tag.use_tagging,
        gpu_devices=CONFIG.get('gpu_devices', None),
        global_video_embed=CONFIG.indexing.get('global_video_embed', True)
    )
    
    # Step 4: Encode all videos in parallel across GPUs
    print("Starting multi-GPU batch encoding...")
    video_dataset = encoder.encode_videos(
        force=force_reencoding,
        force_video=force_video,
        force_audio=force_audio,
        force_caption=force_caption,
        force_text=force_text
    )
    
    # Step 5: Save each video to its own pickle file
    print(f"\n{'='*60}")
    print("Saving individual pickle files...")
    print(f"{'='*60}\n")
    
    for dp in tqdm(video_dataset.video_datapoints, desc="Saving pickles"):
        # Create a single-video dataset for this video
        single_video_dataset = VideoDataset([])
        single_video_dataset.video_datapoints = [dp]
        single_video_dataset.encoded = True
        
        # Get the pickle path for this video
        pickle_path = video_pickle_paths[dp.video_path]
        
        # Save to pickle
        single_video_dataset.save_to_pickle(pickle_path)
        print(f"Saved {os.path.basename(pickle_path)}")
    
    # Cleanup
    del encoder
    del dataset
    del video_dataset
    
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n{'='*60}")
    print(f"Multi-GPU batch encoding complete! Processed {len(videos_to_process)} videos")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    video_dir = "../../tnanni/ego4d_data/v2/full_scale"
    save_dir = "../../tnanni/ego4d_data/v2/multigpu_test"
    output_pkl_file = os.path.join(save_dir, "merged_10_video.pkl")
    
    # Option 1: Re-encode everything
    force_reencoding = True
    
    # Option 2: Only update missing embeddings (default)
    # force_reencoding = False
    
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
    
    encode_batch(video_dir, save_dir, force_reencoding=force_reencoding)
    merged = merge_pickles(input_dir=save_dir, output_path=output_pkl_file, recursive=False)
    logging.info("Merging pickle files completed")
