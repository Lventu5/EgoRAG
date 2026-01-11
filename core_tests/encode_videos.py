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
    os.makedirs(save_dir, exist_ok=True)

    video_ids = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))[:150]
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
    }
    save_config_snapshot(CONFIG, config_txt_path, extra_info)
    print(f"Saved config snapshot to {config_txt_path}")

    for video in tqdm(video_ids):
        print("-"*50)
        print(f"Encoding video {video}")
        print("-"*50)
        
        base = os.path.splitext(os.path.basename(video))[0]
        pickle_path = f"{save_dir}/{base}_encoded.pkl"
        
        # Determine if we should load existing pickle
        any_force = force_reencoding or force_video or force_audio or force_caption or force_text
        
        # Check if pickle already exists and load it if not forcing complete re-encoding
        if os.path.exists(pickle_path) and not force_reencoding:
            print(f"Found existing pickle at {pickle_path}, loading and updating...")
            encoder = MultiModalEncoder(
                pickle_path=pickle_path, 
                max_workers=2, 
                use_tagging=CONFIG.indexing.tag.use_tagging,
                gpu_devices=CONFIG.get('gpu_devices', None)
            )
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
            encoder = MultiModalEncoder(
                dataset, 
                max_workers=4,
                gpu_devices=CONFIG.get('gpu_devices', None)
            )
            video_dataset = encoder.encode_videos(
                force=force_reencoding,
                force_video=force_video,
                force_audio=force_audio,
                force_caption=force_caption,
                force_text=force_text
            )
        video_dataset.save_to_pickle(pickle_path)

        del encoder
        if 'dataset' in locals():
            del dataset
        del video_dataset
        
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()  # Double gc to ensure everything is freed

if __name__ == "__main__":
    video_dir = "/cluster/project/cvg/data/ego4d/nlq_validation/v2/full_scale/"
    save_dir = "../../tnanni/ego4d_data/v2/full_validation"
    output_pkl_file = os.path.join(save_dir, "merged_validation.pkl")
    
    # Option 1: Re-encode everything
    # force_reencoding = True
    
    # Option 2: Only update missing embeddings (default)
    force_reencoding = True
    
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
    merged = merge_pickles(input_dir=save_dir, output_path=output_pkl_file, recursive=False)
    logging.info("Merging pickle files completed")
    
    # With modality-specific control:
    # encode(video_dir, save_dir, 
    #        force_video=True, force_audio=False, 
    #        force_caption=False, force_text=False)
    
    # Clean up local cache after encoding
    # cleanup_smart_cache(cache_info, verbose=True)
