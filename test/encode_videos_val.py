"""
Encode only the videos referenced in an Ego4D NLQ JSON (validation set).

This script:
- Reads the specified JSON file (default: `ego4d_data/v2/annotations/nlq_val.json`).
- Extracts all video IDs mentioned in the annotation file.
- Finds matching MP4 files under the provided `video_dir`.
- Runs `MultiModalEncoder` on each matched video and saves the encoded pickle to `save_dir`.

Usage:
    python test/encode_videos_val.py --json path/to/nlq_val.json \
        --video-dir ../ego4d_data/v2/full_scale --save-dir ../ego4d_data/v2/val_encoded

Options:
    --limit N    : only encode first N matched videos (useful for testing)
"""
import json
import os
import glob
import argparse
import gc
import logging

from utils.cache_manager import setup_smart_cache, cleanup_smart_cache
from indexing.multimodal_encoder import MultiModalEncoder
from data.video_dataset import VideoDataset
from utils.memory_monitor import MemoryMonitor
from tqdm import tqdm
import torch


def _extract_video_ids(obj, collector: set):
    """Recursively extract video id strings from a JSON object.
    Looks for common keys like 'video_uid', 'video', 'video_id' and also
    any string that looks like an Ego4D video uid (hex + dashes).
    """
    import re

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "video_uid" and isinstance(v, str):
                collector.add(v)
            else:
                _extract_video_ids(v, collector)

    elif isinstance(obj, list):
        for item in obj:
            _extract_video_ids(item, collector)


def find_video_files_for_ids(video_ids, video_dir):
    files = []
    for vid in video_ids:
        matches = glob.glob(os.path.join(video_dir, f"{vid}*.mp4"))
        if not matches:
            matches = glob.glob(os.path.join(video_dir, f"*{vid}*.mp4"))
        for m in matches:
            files.append(os.path.abspath(m))
    return sorted(list(dict.fromkeys(files)))


def encode(json_path, video_dir, save_dir, limit=None, force_reencoding=False,
           force_video=None, force_audio=None, force_caption=None, force_text=None):
    """
    Encode videos referenced in an Ego4D NLQ JSON file.
    
    Args:
        json_path: Path to JSON annotation file
        video_dir: Directory containing video files
        save_dir: Directory to save encoded pickles
        limit: Optional limit on number of videos to encode
        force_reencoding: If True, re-encode all modalities (default for all force_* params)
        force_video: If True, force re-encode video embeddings. If None, uses force_reencoding.
        force_audio: If True, force re-encode audio embeddings. If None, uses force_reencoding.
        force_caption: If True, force re-encode captions. If None, uses force_reencoding.
        force_text: If True, force re-encode text embeddings. If None, uses force_reencoding.
    """
    os.makedirs(save_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    video_ids = set()
    _extract_video_ids(data, video_ids)

    video_files = find_video_files_for_ids(video_ids, video_dir)
    if limit is not None:
        video_files = video_files[:limit]

    print(f"Found {len(video_files)} matching video files for {len(video_ids)} ids")

    print("="*80)
    MemoryMonitor.log_memory("[INITIAL STATE] ")
    print("="*80)

    for video in tqdm(video_files, desc="Encoding selected videos"):
        print("-" * 50)
        print(f"Encoding video {video}")
        print("-" * 50)
        
        base = os.path.splitext(os.path.basename(video))[0]
        pickle_path = os.path.join(save_dir, f"{base}_encoded.pkl")
        
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

        # cleanup per-video
        MemoryMonitor.log_memory("[BEFORE cleanup] ")
        del encoder
        if 'dataset' in locals():
            del dataset
        del video_dataset
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        MemoryMonitor.log_memory("[AFTER cleanup] ")
        print("=" * 80)


if __name__ == "__main__":
    cache_info = setup_smart_cache(verbose=True)
    video_directory = "../ego4d_data/v2/full_scale"
    save_directory = "../ego4d_data/v2/ep_mem_val_encoded"
    nlq_val_directory = "../ego4d_data/v2/annotations/nlq_val.json"
    
    # Option 1: Re-encode everything
    # force_reencoding = True
    
    # Option 2: Only update missing embeddings (default)
    force_reencoding = False
    
    # Option 3: Fine-grained control - force specific modalities
    # Example: Only re-encode video embeddings
    # force_video = True
    # force_audio = False
    # force_caption = False
    # force_text = False
    
    try:
        encode(nlq_val_directory, video_directory, save_directory, 
               limit=None, force_reencoding=force_reencoding)
        
        # With modality-specific control:
        # encode(nlq_val_directory, video_directory, save_directory,
        #        limit=None, force_video=True, force_audio=False,
        #        force_caption=False, force_text=False)
    finally:
        cleanup_smart_cache(cache_info, verbose=True)

