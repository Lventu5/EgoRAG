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


def _remove_legacy_text_keys(video_dataset: VideoDataset, legacy_keys: list[str]) -> None:
    """Remove legacy text embedding keys from all scene/window/global embeddings in-place."""
    for dp in video_dataset.video_datapoints:
        for scene_data in dp.scene_embeddings.values():
            for k in legacy_keys:
                scene_data.pop(k, None)
        for window_data in getattr(dp, "window_embeddings", {}).values():
            for k in legacy_keys:
                window_data.pop(k, None)
        for k in legacy_keys:
            dp.global_embeddings.pop(k, None)


def encode(video_dir, save_dir, force_reencoding=False, force_video=None, force_audio=None,
           force_caption=None, force_text=None, remove_legacy_text=False):
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
        remove_legacy_text: If True, remove the old "text" key (gemma) after encoding.
    """
    os.makedirs(save_dir, exist_ok=True)
    target_subjects = ["A1_JAKE", "A2_ALICE", "A3_TASHA", "A4_LUCIA", "A5_KATRINA", "A6_SHURE"]
    
    all_videos = []
    for subject in target_subjects:
        subject_path = os.path.join(video_dir, subject)
        if os.path.exists(subject_path):
            videos = glob.glob(os.path.join(subject_path, "**/*.mp4"), recursive=True)
            all_videos.extend(videos)
    
    video_ids = sorted(all_videos)
    
    print(f"Total found videos (A1-A6): {len(all_videos)}")

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
                max_workers=2,
                use_tagging=CONFIG.indexing.tag.use_tagging,
                global_video_embed=False,
            )
            video_dataset = encoder.encode_videos(
                force=force_reencoding,
                force_video=force_video,
                force_audio=force_audio,
                force_caption=force_caption,
                force_text=force_text
            )
        if remove_legacy_text:
            _remove_legacy_text_keys(video_dataset, legacy_keys=["text"])
        video_dataset.save_to_pickle(pickle_path)

        del encoder
        if 'dataset' in locals():
            del dataset
        del video_dataset
        
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()  # Double gc to ensure everything is freed

if __name__ == "__main__":
    video_dir = "/cluster/project/cvg/data/EgoLife"
    save_dir = "../../tnanni/ego4d_data/v2/egolife_full_qwen3"
    output_pkl_file = os.path.join(save_dir, "merged_validation.pkl")

    # Re-encode only text with Qwen3, keep all other embeddings from the copied pkls
    encode(video_dir, save_dir,
           force_video=False,
           force_audio=False,
           force_caption=False,
           force_text=True,
           remove_legacy_text=True)
    merged = merge_pickles(input_dir=save_dir, output_path=output_pkl_file, recursive=False)
    logging.info("Merging pickle files completed")
