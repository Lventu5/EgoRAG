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
from utils.merge_pickles import merge_pickles
from configuration.config import CONFIG, save_config_snapshot
# Setup smart cache (only copies large models like LLaVA)
# cache_info = setup_smart_cache(verbose=True)

# Now safe to import (will use fast cache)
import glob
import os
from multiprocessing import Process
import multiprocessing as mp

def encode(video_dir, save_dir, force_reencoding=False):
    import torch
    from data.video_dataset import VideoDataset

    os.makedirs(save_dir, exist_ok=True)

    video_paths = glob.glob(os.path.join(video_dir, "*.mp4"))
    dataset = VideoDataset(video_paths)

    num_gpus = torch.cuda.device_count()
    splits = split_round_robin(dataset.video_datapoints, num_gpus)

    processes = []
    for gpu_id, split in enumerate(splits):
        if not split:
            continue

        p = Process(
            target=encode_worker,
            args=(gpu_id, split, save_dir, force_reencoding),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    

def encode_worker(gpu_id, video_datapoints, save_dir, force):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from data.video_dataset import VideoDataset
    from indexing.multimodal_encoder import MultiModalEncoder

    print(f"[Worker {gpu_id}] CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"[Worker {gpu_id}] Visible GPU count = {torch.cuda.device_count()}")
    print(f"[Worker {gpu_id}] Current device = cuda:{torch.cuda.current_device()}")
    print(f"[Worker {gpu_id}] Device name = {torch.cuda.get_device_name(0)}")
    print(f"[Worker {gpu_id}] Process PID = {os.getpid()}")

    dataset = VideoDataset(video_files=[])
    dataset.video_datapoints = video_datapoints

    device_spec = f"cuda:0"
    encoder = MultiModalEncoder(
        video_dataset=dataset,
        device=device_spec,
        max_workers=1,
        use_tagging=CONFIG.indexing.tag.use_tagging,
        global_video_embed=False,
    )

    encoder.encode_videos(
        force=force,
        save_dir=save_dir, 
    )


def split_round_robin(items, n):
    return [items[i::n] for i in range(n)]



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    video_dir = "/cluster/project/cvg/data/ego4d/nlq_validation/v2/full_scale"
    save_dir = "../../tnanni/ego4d_data/v2/full_validation"
    output_pkl_file = os.path.join(save_dir, "merged_videos.pkl")
    
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
    
    encode(video_dir, save_dir, force_reencoding=force_reencoding)
    merged = merge_pickles(input_dir=save_dir, output_path=output_pkl_file, recursive=False)
    logging.info("Merging pickle files completed")
