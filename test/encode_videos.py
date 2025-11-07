"""
Code to encode the videos in a specific directory
"""
from indexing.multimodal_encoder import MultiModalEncoder
from data.video_dataset import VideoDataset
import glob
import os
import numpy as np

import torch
from tqdm import tqdm

def encode(video_dir, save_dir):
    video_ids = glob.glob(os.path.join(video_dir, "*.mp4"))
    print(f"Found {len(video_ids)} videos")

    for video in tqdm(video_ids[9:]):
        print("-"*50)
        print(f"Encoding video {video}")
        print("-"*50)
        dataset = VideoDataset([video])
<<<<<<< HEAD
        encoder = MultiModalEncoder(dataset, max_workers=4)
=======
        encoder = MultiModalEncoder(dataset, max_workers=1)
        # encoder.load_models()
>>>>>>> b045e90 (First captioner2 commit)
        video_dataset = encoder.encode_videos()
        # Remove the original extension (e.g. .mp4) from the video basename
        base = os.path.splitext(os.path.basename(video))[0]
        pickle_path = f"{save_dir}/{base}_encoded.pkl"
        video_dataset.save_to_pickle(pickle_path)
<<<<<<< HEAD
=======
        # encoder.unload_models()
>>>>>>> b045e90 (First captioner2 commit)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    video_dir = "../ego4d_data/v2/full_scale"
    save_dir = "../ego4d_data/v2/noframe_encoded_videos"
    encode(video_dir, save_dir)
