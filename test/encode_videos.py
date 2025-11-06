"""
Code to encode the videos in a specific directory
"""
from indexing.multimodal_encoder import MultiModalEncoder
from data.video_dataset import VideoDataset
import glob
import os
import torch
from tqdm import tqdm

def encode(video_dir, save_dir):
    video_ids = glob.glob(os.path.join(video_dir, "*.mp4"))
    print(f"Found {len(video_ids)} videos")

    for video in tqdm(video_ids):
        print("-"*50)
        print(f"Encoding video {video}")
        print("-"*50)
        dataset = VideoDataset([video])
        encoder = MultiModalEncoder(dataset, max_workers=2)
        encoder.load_models()
        video_dataset = encoder.encode_videos()
        pickle_path = f"{save_dir}/{video}_encoded.pkl"
        video_dataset.save_to_pickle(pickle_path)
        encoder.unload_models()
        torch.cuda.empty_cache()
        break

if __name__ == "__main__":
    video_dir = "../ego4d_data/v2/full_scale"
    save_dir = "../ego4d_data/v2/encoded_videos"
    encode(video_dir, save_dir)
