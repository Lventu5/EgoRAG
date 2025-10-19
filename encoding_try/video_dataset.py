import torch
from torch.utils.data import Dataset
import os
from typing import Dict, List
import numpy as np
import cv2


class Scene:
    def __init__(self, start_time: float, end_time: float, frames: List[int] | None = None):
        self.start_time = start_time
        self.end_time = end_time
        self.frames = frames or []


class VideoDataPoint:
    def __init__(
            self, 
            video_path: str, 
            scenes: List[Scene] | None = None
        ):
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.scenes = scenes or []

        self.global_embeddings: Dict[str, torch.Tensor | None] = {
            "video": None,
            "audio": None,
            "text": None
        }

        # scene-level embeddings
        self.scene_embeddings: Dict[str, Dict] = {} 

        for i, _ in enumerate(self.scenes):
            self.scene_embeddings[f"scene_{i}"] = {
                "video": None,
                "audio": None,
                "text": None,
                "image": {}
            }
            

class VideoDataset(Dataset):
    def __init__(self, video_files: List[str], scenes_per_video: Dict[str, List[Scene]] | None = None):
        self.video_files = video_files
        self.scenes_per_video = scenes_per_video or {}

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        scenes = self.scenes_per_video.get(video_file, [])
        return VideoDataPoint(video_file, scenes)
