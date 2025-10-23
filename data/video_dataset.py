import torch
from torch.utils.data import Dataset
import os
from typing import Dict, List, Optional
import numpy as np


class Scene:
    """
    Rappresenta una singola scena di un video.
    Include tempi (in secondi) e frame di inizio/fine (in interi).
    """
    def __init__(
        self,
        start_time: float,
        end_time: float,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        frames: Optional[List[int]] = None
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.start_frame = start_frame if start_frame is not None else 0
        self.end_frame = end_frame if end_frame is not None else 0
        self.frames = frames or []

    def __repr__(self):
        return (
            f"Scene(start={self.start_time:.2f}s, end={self.end_time:.2f}s, "
            f"frames={self.start_frame}-{self.end_frame}, total_frames={len(self.frames)})"
        )


class VideoDataPoint:
    """
    Contiene tutti i dati relativi a un singolo video:
      - path e nome file
      - lista delle scene
      - embeddings globali (video/audio/text)
      - embeddings per ogni scena
    """
    def __init__(self, video_path: str, scenes: Optional[List[Scene]] = None):
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.scenes = scenes or []

        # Embeddings globali (media sulle scene)
        self.global_embeddings: Dict[str, Optional[torch.Tensor]] = {
            "video": None,
            "audio": None,
            "text": None,
            "caption": None,
        }

        # Embeddings per ogni scena
        self.scene_embeddings: Dict[str, Dict] = {
            f"scene_{i}": {
                "video": None,
                "audio": None,
                "text": None,
                "transcript": "",
                "image": {},
                "meta": {},
                "caption": None,
            }
            for i, _ in enumerate(self.scenes)
        }

    def __repr__(self):
        return f"VideoDataPoint(name={self.video_name}, scenes={len(self.scenes)})"


class VideoDataset(Dataset):
    """
    Dataset di video per il MultiModalEncoder.
    Contiene una lista di VideoDataPoint (uno per video).
    """
    def __init__(self, video_files: List[str], scenes_per_video: Optional[Dict[str, List[Scene]]] = None):
        self.video_files = video_files
        self.video_datapoints: List[VideoDataPoint] = []
        self.scenes_per_video = scenes_per_video or {}

        for path in video_files:
            scenes = self.scenes_per_video.get(path, [])
            self.video_datapoints.append(VideoDataPoint(path, scenes))

    def __len__(self):
        return len(self.video_datapoints)

    def __getitem__(self, idx):
        return self.video_datapoints[idx]
