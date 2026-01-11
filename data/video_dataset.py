import torch
from torch.utils.data import Dataset
import os
import pickle
from typing import Dict, List, Optional
import numpy as np
from data.query import Query, QueryDataset


class Scene:
    """
    Rappresenta una singola scena di un video.
    Include tempi (in secondi) e frame di inizio/fine (in interi).
    """
    def __init__(
        self,
        scene_id: str,
        start_time: float,
        end_time: float,
        video_name: Optional[str] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        frames: Optional[List[int]] = None
    ):
        self.scene_id = scene_id
        self.video_name = video_name
        self.start_time = start_time
        self.end_time = end_time
        self.start_frame = start_frame if start_frame is not None else 0
        self.end_frame = end_frame if end_frame is not None else 0
        self.frames = frames or []

    def __repr__(self):
        return (
            f"Video name= {self.video_name}, "
            f"Scene id= {self.scene_id}, "
            f"start={self.start_time:.2f}s, end={self.end_time:.2f}s, "
            f"frames={self.start_frame}-{self.end_frame}, total_frames={len(self.frames)}"
        )
    
    def __hash__(self):
        # Use scene_id for hashing - scenes with same ID should hash the same
        return hash(self.scene_id)
    
    def __eq__(self, other):
        if not isinstance(other, Scene):
            return False
        return self.scene_id == other.scene_id


class Window:
    """
    Represents a sliding window over multiple consecutive scenes.
    Contains a unique window ID, start/end times, and the list of scene_ids
    that fall within this window.
    """
    def __init__(
        self,
        window_id: str,
        start_time: float,
        end_time: float,
        scene_ids: Optional[List[str]] = None
    ):
        self.window_id = window_id
        self.start_time = start_time
        self.end_time = end_time
        self.scene_ids = scene_ids or []

    def __repr__(self):
        return (
            f"Window id={self.window_id}, "
            f"start={self.start_time:.2f}s, end={self.end_time:.2f}s, "
            f"scenes={self.scene_ids}"
        )
    
    def __hash__(self):
        return hash(self.window_id)
    
    def __eq__(self, other):
        if not isinstance(other, Window):
            return False
        return self.window_id == other.window_id


class VideoDataPoint:
    """
    Contiene tutti i dati relativi a un singolo video:
      - path e nome file
      - lista delle scene
      - embeddings globali (video/audio/text)
      - embeddings per ogni scena
    """
    def __init__(self, video_path: str, scenes: Optional[Dict[str, Scene]] = None):
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.video_uid = os.path.splitext(self.video_name)[0]
        self.scenes = scenes or {}
        
        # Flag to indicate if this video has an audio track
        self.has_audio = True  # Assume True until proven otherwise during encoding

        # Windows: intermediate granularity between video and scenes
        self.windows: List[Window] = []
        
        # Embeddings per window (keyed by window_id)
        self.window_embeddings: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}

        # Embeddings globali (media sulle scene)
        self.global_embeddings: Dict[str, Optional[torch.Tensor | str]] = {
            "video": None,
            "audio": None,
            "text": None,  # Single text embedding from screenplay summary
            "text_raw": "",  # LLM-generated screenplay text
            "caption": None,
            "caption_text": "",
            "tags": None,
        }

        # Embeddings per ogni scena
        self.scene_embeddings: Dict[str, Dict] = {
            f"scene_{i}": {
                "video": None,
                "audio": None,
                "text": None,  # Single text embedding from screenplay summary
                "text_raw": "",  # LLM-generated screenplay text
                "image": {},
                "meta": {},
                "caption": None,
                "transcript": "",
                "caption_text": "",
                "tags": None,
            }
            for i, _ in enumerate(self.scenes)
        }
        self.queries: List[Query] = []

    def __repr__(self):
        return f"VideoDataPoint(name={self.video_name}, scenes={len(self.scenes)})"
    
    def get_scene_by_id(self, scene_id: str | int) -> Optional[Scene]:
        if isinstance(scene_id, int):
            scene_id = f"scene_{scene_id}"
        return self.scenes.get(scene_id, None)


class VideoDataset(Dataset):
    """
    Dataset di video per il MultiModalEncoder.
    Contiene una lista di VideoDataPoint (uno per video).
    """
    def __init__(self, video_files: List[str], scenes_per_video: Optional[Dict[str, Dict[str, Scene]]] = None, QueryDataset: Optional["QueryDataset"] = None):
        self.video_files = video_files
        self.video_datapoints: List[VideoDataPoint] = []
        self.scenes_per_video = scenes_per_video or {}
        self.encoded = False
        self.query_dataset = QueryDataset

        for path in video_files:
            scenes = self.scenes_per_video.get(path, {})
            self.video_datapoints.append(VideoDataPoint(path, scenes))

    def __len__(self):
        return len(self.video_datapoints)

    def __getitem__(self, idx):
        return self.video_datapoints[idx]
    
    def get_uids(self) -> List[str]:
        return [os.path.splitext(os.path.basename(v))[0] for v in self.video_files]
    
    def save_to_pickle(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Avoid saving large raw keyframes if present in scene_embeddings.
        # Permanently remove 'keyframes' entries prior to pickling to keep
        # the saved file small. NOTE: this mutates the in-memory dataset and
        # will not restore keyframes after saving.
        for dp in self.video_datapoints:
            se = getattr(dp, "scene_embeddings", {})
            if not isinstance(se, dict):
                continue
            for sid, scene_dict in list(se.items()):
                if isinstance(scene_dict, dict) and "keyframes" in scene_dict:
                    try:
                        scene_dict.pop("keyframes")
                    except Exception:
                        # if deletion fails, skip and continue
                        continue

        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_pickle(file_path: str) -> "VideoDataset":
        with open(file_path, 'rb') as f:
            ds = pickle.load(f)
        
        # Normalize video_name and video_uid in case they still have extensions
        for dp in ds.video_datapoints:
            if hasattr(dp, 'video_name') and '.' in dp.video_name:
                dp.video_name = os.path.splitext(dp.video_name)[0]
            if hasattr(dp, 'video_uid') and '.' in dp.video_uid:
                dp.video_uid = os.path.splitext(dp.video_uid)[0]
        
        return ds

    @staticmethod
    def save_datapoint_to_pickle(dp: VideoDataPoint, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Cleanup keyframes prima del salvataggio
        se = getattr(dp, "scene_embeddings", {})
        if isinstance(se, dict):
            for sid, scene_dict in se.items():
                if isinstance(scene_dict, dict):
                    scene_dict.pop("keyframes", None)

        with open(file_path, "wb") as f:
            pickle.dump(dp, f)

