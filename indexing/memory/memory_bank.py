"""
SceneMemoryBank: Modular indexing system for video scene embeddings.

This module provides efficient in-memory storage and querying of multi-modal
scene embeddings at both video and scene levels.
"""

import pickle
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
from collections import defaultdict

from data.video_dataset import VideoDataset
from indexing.utils.logging import get_logger

logger = get_logger(__name__)


class SceneMemoryBank:
    """
    In-memory index for multi-modal video scene embeddings.
    
    Supports hierarchical queries at video and scene levels for efficient retrieval.
    """
    
    def __init__(self, video_dataset: VideoDataset):
        """
        Initialize the memory bank with a video dataset.
        
        Args:
            video_dataset: VideoDataset containing encoded videos and scenes
        """
        self.dataset = video_dataset
        # Per-modality indices: modality -> {id -> embedding}
        self.indices: Dict[str, Dict[str, torch.Tensor]] = {}
        # Per-modality metadata: modality -> {id -> metadata dict}
        self.metadata: Dict[str, Dict[str, dict]] = {}
        self._built = False
    
    def build_indices(self) -> None:
        """
        Build per-modality indices from the dataset.
        
        Creates separate indices for each modality (video, audio, text, caption) at
        both video-level and scene-level granularity.
        """
        logger.info("Building SceneMemoryBank indices...")
        
        # Initialize index structures
        modalities = ["video", "audio", "text", "caption"]
        for mod in modalities:
            self.indices[mod] = {}
            self.metadata[mod] = {}
        
        # Iterate over all video datapoints
        for dp in self.dataset.video_datapoints:
            video_name = dp.video_name
            
            # Store video-level embeddings
            if dp.global_embeddings:
                for mod in modalities:
                    if mod in dp.global_embeddings and dp.global_embeddings[mod] is not None:
                        self.indices[mod][video_name] = dp.global_embeddings[mod]
                        self.metadata[mod][video_name] = {
                            "type": "video",
                            "video_name": video_name,
                            "video_path": dp.video_path,
                            "num_scenes": len(dp.scenes)
                        }
            
            # Store scene-level embeddings
            for scene_id, scene_data in dp.scene_embeddings.items():
                scene_key = f"{video_name}_{scene_id}"
                scene_obj = dp.scenes.get(scene_id)
                
                for mod in modalities:
                    if mod in scene_data and scene_data[mod] is not None:
                        self.indices[mod][scene_key] = scene_data[mod]
                        
                        # Build metadata
                        meta = {
                            "type": "scene",
                            "video_name": video_name,
                            "scene_id": scene_id,
                            "transcript": scene_data.get("transcript", ""),
                            "caption_text": scene_data.get("caption_text", ""),
                        }
                        
                        # Add scene timing info if available
                        if scene_obj:
                            meta.update({
                                "start_time": scene_obj.start_time,
                                "end_time": scene_obj.end_time,
                                "start_frame": scene_obj.start_frame,
                                "end_frame": scene_obj.end_frame,
                            })
                        
                        # Add meta dict if present
                        if "meta" in scene_data:
                            meta.update(scene_data["meta"])
                        
                        self.metadata[mod][scene_key] = meta
        
        self._built = True
        
        # Log statistics
        for mod in modalities:
            video_count = sum(1 for m in self.metadata[mod].values() if m.get("type") == "video")
            scene_count = sum(1 for m in self.metadata[mod].values() if m.get("type") == "scene")
            logger.info(f"  [{mod}] Indexed {video_count} videos, {scene_count} scenes")
        
        logger.info("SceneMemoryBank indices built successfully.")
    
    def query_video_level(self, modality: str) -> Tuple[List[str], torch.Tensor]:
        """
        Query all video-level embeddings for a given modality.
        
        Args:
            modality: One of 'video', 'audio', 'text', 'caption'
            
        Returns:
            Tuple of (video_names, embeddings_tensor)
        """
        if not self._built:
            raise RuntimeError("Indices not built. Call build_indices() first.")
        
        if modality not in self.indices:
            raise ValueError(f"Unknown modality: {modality}")
        
        video_names = []
        embeddings = []
        
        for key, emb in self.indices[modality].items():
            meta = self.metadata[modality][key]
            if meta.get("type") == "video":
                video_names.append(key)
                embeddings.append(emb)
        
        if not embeddings:
            return [], torch.tensor([])
        
        return video_names, torch.stack(embeddings)
    
    def query_scene_level(
        self, 
        video_name: str, 
        modality: str,
        filters: Optional[Dict] = None
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Query scene-level embeddings for a specific video and modality.
        
        Args:
            video_name: Name of the video
            modality: One of 'video', 'audio', 'text', 'caption'
            filters: Optional dict with filter criteria (e.g., scene_type_in, speech_type_in)
            
        Returns:
            Tuple of (scene_ids, embeddings_tensor)
        """
        if not self._built:
            raise RuntimeError("Indices not built. Call build_indices() first.")
        
        if modality not in self.indices:
            raise ValueError(f"Unknown modality: {modality}")
        
        scene_ids = []
        embeddings = []
        
        for key, emb in self.indices[modality].items():
            meta = self.metadata[modality][key]
            
            # Check if this is a scene for the requested video
            if meta.get("type") == "scene" and meta.get("video_name") == video_name:
                # Apply filters if provided
                if filters:
                    # Filter by scene_type
                    if "scene_type_in" in filters:
                        scene_type = meta.get("scene_type")
                        if scene_type not in filters["scene_type_in"]:
                            continue
                    
                    # Filter by speech_type
                    if "speech_type_in" in filters:
                        speech_type = meta.get("speech_type")
                        if speech_type not in filters["speech_type_in"]:
                            continue
                
                scene_ids.append(key)
                embeddings.append(emb)
        
        if not embeddings:
            return [], torch.tensor([])
        
        return scene_ids, torch.stack(embeddings)
    
    def get_scene_metadata(self, scene_key: str, modality: str = "video") -> Optional[dict]:
        """
        Retrieve metadata for a specific scene.
        
        Args:
            scene_key: Scene identifier (format: video_name_scene_id)
            modality: Modality to query (default: 'video')
            
        Returns:
            Metadata dictionary or None if not found
        """
        if not self._built:
            raise RuntimeError("Indices not built. Call build_indices() first.")
        
        return self.metadata.get(modality, {}).get(scene_key)
    
    def serialize(self, path: str) -> None:
        """
        Serialize the memory bank to disk.
        
        Args:
            path: File path to save the pickled memory bank
        """
        if not self._built:
            logger.warning("Serializing memory bank before indices are built.")
        
        data = {
            "indices": self.indices,
            "metadata": self.metadata,
            "built": self._built
        }
        
        with open(path, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"SceneMemoryBank serialized to {path}")
    
    def load(self, path: str) -> None:
        """
        Load a serialized memory bank from disk.
        
        Args:
            path: File path to the pickled memory bank
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.indices = data["indices"]
        self.metadata = data["metadata"]
        self._built = data.get("built", False)
        
        logger.info(f"SceneMemoryBank loaded from {path}")
        
        # Log statistics
        for mod in self.indices.keys():
            video_count = sum(1 for m in self.metadata[mod].values() if m.get("type") == "video")
            scene_count = sum(1 for m in self.metadata[mod].values() if m.get("type") == "scene")
            logger.info(f"  [{mod}] Loaded {video_count} videos, {scene_count} scenes")
    
    def is_built(self) -> bool:
        """Check if indices have been built."""
        return self._built
