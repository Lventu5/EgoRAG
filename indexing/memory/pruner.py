"""
Scene Pruner: Saliency-based filtering for memory management.

This module implements scene saliency computation and dataset pruning to
keep only the most relevant scenes for retrieval.
"""

import numpy as np
import torch
from typing import Dict, List
from collections import defaultdict

from data.video_dataset import VideoDataset, VideoDataPoint
from indexing.utils.logging import get_logger

logger = get_logger(__name__)


def compute_scene_saliency(scene_data: dict) -> float:
    """
    Compute a saliency score for a scene based on multi-modal features.
    
    Args:
        scene_data: Dictionary containing scene embeddings and metadata with keys:
            - video: video embedding tensor
            - audio: audio embedding tensor
            - text: text embedding tensor
            - caption: caption embedding tensor
            - transcript: transcript string
            - caption_text: caption string
            
    Returns:
        Saliency score (higher = more salient/important)
    """
    saliency = 0.0
    
    # 1. Speech presence: Scenes with speech are generally more informative
    transcript = scene_data.get("transcript", "")
    speech_weight = 0.3
    if len(transcript) > 20:  # Meaningful speech threshold
        saliency += speech_weight * min(len(transcript) / 200.0, 1.0)  # Normalize
    
    # 2. Video embedding variance: Higher variance indicates more visual activity
    video_emb = scene_data.get("video")
    if video_emb is not None and isinstance(video_emb, torch.Tensor):
        variance_weight = 0.25
        # Use L2 norm as proxy for information content
        norm = torch.norm(video_emb).item()
        # Normalize by a typical norm value (empirically ~10-20 for XCLIP)
        saliency += variance_weight * min(norm / 15.0, 1.0)
    
    # 3. Caption informativeness: Longer, more descriptive captions suggest richer content
    caption_text = scene_data.get("caption_text", "")
    caption_weight = 0.2
    if len(caption_text) > 10:
        # Favor descriptive captions
        saliency += caption_weight * min(len(caption_text) / 100.0, 1.0)
    
    # 4. Audio presence: Non-silent audio indicates activity
    audio_emb = scene_data.get("audio")
    if audio_emb is not None and isinstance(audio_emb, torch.Tensor):
        audio_weight = 0.15
        audio_norm = torch.norm(audio_emb).item()
        # Non-zero audio embedding suggests meaningful audio
        saliency += audio_weight * min(audio_norm / 10.0, 1.0)
    
    # 5. Text semantic richness: Embedding norm as proxy
    text_emb = scene_data.get("text")
    if text_emb is not None and isinstance(text_emb, torch.Tensor):
        text_weight = 0.1
        text_norm = torch.norm(text_emb).item()
        saliency += text_weight * min(text_norm / 5.0, 1.0)
    
    return saliency


def should_keep_scene(scene_data: dict, cfg: dict) -> bool:
    """
    Determine if a scene should be kept based on minimum thresholds.
    
    Args:
        scene_data: Dictionary containing scene embeddings and metadata
        cfg: Configuration dict with keys:
            - memory.saliency.min_frames: minimum number of frames
            - memory.saliency.min_audio_sec: minimum audio duration
            
    Returns:
        True if scene meets minimum criteria, False otherwise
    """
    # Extract configuration thresholds
    min_frames = cfg.get("memory", {}).get("saliency", {}).get("min_frames", 8)
    min_audio_sec = cfg.get("memory", {}).get("saliency", {}).get("min_audio_sec", 0.5)
    
    # Check frame count (if metadata available)
    if "keyframes" in scene_data:
        keyframes = scene_data["keyframes"]
        if isinstance(keyframes, np.ndarray) and len(keyframes) < min_frames:
            return False
    
    # Check audio duration (if timing metadata available)
    if "start_time" in scene_data and "end_time" in scene_data:
        duration = scene_data["end_time"] - scene_data["start_time"]
        if duration < min_audio_sec:
            return False
    
    # Check if scene has any embeddings at all
    has_embedding = any(
        scene_data.get(mod) is not None 
        for mod in ["video", "audio", "text", "caption"]
    )
    if not has_embedding:
        return False
    
    return True


def prune_dataset(video_dataset: VideoDataset, keep_ratio: float, cfg: dict) -> None:
    """
    Prune scenes from the dataset based on saliency scores.
    
    Keeps the top keep_ratio proportion of scenes per video based on saliency.
    Modifies the dataset in-place by removing low-saliency scenes from
    dp.scenes and dp.scene_embeddings.
    
    Args:
        video_dataset: VideoDataset to prune
        keep_ratio: Proportion of scenes to keep (0.0-1.0)
        cfg: Configuration dictionary
    """
    if not 0.0 <= keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio must be in [0, 1], got {keep_ratio}")
    
    logger.info(f"Pruning dataset with keep_ratio={keep_ratio}...")
    
    total_scenes_before = 0
    total_scenes_after = 0
    
    for dp in video_dataset.video_datapoints:
        scene_ids = list(dp.scene_embeddings.keys())
        total_scenes_before += len(scene_ids)
        
        if not scene_ids:
            continue
        
        # Compute saliency for each scene
        scene_saliencies = []
        for scene_id in scene_ids:
            scene_data = dp.scene_embeddings[scene_id]
            
            # First check if scene meets minimum criteria
            if not should_keep_scene(scene_data, cfg):
                scene_saliencies.append((scene_id, -1.0))  # Mark for removal
            else:
                saliency = compute_scene_saliency(scene_data)
                scene_saliencies.append((scene_id, saliency))
        
        # Sort by saliency (descending)
        scene_saliencies.sort(key=lambda x: x[1], reverse=True)
        
        # Determine how many to keep
        num_to_keep = max(1, int(len(scene_ids) * keep_ratio))  # Keep at least 1 scene
        
        # Get scenes to keep
        scenes_to_keep = set()
        for scene_id, saliency in scene_saliencies[:num_to_keep]:
            if saliency >= 0:  # Only keep scenes that passed minimum criteria
                scenes_to_keep.add(scene_id)
        
        # Remove pruned scenes
        scenes_removed = []
        for scene_id in scene_ids:
            if scene_id not in scenes_to_keep:
                if scene_id in dp.scene_embeddings:
                    del dp.scene_embeddings[scene_id]
                if scene_id in dp.scenes:
                    del dp.scenes[scene_id]
                scenes_removed.append(scene_id)
        
        total_scenes_after += len(scenes_to_keep)
        
        if scenes_removed:
            logger.debug(
                f"Video {dp.video_name}: removed {len(scenes_removed)}/{len(scene_ids)} scenes"
            )
    
    # Re-compute global embeddings after pruning
    for dp in video_dataset.video_datapoints:
        if dp.scene_embeddings:
            dp.global_embeddings = _aggregate_embeddings(dp.scene_embeddings)
    
    pruned_count = total_scenes_before - total_scenes_after
    pruned_pct = (pruned_count / total_scenes_before * 100) if total_scenes_before > 0 else 0
    
    logger.info(
        f"Pruning complete: {total_scenes_before} -> {total_scenes_after} scenes "
        f"({pruned_count} removed, {pruned_pct:.1f}%)"
    )


def _aggregate_embeddings(scene_embeddings: dict) -> dict:
    """
    Aggregate scene embeddings to create global video embeddings.
    
    Args:
        scene_embeddings: Dict of scene_id -> scene_data
        
    Returns:
        Dictionary with aggregated embeddings per modality
    """
    global_embs = {"video": [], "audio": [], "text": [], "caption": []}
    
    for scene_data in scene_embeddings.values():
        for key in global_embs.keys():
            if scene_data.get(key) is not None:
                global_embs[key].append(scene_data[key])
    
    aggregated = {}
    for key, embs in global_embs.items():
        if embs:
            aggregated[key] = torch.stack(embs).mean(dim=0)
        else:
            aggregated[key] = None
    
    return aggregated
