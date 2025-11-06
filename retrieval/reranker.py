"""
Reranker: Confidence-aware reranking with temporal reasoning.

This module provides functions and classes for reranking retrieved scenes
based on modality agreement and temporal proximity.
"""

import numpy as np
from typing import Dict, Optional
from data.video_dataset import Scene
from indexing.utils.logging import get_logger

logger = get_logger(__name__)


def modality_agreement(scores_by_modality: Dict[str, float]) -> float:
    """
    Compute consensus score across modalities with std-dev penalty.
    
    High agreement means all modalities give similar scores, indicating
    confident retrieval. Low agreement suggests uncertainty.
    
    Args:
        scores_by_modality: Dict mapping modality name to score
        
    Returns:
        Agreement score (mean - std_dev)
    """
    if not scores_by_modality:
        return 0.0
    
    scores = list(scores_by_modality.values())
    
    if len(scores) == 1:
        return scores[0]
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    # Penalize high variance (low agreement)
    agreement = mean_score - std_score
    
    return float(agreement)


def temporal_decay(delta_sec: float, half_life: float) -> float:
    """
    Compute temporal decay factor based on time difference.
    
    Uses exponential decay: 0.5 ** (delta / half_life)
    Scenes closer to query time hint receive higher scores.
    
    Args:
        delta_sec: Absolute time difference in seconds
        half_life: Half-life for decay in seconds
        
    Returns:
        Decay factor in [0, 1]
    """
    if half_life <= 0:
        return 1.0
    
    return 0.5 ** (delta_sec / half_life)


class Reranker:
    """
    Reranks retrieved scenes using multi-factor scoring.
    
    Combines:
    - Base fused score (from retrieval)
    - Modality agreement (confidence)
    - Temporal proximity (if time hint available)
    """
    
    def __init__(
        self, 
        alpha: float = 0.70,
        beta: float = 0.20, 
        gamma: float = 0.10,
        half_life_sec: float = 86400.0
    ):
        """
        Initialize reranker with scoring weights.
        
        Args:
            alpha: Weight for base fused score
            beta: Weight for modality agreement
            gamma: Weight for temporal decay
            half_life_sec: Half-life for temporal decay (default: 1 day)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.half_life_sec = half_life_sec
        
        # Normalize weights
        total = alpha + beta + gamma
        self.alpha /= total
        self.beta /= total
        self.gamma /= total
        
        logger.info(
            f"Reranker initialized: alpha={self.alpha:.3f}, beta={self.beta:.3f}, "
            f"gamma={self.gamma:.3f}, half_life={half_life_sec}s"
        )
    
    def score_scene(
        self,
        query_time_hint: Optional[float],
        scene: Scene,
        fused_score: float,
        scores_by_modality: Dict[str, float]
    ) -> float:
        """
        Compute reranked score for a scene.
        
        Args:
            query_time_hint: Optional time hint in seconds (None if not available)
            scene: Scene object with timing information
            fused_score: Base fused retrieval score
            scores_by_modality: Dict of scores per modality
            
        Returns:
            Reranked score
        """
        # Base fused score component
        base = fused_score
        
        # Modality agreement component
        agree = modality_agreement(scores_by_modality)
        
        # Temporal decay component
        if query_time_hint is not None and hasattr(scene, "start_time"):
            # Use scene midpoint for comparison
            scene_midpoint = (scene.start_time + scene.end_time) / 2.0
            delta = abs(scene_midpoint - query_time_hint)
            decay = temporal_decay(delta, self.half_life_sec)
        else:
            decay = 1.0  # No temporal information, no penalty
        
        # Weighted combination
        final_score = self.alpha * base + self.beta * agree + self.gamma * decay
        
        return final_score
