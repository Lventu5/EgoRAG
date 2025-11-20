"""
Scene Merger Module

This module provides functionality to merge consecutive scenes in retrieval results
to improve IoU-based metrics by creating larger temporal windows that better overlap
with ground truth moments.
"""

import logging
from typing import List, Tuple
from data.video_dataset import Scene


class MergedScene(Scene):
    """
    Represents a scene created by merging multiple consecutive scenes.
    Inherits from Scene and adds information about the constituent scenes.
    """
    def __init__(
        self,
        merged_scenes: List[Scene],
        scene_id: str = None,
    ):
        if not merged_scenes:
            raise ValueError("Cannot create MergedScene from empty list")
        
        # Sort scenes by start time to ensure correct ordering
        sorted_scenes = sorted(merged_scenes, key=lambda s: s.start_time)
        
        # The merged scene spans from the earliest start to the latest end
        start_time = sorted_scenes[0].start_time
        end_time = sorted_scenes[-1].end_time
        start_frame = sorted_scenes[0].start_frame
        end_frame = sorted_scenes[-1].end_frame
        
        # Combine all frames
        all_frames = []
        for scene in sorted_scenes:
            all_frames.extend(scene.frames)
        
        # Generate a composite scene_id if not provided
        if scene_id is None:
            scene_ids = [s.scene_id for s in sorted_scenes]
            scene_id = f"merged_{'_'.join(scene_ids)}"
        
        super().__init__(
            scene_id=scene_id,
            start_time=start_time,
            end_time=end_time,
            start_frame=start_frame,
            end_frame=end_frame,
            frames=all_frames
        )
        
        self.constituent_scenes = sorted_scenes
        self.num_merged = len(sorted_scenes)
    
    def __repr__(self):
        return (
            f"MergedScene(id={self.scene_id}, "
            f"start={self.start_time:.2f}s, end={self.end_time:.2f}s, "
            f"merged_count={self.num_merged})"
        )


class SceneMerger:
    """
    Merges consecutive scenes in top-K retrieval results.
    
    This improves IoU-based metrics by creating larger temporal windows
    that have better overlap with ground truth moments.
    """
    
    def __init__(
        self, 
        max_gap: float = 0.5,
        min_scenes_to_merge: int = 2,
        max_scenes_to_merge: int = 5,
    ):
        """
        Args:
            max_gap: Maximum time gap (in seconds) between scenes to consider them consecutive
            min_scenes_to_merge: Minimum number of scenes required to create a merge (default: 2)
            max_scenes_to_merge: Maximum number of scenes to merge into one (default: 5)
        """
        self.max_gap = max_gap
        self.min_scenes_to_merge = min_scenes_to_merge
        self.max_scenes_to_merge = max_scenes_to_merge
        logging.info(
            f"SceneMerger initialized with max_gap={max_gap}s, "
            f"min_scenes={min_scenes_to_merge}, max_scenes={max_scenes_to_merge}"
        )
    
    def _are_consecutive(self, scene1: Scene, scene2: Scene) -> bool:
        """
        Check if two scenes are consecutive (scene2 follows scene1).
        
        Scenes are considered consecutive if:
        1. scene2 starts after scene1
        2. The gap between them is <= max_gap
        """
        gap = scene2.start_time - scene1.end_time
        return 0 <= gap <= self.max_gap
    
    def _find_consecutive_groups(self, scenes: List[Scene]) -> List[List[int]]:
        """
        Find groups of consecutive scene indices.
        
        Args:
            scenes: List of Scene objects (already sorted by start_time)
            
        Returns:
            List of groups, where each group is a list of indices into the scenes list
        """
        if not scenes:
            return []
        
        groups = []
        current_group = [0]
        
        for i in range(1, len(scenes)):
            if self._are_consecutive(scenes[i-1], scenes[i]):
                current_group.append(i)
                # Limit group size
                if len(current_group) >= self.max_scenes_to_merge:
                    groups.append(current_group)
                    current_group = []
            else:
                if len(current_group) >= self.min_scenes_to_merge:
                    groups.append(current_group)
                current_group = [i]
        
        # Don't forget the last group
        if len(current_group) >= self.min_scenes_to_merge:
            groups.append(current_group)
        
        return groups
    
    def merge_top_k_scenes(
        self, 
        top_k_scenes: List[Tuple[Scene, float]],
        preserve_scores: bool = True,
        score_aggregation: str = "max"
    ) -> List[Tuple[Scene, float]]:
        """
        Merge consecutive scenes in a top-K scene list.
        
        Args:
            top_k_scenes: List of (Scene, score) tuples from retrieval
            preserve_scores: If True, keep individual scene scores; if False, aggregate them
            score_aggregation: How to aggregate scores when merging ("max", "mean", "sum")
            
        Returns:
            List of (Scene, score) tuples with consecutive scenes merged
        """
        if not top_k_scenes:
            return []
        
        # Separate scenes and scores
        scenes = [scene for scene, _ in top_k_scenes]
        scores = [score for _, score in top_k_scenes]
        
        # Sort by start time (and keep track of original order via scores)
        sorted_indices = sorted(range(len(scenes)), key=lambda i: scenes[i].start_time)
        sorted_scenes = [scenes[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        
        # Find consecutive groups
        consecutive_groups = self._find_consecutive_groups(sorted_scenes)
        
        if not consecutive_groups:
            # No merging needed, return original
            return top_k_scenes
        
        # Track which indices are part of a group
        indices_in_groups = set()
        for group in consecutive_groups:
            indices_in_groups.update(group)
        
        # Build result list
        result = []
        processed_indices = set()
        
        for i in range(len(sorted_scenes)):
            if i in processed_indices:
                continue
            
            # Check if this index is part of a consecutive group
            in_group = False
            for group in consecutive_groups:
                if i in group:
                    # Merge this group
                    group_scenes = [sorted_scenes[idx] for idx in group]
                    group_scores = [sorted_scores[idx] for idx in group]
                    
                    merged_scene = MergedScene(group_scenes)
                    
                    # Aggregate scores
                    if score_aggregation == "max":
                        merged_score = max(group_scores)
                    elif score_aggregation == "mean":
                        merged_score = sum(group_scores) / len(group_scores)
                    elif score_aggregation == "sum":
                        merged_score = sum(group_scores)
                    else:
                        raise ValueError(f"Unknown score_aggregation: {score_aggregation}")
                    
                    result.append((merged_scene, merged_score))
                    processed_indices.update(group)
                    in_group = True
                    
                    logging.debug(
                        f"Merged {len(group)} scenes: {merged_scene.scene_id} "
                        f"[{merged_scene.start_time:.2f}s - {merged_scene.end_time:.2f}s] "
                        f"with score {merged_score:.4f}"
                    )
                    break
            
            if not in_group:
                # Keep individual scene
                result.append((sorted_scenes[i], sorted_scores[i]))
                processed_indices.add(i)
        
        # Sort result by score (descending) to maintain top-K ordering
        result.sort(key=lambda x: x[1], reverse=True)
        
        return result
    
    def merge_batch_results(
        self,
        batch_results: List[List[Tuple[Scene, float]]],
        **merge_kwargs
    ) -> List[List[Tuple[Scene, float]]]:
        """
        Merge consecutive scenes for a batch of retrieval results.
        
        Args:
            batch_results: List where each element is a list of (Scene, score) tuples
            **merge_kwargs: Additional arguments passed to merge_top_k_scenes
            
        Returns:
            Batch results with consecutive scenes merged
        """
        merged_batch = []
        for query_results in batch_results:
            merged_results = self.merge_top_k_scenes(query_results, **merge_kwargs)
            merged_batch.append(merged_results)
        
        return merged_batch


def merge_consecutive_scenes(
    top_k_scenes: List[Tuple[Scene, float]],
    max_gap: float = 0.5,
    min_scenes_to_merge: int = 2,
    max_scenes_to_merge: int = 5,
    score_aggregation: str = "max"
) -> List[Tuple[Scene, float]]:
    """
    Convenience function to merge consecutive scenes.
    
    Args:
        top_k_scenes: List of (Scene, score) tuples from retrieval
        max_gap: Maximum time gap (in seconds) between scenes to consider them consecutive
        min_scenes_to_merge: Minimum number of scenes required to create a merge
        max_scenes_to_merge: Maximum number of scenes to merge into one
        score_aggregation: How to aggregate scores when merging ("max", "mean", "sum")
        
    Returns:
        List of (Scene, score) tuples with consecutive scenes merged
    """
    merger = SceneMerger(
        max_gap=max_gap,
        min_scenes_to_merge=min_scenes_to_merge,
        max_scenes_to_merge=max_scenes_to_merge
    )
    return merger.merge_top_k_scenes(
        top_k_scenes,
        score_aggregation=score_aggregation
    )
