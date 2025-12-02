"""
Multi-Resolution Retrieval Module

Enables retrieval across multiple scene granularities to balance 
precision (short scenes) and recall (long scenes).

Strategy:
1. Index videos at multiple scene lengths (e.g., 6s, 12s, 24s)
2. At retrieval, search across all resolutions
3. Fuse results from different resolutions using configurable strategies
"""

import logging
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from data.video_dataset import Scene, VideoDataset, VideoDataPoint


@dataclass
class MultiResolutionScene:
    """
    A scene with resolution metadata.
    """
    scene: Scene
    resolution: float  # Scene length in seconds
    video_name: str
    score: float = 0.0
    
    @property
    def scene_id(self) -> str:
        return f"{self.resolution}s_{self.scene.scene_id}"
    
    @property
    def start_time(self) -> float:
        return self.scene.start_time
    
    @property
    def end_time(self) -> float:
        return self.scene.end_time


class VirtualMultiResolutionScene(Scene):
    """
    A virtual scene created by aggregating base scenes to form a larger resolution.
    This allows multi-resolution retrieval without re-encoding the video.
    """
    def __init__(
        self,
        base_scenes: List[Scene],
        resolution: float,
        video_name: str,
    ):
        if not base_scenes:
            raise ValueError("base_scenes cannot be empty")
        
        # Sort by start time
        sorted_scenes = sorted(base_scenes, key=lambda s: s.start_time)
        
        start_time = sorted_scenes[0].start_time
        end_time = sorted_scenes[-1].end_time
        start_frame = sorted_scenes[0].start_frame
        end_frame = sorted_scenes[-1].end_frame
        
        scene_id = f"virt_{resolution}s_{sorted_scenes[0].scene_id}_to_{sorted_scenes[-1].scene_id}"
        
        super().__init__(
            scene_id=scene_id,
            start_time=start_time,
            end_time=end_time,
            start_frame=start_frame,
            end_frame=end_frame,
            frames=[]
        )
        
        self.base_scenes = sorted_scenes
        self.resolution = resolution
        self.video_name = video_name
        self.num_base_scenes = len(sorted_scenes)


class MultiResolutionIndex:
    """
    Creates virtual multi-resolution scenes from a base-resolution dataset.
    
    Given a dataset encoded at a base resolution (e.g., 12s scenes), this class
    creates virtual scenes at higher resolutions (e.g., 24s, 36s) by aggregating
    consecutive base scenes.
    
    The embeddings for virtual scenes are computed as the mean of constituent
    base scene embeddings.
    """
    
    def __init__(
        self,
        video_dataset: VideoDataset,
        base_resolution: float,
        target_resolutions: List[float],
        overlap_factor: float = 0.5,
    ):
        """
        Args:
            video_dataset: The base video dataset with encoded scenes
            base_resolution: The scene length used during encoding (seconds)
            target_resolutions: List of additional resolutions to create (seconds)
            overlap_factor: How much virtual scenes should overlap (0.5 = 50% overlap)
        """
        self.video_dataset = video_dataset
        self.base_resolution = base_resolution
        self.target_resolutions = sorted([r for r in target_resolutions if r > base_resolution])
        self.overlap_factor = overlap_factor
        
        # Cache for virtual scene embeddings
        self._virtual_embeddings: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
        # video_name -> resolution -> scene_id -> embeddings_dict
        
        # Cache for virtual scenes
        self._virtual_scenes: Dict[str, Dict[float, List[VirtualMultiResolutionScene]]] = {}
        # video_name -> resolution -> list of virtual scenes
        
        logging.info(
            f"MultiResolutionIndex: base={base_resolution}s, "
            f"targets={self.target_resolutions}, overlap={overlap_factor}"
        )
        
        # Build virtual scenes for all videos
        self._build_virtual_scenes()
    
    def _build_virtual_scenes(self):
        """Build virtual scenes for all videos at all target resolutions."""
        for dp in self.video_dataset.video_datapoints:
            self._virtual_scenes[dp.video_name] = {}
            self._virtual_embeddings[dp.video_name] = {}
            
            for resolution in self.target_resolutions:
                virtual_scenes, virtual_embeddings = self._create_virtual_scenes_for_video(
                    dp, resolution
                )
                self._virtual_scenes[dp.video_name][resolution] = virtual_scenes
                self._virtual_embeddings[dp.video_name][resolution] = virtual_embeddings
    
    def _create_virtual_scenes_for_video(
        self,
        dp: VideoDataPoint,
        target_resolution: float
    ) -> Tuple[List[VirtualMultiResolutionScene], Dict[str, Dict[str, torch.Tensor]]]:
        """
        Create virtual scenes for a single video at a target resolution.
        
        Args:
            dp: VideoDataPoint with base-resolution scenes
            target_resolution: Target scene length in seconds
            
        Returns:
            Tuple of (list of virtual scenes, dict of virtual embeddings)
        """
        # Sort base scenes by start time
        base_scenes = sorted(dp.scenes.values(), key=lambda s: s.start_time)
        
        if not base_scenes:
            return [], {}
        
        # Calculate how many base scenes per virtual scene
        scenes_per_virtual = max(1, int(target_resolution / self.base_resolution))
        
        # Calculate stride (overlap)
        stride = max(1, int(scenes_per_virtual * (1 - self.overlap_factor)))
        
        virtual_scenes = []
        virtual_embeddings = {}
        
        for start_idx in range(0, len(base_scenes), stride):
            end_idx = min(start_idx + scenes_per_virtual, len(base_scenes))
            
            if end_idx - start_idx < 1:
                continue
            
            # Create virtual scene from base scenes
            constituent_scenes = base_scenes[start_idx:end_idx]
            virtual_scene = VirtualMultiResolutionScene(
                base_scenes=constituent_scenes,
                resolution=target_resolution,
                video_name=dp.video_name,
            )
            virtual_scenes.append(virtual_scene)
            
            # Compute aggregated embeddings
            virtual_embeddings[virtual_scene.scene_id] = self._aggregate_embeddings(
                dp, constituent_scenes
            )
        
        logging.debug(
            f"Created {len(virtual_scenes)} virtual scenes at {target_resolution}s "
            f"for {dp.video_name} (stride={stride})"
        )
        
        return virtual_scenes, virtual_embeddings
    
    def _aggregate_embeddings(
        self,
        dp: VideoDataPoint,
        scenes: List[Scene]
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Aggregate embeddings from multiple base scenes.
        Uses mean pooling across scene embeddings.
        """
        aggregated = {}
        modalities = ["video", "audio", "text", "caption"]
        
        for modality in modalities:
            embeddings = []
            for scene in scenes:
                scene_data = dp.scene_embeddings.get(scene.scene_id, {})
                emb = scene_data.get(modality)
                if emb is not None and isinstance(emb, torch.Tensor):
                    embeddings.append(emb)
            
            if embeddings:
                # Stack and mean pool
                stacked = torch.stack(embeddings, dim=0)
                aggregated[modality] = stacked.mean(dim=0)
            else:
                aggregated[modality] = None
        
        return aggregated
    
    def get_virtual_scenes(
        self,
        video_name: str,
        resolution: float
    ) -> List[VirtualMultiResolutionScene]:
        """Get virtual scenes for a video at a specific resolution."""
        if video_name not in self._virtual_scenes:
            return []
        return self._virtual_scenes[video_name].get(resolution, [])
    
    def get_virtual_embedding(
        self,
        video_name: str,
        resolution: float,
        scene_id: str,
        modality: str
    ) -> Optional[torch.Tensor]:
        """Get embedding for a virtual scene."""
        if video_name not in self._virtual_embeddings:
            return None
        res_dict = self._virtual_embeddings[video_name].get(resolution, {})
        scene_dict = res_dict.get(scene_id, {})
        return scene_dict.get(modality)
    
    def get_all_scenes_for_video(
        self,
        video_name: str,
        include_base: bool = True
    ) -> Dict[float, List[Scene]]:
        """
        Get all scenes (base + virtual) for a video, organized by resolution.
        
        Args:
            video_name: Name of the video
            include_base: Whether to include base-resolution scenes
            
        Returns:
            Dict mapping resolution -> list of scenes
        """
        result = {}
        
        # Add base resolution scenes
        if include_base:
            dp = self.video_dataset.get_datapoint_by_name(video_name)
            if dp:
                result[self.base_resolution] = list(dp.scenes.values())
        
        # Add virtual scenes
        if video_name in self._virtual_scenes:
            for resolution, scenes in self._virtual_scenes[video_name].items():
                result[resolution] = scenes
        
        return result
    
    def get_embedding_for_scene(
        self,
        video_name: str,
        scene: Scene,
        modality: str
    ) -> Optional[torch.Tensor]:
        """
        Get embedding for any scene (base or virtual).
        """
        if isinstance(scene, VirtualMultiResolutionScene):
            return self.get_virtual_embedding(
                video_name, scene.resolution, scene.scene_id, modality
            )
        else:
            # Base scene - get from original dataset
            dp = self.video_dataset.get_datapoint_by_name(video_name)
            if dp:
                scene_data = dp.scene_embeddings.get(scene.scene_id, {})
                return scene_data.get(modality)
        return None


class MultiResolutionRetriever:
    """
    Retrieves scenes across multiple resolutions and fuses results.
    """
    
    def __init__(
        self,
        multi_res_index: MultiResolutionIndex,
        fusion_strategy: str = "rrf",  # "rrf", "max", "weighted"
        resolution_weights: Optional[Dict[float, float]] = None,
    ):
        """
        Args:
            multi_res_index: The multi-resolution index
            fusion_strategy: How to fuse results from different resolutions
            resolution_weights: Optional weights for each resolution (for "weighted" strategy)
        """
        self.index = multi_res_index
        self.fusion_strategy = fusion_strategy
        self.resolution_weights = resolution_weights or {}
        
        logging.info(f"MultiResolutionRetriever: strategy={fusion_strategy}")
    
    def retrieve_scenes(
        self,
        video_name: str,
        query_embedding: torch.Tensor,
        modality: str,
        top_k: int = 10,
        device: str = "cuda"
    ) -> List[Tuple[Scene, float]]:
        """
        Retrieve top-k scenes across all resolutions.
        
        Args:
            video_name: Name of the video to search
            query_embedding: Query embedding tensor
            modality: Modality to use for retrieval
            top_k: Number of scenes to return
            device: Device for computation
            
        Returns:
            List of (scene, score) tuples, sorted by score descending
        """
        from torch.nn.functional import cosine_similarity
        
        all_results: Dict[float, List[Tuple[Scene, float]]] = {}
        
        # Get scenes at all resolutions
        all_scenes = self.index.get_all_scenes_for_video(video_name, include_base=True)
        
        query_emb = query_embedding.to(device)
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        
        for resolution, scenes in all_scenes.items():
            resolution_results = []
            
            for scene in scenes:
                emb = self.index.get_embedding_for_scene(video_name, scene, modality)
                if emb is None:
                    continue
                
                emb = emb.to(device)
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                
                # Compute cosine similarity
                sim = cosine_similarity(query_emb, emb, dim=-1).item()
                resolution_results.append((scene, sim))
            
            # Sort by score
            resolution_results.sort(key=lambda x: x[1], reverse=True)
            all_results[resolution] = resolution_results[:top_k * 2]  # Keep more for fusion
        
        # Fuse results from different resolutions
        fused = self._fuse_results(all_results, top_k)
        
        return fused
    
    def _fuse_results(
        self,
        results_by_resolution: Dict[float, List[Tuple[Scene, float]]],
        top_k: int
    ) -> List[Tuple[Scene, float]]:
        """
        Fuse results from different resolutions.
        """
        if self.fusion_strategy == "rrf":
            return self._fuse_rrf(results_by_resolution, top_k)
        elif self.fusion_strategy == "max":
            return self._fuse_max(results_by_resolution, top_k)
        elif self.fusion_strategy == "weighted":
            return self._fuse_weighted(results_by_resolution, top_k)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
    
    def _fuse_rrf(
        self,
        results_by_resolution: Dict[float, List[Tuple[Scene, float]]],
        top_k: int,
        k: int = 60
    ) -> List[Tuple[Scene, float]]:
        """
        Reciprocal Rank Fusion across resolutions.
        """
        # Group by temporal position (start_time, end_time)
        position_scores: Dict[Tuple[float, float], Tuple[Scene, float]] = {}
        
        for resolution, results in results_by_resolution.items():
            for rank, (scene, score) in enumerate(results):
                position = (scene.start_time, scene.end_time)
                rrf_score = 1.0 / (k + rank + 1)
                
                if position not in position_scores:
                    position_scores[position] = (scene, rrf_score)
                else:
                    existing_scene, existing_score = position_scores[position]
                    # Keep the scene from the resolution with higher score
                    if rrf_score > existing_score:
                        position_scores[position] = (scene, existing_score + rrf_score)
                    else:
                        position_scores[position] = (existing_scene, existing_score + rrf_score)
        
        # Sort by aggregated score
        fused = list(position_scores.values())
        fused.sort(key=lambda x: x[1], reverse=True)
        
        return fused[:top_k]
    
    def _fuse_max(
        self,
        results_by_resolution: Dict[float, List[Tuple[Scene, float]]],
        top_k: int
    ) -> List[Tuple[Scene, float]]:
        """
        Take max score across resolutions for each temporal position.
        """
        position_scores: Dict[Tuple[float, float], Tuple[Scene, float]] = {}
        
        for resolution, results in results_by_resolution.items():
            for scene, score in results:
                position = (scene.start_time, scene.end_time)
                
                if position not in position_scores:
                    position_scores[position] = (scene, score)
                else:
                    existing_scene, existing_score = position_scores[position]
                    if score > existing_score:
                        position_scores[position] = (scene, score)
        
        fused = list(position_scores.values())
        fused.sort(key=lambda x: x[1], reverse=True)
        
        return fused[:top_k]
    
    def _fuse_weighted(
        self,
        results_by_resolution: Dict[float, List[Tuple[Scene, float]]],
        top_k: int
    ) -> List[Tuple[Scene, float]]:
        """
        Weighted average of scores based on resolution weights.
        """
        position_scores: Dict[Tuple[float, float], Tuple[Scene, float, float]] = {}
        # (scene, weighted_sum, weight_sum)
        
        for resolution, results in results_by_resolution.items():
            weight = self.resolution_weights.get(resolution, 1.0)
            
            for scene, score in results:
                position = (scene.start_time, scene.end_time)
                
                if position not in position_scores:
                    position_scores[position] = (scene, score * weight, weight)
                else:
                    existing_scene, existing_weighted_sum, existing_weight_sum = position_scores[position]
                    new_weighted_sum = existing_weighted_sum + score * weight
                    new_weight_sum = existing_weight_sum + weight
                    # Keep scene from higher-weighted resolution
                    if weight > existing_weight_sum / max(1, len([r for r in results_by_resolution])):
                        position_scores[position] = (scene, new_weighted_sum, new_weight_sum)
                    else:
                        position_scores[position] = (existing_scene, new_weighted_sum, new_weight_sum)
        
        # Compute final scores
        fused = [
            (scene, weighted_sum / weight_sum if weight_sum > 0 else 0)
            for scene, weighted_sum, weight_sum in position_scores.values()
        ]
        fused.sort(key=lambda x: x[1], reverse=True)
        
        return fused[:top_k]


def create_multi_resolution_index(
    video_dataset: VideoDataset,
    base_resolution: float,
    multipliers: List[float] = [2.0, 3.0],
    overlap_factor: float = 0.5
) -> MultiResolutionIndex:
    """
    Convenience function to create a multi-resolution index.
    
    Args:
        video_dataset: The base video dataset
        base_resolution: Scene length used during encoding (from config)
        multipliers: Multipliers for target resolutions (e.g., [2.0, 3.0] creates 2x and 3x scenes)
        overlap_factor: Overlap between virtual scenes
        
    Returns:
        MultiResolutionIndex
    """
    target_resolutions = [base_resolution * m for m in multipliers]
    
    return MultiResolutionIndex(
        video_dataset=video_dataset,
        base_resolution=base_resolution,
        target_resolutions=target_resolutions,
        overlap_factor=overlap_factor,
    )
