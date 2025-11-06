"""
ChainRetriever: Query-aware multi-hop retrieval with Chain-of-Retrieval.

This module implements multi-hop retrieval where the system generates
sub-questions and retrieves across multiple iterations to build context.
"""

from typing import List, Dict, Tuple
from functools import lru_cache
import torch
from collections import defaultdict

from data.query import Query
from data.video_dataset import Scene
from indexing.memory.memory_bank import SceneMemoryBank
from indexing.utils.logging import get_logger

logger = get_logger(__name__)


class ChainRetriever:
    """
    Multi-hop retriever that generates sub-questions and accumulates candidates.
    
    Implements Chain-of-Retrieval pattern:
    1. Initial retrieval with original query
    2. Generate sub-questions
    3. Retrieve for each sub-question
    4. Fuse and rank all candidates
    """
    
    def __init__(
        self,
        base_retriever,  # HierarchicalRetriever instance
        rewriter,  # QueryRewriterLLM instance
        memory_bank: SceneMemoryBank,
        cfg: dict
    ):
        """
        Initialize chain retriever.
        
        Args:
            base_retriever: HierarchicalRetriever for single-hop retrieval
            rewriter: QueryRewriterLLM for generating sub-questions
            memory_bank: SceneMemoryBank for efficient querying
            cfg: Configuration dictionary
        """
        self.base_retriever = base_retriever
        self.rewriter = rewriter
        self.memory_bank = memory_bank
        self.cfg = cfg
        
        self.max_hops = cfg.get("retrieval", {}).get("max_hops", 2)
        self.top_k_videos = cfg.get("retrieval", {}).get("top_k_videos", 3)
        self.top_k_scenes = cfg.get("retrieval", {}).get("top_k_scenes", 2)
        self.max_ctx_scenes = cfg.get("qa", {}).get("max_ctx_scenes", 4)
        
        # LRU caches for embeddings and retrieval results
        self._embedding_cache = {}
        self._retrieval_cache = {}
        
        logger.info(
            f"ChainRetriever initialized: max_hops={self.max_hops}, "
            f"top_k_videos={self.top_k_videos}, top_k_scenes={self.top_k_scenes}"
        )
    
    def retrieve(
        self,
        query: Query,
        modalities: List[str] = ["video", "text", "audio"]
    ) -> Dict:
        """
        Perform multi-hop retrieval for a query.
        
        Args:
            query: Query object
            modalities: List of modalities to use
            
        Returns:
            Dictionary with keys:
                - "videos": List of (video_name, score) tuples
                - "scenes": List of (video_name, Scene, score) tuples
                - "chain": List of sub-questions asked
                - "scores_by_modality": Dict mapping scene_key to per-modality scores
        """
        logger.info(f"Chain retrieval for query {query.qid}: {query.query_text}")
        
        # Track all candidates across hops
        all_candidates = []
        candidate_scores = defaultdict(lambda: defaultdict(float))  # scene_key -> {modality -> score}
        sub_questions = [query.query_text]  # Start with original query
        
        # Step 1: Initial retrieval with original query
        logger.debug("Hop 0: Initial retrieval with original query")
        initial_results = self._retrieve_single_hop(
            query.query_text,
            modalities,
            top_k_videos=self.top_k_videos,
            top_k_scenes=1  # Get 1 scene per video initially
        )
        
        # Collect candidates from initial retrieval
        for video_name, scenes in initial_results.items():
            for scene, score, mod_scores in scenes:
                scene_key = f"{video_name}_{scene.scene_id}"
                all_candidates.append((video_name, scene, score))
                
                # Store per-modality scores
                for mod, mod_score in mod_scores.items():
                    candidate_scores[scene_key][mod] = mod_score
        
        # Step 2: Generate sub-questions for follow-up hops
        if self.max_hops > 1:
            logger.debug(f"Generating {self.max_hops - 1} sub-questions")
            try:
                subqs = self.rewriter.subquestions(query.query_text, num_hops=self.max_hops - 1)
                sub_questions.extend(subqs)
                logger.debug(f"Generated sub-questions: {subqs}")
            except Exception as e:
                logger.warning(f"Failed to generate sub-questions: {e}")
        
        # Step 3: Retrieve for each sub-question
        for hop_idx, subq in enumerate(sub_questions[1:], start=1):
            logger.debug(f"Hop {hop_idx}: Retrieving for sub-question: {subq}")
            
            # Retrieve for this sub-question
            hop_results = self._retrieve_single_hop(
                subq,
                modalities,
                top_k_videos=self.top_k_videos,
                top_k_scenes=self.top_k_scenes
            )
            
            # Accumulate candidates
            for video_name, scenes in hop_results.items():
                for scene, score, mod_scores in scenes:
                    scene_key = f"{video_name}_{scene.scene_id}"
                    all_candidates.append((video_name, scene, score))
                    
                    # Aggregate per-modality scores (max)
                    for mod, mod_score in mod_scores.items():
                        candidate_scores[scene_key][mod] = max(
                            candidate_scores[scene_key][mod],
                            mod_score
                        )
        
        # Step 4: Deduplicate and rank candidates
        unique_candidates = self._deduplicate_candidates(all_candidates)
        
        # Step 5: Fuse scores if Fuser is available
        if hasattr(self.base_retriever, 'fuser') and self.base_retriever.fuser:
            logger.debug("Fusing candidate scores")
            unique_candidates = self._fuse_candidates(unique_candidates, modalities)
        
        # Keep top candidates
        unique_candidates = unique_candidates[:self.max_ctx_scenes]
        
        # Extract unique videos
        video_scores = {}
        for video_name, scene, score in unique_candidates:
            if video_name not in video_scores:
                video_scores[video_name] = score
            else:
                video_scores[video_name] = max(video_scores[video_name], score)
        
        videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)
        
        logger.info(
            f"Chain retrieval complete: {len(videos)} videos, "
            f"{len(unique_candidates)} scenes, {len(sub_questions)} hops"
        )
        
        return {
            "videos": videos,
            "scenes": unique_candidates,
            "chain": sub_questions,
            "scores_by_modality": dict(candidate_scores)
        }
    
    def _retrieve_single_hop(
        self,
        query_text: str,
        modalities: List[str],
        top_k_videos: int,
        top_k_scenes: int
    ) -> Dict[str, List[Tuple[Scene, float, Dict[str, float]]]]:
        """
        Perform single-hop retrieval for a query text.
        
        Args:
            query_text: Query string
            modalities: List of modalities
            top_k_videos: Number of videos to retrieve
            top_k_scenes: Number of scenes per video
            
        Returns:
            Dict mapping video_name to list of (Scene, score, mod_scores) tuples
        """
        # Check cache
        cache_key = (query_text, tuple(modalities), top_k_videos, top_k_scenes)
        if cache_key in self._retrieval_cache:
            return self._retrieval_cache[cache_key]
        
        # Create temporary Query object for retrieval
        from data.query import QueryDataset
        temp_query = Query(qid="temp", query_text=query_text)
        temp_dataset = QueryDataset([temp_query])
        
        # Perform retrieval using base retriever
        try:
            results = self.base_retriever.retrieve_hierarchically(
                temp_dataset,
                modalities=modalities,
                top_k_videos=top_k_videos,
                top_k_scenes=top_k_scenes
            )
            
            # Extract results for our query
            query_results = results.get("temp", {})
            fused_results = query_results.get("fused", [])
            
            # Organize by video
            video_scenes = {}
            for video_name, video_score, scenes in fused_results:
                scene_list = []
                for scene, scene_score in scenes:
                    # Get per-modality scores (placeholder - would need to extract from retriever)
                    mod_scores = {mod: scene_score for mod in modalities}
                    scene_list.append((scene, scene_score, mod_scores))
                video_scenes[video_name] = scene_list
            
            # Cache results
            self._retrieval_cache[cache_key] = video_scenes
            
            return video_scenes
            
        except Exception as e:
            logger.error(f"Single-hop retrieval failed: {e}")
            return {}
    
    def _deduplicate_candidates(
        self,
        candidates: List[Tuple[str, Scene, float]]
    ) -> List[Tuple[str, Scene, float]]:
        """
        Remove duplicate scenes and keep highest score.
        
        Args:
            candidates: List of (video_name, Scene, score) tuples
            
        Returns:
            Deduplicated list sorted by score descending
        """
        scene_best = {}
        
        for video_name, scene, score in candidates:
            scene_key = f"{video_name}_{scene.scene_id}"
            
            if scene_key not in scene_best or score > scene_best[scene_key][2]:
                scene_best[scene_key] = (video_name, scene, score)
        
        # Sort by score descending
        unique = sorted(scene_best.values(), key=lambda x: x[2], reverse=True)
        
        return unique
    
    def _fuse_candidates(
        self,
        candidates: List[Tuple[str, Scene, float]],
        modalities: List[str]
    ) -> List[Tuple[str, Scene, float]]:
        """
        Apply fusion to re-score candidates (if fuser available).
        
        Args:
            candidates: List of (video_name, Scene, score) tuples
            modalities: List of modalities used
            
        Returns:
            Re-scored candidates
        """
        # Placeholder - actual fusion would require per-modality scores
        # For now, just return candidates as-is
        return candidates
    
    def clear_cache(self):
        """Clear all caches."""
        self._embedding_cache.clear()
        self._retrieval_cache.clear()
        logger.debug("Cleared retrieval caches")
