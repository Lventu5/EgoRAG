"""
QA Pipeline: Orchestrates retrieval, reranking, and answer generation.

This module provides end-to-end QA over video content using the full
EgoRAG pipeline.
"""

from typing import Dict, List, Optional
from data.query import Query
from data.video_dataset import VideoDataset
from indexing.qa.formatter import build_qa_context
from indexing.qa.generator import QAGenerator
from indexing.utils.logging import get_logger

logger = get_logger(__name__)


class QAPipeline:
    """
    End-to-end pipeline for video QA.
    
    Orchestrates:
    1. Multi-hop retrieval (ChainRetriever)
    2. Reranking (Reranker)
    3. Context formatting
    4. Answer generation
    5. Validation and self-healing
    """
    
    def __init__(
        self,
        chain_retriever,  # ChainRetriever instance
        reranker,  # Reranker instance
        generator: QAGenerator,
        video_dataset: VideoDataset,
        cfg: dict
    ):
        """
        Initialize QA pipeline.
        
        Args:
            chain_retriever: ChainRetriever for multi-hop retrieval
            reranker: Reranker for confidence-aware reranking
            generator: QAGenerator for answer generation
            video_dataset: VideoDataset with encoded videos
            cfg: Configuration dictionary
        """
        self.chain_retriever = chain_retriever
        self.reranker = reranker
        self.generator = generator
        self.video_dataset = video_dataset
        self.cfg = cfg
        
        self.max_ctx_scenes = cfg.get("qa", {}).get("max_ctx_scenes", 4)
        self.max_tokens = cfg.get("qa", {}).get("max_tokens", 2048)
        
        logger.info("QAPipeline initialized")
    
    def answer(
        self,
        query: Query,
        modalities: List[str] = ["video", "text", "audio"]
    ) -> Dict:
        """
        Answer a query using the full pipeline.
        
        Args:
            query: Query object
            modalities: List of modalities to use for retrieval
            
        Returns:
            Dictionary containing:
                - "answer": Generated answer text
                - "validation": Validation results
                - "candidates": Retrieved candidates info
                - "context": Context used for generation
        """
        logger.info(f"Answering query {query.qid}: {query.query_text}")
        
        # Step 1: Multi-hop retrieval
        logger.debug("Step 1: Chain retrieval")
        retrieval_results = self.chain_retriever.retrieve(query, modalities)
        
        # Step 2: Rerank candidates
        logger.debug("Step 2: Reranking")
        reranked_scenes = self._rerank_scenes(
            query,
            retrieval_results["scenes"],
            retrieval_results.get("scores_by_modality", {})
        )
        
        # Step 3: Format context
        logger.debug("Step 3: Building context")
        context = build_qa_context(
            query,
            reranked_scenes[:self.max_ctx_scenes],
            self.video_dataset,
            max_scenes=self.max_ctx_scenes,
            max_tokens=self.max_tokens
        )
        
        # Step 4: Generate answer
        logger.debug("Step 4: Generating answer")
        generation_result = self.generator.generate(query, context)
        answer = generation_result["answer"]
        rationale = generation_result["rationale"]
        
        # Step 5: Validate answer
        logger.debug("Step 5: Validating answer")
        validation = self.generator.validate(answer, context)
        
        # Step 6: Self-healing if validation fails
        if not validation["supported"]:
            logger.warning(f"Answer not well-supported for query {query.qid}")
            # Optional: Implement self-healing logic here
            # For now, we just log the issue
        
        result = {
            "query_id": query.qid,
            "query_text": query.query_text,
            "answer": answer,
            "rationale": rationale,
            "validation": validation,
            "candidates": {
                "videos": retrieval_results.get("videos", []),
                "scenes": [
                    {
                        "video_name": vn,
                        "scene_id": scene.scene_id,
                        "score": score,
                        "start_time": scene.start_time,
                        "end_time": scene.end_time
                    }
                    for vn, scene, score in reranked_scenes[:self.max_ctx_scenes]
                ],
                "chain": retrieval_results.get("chain", [])
            },
            "context": context
        }
        
        logger.info(f"Completed answering query {query.qid}")
        return result
    
    def _rerank_scenes(
        self,
        query: Query,
        scenes: List,
        scores_by_modality: Dict
    ) -> List:
        """
        Rerank scenes using the reranker.
        
        Args:
            query: Query object
            scenes: List of (video_name, Scene, score) tuples
            scores_by_modality: Scores per modality for each scene
            
        Returns:
            Reranked list of scenes
        """
        reranked = []
        
        for video_name, scene, fused_score in scenes:
            # Get per-modality scores for this scene (if available)
            scene_key = f"{video_name}_{scene.scene_id}"
            mod_scores = scores_by_modality.get(scene_key, {})
            
            # Compute reranked score
            new_score = self.reranker.score_scene(
                query_time_hint=query.time_hint_sec,
                scene=scene,
                fused_score=fused_score,
                scores_by_modality=mod_scores
            )
            
            reranked.append((video_name, scene, new_score))
        
        # Sort by new score (descending)
        reranked.sort(key=lambda x: x[2], reverse=True)
        
        return reranked
