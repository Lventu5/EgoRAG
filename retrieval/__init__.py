"""
Retrieval module for EgoRAG.

This module contains components for hierarchical video retrieval:
- HierarchicalRetriever: Main retrieval engine
- Fuser: Multi-modal score fusion
- QueryRewriter: Query decomposition and rewriting
- SceneMerger: Consecutive scene merging for improved IoU metrics
- MultiResolution: Multi-resolution scene retrieval for balanced precision/recall
"""

from .hierarchical_retriever import HierarchicalRetriever
from .fuser import Fuser, BaseFuser, FuserRRF, FuserMeanImputation, FuserExcludeMissing
from .rewriter import QueryRewriterLLM
from .scene_merger import SceneMerger, MergedScene, merge_consecutive_scenes

__all__ = [
    'HierarchicalRetriever',
    'Fuser',
    'BaseFuser',
    'FuserRRF',
    'FuserMeanImputation',
    'FuserExcludeMissing',
    'QueryRewriterLLM',
    'SceneMerger',
    'MergedScene',
    'merge_consecutive_scenes',
    'MultiResolutionIndex',
    'MultiResolutionRetriever',
    'MultiResolutionScene',
    'VirtualMultiResolutionScene',
    'create_multi_resolution_index',
]
