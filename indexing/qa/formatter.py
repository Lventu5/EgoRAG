"""
QA Formatter: Builds context strings from retrieved scenes for LLM input.

This module formats retrieved scenes into bounded-length context strings
suitable for QA generation.
"""

from typing import List, Tuple
from data.query import Query
from data.video_dataset import Scene, VideoDataset
from indexing.utils.logging import get_logger

logger = get_logger(__name__)


def build_qa_context(
    query: Query,
    scenes: List[Tuple[str, Scene, float]],
    video_dataset: VideoDataset,
    max_scenes: int = 4,
    max_tokens: int = 2048
) -> str:
    """
    Build a bounded-length context string from retrieved scenes.
    
    Args:
        query: The query object
        scenes: List of (video_name, Scene, score) tuples
        video_dataset: VideoDataset for accessing metadata
        max_scenes: Maximum number of scenes to include
        max_tokens: Approximate maximum token budget (rough estimate: 1 token ≈ 4 chars)
        
    Returns:
        Formatted context string
    """
    context_parts = []
    context_parts.append(f"Query: {query.query_text}\n")
    context_parts.append("Relevant video scenes:\n\n")
    
    # Approximate character budget (rough heuristic)
    char_budget = max_tokens * 4
    chars_used = len("".join(context_parts))
    
    scenes_included = 0
    for video_name, scene, score in scenes[:max_scenes]:
        if scenes_included >= max_scenes:
            break
        
        # Find the video datapoint
        dp = None
        for candidate_dp in video_dataset.video_datapoints:
            if candidate_dp.video_name == video_name:
                dp = candidate_dp
                break
        
        if dp is None:
            logger.warning(f"Could not find video datapoint for {video_name}")
            continue
        
        # Find scene data
        scene_data = dp.scene_embeddings.get(scene.scene_id)
        if scene_data is None:
            logger.warning(f"Could not find scene data for {scene.scene_id} in {video_name}")
            continue
        
        # Build scene context
        scene_context = f"Scene {scenes_included + 1} (from {video_name}):\n"
        scene_context += f"  Time: {scene.start_time:.2f}s - {scene.end_time:.2f}s\n"
        
        # Add transcript if available
        transcript = scene_data.get("transcript", "")
        if transcript:
            scene_context += f"  Transcript: {transcript}\n"
        
        # Add caption if available
        caption_text = scene_data.get("caption_text", "")
        if caption_text:
            scene_context += f"  Visual description: {caption_text}\n"
        
        scene_context += f"  Relevance score: {score:.3f}\n\n"
        
        # Check if adding this scene would exceed budget
        if chars_used + len(scene_context) > char_budget:
            logger.debug(f"Context budget exceeded, stopping at {scenes_included} scenes")
            break
        
        context_parts.append(scene_context)
        chars_used += len(scene_context)
        scenes_included += 1
    
    if scenes_included == 0:
        context_parts.append("No relevant scenes found.\n")
    
    context = "".join(context_parts)
    logger.debug(f"Built context with {scenes_included} scenes, {len(context)} characters")
    
    return context
