"""
Memory Editor: Self-healing memory with contradiction detection.

This module detects inconsistencies in generated answers and proposes
patches to improve memory coherence.
"""

from typing import List, Dict, Tuple
from data.video_dataset import VideoDataset, Scene
from indexing.utils.logging import get_logger

logger = get_logger(__name__)


def detect_contradictions(answer: str, context: str) -> List[dict]:
    """
    Detect contradictions or inconsistencies between answer and context.
    
    Args:
        answer: Generated answer text
        context: Context string used for generation
        
    Returns:
        List of issue dictionaries with keys:
            - "type": Issue type (timestamp, entity, action)
            - "detail": Description of the issue
    """
    issues = []
    
    # Placeholder implementation - basic heuristics
    # TODO: Implement more sophisticated contradiction detection
    
    answer_lower = answer.lower()
    context_lower = context.lower()
    
    # 1. Check for temporal contradictions
    # Look for time references in answer that don't match context
    time_keywords = ["before", "after", "during", "while", "at"]
    for keyword in time_keywords:
        if keyword in answer_lower and keyword not in context_lower:
            issues.append({
                "type": "timestamp",
                "detail": f"Answer mentions temporal relation '{keyword}' not found in context"
            })
    
    # 2. Check for entity contradictions
    # Simple check: if answer mentions specific names/entities not in context
    # This is a very naive implementation
    answer_words = set(answer_lower.split())
    context_words = set(context_lower.split())
    
    # Look for capitalized words (potential entities) in answer not in context
    answer_tokens = answer.split()
    for token in answer_tokens:
        if token[0].isupper() and len(token) > 3:  # Potential entity
            if token.lower() not in context_lower:
                issues.append({
                    "type": "entity",
                    "detail": f"Answer mentions entity '{token}' not found in context"
                })
    
    # 3. Check for action contradictions
    # Look for action verbs in answer that contradict context
    action_keywords = ["did", "does", "doing", "made", "makes", "went", "goes"]
    for keyword in action_keywords:
        if keyword in answer_lower:
            # Check if there's a negation nearby in context
            if f"not {keyword}" in context_lower or f"didn't" in context_lower:
                issues.append({
                    "type": "action",
                    "detail": f"Answer implies action '{keyword}' that may be contradicted in context"
                })
    
    if issues:
        logger.info(f"Detected {len(issues)} potential contradictions")
    
    return issues


def propose_memory_patches(
    issues: List[dict],
    scenes: List[Tuple[str, Scene, float]],
    video_dataset: VideoDataset
) -> List[dict]:
    """
    Propose memory patches to address detected issues.
    
    Args:
        issues: List of detected issues from detect_contradictions
        scenes: List of (video_name, Scene, score) tuples used for answer
        video_dataset: VideoDataset for accessing and modifying metadata
        
    Returns:
        List of patch dictionaries with keys:
            - "scene_key": Identifier for scene to patch
            - "patch_type": Type of patch (add_alias, add_tag, adjust_time)
            - "patch_data": Data for the patch
    """
    patches = []
    
    for issue in issues:
        issue_type = issue["type"]
        detail = issue["detail"]
        
        if issue_type == "entity":
            # Propose adding entity alias to scene metadata
            # Extract entity name from detail
            if "entity '" in detail:
                entity = detail.split("entity '")[1].split("'")[0]
                
                # Add patch for each scene
                for video_name, scene, _ in scenes:
                    patches.append({
                        "scene_key": f"{video_name}_{scene.scene_id}",
                        "patch_type": "add_entity_tag",
                        "patch_data": {"entity": entity, "confidence": "low"}
                    })
        
        elif issue_type == "timestamp":
            # Propose adjusting temporal metadata or adding temporal markers
            for video_name, scene, _ in scenes:
                patches.append({
                    "scene_key": f"{video_name}_{scene.scene_id}",
                    "patch_type": "add_temporal_marker",
                    "patch_data": {"marker": "ambiguous_time", "detail": detail}
                })
        
        elif issue_type == "action":
            # Propose adding action clarification
            for video_name, scene, _ in scenes:
                patches.append({
                    "scene_key": f"{video_name}_{scene.scene_id}",
                    "patch_type": "add_action_clarification",
                    "patch_data": {"detail": detail}
                })
    
    logger.info(f"Proposed {len(patches)} memory patches")
    return patches


def apply_patches(video_dataset: VideoDataset, patches: List[dict]) -> None:
    """
    Apply memory patches to the video dataset.
    
    Mutates scene_embeddings metadata fields. Does not alter embeddings themselves.
    
    Args:
        video_dataset: VideoDataset to patch
        patches: List of patch dictionaries from propose_memory_patches
    """
    applied_count = 0
    
    for patch in patches:
        scene_key = patch["scene_key"]
        patch_type = patch["patch_type"]
        patch_data = patch["patch_data"]
        
        # Parse scene_key to find video and scene
        parts = scene_key.rsplit("_", 1)
        if len(parts) != 2:
            logger.warning(f"Invalid scene_key format: {scene_key}")
            continue
        
        video_name = parts[0]
        scene_id = parts[1]
        
        # Find the video datapoint
        dp = None
        for candidate_dp in video_dataset.video_datapoints:
            if candidate_dp.video_name == video_name:
                dp = candidate_dp
                break
        
        if dp is None:
            logger.warning(f"Could not find video {video_name} for patching")
            continue
        
        # Find the scene embedding
        if scene_id not in dp.scene_embeddings:
            logger.warning(f"Could not find scene {scene_id} in {video_name}")
            continue
        
        scene_data = dp.scene_embeddings[scene_id]
        
        # Ensure meta dict exists
        if "meta" not in scene_data:
            scene_data["meta"] = {}
        
        # Apply patch based on type
        if patch_type == "add_entity_tag":
            if "entities" not in scene_data["meta"]:
                scene_data["meta"]["entities"] = []
            scene_data["meta"]["entities"].append(patch_data)
            applied_count += 1
        
        elif patch_type == "add_temporal_marker":
            if "temporal_markers" not in scene_data["meta"]:
                scene_data["meta"]["temporal_markers"] = []
            scene_data["meta"]["temporal_markers"].append(patch_data)
            applied_count += 1
        
        elif patch_type == "add_action_clarification":
            if "action_clarifications" not in scene_data["meta"]:
                scene_data["meta"]["action_clarifications"] = []
            scene_data["meta"]["action_clarifications"].append(patch_data)
            applied_count += 1
        
        else:
            logger.warning(f"Unknown patch type: {patch_type}")
    
    logger.info(f"Applied {applied_count}/{len(patches)} memory patches")
