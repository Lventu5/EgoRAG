"""
Scene Tagging: Automatic scene-type and dialogue classification.

This module tags scenes with semantic types (cooking, meeting, etc.) and
dialogue characteristics (monologue, dialogue, no_speech).
"""

import re
from typing import Set
from data.video_dataset import VideoDataPoint
from indexing.utils.logging import get_logger

logger = get_logger(__name__)


# Scene type keywords
SCENE_TYPE_KEYWORDS = {
    "cooking": ["cook", "recipe", "kitchen", "food", "ingredient", "meal", "prepare", "eat", "dish"],
    "meeting": ["meeting", "discuss", "presentation", "conference", "call", "zoom", "talk", "conversation"],
    "commute": ["drive", "driving", "car", "bus", "train", "travel", "road", "traffic", "walk"],
    "idle": ["wait", "waiting", "pause", "still", "quiet", "silence", "nothing"],
}


def tag_scene_types(dp: VideoDataPoint) -> dict:
    """
    Tag each scene with a semantic scene type.
    
    Sets dp.scene_embeddings[sid]["meta"]["scene_type"] to one of:
    - "cooking"
    - "meeting"
    - "commute"
    - "idle"
    - "other"
    
    Uses keyword heuristics over transcript and caption text.
    
    Args:
        dp: VideoDataPoint to tag
        
    Returns:
        Dict mapping scene_id -> scene_type
    """
    scene_types = {}
    tagged_count = 0
    
    for scene_id, scene_data in dp.scene_embeddings.items():
        # Ensure meta dict exists
        if "meta" not in scene_data:
            scene_data["meta"] = {}
        
        # Gather text for analysis
        transcript = scene_data.get("transcript", "").lower()
        caption = scene_data.get("caption_text", "").lower()
        combined_text = transcript + " " + caption
        
        # Count keyword matches for each type
        type_scores = {}
        for scene_type, keywords in SCENE_TYPE_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            type_scores[scene_type] = score
        
        # Assign type based on highest score
        if all(score == 0 for score in type_scores.values()):
            scene_type = "other"
        else:
            scene_type = max(type_scores, key=type_scores.get)
        
        scene_data["meta"]["scene_type"] = scene_type
        scene_types[scene_id] = scene_type
        tagged_count += 1
    
    logger.debug(f"Tagged {tagged_count} scenes in video {dp.video_name} with scene types")
    return scene_types


def tag_dialogue_roles(dp: VideoDataPoint) -> dict:
    """
    Tag each scene with dialogue/speech type.
    
    Sets dp.scene_embeddings[sid]["meta"]["speech_type"] to one of:
    - "dialogue": Multiple speakers detected
    - "monologue": Single speaker
    - "no_speech": No speech detected
    
    Uses diarization markers if present, otherwise heuristics.
    
    Args:
        dp: VideoDataPoint to tag
        
    Returns:
        Dict mapping scene_id -> speech_type
    """
    speech_types = {}
    tagged_count = 0
    
    for scene_id, scene_data in dp.scene_embeddings.items():
        # Ensure meta dict exists
        if "meta" not in scene_data:
            scene_data["meta"] = {}
        
        transcript = scene_data.get("transcript", "")
        
        # Check for no speech
        if not transcript or len(transcript.strip()) < 10:
            speech_type = "no_speech"
        else:
            # Look for diarization markers (e.g., "Speaker 1:", "[Person]:")
            speaker_patterns = [
                r"speaker\s*\d+:",
                r"\[.*?\]:",
                r"person\s*\d+:",
            ]
            
            speakers = set()
            for pattern in speaker_patterns:
                matches = re.findall(pattern, transcript.lower())
                speakers.update(matches)
            
            if len(speakers) > 1:
                speech_type = "dialogue"
            elif len(speakers) == 1:
                speech_type = "monologue"
            else:
                # No explicit markers - use heuristics
                # Check for conversational markers
                conversational_markers = ["said", "asked", "replied", "told", "answered"]
                has_conversation = any(marker in transcript.lower() for marker in conversational_markers)
                
                if has_conversation:
                    speech_type = "dialogue"
                else:
                    speech_type = "monologue"
        
        scene_data["meta"]["speech_type"] = speech_type
        speech_types[scene_id] = speech_type
        tagged_count += 1
    
    logger.debug(f"Tagged {tagged_count} scenes in video {dp.video_name} with speech types")
    return speech_types


def tag_all_scenes(dp: VideoDataPoint) -> None:
    """
    Apply all tagging functions to a video datapoint.
    
    Convenience function that calls both tag_scene_types and tag_dialogue_roles.
    
    Args:
        dp: VideoDataPoint to tag
    """
    tag_scene_types(dp)
    tag_dialogue_roles(dp)
    logger.info(f"Completed tagging for video {dp.video_name}")
