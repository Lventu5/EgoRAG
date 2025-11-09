"""
Utility functions for video processing.
"""

import logging
from moviepy.editor import VideoFileClip


def has_audio_track(video_path: str) -> bool:
    """
    Check if a video file has an audio track.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        True if video has audio, False otherwise
    """
    try:
        with VideoFileClip(video_path) as vid:
            has_audio = vid.audio is not None
            
        if not has_audio:
            logging.info(f"Video {video_path} has no audio track")
        
        return has_audio
    except Exception as e:
        logging.warning(f"Could not check audio for {video_path}: {e}")
        # Assume True to attempt audio extraction (will fail gracefully)
        return True


def get_video_info(video_path: str) -> dict:
    """
    Get basic information about a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video metadata:
        - has_audio: bool
        - duration: float (seconds)
        - fps: float
        - size: tuple (width, height)
    """
    try:
        with VideoFileClip(video_path) as vid:
            info = {
                "has_audio": vid.audio is not None,
                "duration": vid.duration,
                "fps": vid.fps,
                "size": vid.size,
            }
        return info
    except Exception as e:
        logging.error(f"Could not get video info for {video_path}: {e}")
        return {
            "has_audio": True,  # Assume True
            "duration": 0.0,
            "fps": 0.0,
            "size": (0, 0),
        }
