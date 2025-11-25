"""
Scene detection and manipulation utilities.

This module provides scene detection strategies for video processing in RAG systems.
Keeps the Scene class definition in data.video_dataset for backward compatibility.
"""

import os
import logging
import traceback
from typing import Dict, List, Tuple


class SceneDetector:
    """
    Utility class for detecting scenes in videos using various strategies.
    """
    
    @staticmethod
    def detect_scenes(
        video_path: str,
        method: str = "temporal",
        existing_scenes: Dict = None,
        **kwargs
    ) -> Dict:
        """
        Detect scenes in a video using the specified method.
        
        Args:
            video_path: Path to the video file
            method: Detection method - supports:
                - "pyscenedetect": Content-based detection
                - "temporal": Fixed temporal window splitting
            existing_scenes: Pre-existing scenes to use (highest priority)
            **kwargs: Additional method-specific parameters
                For "temporal": window_size (float, default=30.0 seconds)
                For "pyscenedetect": threshold (float, default=25.0)
            
        Returns:
            Dictionary mapping scene_id to Scene objects
        """
        # Import Scene here to avoid circular imports
        from data.video_dataset import Scene
        
        # Use pre-existing scenes if available
        if existing_scenes:
            logging.info(
                f"Using {len(existing_scenes)} pre-existing scenes for video {os.path.basename(video_path)}"
            )
            return existing_scenes
        
        # Method: Content-based detection using pyscenedetect
        if method == "pyscenedetect":
            return SceneDetector._pyscenedetect(video_path, Scene, **kwargs)
        # Method: Fixed temporal window
        elif method == "temporal":
            print(f"calling with {kwargs}")
            return SceneDetector._temporal_window(video_path, Scene, **kwargs)
        else:
            logging.error(f"Unknown scene detection method: {method}")
            return {}
    
    @staticmethod
    def _pyscenedetect(video_path: str, Scene, threshold: float = 25.0) -> Dict:
        """
        Content-based scene detection using pyscenedetect.
        
        Args:
            video_path: Path to the video file
            Scene: Scene class (passed to avoid circular import)
            threshold: Detection threshold for ContentDetector
            
        Returns:
            Dictionary of scene_id -> Scene objects
            
        Note: This can produce scenes of very variable length, which may not
        be ideal for RAG systems that require consistent granularity.
        """
        try:
            from scenedetect import detect, ContentDetector
            
            scene_list = detect(video_path, ContentDetector(threshold=threshold))
            logging.info(
                f"Scene detection (pyscenedetect) for video {os.path.basename(video_path)}, "
                f"found {len(scene_list)} scenes"
            )
            
            scenes = {
                f"scene_{i}": Scene(
                    scene_id=f"scene_{i}",
                    start_time=start.get_seconds(),
                    end_time=end.get_seconds(),
                    start_frame=start.get_frames(),
                    end_frame=end.get_frames(),
                )
                for i, (start, end) in enumerate(scene_list)
            }
            
            return scenes
            
        except ImportError:
            logging.error(
                "pyscenedetect not installed. Install with: pip install scenedetect"
            )
            return {}
        except Exception as e:
            logging.error(f"{'='*100} \n Scene detection failed for {video_path}: {e}")
            logging.error(traceback.format_exc())
            return {}
    
    @staticmethod
    def _temporal_window(video_path: str, Scene, window_size: float = 20.0) -> Dict:
        """
        Split video into fixed temporal windows (scenes of equal duration).
        
        Args:
            video_path: Path to the video file
            Scene: Scene class (passed to avoid circular import)
            window_size: Duration of each scene in seconds (default=30.0)
            
        Returns:
            Dictionary of scene_id -> Scene objects
            
        Note: This produces scenes of consistent length, which is ideal for
        RAG systems requiring uniform granularity. The last scene may be shorter
        than window_size if the video duration is not evenly divisible.
        """
        print(f"Splitting video into fixed temporal windows of {window_size} seconds.")
        try:
            # Get video duration
            from moviepy.editor import VideoFileClip
            
            with VideoFileClip(video_path) as video:
                duration = video.duration
                fps = video.fps
            
            # Calculate number of scenes
            num_scenes = int(duration / window_size) + (1 if duration % window_size > 0 else 0)
            
            logging.info(
                f"Temporal window scene detection for video {os.path.basename(video_path)}: "
                f"duration={duration:.2f}s, window={window_size}s, scenes={num_scenes}"
            )
            
            scenes = {}
            for i in range(num_scenes):
                start_time = i * window_size
                end_time = min((i + 1) * window_size, duration)
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)
                
                scenes[f"scene_{i}"] = Scene(
                    scene_id=f"scene_{i}",
                    start_time=start_time,
                    end_time=end_time,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            
            return scenes
            
        except ImportError:
            logging.error(
                "moviepy not installed. Install with: pip install moviepy"
            )
            return {}
        except Exception as e:
            logging.error(f"{'='*100} \n Temporal window scene detection failed for {video_path}: {e}")
            logging.error(traceback.format_exc())
            return {}


