import os
import logging
import tempfile
import torch
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu
from scenedetect import detect, ContentDetector
from concurrent.futures import ThreadPoolExecutor, as_completed

from data.video_dataset import VideoDataset, VideoDataPoint, Scene
from indexing.utils.logging_formatter import LevelAwareFormatter
from indexing.components.video_encoder import VideoEncoder
from indexing.components.audio_encoder import AudioEncoder # <--- UPDATED
from indexing.components.text_encoder import TextEncoder
from indexing.components.visual_captioner import VisualCaptioner

# --- Setup logging (can be moved to a main file) ---
handler = logging.StreamHandler()
handler.setFormatter(LevelAwareFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
# ---

class MultiModalEncoder:
    """
    Orchestrates the hierarchical, multi-modal encoding of long videos.
    
    This class manages the full pipeline:
    1.  Scene Detection
    2.  Asset Extraction (frames, audio)
    3.  Delegation to atomic components (Video, Audio, Text, Captioner)
    4.  Aggregation of results into a VideoDataPoint.
    """
    def __init__(
        self,
        video_dataset: VideoDataset,
        device: str = "cuda",
        max_frames_per_scene: int = 96,
        max_temporal_segments: int = 8,
        audio_sr: int = 48000, # <-- NEW: Pass config down
        asr_sr: int = 16000,     # <-- NEW: Pass config down
        max_workers: int = 2,
    ):
        if video_dataset is None or len(video_dataset) == 0:
            raise ValueError("Video dataset is empty or not provided.")
        
        self.dataset = video_dataset
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_workers = max_workers

        # 1. Instantiate Atomic Components
        self.video_encoder = VideoEncoder(
            device=self.device,
            max_frames_per_scene=max_frames_per_scene,
            max_temporal_segments=max_temporal_segments
        )
        self.audio_encoder = AudioEncoder( # <-- UPDATED
            device=self.device,
            audio_sr=audio_sr,
            asr_sr=asr_sr
        )
        self.text_encoder = TextEncoder(device=self.device)
        self.captioner = VisualCaptioner(device=self.device)
        
        logging.info(f"MultiModalEncoder initialized with {max_workers} workers.")

    def load_models(self):
        """
        Loads all component models in parallel.
        This is a heavy operation.
        """
        logging.info("Loading all models...")
        # (In a real scenario, you might parallelize this)
        self.video_encoder.load_models()
        self.audio_encoder.load_models()
        self.text_encoder.load_models()
        self.captioner.load_models()
        logging.info("All models loaded successfully.")

    def _detect_scenes(self, video_path: str) -> list[Scene]:
        """Detects content-based scenes and returns Scene objects."""
        try:
            scene_list = detect(video_path, ContentDetector())
            return [
                Scene(
                    start_time=start.get_seconds(),
                    end_time=end.get_seconds(),
                    start_frame=start.get_frames(),
                    end_frame=end.get_frames(),
                )
                for start, end in scene_list
            ]
        except Exception as e:
            logging.error(f"Scene detection failed for {video_path}: {e}")
            return []

    def _extract_frames(self, video_path: str, start_frame: int, end_frame: int, max_frames: int) -> tuple[np.ndarray, np.ndarray]:
        """Extracts frames for a given scene."""
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames_in_scene = end_frame - start_frame
        
        if num_frames_in_scene > max_frames:
            indices = np.linspace(start_frame, end_frame - 1, max_frames, dtype=int)
        else:
            indices = np.arange(start_frame, end_frame)
            
        frames = vr.get_batch(indices).asnumpy()
        return frames

    def _encode_scene(self, video_path: str, scene: Scene) -> dict | None:
        """
        Orchestrates the encoding of a single scene by delegating
        to the modular components. (UPDATED)
        """
        scene_key = f"scene_{scene.start_frame}_{scene.end_frame}"
        try:
            frames = self._extract_frames(
                video_path, 
                scene.start_frame, 
                scene.end_frame, 
                self.video_encoder.max_frames_per_scene
            )
            
            video_data = self.video_encoder.encode(frames)
            
            audio_data = self.audio_encoder.encode(
                video_path,
                scene.start_time,
                scene.end_time
            )
            
            caption = self.captioner.encode(frames)
            
            transcript = audio_data["transcript"]
            full_text = f"Transcript: {transcript}. Visuals: {caption}"
            
            text_embedding = self.text_encoder.encode(full_text)
            
            return {
                "video": video_data["video"],
                "audio": audio_data["audio_embedding"],
                "text": text_embedding,
                "caption": caption,
                "transcript": transcript,
                "keyframes": video_data["image"],
            }
        except Exception as e:
            logging.error(f"Failed to encode scene {scene_key}: {e}")
            return None
        finally:
            # No more temp files to clean up
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def _aggregate_embeddings(self, scene_embeddings: dict) -> dict:
        """Aggregates scene embeddings to create global video embeddings."""
        global_embs = {"video": [], "audio": [], "text": []}
        for scene_data in scene_embeddings.values():
            for key in global_embs.keys():
                if scene_data[key] is not None:
                    global_embs[key].append(scene_data[key])
        
        aggregated = {}
        for key, embs in global_embs.items():
            if embs:
                aggregated[key] = torch.stack(embs).mean(dim=0)
            else:
                aggregated[key] = None # Or a zero vector
                
        return aggregated

    def encode_videos(self) -> VideoDataset:
        for dp in tqdm(self.dataset.video_datapoints, desc="Encoding Videos"):
            video_path = dp.video_path
            logging.info(f"Processing video: {video_path}")
            
            dp.scenes = self._detect_scenes(video_path)
            if not dp.scenes:
                logging.warning(f"No scenes detected for {video_path}. Skipping.")
                continue

            futures = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for i, scene in enumerate(dp.scenes):
                    sid = f"scene_{i}"
                    futures[executor.submit(self._encode_scene, dp.video_path, scene)] = sid

                for f in tqdm(as_completed(futures), total=len(futures), desc=f"Scenes ({dp.video_name})"):
                    sid = futures[f]
                    try:
                        scene_out = f.result()
                        if scene_out:
                            dp.scene_embeddings[sid] = scene_out
                        else:
                            logging.warning(f"[SKIP] scena {sid} vuota.")
                    except Exception as e:
                        logging.error(f"[ERROR] scena {sid} fallita: {e}")
                
            if not dp.scene_embeddings:
                logging.warning(f"No scenes were successfully encoded for {video_path}.")
                continue

            dp.global_embeddings = self._aggregate_embeddings(dp.scene_embeddings)
        
        return self.dataset