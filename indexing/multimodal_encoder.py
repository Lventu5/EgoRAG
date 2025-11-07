import gc
import os
import logging
import traceback
import torch
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu
from scenedetect import detect, ContentDetector
from concurrent.futures import ThreadPoolExecutor, as_completed

from data.video_dataset import VideoDataset, VideoDataPoint, Scene
from indexing.components.model_registry import ModelRegistry
from indexing.components.video_encoder import VideoEncoder
from indexing.components.audio_encoder import AudioEncoder
from indexing.components.text_encoder import TextEncoder
from indexing.components.visual_captioner import VisualCaptioner
from indexing.analytics.tagging import tag_scene_types, tag_dialogue_roles

# --- Setup logging (can be moved to a main file) ---
# handler = logging.StreamHandler()
# handler.setFormatter(LevelAwareFormatter())
# logging.basicConfig(level=logging.INFO, handlers=[handler])
# # ---

class MultiModalEncoder:
    """
    Orchestrates the hierarchical, multi-modal encoding of long videos.
    
    This class manages the full pipeline:
    1.  Scene Detection
    2.  Asset Extraction (frames, audio)
    3.  Delegation to atomic components (Video, Audio, Text, Captioner)
    4.  Scene Tagging (types and dialogue roles)
    5.  Aggregation of results into a VideoDataPoint.
    
    Uses ModelRegistry for efficient model lifecycle management.
    """
    def __init__(
        self,
        video_dataset: VideoDataset,
        device: str = "cuda",
        max_frames_per_scene: int = 96,
        max_temporal_segments: int = 8,
        audio_sr: int = 48000, 
        asr_sr: int = 16000,
        max_workers: int = 2,
        apply_tagging: bool = True,
    ):
        if video_dataset is None or len(video_dataset) == 0:
            raise ValueError("Video dataset is empty or not provided.")
        
        self.dataset = video_dataset
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_workers = max_workers
        self.apply_tagging = apply_tagging

        # IMPORTANT: VideoReader (decord) is NOT thread-safe
        # Using max_workers > 1 can cause segmentation faults
        if max_workers > 1:
            logging.warning(
                "WARNING: max_workers > 1 may cause segmentation faults! "
                "VideoReader is not thread-safe when shared across threads. "
                "Consider using max_workers=1 for stable execution."
            )

        # Initialize ModelRegistry
        self.registry = ModelRegistry()

        # 1. Instantiate Atomic Components (without loading models)
        self.video_encoder = VideoEncoder(
            device=self.device,
            max_frames_per_scene=max_frames_per_scene,
            max_temporal_segments=max_temporal_segments
        )
        self.audio_encoder = AudioEncoder(
            device=self.device,
            audio_sr=audio_sr,
            asr_sr=asr_sr
        )
        self.text_encoder = TextEncoder(device=self.device)
        self.captioner = VisualCaptioner(device=self.device)
        
        logging.info(f"MultiModalEncoder initialized with {max_workers} workers and ModelRegistry.")

    def load_models(self):
        """
        Loads all component models. 
        Note: With ModelRegistry, models are loaded lazily on first use.
        This method can be used to pre-warm the registry if needed.
        """
        logging.info("Models will be loaded lazily via ModelRegistry.")
        # Optional: pre-load all models
        # self.video_encoder.load_models()
        # self.audio_encoder.load_models()
        # self.text_encoder.load_models()
        # self.captioner.load_models()

    def _detect_scenes(self, video_path: str) -> dict[str, Scene]:
        """Detects content-based scenes and returns Scene objects."""
        try:
            scene_list = detect(video_path, ContentDetector(threshold=25.0))
            logging.info(
                f"Scene extraction for video {os.path.basename(video_path)}, found {len(scene_list)} scenes"
            )
            return {
                f"scene_{i}": Scene(
                    scene_id=f"scene_{i}",
                    start_time=start.get_seconds(),
                    end_time=end.get_seconds(),
                    start_frame=start.get_frames(),
                    end_frame=end.get_frames(),
                )
                for i, (start, end) in enumerate(scene_list)
            }
        except Exception as e:
            logging.error(f"{'='*100} \n Scene detection failed for {video_path}: {e}")
            logging.error(traceback.format_exc())
            return {}

    def _extract_frames(self, vr: VideoReader, start_frame: int, end_frame: int, max_frames: int) -> tuple[np.ndarray, np.ndarray]:
        """Extracts frames for a given scene."""
        # vr = VideoReader(video_path, ctx=cpu(0))
        num_frames_in_scene = end_frame - start_frame
        
        if num_frames_in_scene > max_frames:
            indices = np.linspace(start_frame, end_frame - 1, max_frames, dtype=int)
        else:
            indices = np.arange(start_frame, end_frame)
            
        frames = vr.get_batch(indices).asnumpy()
        return frames

    def _encode_video_stage(self, video_path: str, dp: VideoDataPoint, vr: VideoReader) -> None:
        """
        Stage 1: Extract frames and encode all scenes with video encoder.
        Stores raw video embeddings and keyframes for later stages.
        """
        logging.info(f"[Stage 1] Video encoding for {video_path}")
        futures = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for sid, scene in dp.scenes.items():
                futures[executor.submit(self._extract_and_encode_video, scene, vr)] = sid

            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Video Encoding ({dp.video_name})"):
                sid = futures[f]
                try:
                    video_data = f.result()
                    if video_data:
                        dp.scene_embeddings[sid] = {"video": video_data["video"], "keyframes": video_data["keyframes"]}
                    else:
                        logging.warning(f"[SKIP] scene {sid}, video encoding failed.")
                except Exception as e:
                    logging.error(f"[ERROR] scene {sid} video encoding failed: {e}")
                    logging.error(traceback.format_exc())

    def _extract_and_encode_video(self, scene: Scene, vr: VideoReader) -> dict | None:
        """Helper: Extract frames and encode with video encoder."""
        try:
            frames = self._extract_frames(vr, scene.start_frame, scene.end_frame, self.video_encoder.max_frames_per_scene)
            video_data = self.video_encoder.encode(frames)
            del frames
            return {"video": video_data["video"], "keyframes": video_data["image"]}
        except Exception as e:
            logging.error(f"Failed to encode video for scene: {e}")
            return None

    def _encode_audio_stage(self, video_path: str, dp: VideoDataPoint) -> None:
        """
        Stage 2: Encode all scenes with audio encoder.
        Requires video_path and scene timing information.
        """
        logging.info(f"[Stage 2] Audio encoding for {video_path}")
        futures = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for sid, scene in dp.scenes.items():
                futures[executor.submit(self.audio_encoder.encode, video_path, scene.start_time, scene.end_time)] = sid

            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Audio Encoding ({dp.video_name})"):
                sid = futures[f]
                try:
                    audio_data = f.result()
                    if audio_data and sid in dp.scene_embeddings:
                        dp.scene_embeddings[sid]["audio"] = audio_data["audio_embedding"]
                        dp.scene_embeddings[sid]["transcript"] = audio_data["transcript"]
                    else:
                        logging.warning(f"[SKIP] scene {sid}, audio encoding failed.")
                except Exception as e:
                    logging.error(f"[ERROR] scene {sid} audio encoding failed: {e}")
                    logging.error(traceback.format_exc())

    def _encode_caption_stage(self, dp: VideoDataPoint) -> None:
        """
        Stage 3: Generate captions from keyframes for all scenes.
        Requires keyframes from video encoding stage.
        """
        logging.info(f"[Stage 3] Caption generation for {dp.video_name}")
        futures = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for sid, scene_data in dp.scene_embeddings.items():
                if "keyframes" in scene_data:
                    futures[executor.submit(self.captioner.encode, scene_data["keyframes"])] = sid

            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Caption Generation ({dp.video_name})"):
                sid = futures[f]
                try:
                    caption = f.result()
                    if sid in dp.scene_embeddings:
                        dp.scene_embeddings[sid]["caption_text"] = caption
                    else:
                        logging.warning(f"[SKIP] scene {sid}, caption generation failed.")
                except Exception as e:
                    logging.error(f"[ERROR] scene {sid} caption generation failed: {e}")
                    logging.error(traceback.format_exc())

    def _encode_text_stage(self, dp: VideoDataPoint) -> None:
        """
        Stage 4: Encode all text (captions + transcripts) with text encoder.
        Requires caption_text and transcript from previous stages.
        """
        logging.info(f"[Stage 4] Text encoding for {dp.video_name}")
        futures = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for sid, scene_data in dp.scene_embeddings.items():
                caption = scene_data.get("caption_text", "")
                transcript = scene_data.get("transcript", "")
                full_text = f"Transcript: {transcript}. Visuals: {caption}"
                futures[executor.submit(self._encode_text_pair, full_text, caption)] = sid

            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Text Encoding ({dp.video_name})"):
                sid = futures[f]
                try:
                    text_emb, caption_emb = f.result()
                    if sid in dp.scene_embeddings:
                        dp.scene_embeddings[sid]["text"] = text_emb
                        dp.scene_embeddings[sid]["caption"] = caption_emb
                    else:
                        logging.warning(f"[SKIP] scene {sid}, text encoding failed.")
                except Exception as e:
                    logging.error(f"[ERROR] scene {sid} text encoding failed: {e}")
                    logging.error(traceback.format_exc())

    def _tag_scenes_stage(self, dp: VideoDataPoint) -> None:
        """
        Stage 5: Tag scenes with scene types and dialogue roles.
        Requires caption_text and transcript from previous stages.
        """
        if not self.apply_tagging:
            logging.info(f"[Stage 5] Skipping scene tagging for {dp.video_name} (disabled)")
            return
            
        logging.info(f"[Stage 5] Scene tagging for {dp.video_name}")
        
        # Tag scene types
        scene_types = tag_scene_types(dp)
        
        # Tag dialogue roles
        dialogue_roles = tag_dialogue_roles(dp)
        
        # Store in scene metadata
        for sid in dp.scene_embeddings.keys():
            if "meta" not in dp.scene_embeddings[sid]:
                dp.scene_embeddings[sid]["meta"] = {}
            
            dp.scene_embeddings[sid]["meta"]["scene_type"] = scene_types.get(sid, "other")
            dp.scene_embeddings[sid]["meta"]["speech_type"] = dialogue_roles.get(sid, "no_speech")
        
        logging.info(f"[Stage 5] Tagged {len(dp.scene_embeddings)} scenes for {dp.video_name}")


    def _encode_text_pair(self, full_text: str, caption: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Helper: Encode both full text and caption separately."""
        text_embedding = self.text_encoder.encode(full_text)
        caption_emb = self.text_encoder.encode(caption) if caption else torch.zeros(384, dtype=torch.float32)
        return text_embedding, caption_emb

    def _aggregate_embeddings(self, scene_embeddings: dict) -> dict:
        """Aggregates scene embeddings to create global video embeddings."""
        global_embs = {"video": [], "audio": [], "text": [], "caption": []}
        for scene_data in scene_embeddings.values():
            for key in global_embs.keys():
                if scene_data[key] is not None:
                    global_embs[key].append(scene_data[key])
        
        aggregated = {}
        for key, embs in global_embs.items():
            if embs:
                aggregated[key] = torch.stack(embs).mean(dim=0)
            else:
                aggregated[key] = None 
                
        return aggregated

    def encode_videos(self) -> VideoDataset:
        """
        Orchestrates encoding in stages using ModelRegistry for efficient memory management.
        
        Stages:
        1. Video encoding (extract frames + encode)
        2. Audio encoding
        3. Caption generation
        4. Text encoding (captions + transcripts)
        5. Scene tagging (types and dialogue roles)
        
        Models are managed via ModelRegistry - no manual load/unload needed per stage.
        """
        for dp in tqdm(self.dataset.video_datapoints, desc="Encoding Videos"):
            video_path = dp.video_path
            logging.info(f"Processing video: {video_path}")
            
            # Detect scenes for all videos first
            dp.scenes = self._detect_scenes(video_path)
            if not dp.scenes:
                logging.warning(f"No scenes detected for {video_path}. Skipping.")
                continue
            for scene in dp.scenes.values():
                logging.info(f"Detected scene {scene.scene_id}: frames {scene.start_frame}-{scene.end_frame}, time {scene.start_time:.2f}-{scene.end_time:.2f}s")
            
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            
            # Stage 1: Video Encoding (ModelRegistry handles model lifecycle)
            self._encode_video_stage(video_path, dp, vr)
            
            if not dp.scene_embeddings:
                logging.warning(f"No scenes were successfully encoded for {video_path} (video stage).")
                del vr
                continue
            
            # Stage 2: Audio Encoding
            self._encode_audio_stage(video_path, dp)
            
            # Stage 3: Caption Generation
            self._encode_caption_stage(dp)
            
            # Stage 4: Text Encoding
            self._encode_text_stage(dp)
            
            # Stage 5: Scene Tagging
            self._tag_scenes_stage(dp)
            
            del vr
            
            # Aggregate embeddings
            dp.global_embeddings = self._aggregate_embeddings(dp.scene_embeddings)
            
            # Periodic GPU cleanup (ModelRegistry handles model caching)
            if self.device == "cuda":
                torch.cuda.empty_cache()

        self.dataset.encoded = True
        logging.info(f"Encoding complete for {len(self.dataset.video_datapoints)} videos.")
        
        return self.dataset
    
    def unload_models(self):
        """
        Free heavy encoder models from GPU.
        With ModelRegistry, this unloads all registered models.
        """
        logging.info("Unloading all models via ModelRegistry...")
        self.registry.unload_all()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        logging.info("All models unloaded successfully.")