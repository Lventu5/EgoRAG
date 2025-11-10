import gc
import os
import logging
import traceback
import torch
import numpy as np
from threading import Lock
from tqdm import tqdm
from decord import VideoReader, cpu
from concurrent.futures import ThreadPoolExecutor, as_completed

from data.video_dataset import VideoDataset, VideoDataPoint, Scene
from utils.scene_utils import SceneDetector
from indexing.utils.logging import LevelAwareFormatter
from indexing.components.video_encoder import VideoEncoder
from indexing.components.audio_encoder import AudioEncoder
from indexing.components.text_encoder import TextEncoder
from indexing.components.visual_captioner import VisualCaptioner as VisualCaptioner1
from indexing.components.visual_captioner2 import VisualCaptioner as VisualCaptioner2
from configuration.config import CONFIG

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
        max_workers: int = 2,
    ):
        if video_dataset is None or len(video_dataset) == 0:
            raise ValueError("Video dataset is empty or not provided.")
        
        self.dataset = video_dataset
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_workers = max_workers
        self.video_reader_lock = Lock()

        # 1. Instantiate Atomic Components
        self.video_encoder = VideoEncoder(
            device=self.device,
        )
        self.audio_encoder = AudioEncoder( 
            device=self.device,
        )
        self.text_encoder = TextEncoder(
            device=self.device
        )
        
        # Select captioner based on config
        self.use_captioner = CONFIG.indexing.caption.use_captioner
        if self.use_captioner == "captioner1":
            self.captioner = VisualCaptioner1(device=self.device)
        elif self.use_captioner == "captioner2":
            self.captioner = VisualCaptioner2(device=self.device)
        else:
            raise ValueError(f"Unknown captioner: {self.use_captioner}. Use 'captioner1' or 'captioner2'")
        
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

    def _encode_video_stage(self, video_path: str, dp: VideoDataPoint) -> None:
        """
        Stage 1: Extract frames and encode all scenes with video encoder.
        Two modes:
        1. Frame extraction mode (use_video_clips=False): Pre-extract frames using VideoReader
        2. Video clip mode (use_video_clips=True): Pass video_path to encoder, which creates clips
        """
        logging.info(f"[Stage 1] Video encoding for {video_path} (use_clips={self.video_encoder.use_video_clips})")
        
        # Temporary storage for keyframes (needed for captioner1)
        dp._temp_keyframes = {}
        
        if self.video_encoder.use_video_clips:
            # Mode 2: Video clip mode - encoder handles everything
            futures = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for sid, scene in dp.scenes.items():
                    futures[executor.submit(self._encode_video_from_clip, video_path, scene)] = sid

                for f in tqdm(as_completed(futures), total=len(futures), desc=f"Video Encoding ({dp.video_name})"):
                    sid = futures[f]
                    try:
                        video_data = f.result()
                        if video_data:
                            dp.scene_embeddings[sid] = {"video": video_data["video"]}
                            dp._temp_keyframes[sid] = video_data["keyframes"]
                        else:
                            logging.warning(f"[SKIP] scene {sid}, video encoding failed.")
                    except Exception as e:
                        logging.error(f"[ERROR] scene {sid} video encoding failed: {e}")
                        logging.error(traceback.format_exc())
        else:
            # Mode 1: Frame extraction mode - use VideoReader with lock
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            
            futures = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for sid, scene in dp.scenes.items():
                    futures[executor.submit(self._extract_and_encode_video, scene, vr)] = sid

                for f in tqdm(as_completed(futures), total=len(futures), desc=f"Video Encoding ({dp.video_name})"):
                    sid = futures[f]
                    try:
                        video_data = f.result()
                        if video_data:
                            dp.scene_embeddings[sid] = {"video": video_data["video"]}
                            dp._temp_keyframes[sid] = video_data["keyframes"]
                        else:
                            logging.warning(f"[SKIP] scene {sid}, video encoding failed.")
                    except Exception as e:
                        logging.error(f"[ERROR] scene {sid} video encoding failed: {e}")
                        logging.error(traceback.format_exc())
            
            del vr

    def _extract_and_encode_video(self, scene: Scene, vr: VideoReader) -> dict | None:
        """
        Helper: Extract frames and encode with video encoder (frame extraction mode).
        Frame extraction is serialized via lock, but encoding is parallel.
        """
        try:
            # Serialize frame extraction to avoid Decord segfaults
            with self.video_reader_lock:
                frames = self._extract_frames(vr, scene.start_frame, scene.end_frame, self.video_encoder.max_frames_per_scene)
            
            # Encoding happens in parallel (no lock needed)
            video_data = self.video_encoder.encode(frames=frames)
            del frames
            # video_data["image"] holds embedding(s); video_data["keyframes"] holds the actual raw frames
            return {"video": video_data["video"], "keyframes": video_data["keyframes"]}
        except Exception as e:
            logging.error(f"Failed to encode video for scene: {e}")
            logging.error(traceback.format_exc())
            return None

    def _encode_video_from_clip(self, video_path: str, scene: Scene) -> dict | None:
        """
        Helper: Encode video using clip extraction mode.
        The encoder handles clip extraction internally.
        """
        try:
            video_data = self.video_encoder.encode(video_path=video_path, scene=scene)
            if video_data is None:
                return None
            return {"video": video_data["video"], "keyframes": video_data["keyframes"]}
        except Exception as e:
            logging.error(f"Failed to encode video clip for scene {scene.scene_id}: {e}")
            logging.error(traceback.format_exc())
            return None

    def _encode_audio_stage(self, video_path: str, dp: VideoDataPoint) -> None:
        """
        Stage 2: Encode all scenes with audio encoder.
        Requires video_path and scene timing information.
        Gracefully handles videos without audio tracks.
        """
        logging.info(f"[Stage 2] Audio encoding for {video_path}")
        
        # Track if this video has audio at all
        video_has_audio = None  # Unknown until we try first scene
        scenes_with_audio = 0
        
        futures = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for sid, scene in dp.scenes.items():
                futures[executor.submit(self.audio_encoder.encode, video_path, scene.start_time, scene.end_time)] = sid

            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Audio Encoding ({dp.video_name})"):
                sid = futures[f]
                try:
                    audio_data = f.result()
                    
                    # Check if this is a video without audio
                    if video_has_audio is None:
                        video_has_audio = audio_data.get("has_audio", True)
                        if not video_has_audio:
                            logging.info(f"Video {dp.video_name} has no audio track - skipping audio encoding")
                    
                    if audio_data and sid in dp.scene_embeddings:
                        if audio_data.get("has_audio", True):
                            dp.scene_embeddings[sid]["audio"] = audio_data["audio_embedding"]
                            dp.scene_embeddings[sid]["transcript"] = audio_data["transcript"]
                            scenes_with_audio += 1
                        else:
                            # No audio - set to None explicitly
                            dp.scene_embeddings[sid]["audio"] = None
                            dp.scene_embeddings[sid]["transcript"] = ""
                    else:
                        logging.warning(f"[SKIP] scene {sid}, audio encoding failed.")
                except Exception as e:
                    logging.error(f"[ERROR] scene {sid} audio encoding failed: {e}")
                    logging.error(traceback.format_exc())
        
        # Log summary
        if video_has_audio:
            logging.info(f"Audio encoding complete: {scenes_with_audio}/{len(dp.scenes)} scenes have audio")
        else:
            logging.info(f"Video {dp.video_name} has no audio track - all audio embeddings set to None")
        
        # Store metadata about audio availability
        dp.has_audio = video_has_audio if video_has_audio is not None else False

    def _encode_caption_stage(self, video_path: str, dp: VideoDataPoint) -> None:
        """
        Stage 3: Generate captions from keyframes (captioner1) or video scenes (captioner2).
        Requires keyframes from temporary storage (deleted after this stage for captioner1).
        """
        logging.info(f"[Stage 3] Caption generation for {dp.video_name} using {self.use_captioner}")
        futures = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            if self.use_captioner == "captioner1":
                # BLIP captioner: uses keyframes
                for sid in dp.scene_embeddings.keys():
                    if sid in dp._temp_keyframes:
                        futures[executor.submit(self.captioner.encode, dp._temp_keyframes[sid])] = sid
            
            elif self.use_captioner == "captioner2":
                # LLaVA captioner: uses video path + scene timing
                for sid, scene in dp.scenes.items():
                    if sid in dp.scene_embeddings:
                        futures[executor.submit(self.captioner.encode, video_path, scene)] = sid

            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Caption Generation ({dp.video_name})"):
                sid = futures[f]
                try:
                    caption = f.result()
                    if sid in dp.scene_embeddings:
                        dp.scene_embeddings[sid]["caption_text"] = caption
                        # Free keyframes after caption generation to avoid storing large
                        # raw frames inside the VideoDataPoint (we only needed them for
                        # captioning). This keeps the pickled dataset small.
                        if "keyframes" in dp.scene_embeddings[sid]:
                            try:
                                del dp.scene_embeddings[sid]["keyframes"]
                                logging.debug(f"Freed keyframes for scene {sid} in {dp.video_name}")
                            except Exception:
                                logging.warning(f"Could not delete keyframes for scene {sid}")
                    else:
                        logging.warning(f"[SKIP] scene {sid}, caption generation failed.")
                except Exception as e:
                    logging.error(f"[ERROR] scene {sid} caption generation failed: {e}")
                    logging.error(traceback.format_exc())
        
        # Delete keyframes immediately after caption generation (only for captioner1)
        if self.use_captioner == "captioner1":
            del dp._temp_keyframes
            logging.info(f"[Stage 3] Keyframes cleaned up for {dp.video_name}")


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
        Orchestrates encoding in stages to load/unload models one at a time.
        
        Stages:
        1. Video encoding + Caption generation (if both use LLaVA, share the model)
        2. Audio encoding
        3. Text encoding (captions + transcripts)
        """
        for dp in tqdm(self.dataset.video_datapoints, desc="Encoding Videos"):
            video_path = dp.video_path
            logging.info(f"Processing video: {video_path}")
            
            # Use pre-existing scenes if available, otherwise detect them
            if dp.scenes:
                logging.info(f"Video {os.path.basename(video_path)} has {len(dp.scenes)} pre-existing scenes")
                # Scenes already set from dataset, no need to detect
            else:
                logging.info(f"No pre-existing scenes found for {os.path.basename(video_path)}, using pyscenedetect")
                dp.scenes = SceneDetector.detect_scenes(
                    video_path=video_path,
                    method="pyscenedetect",
                    existing_scenes=None
                )
            
            if not dp.scenes:
                logging.warning(f"No scenes detected for {video_path}. Skipping.")
                continue
            # for scene in dp.scenes.values():
            #     logging.info(f"Detected scene {scene.scene_id}: frames {scene.start_frame}-{scene.end_frame}, time {scene.start_time:.2f}-{scene.end_time:.2f}s")
            
            # Stage 1: Video Encoding + Caption Generation
            # If both use LLaVA (video model is llava-video and captioner is captioner2), keep model loaded
            video_model_name = CONFIG.indexing.video.model_name
            both_use_llava = (video_model_name == "llava-video" and self.use_captioner == "captioner2")
            
            self.video_encoder.load_models()
            self._encode_video_stage(video_path, dp)
            
            if not dp.scene_embeddings:
                logging.warning(f"No scenes were successfully encoded for {video_path} (video stage).")
                self.unload_model("video")
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
                continue
            
            # Stage 1b: Caption Generation immediately after video
            # (keyframes are deleted inside this stage for captioner1)
            if both_use_llava:
                logging.info(f"[Optimization] Video and Caption both use LLaVA - keeping model loaded")
                # Share the LLaVA model between video encoder and captioner
                self.captioner.model = self.video_encoder.model
                self.captioner.processor = self.video_encoder.processor
            else:
                # Load captioner models separately
                self.captioner.load_models()
            
            self._encode_caption_stage(video_path, dp)
            
            # Unload models after both video and caption are done
            if both_use_llava:
                # Unload shared LLaVA model
                self.unload_model("video")
                # Don't unload captioner separately since it shares the model
                if hasattr(self.captioner, 'model'):
                    del self.captioner.model
                if hasattr(self.captioner, 'processor'):
                    del self.captioner.processor
            else:
                # Unload video and caption separately
                self.unload_model("video")
                self.unload_model("caption")
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Stage 2: Audio Encoding
            self.audio_encoder.load_models()
            self._encode_audio_stage(video_path, dp)
            self.unload_model("audio")
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Stage 3: Text Encoding
            self.text_encoder.load_models()
            self._encode_text_stage(dp)
            self.unload_model("text")
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Aggregate embeddings
            dp.global_embeddings = self._aggregate_embeddings(dp.scene_embeddings)

        self.dataset.encoded = True
        
        return self.dataset
    
    def unload_model(self, modality: str) -> None:
        """Free heavy encoder models from GPU."""
        if modality == "text" and hasattr(self, "text_encoder"):
            del self.text_encoder
        if modality == "video" and hasattr(self, "video_encoder"):
            del self.video_encoder
        if modality == "audio" and hasattr(self, "audio_encoder"):
            del self.audio_encoder
        if modality == "caption" and hasattr(self, "captioner"):
            del self.captioner
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()