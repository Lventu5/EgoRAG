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
        use_video_captioning: bool = True,
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
        self.captioner = VisualCaptioner(
            device=self.device,
            use_video_model=use_video_captioning
        )
        
        caption_mode = "video-aware (BLIP-2)" if use_video_captioning else "legacy (BLIP)"
        logging.info(f"MultiModalEncoder initialized with {max_workers} workers and ModelRegistry.")
        logging.info(f"  Caption mode: {caption_mode}")

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
            logging.info(f"Starting scene detection for: {video_path}")
            scene_list = detect(video_path, ContentDetector(threshold=25.0))
            logging.info(
                f"✓ Scene detection successful for {os.path.basename(video_path)}: "
                f"found {len(scene_list)} scenes"
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
        except FileNotFoundError:
            logging.error(f"✗ Scene detection failed - Video file not found: {video_path}")
            return {}
        except Exception as e:
            logging.error(f"✗ Scene detection failed for {os.path.basename(video_path)}")
            logging.error(f"  Error type: {type(e).__name__}")
            logging.error(f"  Error message: {str(e)}")
            logging.error(f"  Video path: {video_path}")
            logging.debug(f"Full traceback:\n{traceback.format_exc()}")
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
        Stores raw video embeddings. Keyframes are temporarily stored for caption generation
        but will be cleaned up immediately after.
        """
        logging.info(f"[Stage 1/5] Video encoding for {dp.video_name}")
        logging.info(f"  Processing {len(dp.scenes)} scenes...")
        
        futures = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for sid, scene in dp.scenes.items():
                futures[executor.submit(self._extract_and_encode_video, scene, vr)] = sid

            success_count = 0
            fail_count = 0
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Video Encoding ({dp.video_name})"):
                sid = futures[f]
                try:
                    video_data = f.result()
                    if video_data:
                        # Store embeddings and keyframes (keyframes will be cleaned after caption stage)
                        dp.scene_embeddings[sid] = {
                            "video": video_data["video"],
                            "keyframes": video_data["keyframes"]  # Temporary - will be deleted after captioning
                        }
                        success_count += 1
                    else:
                        logging.warning(f"  ⚠ Scene {sid} skipped - video encoding returned None")
                        fail_count += 1
                except Exception as e:
                    logging.error(f"  ✗ Scene {sid} video encoding failed")
                    logging.error(f"    Error type: {type(e).__name__}")
                    logging.error(f"    Error message: {str(e)}")
                    logging.debug(f"    Full traceback:\n{traceback.format_exc()}")
                    fail_count += 1
            
            logging.info(f"  Stage 1 complete: {success_count} succeeded, {fail_count} failed")

    def _extract_and_encode_video(self, scene: Scene, vr: VideoReader) -> dict | None:
        """
        Helper: Extract frames and encode with video encoder.
        
        Returns both video embeddings and representative keyframes.
        Note: Keyframes are raw frame arrays needed for caption generation,
        but they will be deleted after captions are created to save memory.
        """
        try:
            frames = self._extract_frames(vr, scene.start_frame, scene.end_frame, self.video_encoder.max_frames_per_scene)
            video_data = self.video_encoder.encode(frames)
            del frames  # Clean up full frame set immediately
            return {"video": video_data["video"], "keyframes": video_data["keyframes"]}
        except AttributeError as e:
            logging.error(f"  ✗ Video encoding failed for scene {scene.scene_id}")
            logging.error(f"    Likely cause: Model not loaded properly")
            logging.error(f"    Error: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"  ✗ Video encoding failed for scene {scene.scene_id}")
            logging.error(f"    Scene time range: {scene.start_time:.2f}s - {scene.end_time:.2f}s")
            logging.error(f"    Frame range: {scene.start_frame} - {scene.end_frame}")
            logging.error(f"    Error type: {type(e).__name__}")
            logging.error(f"    Error message: {str(e)}")
            return None

    def _encode_audio_stage(self, video_path: str, dp: VideoDataPoint) -> None:
        """
        Stage 2: Encode all scenes with audio encoder.
        Requires video_path and scene timing information.
        """
        logging.info(f"[Stage 2/5] Audio encoding for {dp.video_name}")
        logging.info(f"  Processing {len(dp.scenes)} scenes...")
        
        futures = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for sid, scene in dp.scenes.items():
                futures[executor.submit(self.audio_encoder.encode, video_path, scene.start_time, scene.end_time)] = sid

            success_count = 0
            fail_count = 0
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Audio Encoding ({dp.video_name})"):
                sid = futures[f]
                try:
                    audio_data = f.result()
                    if audio_data and sid in dp.scene_embeddings:
                        dp.scene_embeddings[sid]["audio"] = audio_data["audio_embedding"]
                        dp.scene_embeddings[sid]["transcript"] = audio_data["transcript"]
                        success_count += 1
                    else:
                        if sid not in dp.scene_embeddings:
                            logging.warning(f"  ⚠ Scene {sid} skipped - not found in scene_embeddings (video stage failed)")
                        else:
                            logging.warning(f"  ⚠ Scene {sid} skipped - audio encoding returned None")
                        fail_count += 1
                except Exception as e:
                    logging.error(f"  ✗ Scene {sid} audio encoding failed")
                    logging.error(f"    Error type: {type(e).__name__}")
                    logging.error(f"    Error message: {str(e)}")
                    logging.debug(f"    Full traceback:\n{traceback.format_exc()}")
                    fail_count += 1
            
            logging.info(f"  Stage 2 complete: {success_count} succeeded, {fail_count} failed")

    def _encode_caption_stage(self, dp: VideoDataPoint) -> None:
        """
        Stage 3: Generate captions from keyframes for all scenes.
        Requires keyframes from video encoding stage.
        """
        logging.info(f"[Stage 3/5] Caption generation for {dp.video_name}")
        
        scenes_with_keyframes = [sid for sid, data in dp.scene_embeddings.items() if "keyframes" in data]
        logging.info(f"  Processing {len(scenes_with_keyframes)} scenes with keyframes...")
        
        futures = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for sid, scene_data in dp.scene_embeddings.items():
                if "keyframes" in scene_data:
                    futures[executor.submit(self.captioner.encode, scene_data["keyframes"])] = sid

            success_count = 0
            fail_count = 0
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Caption Generation ({dp.video_name})"):
                sid = futures[f]
                try:
                    caption = f.result()
                    if sid in dp.scene_embeddings:
                        dp.scene_embeddings[sid]["caption_text"] = caption
                        success_count += 1
                    else:
                        logging.warning(f"  ⚠ Scene {sid} skipped - not found in scene_embeddings")
                        fail_count += 1
                except Exception as e:
                    logging.error(f"  ✗ Scene {sid} caption generation failed")
                    logging.error(f"    Error type: {type(e).__name__}")
                    logging.error(f"    Error message: {str(e)}")
                    logging.debug(f"    Full traceback:\n{traceback.format_exc()}")
                    fail_count += 1
            
            logging.info(f"  Stage 3 complete: {success_count} succeeded, {fail_count} failed")
        
        # Clean up keyframes from memory to prevent bloat in pickle files
        self._cleanup_keyframes(dp)

    def _cleanup_keyframes(self, dp: VideoDataPoint) -> None:
        """
        Remove raw keyframe arrays from scene_embeddings after captions are generated.
        This prevents massive memory overhead and bloated pickle files.
        Keyframes are only needed for caption generation and should not persist.
        """
        logging.info(f"[Cleanup] Removing keyframes from memory for {dp.video_name}")
        keyframes_cleaned = 0
        total_memory_freed = 0
        
        for sid, scene_data in dp.scene_embeddings.items():
            if "keyframes" in scene_data:
                # Calculate memory size before deletion (approximate)
                keyframes = scene_data["keyframes"]
                if isinstance(keyframes, np.ndarray):
                    memory_size = keyframes.nbytes / (1024 * 1024)  # Convert to MB
                    total_memory_freed += memory_size
                
                # Delete the keyframes
                del scene_data["keyframes"]
                keyframes_cleaned += 1
        
        logging.info(f"  ✓ Cleaned {keyframes_cleaned} keyframe arrays, freed ~{total_memory_freed:.2f} MB")

    def _encode_text_stage(self, dp: VideoDataPoint) -> None:
        """
        Stage 4: Encode all text (captions + transcripts) with text encoder.
        Requires caption_text and transcript from previous stages.
        """
        logging.info(f"[Stage 4/5] Text encoding for {dp.video_name}")
        logging.info(f"  Processing {len(dp.scene_embeddings)} scenes...")
        
        futures = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for sid, scene_data in dp.scene_embeddings.items():
                caption = scene_data.get("caption_text", "")
                transcript = scene_data.get("transcript", "")
                full_text = f"Transcript: {transcript}. Visuals: {caption}"
                futures[executor.submit(self._encode_text_pair, full_text, caption)] = sid

            success_count = 0
            fail_count = 0
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Text Encoding ({dp.video_name})"):
                sid = futures[f]
                try:
                    text_emb, caption_emb = f.result()
                    if sid in dp.scene_embeddings:
                        dp.scene_embeddings[sid]["text"] = text_emb
                        dp.scene_embeddings[sid]["caption"] = caption_emb
                        success_count += 1
                    else:
                        logging.warning(f"  ⚠ Scene {sid} skipped - not found in scene_embeddings")
                        fail_count += 1
                except Exception as e:
                    logging.error(f"  ✗ Scene {sid} text encoding failed")
                    logging.error(f"    Error type: {type(e).__name__}")
                    logging.error(f"    Error message: {str(e)}")
                    logging.debug(f"    Full traceback:\n{traceback.format_exc()}")
                    fail_count += 1
            
            logging.info(f"  Stage 4 complete: {success_count} succeeded, {fail_count} failed")

    def _tag_scenes_stage(self, dp: VideoDataPoint) -> None:
        """
        Stage 5: Tag scenes with scene types and dialogue roles.
        Requires caption_text and transcript from previous stages.
        """
        if not self.apply_tagging:
            logging.info(f"[Stage 5/5] Skipping scene tagging for {dp.video_name} (disabled)")
            return
            
        logging.info(f"[Stage 5/5] Scene tagging for {dp.video_name}")
        
        try:
            # Tag scene types
            scene_types = tag_scene_types(dp)
            
            # Tag dialogue roles
            dialogue_roles = tag_dialogue_roles(dp)
            
            # Store in scene metadata
            success_count = 0
            for sid in dp.scene_embeddings.keys():
                if "meta" not in dp.scene_embeddings[sid]:
                    dp.scene_embeddings[sid]["meta"] = {}
                
                dp.scene_embeddings[sid]["meta"]["scene_type"] = scene_types.get(sid, "other")
                dp.scene_embeddings[sid]["meta"]["speech_type"] = dialogue_roles.get(sid, "no_speech")
                success_count += 1
            
            logging.info(f"  ✓ Successfully tagged {success_count} scenes")
            
            # Log distribution
            type_dist = {}
            speech_dist = {}
            for sid in dp.scene_embeddings.keys():
                meta = dp.scene_embeddings[sid].get("meta", {})
                st = meta.get("scene_type", "unknown")
                sp = meta.get("speech_type", "unknown")
                type_dist[st] = type_dist.get(st, 0) + 1
                speech_dist[sp] = speech_dist.get(sp, 0) + 1
            
            logging.info(f"  Scene types: {dict(type_dist)}")
            logging.info(f"  Speech types: {dict(speech_dist)}")
            
        except Exception as e:
            logging.error(f"  ✗ Scene tagging failed for {dp.video_name}")
            logging.error(f"    Error type: {type(e).__name__}")
            logging.error(f"    Error message: {str(e)}")
            logging.debug(f"    Full traceback:\n{traceback.format_exc()}")


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
        logging.info("="*80)
        logging.info("Starting multi-modal video encoding pipeline")
        logging.info(f"Total videos to process: {len(self.dataset.video_datapoints)}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Max workers: {self.max_workers}")
        logging.info(f"Scene tagging: {'enabled' if self.apply_tagging else 'disabled'}")
        logging.info("="*80)
        
        total_videos = len(self.dataset.video_datapoints)
        successful_videos = 0
        failed_videos = 0
        
        for idx, dp in enumerate(tqdm(self.dataset.video_datapoints, desc="Encoding Videos"), 1):
            video_path = dp.video_path
            logging.info(f"\n{'='*80}")
            logging.info(f"Processing video {idx}/{total_videos}: {os.path.basename(video_path)}")
            logging.info(f"Full path: {video_path}")
            logging.info(f"{'='*80}")
            
            try:
                # Detect scenes for all videos first
                dp.scenes = self._detect_scenes(video_path)
                if not dp.scenes:
                    logging.error(f"✗ No scenes detected for {video_path}. Skipping video.")
                    failed_videos += 1
                    continue
                
                for scene in dp.scenes.values():
                    logging.debug(f"  Scene {scene.scene_id}: "
                                f"frames [{scene.start_frame}-{scene.end_frame}], "
                                f"time [{scene.start_time:.2f}s-{scene.end_time:.2f}s]")
                
                vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
                
                # Stage 1: Video Encoding (ModelRegistry handles model lifecycle)
                self._encode_video_stage(video_path, dp, vr)
                
                if not dp.scene_embeddings:
                    logging.error(f"✗ No scenes successfully encoded for {video_path} (video stage failed)")
                    logging.error(f"  This usually indicates a problem with frame extraction or video model")
                    del vr
                    failed_videos += 1
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
                
                logging.info(f"✓ Successfully encoded video: {os.path.basename(video_path)}")
                logging.info(f"  Total scenes: {len(dp.scenes)}")
                logging.info(f"  Successfully encoded scenes: {len(dp.scene_embeddings)}")
                successful_videos += 1
                
            except FileNotFoundError as e:
                logging.error(f"✗ Video file not found: {video_path}")
                logging.error(f"  Error: {str(e)}")
                failed_videos += 1
            except Exception as e:
                logging.error(f"✗ Fatal error processing video: {os.path.basename(video_path)}")
                logging.error(f"  Error type: {type(e).__name__}")
                logging.error(f"  Error message: {str(e)}")
                logging.error(f"  Full traceback:\n{traceback.format_exc()}")
                failed_videos += 1

        self.dataset.encoded = True
        
        logging.info("\n" + "="*80)
        logging.info("Encoding pipeline complete!")
        logging.info(f"  Total videos: {total_videos}")
        logging.info(f"  Successful: {successful_videos}")
        logging.info(f"  Failed: {failed_videos}")
        logging.info(f"  Success rate: {100*successful_videos/total_videos:.1f}%")
        logging.info("="*80)
        
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