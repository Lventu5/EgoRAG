import gc
import os
import logging
import traceback
import torch
import numpy as np
import re
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
        

    def _encode_video_stage(self, video_path: str, dp: VideoDataPoint) -> None:
        """
        Stage 1: Encode all scenes with video encoder.
        Each encoder handles frame/clip extraction internally:
        
        - XCLIP: Extracts all frames, clusters to 8 representative frames, encodes
        - Qwen2-VL: Creates temporary clip of scene, loads ALL frames, extracts 
                    single embedding (first token from vision tower, no pooling)
        """
        logging.info(f"[Stage 1] Video encoding for {video_path}")
        
        # Temporary storage for keyframes (needed for captioner1)
        dp._temp_keyframes = {}
        
        futures = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for sid, scene in dp.scenes.items():
                futures[executor.submit(self.video_encoder.encode, video_path, scene)] = sid

            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Video ({dp.video_name})", disable=False):
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
                    raise ValueError("Failed")
        
        # Generate global video embedding for the entire video (except XCLIP)
        if self.video_encoder.model_name == "qwen2-vl":
            logging.info(f"[Video Stage] Encoding global video embedding for entire video ({dp.video_name})...")
            video_data = self.video_encoder.encode_full_video(video_path)
            if video_data:
                dp.global_embeddings["video"] = video_data["video"]
                logging.info(f"[Video Stage] Global video embedding generated for {dp.video_name}")
        elif self.video_encoder.model_name == "xclip":
            logging.info(f"[Video Stage] Using mean pooling for global video embedding (XCLIP)")
        elif self.video_encoder.model_name == "internvideo2":
            logging.info(f"[Video Stage] Encoding global video embedding for entire video ({dp.video_name})...")
            video_data = self.video_encoder.encode_full_video(video_path)
            if video_data:
                dp.global_embeddings["video"] = video_data["video"]
                logging.info(f"[Video Stage] Global video embedding generated for {dp.video_name}")
        else:
            raise ValueError(f"Unknown model {self.video_encoder.model_name}")

    def _encode_audio_stage(self, video_path: str, dp: VideoDataPoint) -> None:
        """
        Stage 2: Encode all scenes with audio encoder.
        Requires video_path and scene timing information.
        Gracefully handles videos without audio tracks.
        Stores audio embeddings, transcripts, audio events, and speaker diarization.
        """
        logging.info(f"[Audio Stage] Audio encoding for {video_path}")
        
        # Track if this video has audio at all
        video_has_audio = None  # Unknown until we try first scene
        scenes_with_audio = 0
        
        futures = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for sid, scene in dp.scenes.items():
                futures[executor.submit(self.audio_encoder.encode, video_path, scene.start_time, scene.end_time)] = sid

            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Audio ({dp.video_name})", disable=False):
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
                            # Store temporary data for screenplay generation
                            dp.scene_embeddings[sid]["_temp_transcript"] = audio_data["transcript"]
                            dp.scene_embeddings[sid]["_temp_audio_events"] = audio_data.get("audio_events", [])
                            dp.scene_embeddings[sid]["_temp_speaker_segments"] = audio_data.get("speaker_segments", [])
                            scenes_with_audio += 1
                        else:
                            # No audio - set to None explicitly
                            dp.scene_embeddings[sid]["audio"] = None
                            dp.scene_embeddings[sid]["_temp_transcript"] = ""
                            dp.scene_embeddings[sid]["_temp_audio_events"] = []
                            dp.scene_embeddings[sid]["_temp_speaker_segments"] = []
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
        
        # Generate global audio embedding for the entire video
        if video_has_audio:
            logging.info(f"[Caption Stage] Encoding global audio embedding for entire video ({dp.video_name})...")
            start_time, end_time = self._get_video_time_bounds(dp, video_path)
            if end_time > start_time:
                duration = end_time - start_time
                logging.info(f"[Caption Stage] Processing {duration:.1f}s of audio...")
                global_audio = self.audio_encoder.encode(video_path, start_time, end_time)
                if global_audio:
                    dp.global_embeddings["audio"] = global_audio.get("audio_embedding")
                    # Store temporary data for screenplay generation
                    dp._temp_global_transcript = global_audio.get("transcript", "")
                    dp._temp_global_audio_events = global_audio.get("audio_events", [])
                    dp._temp_global_speaker_segments = global_audio.get("speaker_segments", [])
                    logging.info(f"[Caption Stage] Global audio embedding generated for {dp.video_name}")

    def _encode_caption_stage(self, video_path: str, dp: VideoDataPoint) -> None:
        """
        Stage 3: Generate visual captions from keyframes (captioner1) or video scenes (captioner2).
        These captions will be combined with audio information in the text stage.
        
        Captioner1 (BLIP): Uses pre-extracted keyframes from video stage
        Captioner2 (Qwen2-VL): Creates temporary video clip of entire scene, 
                               loads all frames, generates caption, then deletes clip
        """
        logging.info(f"[Caption Stage] Visual caption generation for {dp.video_name} using {self.use_captioner}")
        futures = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            if self.use_captioner == "captioner1":
                # BLIP captioner: uses keyframes
                for sid in dp.scene_embeddings.keys():
                    if sid in dp._temp_keyframes:
                        futures[executor.submit(self.captioner.encode, dp._temp_keyframes[sid])] = sid
            
            elif self.use_captioner == "captioner2":
                # Qwen2-VL captioner: use precomputed video embedding when available
                for sid, scene in dp.scenes.items():
                    if sid in dp.scene_embeddings:
                        futures[executor.submit(self.captioner.encode, video_path, scene)] = sid

            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Caption ({dp.video_name})", disable=False):
                sid = futures[f]
                try:
                    caption = f.result()
                    if sid in dp.scene_embeddings:
                        # Store as temporary data for screenplay generation
                        dp.scene_embeddings[sid]["_temp_caption"] = caption
                        if "keyframes" in dp.scene_embeddings[sid]:
                            del dp.scene_embeddings[sid]["keyframes"]
                            logging.debug(f"Freed keyframes for scene {sid} in {dp.video_name}")
                    else:
                        logging.warning(f"[SKIP] scene {sid}, caption generation failed.")
                except Exception as e:
                    logging.error(f"[ERROR] scene {sid} caption generation failed: {e}")
                    logging.error(traceback.format_exc())
        
        # Generate global caption for the entire video
        logging.info(f"[Caption Stage] Generating global caption for entire video ({dp.video_name})...")
        global_caption = self._generate_full_video_caption(video_path, dp)
        # Store as temporary data for screenplay generation
        dp._temp_global_caption = global_caption
        logging.info(f"[Caption Stage] Global caption generated for {dp.video_name}")
        
        if hasattr(dp, "_temp_keyframes"):
            del dp._temp_keyframes
            logging.info(f"[Caption Stage] Keyframes cleaned up for {dp.video_name}")

    def _encode_text_stage(self, dp: VideoDataPoint) -> None:
        """
        Stage 4: Generate screenplay-style summaries and encode with text encoder.
        Combines visual captions, transcripts, audio events, and speaker diarization
        into a coherent narrative description using TextEncoder's LLM.
        """
        logging.info(f"[Text Stage] Generating screenplay summaries for {dp.video_name}")
        
        # Generate screenplay for each scene
        futures = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for sid, scene_data in dp.scene_embeddings.items():
                # Prepare data for screenplay generation
                screenplay_data = {
                    "caption_text": scene_data.get("_temp_caption", ""),
                    "transcript": scene_data.get("_temp_transcript", ""),
                    "audio_events": scene_data.get("_temp_audio_events", []),
                    "speaker_segments": scene_data.get("_temp_speaker_segments", [])
                }
                futures[executor.submit(self.text_encoder.generate_screenplay_summary, screenplay_data)] = sid

            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Text ({dp.video_name})", disable=False):
                sid = futures[f]
                try:
                    screenplay = f.result()
                    if sid in dp.scene_embeddings:
                        # Store the screenplay text and its embedding
                        dp.scene_embeddings[sid]["text_raw"] = screenplay
                        text_emb = self.text_encoder.encode(screenplay)
                        dp.scene_embeddings[sid]["text"] = text_emb
                        
                        # Clean up temporary data
                        for temp_key in ["_temp_caption", "_temp_transcript", "_temp_audio_events", "_temp_speaker_segments"]:
                            if temp_key in dp.scene_embeddings[sid]:
                                del dp.scene_embeddings[sid][temp_key]
                    else:
                        logging.warning(f"[SKIP] scene {sid}, screenplay generation failed.")
                except Exception as e:
                    logging.error(f"[ERROR] scene {sid} screenplay generation failed: {e}")
                    logging.error(traceback.format_exc())
        
        # Generate global screenplay summary for the entire video
        logging.info(f"[Text Stage] Generating global screenplay summary for {dp.video_name}")
        scene_screenplays = [(sid, scene_data.get("text_raw", "")) for sid, scene_data in sorted(dp.scene_embeddings.items())]
        global_screenplay = self.text_encoder.generate_global_screenplay(scene_screenplays)
        dp.global_embeddings["text_raw"] = global_screenplay
        text_emb = self.text_encoder.encode(global_screenplay)
        dp.global_embeddings["text"] = text_emb
        
        # Clean up temporary global data
        for attr in ["_temp_global_caption", "_temp_global_transcript", "_temp_global_audio_events", "_temp_global_speaker_segments"]:
            if hasattr(dp, attr):
                delattr(dp, attr)
        
        logging.info(f"[Text Stage] Global screenplay summary generated for {dp.video_name}")

    def _aggregate_embeddings(self, scene_embeddings: dict) -> dict:
        """Aggregates scene embeddings to create global video embeddings."""
        global_embs = {"video": [], "audio": [], "text": []}
        for scene_data in scene_embeddings.values():
            for key in global_embs.keys():
                if scene_data.get(key) is not None:
                    global_embs[key].append(scene_data[key])
        
        aggregated = {}
        for key, embs in global_embs.items():
            if embs:
                aggregated[key] = torch.stack(embs).mean(dim=0)
            else:
                aggregated[key] = None 
                
        return aggregated

    def _get_video_time_bounds(self, dp: VideoDataPoint, video_path: str) -> tuple[float, float]:
        """Return the start and end time for the full video."""
        if dp.scenes:
            starts = [scene.start_time for scene in dp.scenes.values() if scene is not None]
            ends = [scene.end_time for scene in dp.scenes.values() if scene is not None]
            if starts and ends:
                return max(0.0, min(starts)), max(0.01, max(ends))
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            total_frames = len(vr)
            fps = vr.get_avg_fps()
            del vr
            duration = (total_frames / fps) if fps and fps > 0 else 0.0
            return 0.0, max(0.01, duration)
        except Exception as exc:
            logging.error(f"[Duration] Failed to estimate video duration for {video_path}: {exc}")
            return 0.0, 0.01

    def _generate_full_video_caption(self, video_path: str, dp: VideoDataPoint) -> str:
        """Generate a single caption describing the full video."""
        if self.use_captioner == "captioner1":
            # For BLIP: collect keyframes from all scenes
            keyframes = self._collect_keyframes_from_scenes(dp)
            if keyframes.size == 0:
                logging.warning(f"[Caption] No keyframes for full video {dp.video_name}")
                return ""
            return self.captioner.encode(keyframes)
        
        if self.use_captioner == "captioner2":
            # For LLaVA captioner: summarize all per-scene captions by asking the captioner (LLM)
            captions = []
            for sid in sorted(dp.scene_embeddings.keys()):
                c = dp.scene_embeddings[sid].get("caption_text", "")
                if c:
                    captions.append(f"Scene {sid}: {c}")

            if not captions:
                logging.warning(f"[Caption] No scene captions available for full video {dp.video_name}")
                return ""

            # Build summarization prompt
            prompt_lines = [
                "You are a helpful assistant. Summarize the following scene captions into a concise, coherent paragraph describing the whole video. Keep it short (1-3 sentences).",
                "\nCaptions:\n"
            ]
            prompt_lines.extend(captions)
            full_prompt = "\n".join(prompt_lines)

            try:
                # Use captioner.processor to tokenize the prompt and captioner.model to generate summary
                messages = [{"role": "user", "content": [{"type": "text", "text": full_prompt}]}]
                proc_inputs = self.captioner.processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
                # The processor may return a single Tensor (e.g., input_ids) or a dict
                # of tensors. Normalize to a dict so we can safely move tensors to device.
                if isinstance(proc_inputs, torch.Tensor):
                    proc_inputs = {"input_ids": proc_inputs}
                elif not isinstance(proc_inputs, dict):
                    proc_inputs = dict(proc_inputs)
                
                proc_inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in proc_inputs.items()}

                with torch.inference_mode():
                    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                        gen_ids = self.captioner.model.generate(**proc_inputs, max_new_tokens=128)

                summary = self.captioner.processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()

                # Extract assistant reply if model returns chat-style text
                m = re.search(r"assistant:\s*(.*)$", summary, flags=re.IGNORECASE | re.DOTALL)
                if m:
                    summary = m.group(1).strip()
                else:
                    parts = re.split(r"assistant:\s*", summary, flags=re.IGNORECASE)
                    if parts and len(parts) > 1:
                        summary = parts[-1].strip()

                # Remove trailing separators like lines of dashes
                summary = re.sub(r"\n[-]{3,}.*$", "", summary, flags=re.DOTALL).strip()
                # Cleanup
                del proc_inputs, gen_ids
                if self.device == "cuda":
                    torch.cuda.empty_cache()

                return summary
            except Exception as e:
                logging.error(f"[Caption][full_video] summarization failed: {e}")
                logging.error(traceback.format_exc())
                return ""
        
        logging.error(f"[Caption] Unknown captioner: {self.use_captioner}")
        return ""

    def _collect_keyframes_from_scenes(self, dp: VideoDataPoint, max_frames: int = 32) -> np.ndarray:
        """Concatenate keyframes from all scenes for global captioning."""
        if not hasattr(dp, "_temp_keyframes"):
            return np.array([])
        
        collected = []
        for frames in dp._temp_keyframes.values():
            if isinstance(frames, np.ndarray) and frames.size > 0:
                collected.append(frames)
        
        if not collected:
            return np.array([])
        
        stacked = np.concatenate(collected, axis=0)
        if len(stacked) > max_frames:
            indices = np.linspace(0, len(stacked) - 1, max_frames, dtype=int)
            stacked = stacked[indices]
        return stacked

    def encode_videos(self) -> VideoDataset:
        """
        Orchestrates encoding in stages to load/unload models one at a time.
        
        Stages:
        1. Video encoding:
           - XCLIP: Extracts all frames → clustering → selects 8 frames → encoding (768-dim)
           - Qwen2-VL: Creates temp clip → loads all frames → single embedding (1536-dim, first token)
        
        2. Caption generation (if both video & caption use Qwen2-VL, share the model):
           - Captioner1 (BLIP): Uses keyframes from stage 1
           - Captioner2 (Qwen2-VL): Creates temp clip → loads all frames → generates caption
        
        3. Audio encoding:
           - Extracts audio from scene
           - Whisper transcription + CLAP embedding
        
        4. Text encoding:
           - Encodes transcript + caption with Sentence Transformers
        """
        for dp in tqdm(self.dataset.video_datapoints, desc="Encoding Videos", disable=False):
            # GPU monitor semplice dopo ogni video
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1024**3
                mem_free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - mem
                print(f"[GPU] Allocated: {mem:.2f} GB, Free: {mem_free:.2f} GB")

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
                    method="temporal",
                    existing_scenes=None
                )
            
            if not dp.scenes:
                logging.warning(f"No scenes detected for {video_path}. Skipping.")
                continue

            video_model_name = CONFIG.indexing.video.model_name

            self.video_encoder.load_models()
            self._encode_video_stage(video_path, dp)
            
            if not dp.scene_embeddings:
                logging.warning(f"No scenes were successfully encoded for {video_path} (video stage).")
                self.unload_model("video")
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
                continue
            self.unload_model("video")
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Load captioner models separately (no sharing between video encoder and captioner).
            self.captioner.load_models()
            
            self._encode_caption_stage(video_path, dp)

            self.unload_model("caption")
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Stage 2: Audio Encoding
            # Fast-check if the video has an audio track before loading heavy models
            """
            has_audio = self.audio_encoder.has_audio_track(video_path)

            if not has_audio:
                logging.info(f"Video {video_path} appears to have no audio - skipping audio stage")
                dp.has_audio = False
            else:
                self.audio_encoder.load_models()
                self._encode_audio_stage(video_path, dp)
                self.unload_model("audio")
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            """
            
            # Stage 3: Text Encoding
            self.text_encoder.load_models()
            self._encode_text_stage(dp)
            self.unload_model("text")
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # For XCLIP video: use mean pooling since we don't call the model on full video
            if CONFIG.indexing.video.model_name == "xclip":
                aggregated = self._aggregate_embeddings(dp.scene_embeddings)
                # Only override video embedding, keep the rest (audio, text were computed globally)
                if aggregated.get("video") is not None:
                    dp.global_embeddings["video"] = aggregated["video"]
                    logging.info(f"[Aggregate] Used mean pooling for global video embedding (XCLIP)")

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