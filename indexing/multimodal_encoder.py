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

from data.video_dataset import VideoDataset, VideoDataPoint, Scene, Window
from utils.scene_utils import SceneDetector
from indexing.utils.logging import LevelAwareFormatter
from indexing.components.video_encoder import VideoEncoder
from indexing.components.audio_encoder import AudioEncoder
from indexing.components.text_encoder import TextEncoder
from indexing.components.visual_captioner import VisualCaptioner
from indexing.components.video_tagger import VisionTagger
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
        video_dataset: VideoDataset=None,
        device: str = "cuda",
        max_workers: int = 2,
        pickle_path: str = None,
        use_tagging: bool = True,
    ):
        # Load from pickle if provided
        if pickle_path and os.path.exists(pickle_path):
            logging.info(f"Loading existing VideoDataset from {pickle_path}")
            self.dataset = VideoDataset.load_from_pickle(pickle_path)
            logging.info(f"Loaded {len(self.dataset)} videos from pickle")
        elif video_dataset is not None:
            self.dataset = video_dataset
        else:
            raise ValueError("Either video_dataset or pickle_path must be provided.")
        
        if len(self.dataset) == 0:
            raise ValueError("Video dataset is empty.")
        
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
        
        self.captioner = VisualCaptioner(device=self.device)
        logging.info(f"MultiModalEncoder initialized with {max_workers} workers.")
        
        # VisionTagger initialization
        self.use_tagging = use_tagging
        self.tagger = None 
        self._tagger_loaded = False
        
        if self.use_tagging:
            self.tagger = VisionTagger(device=self.device)
            logging.info("VisionTagger initialized (model will be loaded on first use).")
        

    def _should_encode_stage(self, dp: VideoDataPoint, stage: str) -> bool:
        """Check if a stage needs to be encoded based on existing embeddings."""
        if stage == "video":
            # Check if video embeddings exist
            if dp.global_embeddings.get("video") is not None:
                return False
            for scene_data in dp.scene_embeddings.values():
                if scene_data.get("video") is not None:
                    return False
            return True
        elif stage == "audio":
            # Check if audio embeddings exist
            if dp.global_embeddings.get("audio") is not None:
                return False
            for scene_data in dp.scene_embeddings.values():
                if scene_data.get("audio") is not None:
                    return False
            return True
        elif stage == "caption":
            # Check if captions exist
            if dp.global_embeddings.get("caption_text"):
                return False
            for scene_data in dp.scene_embeddings.values():
                if scene_data.get("caption_text"):
                    return False
            return True
        elif stage == "text":
            # Check if text embeddings exist
            if dp.global_embeddings.get("text") is not None:
                return False
            for scene_data in dp.scene_embeddings.values():
                if scene_data.get("text") is not None:
                    return False
            return True
        return True

    def _encode_video_stage(self, video_path: str, dp: VideoDataPoint, force: bool = False) -> None:
        """
        Stage 1: Encode all scenes with video encoder.
        Each encoder handles frame/clip extraction internally:
        
        - XCLIP: Extracts all frames, clusters to 8 representative frames, encodes
        - Qwen2-VL: Creates temporary clip of scene, loads ALL frames, extracts 
                    single embedding (first token from vision tower, no pooling)
        """
        if not force and not self._should_encode_stage(dp, "video"):
            logging.info(f"[Stage 1] Video embeddings already exist for {video_path}, skipping")
            return
        
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
        if self.video_encoder.model_name == "xclip":
            logging.info(f"[Video Stage] Using mean pooling for global video embedding (XCLIP)")
        elif self.video_encoder.model_name in ["internvideo2-1b", "internvideo2-6b"]:
            logging.info(f"[Video Stage] Encoding global video embedding for entire video ({dp.video_name})...")
            video_data = self.video_encoder.encode_full_video(video_path)
            if video_data:
                dp.global_embeddings["video"] = video_data["video"]
                logging.info(f"[Video Stage] Global video embedding generated for {dp.video_name}")
        else:
            raise ValueError(f"Unknown model {self.video_encoder.model_name}")

    def _encode_audio_stage(self, video_path: str, dp: VideoDataPoint, force: bool = False) -> None:
        """
        Stage 2: Encode all scenes with audio encoder.
        Requires video_path and scene timing information.
        Gracefully handles videos without audio tracks.
        Stores audio embeddings, transcripts, audio events, and speaker diarization.
        """
        if not force and not self._should_encode_stage(dp, "audio"):
            logging.info(f"[Audio Stage] Audio embeddings already exist for {video_path}, skipping")
            return
        
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

    def _encode_caption_stage(self, video_path: str, dp: VideoDataPoint, force: bool = False) -> None:
        """
        Stage 3: Generate visual captions from keyframes (captioner1) or video scenes (captioner2).
        These captions will be combined with audio information in the text stage.
        
        Captioner1 (BLIP): Uses pre-extracted keyframes from video stage
        Captioner2 (Qwen2-VL): Creates temporary video clip of entire scene, 
                               loads all frames, generates caption, then deletes clip
        """
        if not force and not self._should_encode_stage(dp, "caption"):
            logging.info(f"[Caption Stage] Captions already exist for {dp.video_name}, skipping")
            return
        
        logging.info(f"[Caption Stage] Visual caption generation for {dp.video_name}")
        futures = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
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
        global_caption = self._generate_full_video_caption(dp)
        # Store as temporary data for screenplay generation
        dp._temp_global_caption = global_caption
        logging.info(f"[Caption Stage] Global caption generated for {dp.video_name}")
        
        if hasattr(dp, "_temp_keyframes"):
            del dp._temp_keyframes
            logging.info(f"[Caption Stage] Keyframes cleaned up for {dp.video_name}")

    def _encode_text_stage(self, dp: VideoDataPoint, force: bool = False) -> None:
        """
        Stage 4: Generate screenplay-style summaries and encode with text encoder.
        Combines visual captions, transcripts, audio events, and speaker diarization
        into a coherent narrative description using TextEncoder's LLM.
        """
        if not force and not self._should_encode_stage(dp, "text"):
            logging.info(f"[Text Stage] Text embeddings already exist for {dp.video_name}, skipping")
            return
        
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
        
        # Scene-level text stage complete; global/window-level summaries will be generated later.
        logging.info(f"[Text Stage] Scene-level screenplays and embeddings generated for {dp.video_name}")

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

    def _create_windows(
        self, 
        dp: VideoDataPoint, 
        window_size: int = 3, 
        stride: int = 1,
        modalities: list = ["video"]
    ) -> None:
        """
        Creates sliding windows over scenes and computes window embeddings using mean pooling.

        For video embeddings: uses mean pooling of scene embeddings.
        For text embeddings: summarizes scene screenplays using the LLM (similar to global text embedding).
        
        Args:
            dp: The VideoDataPoint to process
            window_size: Number of scenes per window
            stride: Number of scenes to slide between windows
            modalities: List of modalities to compute window embeddings for
        """
        if not dp.scenes:
            logging.warning(f"No scenes available for window creation in {dp.video_name}")
            return
        
        # Sort scenes by start time
        sorted_scene_ids = sorted(
            dp.scenes.keys(), 
            key=lambda sid: dp.scenes[sid].start_time
        )
        
        num_scenes = len(sorted_scene_ids)
        if num_scenes < window_size:
            logging.info(f"Video {dp.video_name} has fewer scenes ({num_scenes}) than window_size ({window_size}), creating single window")
            window_size = num_scenes
        
        dp.windows = []
        dp.window_embeddings = {}
        
        # Generate window ranges
        window_ranges = []
        
        if num_scenes > 0:
            # Standard sliding windows
            for i in range(0, num_scenes - window_size + 1, stride):
                window_ranges.append((i, i + window_size))
            
            # Ensure the tail is covered if the last window didn't reach the end
            last_end = window_ranges[-1][1] if window_ranges else 0
            if last_end < num_scenes:
                logging.info(f"Adding final window to cover tail scenes for {dp.video_name}")
                start = max(0, num_scenes - window_size)
                window_ranges.append((start, num_scenes))
        
        for window_idx, (start_idx, end_idx) in enumerate(window_ranges):
            # Get the scene IDs in this window
            window_scene_ids = sorted_scene_ids[start_idx:end_idx]
            
            # Get start time from first scene and end time from last scene
            first_scene = dp.scenes[window_scene_ids[0]]
            last_scene = dp.scenes[window_scene_ids[-1]]
            
            window_id = f"window_{window_idx}"
            
            # Create Window object
            window = Window(
                window_id=window_id,
                start_time=first_scene.start_time,
                end_time=last_scene.end_time,
                scene_ids=window_scene_ids
            )
            dp.windows.append(window)
            
            # Compute video embeddings for this window (text will be filled later)
            window_embs = {}
            modality_embeddings = []
            for sid in window_scene_ids:
                if sid in dp.scene_embeddings:
                    emb = dp.scene_embeddings[sid].get("video")
                    if emb is not None:
                        if isinstance(emb, torch.Tensor):
                            modality_embeddings.append(emb)
                        else:
                            modality_embeddings.append(torch.tensor(emb))

            if modality_embeddings:
                stacked = torch.stack(modality_embeddings)
                window_embs["video"] = stacked.mean(dim=0)
            else:
                window_embs["video"] = None

            # Reserve placeholders for window text; will be computed in a separate stage
            window_embs["text"] = None
            window_embs["text_raw"] = ""

            dp.window_embeddings[window_id] = window_embs
            
        logging.info(f"Created {len(dp.windows)} windows for video {dp.video_name} (window_size={window_size}, stride={stride})")

    def _create_window_text_embedding(
        self, 
        dp: VideoDataPoint, 
        window_scene_ids: list, 
        window_id: str
    ) -> tuple[torch.Tensor, str]:
        """
        Create a text embedding for a window by summarizing the scene screenplays.
        Similar approach to genera
        ting global video text embeddings.
        
        Args:
            dp: The VideoDataPoint being processed
            window_scene_ids: List of scene IDs in this window
            window_id: ID of the window (for logging)
            
        Returns:
            Tuple of (text_embedding, raw_summary_text) for the window
        """
        # Collect screenplays from scenes in this window
        scene_screenplays = []
        for sid in window_scene_ids:
            if sid in dp.scene_embeddings:
                text_raw = dp.scene_embeddings[sid].get("text_raw", "")
                if text_raw:
                    scene_screenplays.append((sid, text_raw))
        
        if not scene_screenplays:
            logging.warning(f"No screenplays found for window {window_id}, using mean pooling fallback")
            # Fallback to mean pooling of text embeddings
            text_embeddings = []
            for sid in window_scene_ids:
                if sid in dp.scene_embeddings:
                    emb = dp.scene_embeddings[sid].get("text")
                    if emb is not None:
                        if isinstance(emb, torch.Tensor):
                            text_embeddings.append(emb)
                        else:
                            text_embeddings.append(torch.tensor(emb))
            if text_embeddings:
                return torch.stack(text_embeddings).mean(dim=0), "[Mean pooled - no screenplays available]"
            return None, ""
        
        # Generate summary for this window using the text encoder
        window_summary = self.text_encoder.generate_window_screenplay(scene_screenplays, window_id)
        
        # Encode the summary
        text_emb = self.text_encoder.encode(window_summary)
        
        return text_emb, window_summary

    def _encode_window_text_stage(self, dp: VideoDataPoint) -> None:
        """
        Generate window-level screenplays and text embeddings.
        Uses scene-level `text_raw` as input to the window summarizer.
        """
        if not hasattr(dp, 'windows') or not dp.windows:
            logging.info(f"No windows to process for {getattr(dp,'video_name','<unknown>')}")
            return

        logging.info(f"[Window Text Stage] Generating window-level screenplays for {dp.video_name}")
        for window in tqdm(dp.windows, desc = "Window text embeddings"):
            window_id = window.window_id
            # Build scene screenplays for this window
            scene_screenplays = []
            for sid in window.scene_ids:
                scene_text = dp.scene_embeddings.get(sid, {}).get('text_raw', '')
                scene_screenplays.append((sid, scene_text))

            # Use the text encoder's window summarizer
            try:
                window_summary = self.text_encoder.generate_window_screenplay(scene_screenplays, window_id)
                # Store raw text and encoded embedding
                if window_id not in dp.window_embeddings:
                    dp.window_embeddings[window_id] = {"video": None, "text": None, "text_raw": ""}
                dp.window_embeddings[window_id]["text_raw"] = window_summary
                dp.window_embeddings[window_id]["text"] = self.text_encoder.encode(window_summary)
            except Exception as e:
                logging.error(f"[Window Text] Failed for {dp.video_name} {window_id}: {e}")

        logging.info(f"[Window Text Stage] Completed for {dp.video_name}")

    def _encode_global_text_stage(self, dp: VideoDataPoint) -> None:
        """
        Generate a global screenplay and embedding using windows or scenes based on config.
        """
        use_windows = True
        try:
            use_windows = CONFIG.indexing.text.get("use_windows", True)
        except Exception:
            # fallback default
            use_windows = True

        logging.info(f"[Global Text Stage] Using windows for global summary: {use_windows}")

        if use_windows:
            # Collect (id, text) pairs from windows
            scene_pairs = []
            for wid, wdict in dp.window_embeddings.items():
                txt = wdict.get("text_raw", "")
                if txt:
                    scene_pairs.append((wid, txt))
        else:
            # Collect from scenes
            scene_pairs = [(sid, sd.get("text_raw", "")) for sid, sd in sorted(dp.scene_embeddings.items()) if sd.get("text_raw", "")]

        if not scene_pairs:
            logging.warning(f"[Global Text Stage] No window/scene text available for {dp.video_name}")
            return

        try:
            global_screenplay = self.text_encoder.generate_global_screenplay(scene_pairs)
            dp.global_embeddings["text_raw"] = global_screenplay
            dp.global_embeddings["text"] = self.text_encoder.encode(global_screenplay)
            logging.info(f"[Global Text Stage] Global screenplay and embedding generated for {dp.video_name}")
        except Exception as e:
            logging.error(f"[Global Text Stage] Failed for {dp.video_name}: {e}")

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

    def _generate_full_video_caption(self, dp: VideoDataPoint) -> str:
        """Generate a single caption describing the full video."""
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
            "You are a helpful assistant. Summarize the following scene captions into a concise, coherent paragraph describing the whole video. Be concrete: encapsulate the setting, actions performed and how they impact the surroundings, objects used. Keep it short (1-3 sentences).",
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

    def encode_videos(self, force: bool = False, force_video: bool = None, force_audio: bool = None, 
                      force_caption: bool = None, force_text: bool = None) -> VideoDataset:
        """
        Orchestrates encoding in stages to load/unload models one at a time.
        
        Args:
            force: If True, re-encode all stages even if embeddings already exist.
                  If False, skip stages that already have embeddings.
                  Overridden by modality-specific flags.
            force_video: If True, force re-encode video embeddings. If None, use `force`.
            force_audio: If True, force re-encode audio embeddings. If None, use `force`.
            force_caption: If True, force re-encode captions. If None, use `force`.
            force_text: If True, force re-encode text embeddings. If None, use `force`.
        
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
        # Set modality-specific force flags (use `force` as default if not specified)
        _force_video = force if force_video is None else force_video
        _force_audio = force if force_audio is None else force_audio
        _force_caption = force if force_caption is None else force_caption
        _force_text = force if force_text is None else force_text
        # Set modality-specific force flags (use `force` as default if not specified)
        _force_video = force if force_video is None else force_video
        _force_audio = force if force_audio is None else force_audio
        _force_caption = force if force_caption is None else force_caption
        _force_text = force if force_text is None else force_text
        
        for dp in tqdm(self.dataset.video_datapoints, desc="Encoding Videos", disable=False):
            # if torch.cuda.is_available():
            #     mem = torch.cuda.memory_allocated() / 1024**3
            #     mem_free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - mem
            #     print(f"[GPU] Allocated: {mem:.2f} GB, Free: {mem_free:.2f} GB")

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
                logging.error(f"No scenes detected for {video_path}. Skipping.")
                continue

            # Stage 1: Video Encoding 
            self.video_encoder.load_models()
            self._encode_video_stage(video_path, dp)
            
            if not dp.scene_embeddings:
                logging.error(f"No scenes were successfully encoded for {video_path} (video stage).")
                self.unload_model("video")
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
                continue
            self.unload_model("video")
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            

            # Stage 2: Caption generation
            self.captioner.load_models()
            self._encode_caption_stage(video_path, dp, force=_force_caption)

            self.unload_model("caption")
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

            # Stage 3: Audio Encoding
            # Fast-check if the video has an audio track before loading heavy models
            """
            has_audio = self.audio_encoder.has_audio_track(video_path)

            if not has_audio:
                logging.info(f"Video {video_path} appears to have no audio - skipping audio stage")
                dp.has_audio = False
            else:
                self.audio_encoder.load_models()
                self._encode_audio_stage(video_path, dp, force=_force_audio)
                self.unload_model("audio")
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            """
            
            # Stage 4: Text Encoding
            self.text_encoder.load_models()
            self._encode_text_stage(dp, force=_force_text)

            # Tagging: run VisionTagger using the generated screenplay/text
            if self.use_tagging:
                try:
                    if not self._tagger_loaded:
                        logging.info(f"[Tagger] Loading VisionTagger model for {dp.video_name}...")
                        self.tagger.load_model()
                        self._tagger_loaded = True
                    
                    # Tag both global video and individual scenes
                    logging.info(f"[Tagger] Tagging video {dp.video_name} (global + scenes)...")
                    self.tagger.tag_datapoint(dp, tag_scenes=True)
                    logging.info(f"[Tagger] Successfully tagged {dp.video_name}")
                    
                    # Print tags immediately for this video
                    self.tagger.pretty_print_datapoint(dp, show_scenes=True, color=True)
                    
                except Exception as e:
                    logging.error(f"[Tagger] Tagging failed for {dp.video_name}: {e}")
                    logging.error(traceback.format_exc())

            # Unload text encoder after text stage
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
            # Create sliding windows over scenes (video embeddings only);
            # text for windows will be generated in a dedicated window text stage.
            self._create_windows(
                dp,
                window_size=CONFIG.indexing.get("window_size", 3),
                stride=CONFIG.indexing.get("window_stride", 1),
            )

            # Window-level text generation and embeddings
            self._encode_window_text_stage(dp)

            # Global-level text generation (uses windows or scenes per config)
            self._encode_global_text_stage(dp)

            # Unload text encoder after all text-related stages are complete
            self.unload_model("text")
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

        # Unload tagger at the very end after all videos are processed
        if self.use_tagging and self._tagger_loaded:
            # Pretty print the tagging results before unloading
            logging.info("[Tagger] Printing tagging results...")
            self.tagger.pretty_print_dataset(
                self.dataset,
                max_text_len=200,
                show_scenes=True,
                color=True
            )
            
            logging.info("[Tagger] Unloading VisionTagger model after processing all videos")
            self.unload_model("tagger")

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
        if modality == "tagger" and hasattr(self, "tagger") and self.tagger is not None:
            if hasattr(self.tagger, 'unload_model'):
                self.tagger.unload_model()
            del self.tagger
            self.tagger = None
            self._tagger_loaded = False
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()