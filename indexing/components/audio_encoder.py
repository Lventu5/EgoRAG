"""
Audio Encoder with Advanced Features

Required dependencies:
    pip install faster-whisper  # For faster Whisper inference with word timestamps
    pip install pyannote.audio  # For speaker diarization (optional)

Note: The encoder gracefully falls back if optional dependencies are not installed.
"""

import torch
import numpy as np
import logging
import librosa
from moviepy.editor import VideoFileClip
from transformers import (
    AutoModel, 
    AutoProcessor, 
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    AutoFeatureExtractor,
)
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logging.warning("faster-whisper not available, falling back to transformers Whisper")

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logging.warning("pyannote.audio not available, speaker diarization disabled")

from .base_encoder import BaseEncoder
from configuration.config import CONFIG
import subprocess
import shutil

class AudioEncoder(BaseEncoder):
    """
    Encodes raw audio from a video segment IN-MEMORY into multiple representations:
    1.  Transcript: Transcribed speech with word-level timestamps (Whisper Large v3 / faster-whisper).
    2.  Audio Embedding: Acoustic event embeddings (CLAP).
    3.  Audio Events: Detected audio events like sounds, music, etc. (BEATs/AudioMAE).
    4.  Speaker Diarization: Who spoke when (PyAnnote - optional).
    """
    def __init__(
        self, 
        device: str = "cuda",
    ):
        super().__init__(device)
        self.audio_sr = CONFIG.indexing.audio.audio_sample_rate
        self.asr_sr = CONFIG.indexing.audio.asr_sample_rate
        self.use_faster_whisper = CONFIG.indexing.audio.get("use_faster_whisper", True) and FASTER_WHISPER_AVAILABLE
        self.use_audio_events = CONFIG.indexing.audio.get("use_audio_events", True)
        self.use_diarization = CONFIG.indexing.audio.get("use_diarization", False) and PYANNOTE_AVAILABLE
        
        # Models to be loaded
        self.audio_embed_model: AutoModel = None
        self.audio_embed_processor: AutoProcessor = None
        self.asr_model: WhisperForConditionalGeneration = None
        self.asr_processor: WhisperProcessor = None
        self.faster_whisper_model = None
        self.audio_event_model: AutoModel = None
        self.audio_event_processor: AutoFeatureExtractor = None
        self.diarization_pipeline: Pipeline = None

    def load_models(self):
        logging.info(f"[{self.__class__.__name__}] Loading models...")
        
        # 1. Load Whisper (ASR) - Large v3 or faster-whisper
        asr_model_id = CONFIG.indexing.audio.asr_model_id
        
        if self.use_faster_whisper:
            logging.info(f"Loading faster-whisper: {asr_model_id}")
            # faster-whisper uses model size names or paths
            model_size = asr_model_id.split("/")[-1].replace("whisper-", "") if "/" in asr_model_id else asr_model_id.replace("whisper-", "")
            self.faster_whisper_model = WhisperModel(model_size, device=self.device, compute_type="float16" if self.device == "cuda" else "float32")
        else:
            logging.info(f"Loading transformers Whisper: {asr_model_id}")
            self.asr_model = WhisperForConditionalGeneration.from_pretrained(asr_model_id).to(self.device)
            self.asr_processor = WhisperProcessor.from_pretrained(asr_model_id)

            if self.device == "cuda":
                self.asr_model = self.asr_model.half()

            self.asr_model.config.forced_decoder_ids = self.asr_processor.get_decoder_prompt_ids(
                language="en", task="transcribe"
            )
        
        # 2. Load CLAP (Audio Embedding)
        audio_model_id = CONFIG.indexing.audio.audio_model_id
        self.audio_embed_model = AutoModel.from_pretrained(audio_model_id).to(self.device)
        self.audio_embed_processor = AutoProcessor.from_pretrained(audio_model_id)
        
        # 3. Load Audio Event Detection (BEATs or AudioMAE)
        if self.use_audio_events:
            audio_event_model_id = CONFIG.indexing.audio.get("audio_event_model_id", "MIT/ast-finetuned-audioset-10-10-0.4593")
            logging.info(f"Loading audio event model: {audio_event_model_id}")
            self.audio_event_model = AutoModel.from_pretrained(audio_event_model_id).to(self.device)
            self.audio_event_processor = AutoFeatureExtractor.from_pretrained(audio_event_model_id)
        
        # 4. Load PyAnnote for speaker diarization (optional)
        if self.use_diarization:
            diarization_model_id = CONFIG.indexing.audio.get("diarization_model_id", "pyannote/speaker-diarization-3.1")
            logging.info(f"Loading diarization pipeline: {diarization_model_id}")
            self.diarization_pipeline = Pipeline.from_pretrained(diarization_model_id).to(torch.device(self.device))
        
        logging.info(f"[{self.__class__.__name__}] Models loaded.")

    def has_audio_track(self, video_path: str) -> bool:
        """
        Fast check whether a video file has an audio track.
        Prefer using `ffprobe` (fast, no decoding). If `ffprobe` is not
        available, fall back to using MoviePy's `VideoFileClip`.
        """
        # Try ffprobe first
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            try:
                cmd = [
                    ffprobe,
                    "-v", "error",
                    "-select_streams", "a",
                    "-show_entries", "stream=index",
                    "-of", "csv=p=0",
                    video_path,
                ]
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
                return bool(out and out.strip())
            except Exception:
                pass
        try:
            with VideoFileClip(video_path) as vid:
                return vid.audio is not None
        except Exception:
            return False

    def _extract_audio_array(self, video_path: str, start_t: float, end_t: float) -> np.ndarray | None:
        """
        Extracts an audio segment as a numpy array IN-MEMORY.
        Uses the iter_frames workaround for moviepy bug.
        Returns None if video has no audio track.
        """
        try:
            with VideoFileClip(video_path) as vid:
                # Check if video has audio at all
                if vid.audio is None:
                    logging.info(f"Video {video_path} has no audio track (silent video)")
                    return None
                
                if start_t >= vid.duration or start_t >= end_t:
                    logging.warning(f"Invalid time range {start_t}-{end_t} for video {video_path}")
                    return None
                
                clip = vid.subclip(start_t, end_t)
                if clip.audio is None:
                    logging.info(f"No audio found in subclip {start_t}-{end_t} for {video_path}")
                    return None

                # Moviepy workaround: iter_frames() instead of to_soundarray()
                audio_frames = list(clip.audio.iter_frames(fps=self.audio_sr))
                if not audio_frames:
                    logging.warning(f"Audio frames list is empty for {video_path}")
                    return None
                    
                audio_array = np.array(audio_frames)
                clip.close()
                del clip

            # Convert to mono if stereo
            if audio_array.ndim == 2:
                audio_array = audio_array.mean(axis=1)
                
            return audio_array.astype(np.float32)

        except Exception as e:
            logging.error(f"Failed to extract audio array for {video_path} ({start_t}-{end_t}): {e}")
            return None

    def _embed_audio(self, audio_array_48k: np.ndarray) -> np.ndarray | None:
        """Embeds audio data using CLAP."""
        try:
            audio_trimmed_48k, _ = librosa.effects.trim(audio_array_48k, top_db=20)
            if audio_trimmed_48k.size == 0:
                logging.warning("Audio clip silent or too short for CLAP.")
                return None

            inputs_audio = self.audio_embed_processor(
                audio=audio_trimmed_48k, 
                sampling_rate=self.audio_sr, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.inference_mode():
                audio_emb = self.audio_embed_model.get_audio_features(**inputs_audio)
            
            return audio_emb.cpu().numpy().squeeze()
        
        except Exception as e:
            logging.error(f"Error during audio embedding: {e}")
            return None

    def _transcribe_audio(self, audio_array_48k: np.ndarray) -> dict:
        """
        Transcribes audio with word-level timestamps.
        Returns dict with 'text' and 'words' (list of word segments with timestamps).
        """
        try:
            if self.audio_sr != self.asr_sr:
                audio_array_16k = librosa.resample(
                    audio_array_48k, 
                    orig_sr=self.audio_sr, 
                    target_sr=self.asr_sr
                )
            else:
                audio_array_16k = audio_array_48k

            if audio_array_16k.size == 0:
                logging.warning("Audio too short for ASR after resampling.")
                return {"text": "", "words": []}

            if self.use_faster_whisper:
                # faster-whisper provides word-level timestamps
                segments, info = self.faster_whisper_model.transcribe(
                    audio_array_16k,
                    language="en",
                    task="transcribe",
                    word_timestamps=True,
                    vad_filter=True,
                )
                
                transcript_text = ""
                words = []
                for segment in segments:
                    transcript_text += segment.text + " "
                    if hasattr(segment, 'words'):
                        for word in segment.words:
                            words.append({
                                "word": word.word,
                                "start": word.start,
                                "end": word.end,
                                "probability": word.probability
                            })
                
                return {
                    "text": transcript_text.strip(),
                    "words": words
                }
            else:
                # transformers Whisper - basic transcription (no word timestamps in generate)
                inputs = self.asr_processor(
                    audio_array_16k, 
                    sampling_rate=self.asr_sr, 
                    return_tensors="pt"
                )
                dtype = torch.float16 if (self.device == "cuda") else torch.float32
                input_features = inputs.input_features.to(self.device, dtype=dtype)

                with torch.inference_mode():
                    language = CONFIG.indexing.audio.audio_language
                    predicted_ids = self.asr_model.generate(
                        input_features, 
                        language=language, 
                        task="transcribe",
                        return_timestamps=True  # This enables timestamp tokens
                    )

                transcript_list = self.asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)
                transcript = transcript_list[0].strip() if transcript_list else ""
                
                del inputs, input_features, predicted_ids
                torch.cuda.empty_cache()
                
                return {
                    "text": transcript,
                    "words": []  # Word-level timestamps not easily available in transformers
                }

        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            return {"text": "", "words": []}

    def _detect_audio_events(self, audio_array_48k: np.ndarray, top_k: int = 5) -> list:
        """
        Detects audio events (sounds, music, speech, etc.) from audio.
        Returns top-k event labels with confidence scores.
        """
        if not self.use_audio_events or self.audio_event_model is None:
            return []
        
        try:
            # Resample to model's expected sample rate (usually 16kHz for AST)
            target_sr = self.audio_event_processor.sampling_rate if hasattr(self.audio_event_processor, 'sampling_rate') else 16000
            if self.audio_sr != target_sr:
                audio_resampled = librosa.resample(
                    audio_array_48k,
                    orig_sr=self.audio_sr,
                    target_sr=target_sr
                )
            else:
                audio_resampled = audio_array_48k
            
            # Process audio
            inputs = self.audio_event_processor(
                audio_resampled,
                sampling_rate=target_sr,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.inference_mode():
                outputs = self.audio_event_model(**inputs)
                # Get logits - typically from last_hidden_state or logits
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs.last_hidden_state.mean(dim=1)
                
                # Get top-k predictions
                probs = torch.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probs[0], k=top_k)
                
                events = []
                if hasattr(self.audio_event_model.config, 'id2label'):
                    for prob, idx in zip(top_probs, top_indices):
                        label = self.audio_event_model.config.id2label.get(idx.item(), f"event_{idx.item()}")
                        events.append({
                            "label": label,
                            "confidence": float(prob.item())
                        })
                else:
                    # Fallback if no label mapping
                    for prob, idx in zip(top_probs, top_indices):
                        events.append({
                            "label": f"event_{idx.item()}",
                            "confidence": float(prob.item())
                        })
            
            return events
            
        except Exception as e:
            logging.error(f"Error during audio event detection: {e}")
            return []

    def _diarize_speakers(self, audio_array_48k: np.ndarray, duration: float) -> list:
        """
        Performs speaker diarization to identify who spoke when.
        Returns list of speaker segments with timestamps.
        """
        if not self.use_diarization or self.diarization_pipeline is None:
            return []
        
        try:
            # PyAnnote expects dict format with waveform and sample_rate
            audio_dict = {
                "waveform": torch.from_numpy(audio_array_48k).unsqueeze(0).float(),
                "sample_rate": self.audio_sr
            }
            
            # Run diarization
            diarization = self.diarization_pipeline(audio_dict)
            
            # Convert to list of segments
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end
                })
            
            return speaker_segments
            
        except Exception as e:
            logging.error(f"Error during speaker diarization: {e}")
            return []

    def encode(self, video_path: str, start_time: float, end_time: float) -> dict:
        """
        Public method to encode an audio segment directly from a video file.
        
        Args:
            video_path: Path to the full video file.
            start_time: Start time of the scene in seconds.
            end_time: End time of the scene in seconds.
            
        Returns:
            A dictionary containing:
            - 'audio_embedding': CLAP embedding for acoustic features
            - 'transcript': Transcribed text
            - 'transcript_words': Word-level timestamps (if using faster-whisper)
            - 'audio_events': Detected audio events with confidence scores
            - 'speaker_segments': Speaker diarization results
            - 'has_audio': Boolean flag
        """
        audio_array_48k = self._extract_audio_array(video_path, start_time, end_time)
        
        if audio_array_48k is None:
            # Video has no audio track - return None values with flag
            return {
                "audio_embedding": None, 
                "transcript": "",
                "transcript_words": [],
                "audio_events": [],
                "speaker_segments": [],
                "has_audio": False
            }

        # 1. Get CLAP embedding
        audio_embedding = self._embed_audio(audio_array_48k)
        if audio_embedding is None:
            # Audio exists but CLAP failed (e.g., silent segment)
            # Create a zero vector (512-d for CLAP) if embedding fails
            zero_emb = np.zeros(512, dtype=np.float32)
            audio_embedding = torch.tensor(zero_emb, dtype=torch.float32)
        else:
            audio_embedding = torch.tensor(audio_embedding, dtype=torch.float32)

        # 2. Get Whisper transcript with word-level timestamps
        transcript_result = self._transcribe_audio(audio_array_48k)
        
        # 3. Detect audio events (sounds, music, etc.)
        audio_events = self._detect_audio_events(audio_array_48k, top_k=5)
        
        # 4. Speaker diarization (who spoke when)
        duration = end_time - start_time
        speaker_segments = self._diarize_speakers(audio_array_48k, duration)
        
        return {
            "audio_embedding": audio_embedding,
            "transcript": transcript_result["text"],
            "transcript_words": transcript_result["words"],
            "audio_events": audio_events,
            "speaker_segments": speaker_segments,
            "has_audio": True
        }