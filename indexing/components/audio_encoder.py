import torch
import numpy as np
import logging
import librosa
from moviepy.editor import VideoFileClip
from transformers import (
    AutoModel, 
    AutoProcessor, 
    WhisperProcessor, 
    WhisperForConditionalGeneration
)

from .base_encoder import BaseEncoder

class AudioEncoder(BaseEncoder):
    """
    Encodes raw audio from a video segment IN-MEMORY into two representations:
    1.  Transcript: Transcribed speech (Whisper).
    2.  Audio Embedding: Acoustic event embeddings (CLAP).
    """
    def __init__(
        self, 
        device: str = "cuda",
        audio_sr: int = 48000, # Required by CLAP
        asr_sr: int = 16000     # Required by Whisper
    ):
        super().__init__(device)
        self.audio_sr = audio_sr
        self.asr_sr = asr_sr
        
        # Models to be loaded
        self.audio_embed_model: AutoModel = None
        self.audio_embed_processor: AutoProcessor = None
        self.asr_model: WhisperForConditionalGeneration = None
        self.asr_processor: WhisperProcessor = None

    def load_models(self):
        logging.info(f"[{self.__class__.__name__}] Loading models...")
        
        # 1. Load Whisper (ASR)
        asr_model_id = "openai/whisper-base"
        self.asr_model = WhisperForConditionalGeneration.from_pretrained(asr_model_id).to(self.device)
        self.asr_processor = WhisperProcessor.from_pretrained(asr_model_id)
        
        # 2. Load CLAP (Audio Embedding)
        audio_model_id = "laion/clap-htsat-unfused"
        self.audio_embed_model = AutoModel.from_pretrained(audio_model_id).to(self.device)
        self.audio_embed_processor = AutoProcessor.from_pretrained(audio_model_id)
        
        logging.info(f"[{self.__class__.__name__}] Models loaded.")

    def _extract_audio_array(self, video_path: str, start_t: float, end_t: float) -> np.ndarray | None:
        """
        Extracts an audio segment as a numpy array IN-MEMORY.
        Uses the iter_frames workaround for moviepy bug.
        """
        try:
            with VideoFileClip(video_path) as vid:
                if start_t >= vid.duration or start_t >= end_t:
                    logging.warning(f"Invalid time range {start_t}-{end_t} for video {video_path}")
                    return None
                
                clip = vid.subclip(start_t, end_t)
                if clip.audio is None:
                    logging.warning(f"No audio found in subclip {start_t}-{end_t} for {video_path}")
                    return None

                # Moviepy workaround: iter_frames() instead of to_soundarray()
                audio_frames = list(clip.audio.iter_frames(fps=self.audio_sr))
                if not audio_frames:
                    logging.warning(f"Audio frames list is empty for {video_path}")
                    return None
                    
                audio_array = np.array(audio_frames)

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
            # Trim silence for a more representative CLAP embedding
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
            logging.error(f"Error during CLAP audio embedding: {e}")
            return None

    def _transcribe_audio(self, audio_array_48k: np.ndarray) -> str:
        """Transcribes audio using Whisper."""
        try:
            # 1. Resample for Whisper
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
                return ""

            # 2. Transcribe IN-MEMORY
            inputs = self.asr_processor(
                audio_array_16k, 
                sampling_rate=self.asr_sr, 
                return_tensors="pt"
            )
            input_features = inputs.input_features.to(self.device) 

            with torch.inference_mode():
                predicted_ids = self.asr_model.generate(input_features)

            transcript_list = self.asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)
            transcript = transcript_list[0].strip() if transcript_list else ""
            return transcript

        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            return ""

    def encode(self, video_path: str, start_time: float, end_time: float) -> dict:
        """
        Public method to encode an audio segment directly from a video file.
        
        Args:
            video_path: Path to the full video file.
            start_time: Start time of the scene in seconds.
            end_time: End time of the scene in seconds.
            
        Returns:
            A dictionary containing 'audio_embedding' and 'transcript'.
        """
        audio_array_48k = self._extract_audio_array(video_path, start_time, end_time)
        
        if audio_array_48k is None:
            return {"audio_embedding": None, "transcript": ""}

        # 1. Get CLAP embedding
        audio_embedding = self._embed_audio(audio_array_48k)
        if audio_embedding is None:
            # Create a zero vector if CLAP fails
            zero_emb = np.zeros(self.audio_embed_model.config.hidden_size, dtype=np.float32)
            audio_embedding = torch.tensor(zero_emb, dtype=torch.float32)
        else:
            audio_embedding = torch.tensor(audio_embedding, dtype=torch.float32)

        # 2. Get Whisper transcript
        # We use the original (non-trimmed) audio for transcription
        transcript = self._transcribe_audio(audio_array_48k)
        
        return {
            "audio_embedding": audio_embedding,
            "transcript": transcript
        }