import torch
import numpy as np
import logging
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from indexing.utils.clustering import cluster_frames

from .base_encoder import BaseEncoder
from indexing.components.model_registry import get_registry, GPUMemoryGuard
from indexing.utils.logging import get_logger

logger = get_logger(__name__)

class VisualCaptioner(BaseEncoder):
    """
    Generates textual captions from video frames.
    
    Supports two modes:
    1. BLIP (legacy): Captions individual keyframes and combines them
    2. BLIP-2 Video (default): Uses video-aware captioning that understands temporal context
    """
    def __init__(self, device: str = "cuda", max_k_clusters: int = 5, use_video_model: bool = True):
        super().__init__(device)
        self.max_k_clusters = max_k_clusters
        self.use_video_model = use_video_model
        self.registry = get_registry()
        
        if use_video_model:
            # Register BLIP-2 models (better for video understanding)
            self.registry.register("blip2_processor", self._load_blip2_processor)
            self.registry.register("blip2_model", self._load_blip2_model)
            self.processor = None
            self.model = None
        else:
            # Register original BLIP models (backward compatibility)
            self.registry.register("blip_processor", self._load_blip_processor)
            self.registry.register("blip_model", self._load_blip_model)
            self.processor: BlipProcessor = None
            self.model: BlipForConditionalGeneration = None
    
    def _load_blip_processor(self):
        """Loader for BLIP processor (legacy)."""
        model_id = "Salesforce/blip-image-captioning-base"
        return BlipProcessor.from_pretrained(model_id)
    
    def _load_blip_model(self):
        """Loader for BLIP model (legacy)."""
        logger.info(f"[{self.__class__.__name__}] Loading BLIP model (legacy mode)...")
        model_id = "Salesforce/blip-image-captioning-base"
        return BlipForConditionalGeneration.from_pretrained(model_id).to(self.device)
    
    def _load_blip2_processor(self):
        """Loader for BLIP-2 processor."""
        model_id = "Salesforce/blip2-opt-2.7b"
        logger.info(f"[{self.__class__.__name__}] Loading BLIP-2 processor...")
        return Blip2Processor.from_pretrained(model_id)
    
    def _load_blip2_model(self):
        """Loader for BLIP-2 model."""
        logger.info(f"[{self.__class__.__name__}] Loading BLIP-2 model for video captioning...")
        model_id = "Salesforce/blip2-opt-2.7b"
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        return model.eval()

    def load_models(self):
        logger.info(f"[{self.__class__.__name__}] Loading captioning models...")
        if self.use_video_model:
            self.processor = self.registry.get("blip2_processor")
            self.model = self.registry.get("blip2_model")
            logger.info(f"[{self.__class__.__name__}] BLIP-2 models loaded (video-aware mode).")
        else:
            self.processor = self.registry.get("blip_processor")
            self.model = self.registry.get("blip_model")
            logger.info(f"[{self.__class__.__name__}] BLIP models loaded (legacy mode).")

    def encode(self, keyframes: np.ndarray, prompt: str = "a video of") -> str:
        """
        Public method to generate a caption for a set of frames.
        
        Args:
            keyframes: A np.ndarray of shape (num_frames, H, W, C) - video frames
            prompt: An optional text prompt for the captioner.
            
        Returns:
            A string containing the video caption.
        """
        # Ensure models are loaded (lazy loading via ModelRegistry)
        if self.model is None or self.processor is None:
            self.load_models()
            
        with GPUMemoryGuard():
            if len(keyframes) == 0:
                logger.warning("No frames provided to VisualCaptioner.")
                return ""
            
            if self.use_video_model:
                return self._encode_with_video_model(keyframes, prompt)
            else:
                return self._encode_with_legacy_blip(keyframes, prompt)
    
    def _encode_with_video_model(self, frames: np.ndarray, prompt: str) -> str:
        """
        Use BLIP-2 to caption the video sequence with temporal awareness.
        Instead of captioning individual frames, this processes them as a sequence.
        """
        with torch.inference_mode(), torch.autocast(device_type=self.device, enabled=(self.device == "cuda")):
            # Subsample frames if too many (BLIP-2 can handle ~8-16 frames efficiently)
            max_frames = 8
            if len(frames) > max_frames:
                indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
                frames = frames[indices]
            
            # Convert frames to list of PIL images for processing
            from PIL import Image
            pil_frames = [Image.fromarray(frame) for frame in frames]
            
            # Process all frames together - BLIP-2 can handle multiple images
            # We'll caption them in small batches to capture temporal progression
            captions = []
            batch_size = 4
            
            for i in range(0, len(pil_frames), batch_size):
                batch_frames = pil_frames[i:i+batch_size]
                
                # Create a temporal prompt
                if len(batch_frames) > 1:
                    temporal_prompt = f"Question: Describe what is happening in this video sequence. Answer:"
                else:
                    temporal_prompt = f"Question: Describe this scene. Answer:"
                
                # Process batch
                inputs = self.processor(
                    images=batch_frames,
                    text=temporal_prompt,
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate caption
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=3,
                    temperature=0.7,
                    do_sample=False
                )
                
                caption = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
                
                # Clean up the caption
                caption = caption.replace(temporal_prompt, "").strip()
                if caption and caption not in captions:
                    captions.append(caption)
            
            # Combine captions with proper flow
            if len(captions) == 1:
                return captions[0]
            elif len(captions) > 1:
                # Join with progression indicators
                return " Then, ".join(captions)
            else:
                return ""
    
    def _encode_with_legacy_blip(self, keyframes: np.ndarray, prompt: str) -> str:
        """
        Legacy BLIP method: Captions individual keyframes and combines them.
        Kept for backward compatibility.
        """
        captions = []

        with torch.inference_mode(), torch.autocast(device_type=self.device, enabled=(self.device == "cuda")):
            for frame in keyframes:
                inputs = self.processor(images=frame, text=prompt, return_tensors="pt").to(self.device)
                outputs = self.model.generate(**inputs, max_new_tokens=50)
                caption = self.processor.decode(outputs[0], skip_special_tokens=True)
                
                # Post-process to remove the prompt
                caption = caption.replace(prompt, "").strip()
                if caption:
                    captions.append(caption)
        
        # Combine unique captions into a single description
        unique_captions = sorted(list(set(captions)), key=captions.index)
        return ". ".join(unique_captions)