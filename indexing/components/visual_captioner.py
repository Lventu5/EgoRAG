import torch
import numpy as np
import logging
from transformers import BlipProcessor, BlipForConditionalGeneration
from indexing.utils.clustering import cluster_frames

from .base_encoder import BaseEncoder
from indexing.components.model_registry import get_registry, GPUMemoryGuard
from indexing.utils.logging import get_logger

logger = get_logger(__name__)

class VisualCaptioner(BaseEncoder):
    """
    Generates textual captions from video frames using BLIP.
    It clusters frames to find key visual moments and captions them.
    """
    def __init__(self, device: str = "cuda", max_k_clusters: int = 5):
        super().__init__(device)
        self.max_k_clusters = max_k_clusters
        self.registry = get_registry()
        
        # Register model loaders
        self.registry.register("blip_processor", self._load_blip_processor)
        self.registry.register("blip_model", self._load_blip_model)
        
        # Models to be loaded
        self.processor: BlipProcessor = None
        self.model: BlipForConditionalGeneration = None
    
    def _load_blip_processor(self):
        """Loader for BLIP processor."""
        model_id = "Salesforce/blip-image-captioning-base"
        return BlipProcessor.from_pretrained(model_id)
    
    def _load_blip_model(self):
        """Loader for BLIP model."""
        logger.info(f"[{self.__class__.__name__}] Loading BLIP model...")
        model_id = "Salesforce/blip-image-captioning-base"
        return BlipForConditionalGeneration.from_pretrained(model_id).to(self.device)

    def load_models(self):
        logger.info(f"[{self.__class__.__name__}] Loading BLIP models...")
        self.processor = self.registry.get("blip_processor")
        self.model = self.registry.get("blip_model")
        logger.info(f"[{self.__class__.__name__}] Models loaded.")

    # def _cluster_frames_for_captioning(self, frames: np.ndarray) -> np.ndarray:
    #     """
    #     Selects representative frames for captioning.
    #     This uses a simplified clustering (or just subsampling)
    #     to pick keyframes.
    #     """
    #     num_frames = len(frames)
    #     if num_frames == 0:
    #         return np.array([])
        
    #     # Select k indices evenly spaced
    #     k = min(self.max_k_clusters, num_frames)
    #     indices = np.linspace(0, num_frames - 1, k, dtype=int)
        
    #     return frames[indices]

    def encode(self, keyframes: np.ndarray, prompt: str = "a video of") -> str:
        """
        Public method to generate a caption for a set of frames.
        
        Args:
            frames: A np.ndarray of shape (num_frames, H, W, C)
            prompt: An optional text prompt for the captioner.
            
        Returns:
            A string containing the concatenated captions.
        """
        with GPUMemoryGuard():
            if len(keyframes) == 0:
                logger.warning("No frames provided to VisualCaptioner.")
                return ""
                
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