import torch
import numpy as np
import logging
from transformers import BlipProcessor, BlipForConditionalGeneration
from indexing.utils.clustering import cluster_frames

from .base_encoder import BaseEncoder
from configuration.config import CONFIG

class VisualCaptioner(BaseEncoder):
    """
    Generates textual captions from video frames using BLIP.
    It clusters frames to find key visual moments and captions them.
    """
    def __init__(self, device: str = "cuda", max_k_clusters: int = 5):
        super().__init__(device)
        self.max_k_clusters = max_k_clusters
        
        # Models to be loaded
        self.processor: BlipProcessor = None
        self.model: BlipForConditionalGeneration = None

    def load_models(self):
        logging.info(f"[{self.__class__.__name__}] Loading BLIP models...")
        model_id = CONFIG.indexing.caption.caption_model_id
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device)
        logging.info(f"[{self.__class__.__name__}] Models loaded.")

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
        if len(keyframes) == 0:
            logging.warning("No frames provided to VisualCaptioner.")
            return ""
            
        captions = []

        with torch.inference_mode():
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