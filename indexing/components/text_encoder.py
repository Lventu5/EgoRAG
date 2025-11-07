import torch
import logging
from sentence_transformers import SentenceTransformer
from .base_encoder import BaseEncoder
from indexing.components.model_registry import get_registry, GPUMemoryGuard
from indexing.utils.logging import get_logger

logger = get_logger(__name__)

class TextEncoder(BaseEncoder):
    """
    Encodes textual descriptions (transcripts, captions) into
    semantic embeddings using SentenceTransformers (SBERT).
    """
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.model_name = "all-MiniLM-L6-v2"
        self.registry = get_registry()
        
        # Register model loader
        self.registry.register("sbert_text_model", self._load_sbert_model)
        
        # Model to be loaded
        self.sbert_model: SentenceTransformer = None
    
    def _load_sbert_model(self):
        """Loader for SentenceTransformer model."""
        logger.info(f"[{self.__class__.__name__}] Loading {self.model_name}...")
        return SentenceTransformer(self.model_name, device=self.device)

    def load_models(self):
        logger.info(f"[{self.__class__.__name__}] Loading models...")
        self.sbert_model = self.registry.get("sbert_text_model")
        logger.info(f"[{self.__class__.__name__}] Model loaded.")

    def encode(self, text: str) -> torch.Tensor:
        """
        Public method to encode a string of text.
        
        Args:
            text: The input string.
            
        Returns:
            A torch.Tensor containing the embedding.
        """
        # Ensure models are loaded (lazy loading via ModelRegistry)
        if self.sbert_model is None:
            self.load_models()
            
        with GPUMemoryGuard():
            if not text or not isinstance(text, str):
                logger.warning("No valid text provided to TextEncoder.")
                # Return a zero vector of the correct dimension
                return torch.zeros(self.sbert_model.get_sentence_embedding_dimension(), dtype=torch.float32)
                
            with torch.inference_mode():
                embedding = self.sbert_model.encode(
                    text, 
                    convert_to_tensor=True, 
                    device=self.device
                )
            return embedding.cpu()