import os
import torch
import logging
from sentence_transformers import SentenceTransformer
from .base_encoder import BaseEncoder
from configuration.config import CONFIG

class TextEncoder(BaseEncoder):
    """
    Encodes textual descriptions (transcripts, captions) into
    semantic embeddings using SentenceTransformers (SBERT).
    """
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.sbert_model: SentenceTransformer = None
        self.model_name = CONFIG.indexing.text.text_model_id

    def load_models(self):
        logging.info(f"[{self.__class__.__name__}] Loading {self.model_name}...")
        # SentenceTransformer automatically uses SENTENCE_TRANSFORMERS_HOME or HF_HOME
        self.sbert_model = SentenceTransformer(self.model_name, device=self.device)
        logging.info(f"[{self.__class__.__name__}] Model loaded.")

    def encode(self, text: str) -> torch.Tensor:
        """
        Public method to encode a string of text.
        
        Args:
            text: The input string.
            
        Returns:
            A torch.Tensor containing the embedding.
        """
        if not text or not isinstance(text, str):
            logging.warning("No valid text provided to TextEncoder.")
            # Return a zero vector of the correct dimension
            return torch.zeros(self.sbert_model.get_sentence_embedding_dimension(), dtype=torch.float32)
            
        with torch.inference_mode():
            embedding = self.sbert_model.encode(
                text, 
                convert_to_tensor=True, 
                device=self.device
            )
        return embedding.cpu()