import numpy as np
import torch
from transformers import (
    XCLIPModel,
    XCLIPProcessor,
    CLIPVisionModel,
    CLIPImageProcessor
)
import logging
from transformers import logging as hf_logging

from .base_encoder import BaseEncoder
from indexing.utils.logging_formatter import LevelAwareFormatter
from indexing.utils.clustering import choose_k, cluster_frames

class VideoEncoder(BaseEncoder):
    """
    Encodes video frames into two hierarchical representations:
    1.  Temporal "Bag-of-Actions": Spatiotemporal embeddings (XCLIP) 
        of contiguous segments.
    2.  Static "Bag-of-Keyframes": Image embeddings (CLIP) of the
        most representative cluster centroids.
    """
    def __init__(
        self,
        device: str = "cuda",
        max_frames_per_scene: int = 96,
        max_temporal_segments: int = 8,
    ):
        super().__init__(device)
        self.max_frames_per_scene = max_frames_per_scene
        self.max_temporal_segments = max_temporal_segments
        
        # Models to be loaded
        self.video_model: XCLIPModel = None
        self.video_processor: XCLIPProcessor = None
        self.image_model: CLIPVisionModel = None
        self.image_processor: CLIPImageProcessor = None

    def load_models(self):
        logging.info(f"[{self.__class__.__name__}] Loading models...")
        # Load XCLIP (Temporal)
        self.video_model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch16").to(self.device)
        self.video_processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch16")
        
        # Load CLIP-Vision (Static)
        self.image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        logging.info(f"[{self.__class__.__name__}] Models loaded.")

    def _embed_frames_clip(self, frames: np.ndarray) -> np.ndarray:
        inputs = self.image_processor(images=list(frames), return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.image_model(**inputs)
            img_embs = outputs.pooler_output
            
        return img_embs.cpu().numpy()
    
    def _embed_temporal_segments(self, temporal_chunks: list[np.ndarray]) -> np.ndarray:
        """
        Embeds a list of temporally contiguous frame chunks using XCLIP.
        Each chunk is treated as a mini-clip and its embedding is calculated.
        The final embedding is the mean of all mini-clip embeddings.
        """
        segment_embeddings = []
        
        # The loop logic is now simpler and conceptually correct
        for segment_frames in temporal_chunks:
            
            if len(segment_frames) == 0:
                continue
            
            n = len(segment_frames)

            # Subsample this TEMPORAL chunk to 8 frames for XCLIP
            if n > 8:
                # Evenly sample across the temporal segment
                idxs = np.linspace(0, n - 1, 8, dtype=int)
                segment_frames_subsampled = segment_frames[idxs]
            elif n < 8:
                # Pad with the last frame
                padding = [segment_frames[-1]] * (8 - n)
                segment_frames_subsampled = np.concatenate((segment_frames, padding), axis=0)
            else:
                segment_frames_subsampled = segment_frames

            video_inputs = self.video_processor(images=list(segment_frames_subsampled), return_tensors="pt").to(self.device)

            with torch.inference_mode():
                text_inputs = self.video_processor(text = "", return_tensors="pt").to(self.device)
                outputs = self.video_model(
                    pixel_values=video_inputs["pixel_values"],
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"],
                )
                video_embedding = outputs.video_embeds.squeeze(0)
                segment_embeddings.append(video_embedding)

        if not segment_embeddings:
            # Use the dimension of your XCLIP model, e.g., 768
            return np.zeros(768, dtype=np.float32) 

        video_emb = torch.stack(segment_embeddings).mean(dim=0)
        return video_emb.cpu().numpy()


    def _embed_image_clusters(self, frames: np.ndarray, frame_embs: np.ndarray, clusters: dict[int, list[int]]):
        result = {}
        for cid, idxs in clusters.items():
            if not idxs:
                continue
            
            # ... (Questa parte del codice per trovare l'immagine migliore non cambia)
            embs = frame_embs[idxs]
            centroid = embs.mean(axis=0, keepdims=True)
            sims = (embs @ centroid.T) / (
                np.linalg.norm(embs, axis=1, keepdims=True) * np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-8
            )
            best_local_index_within_cluster = int(np.argmax(sims))
            best_local = idxs[best_local_index_within_cluster]
            img = frames[best_local]

            inputs = self.image_processor(images=[img], return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.image_model(**inputs)

                img_emb = outputs.pooler_output.cpu().numpy().squeeze()

            result[f"cluster_{cid}"] = {
                "frame_idx_local": int(best_local),
                "frame_embedding": torch.tensor(img_emb, dtype=torch.float32),
            }
        return result
    
    def encode(self, frames):
        try:
            frame_embs = self._embed_frames_clip(frames)
            clusters = cluster_frames(frame_embs, self.max_temporal_segments)
            keyframe_embedding = self._embed_image_clusters(frames, frame_embs, clusters)

            k = choose_k(len(frames), self.max_temporal_segments)
            if k <= 1 or len(frames) < 8: # If scene is too short, treat as one chunk
                temporal_chunks = [frames] 
            else:
                # This splits the frames into k *temporally contiguous* groups
                temporal_chunks = np.array_split(frames, k, axis=0)
            
            temporal_embedding = self._embed_temporal_segments(temporal_chunks)

            return {
                "image": keyframe_embedding,
                "video": torch.tensor(temporal_embedding, dtype=torch.float32)
            }
        finally:
            if self.device == "cuda":
                torch.cuda.empty_cache()