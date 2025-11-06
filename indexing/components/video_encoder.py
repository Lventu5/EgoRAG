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
from sklearn.cluster import KMeans

from .base_encoder import BaseEncoder
from indexing.utils.logging import LevelAwareFormatter
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
        self.video_model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch16").to(self.device).eval()
        self.video_processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch16")
        
        # Load CLIP-Vision (Static)
        self.image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        logging.info(f"[{self.__class__.__name__}] Models loaded.")

    def _embed_frames_clip(self, frames: np.ndarray) -> np.ndarray:
        inputs = self.image_processor(images=list(frames), return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.image_model(**inputs)
            img_embs = outputs.pooler_output
            
        return img_embs.cpu().numpy()

    def _get_representative_frames(self, frames: np.ndarray, frame_embs: np.ndarray, clusters: KMeans) -> np.ndarray:
        """
        Finds and returns the actual raw frames closest to each cluster centroid.
        """
        representative_frames = []
        if frame_embs.shape[0] == 0:
            return np.array([])
            
        for i in range(clusters.n_clusters):
            centroid = clusters.cluster_centers_[i]
            # Find the index of the frame embedding closest to the centroid
            distances = np.linalg.norm(frame_embs - centroid, axis=1)
            closest_frame_idx = np.argmin(distances)
            # Append the *actual frame*
            representative_frames.append(frames[closest_frame_idx])
            
        return np.array(representative_frames)
    
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
                video_embedding = outputs.video_embeds.squeeze(0).detach().cpu()
                segment_embeddings.append(video_embedding)
                del video_inputs
                del text_inputs
                del outputs
                torch.cuda.empty_cache()

        if not segment_embeddings:
            # Use the dimension of your XCLIP model, e.g., 768
            return np.zeros(768, dtype=np.float32) 

        video_emb = torch.stack(segment_embeddings).mean(dim=0)
        return video_emb.cpu().numpy()


    # In VideoEncoder class
    def _embed_image_clusters(self, frame_embs: np.ndarray, clusters: KMeans) -> np.ndarray:
        """
        Finds the frame embedding closest to each cluster centroid and returns it.
        This reuses the existing embeddings and performs no new model inference.
        """
        centroid_embeddings = []
        
        if frame_embs.shape[0] == 0:
            return np.zeros((0, self.static_model.config.hidden_size), dtype=np.float32)

        for i in range(clusters.n_clusters):
            # 1. Get the true centroid from the KMeans object
            centroid = clusters.cluster_centers_[i]
            
            # 2. Get the indices of all frames in this cluster
            cluster_indices = np.where(clusters.labels_ == i)[0]
            if len(cluster_indices) == 0:
                continue
                
            # 3. Get the pre-computed embeddings for this cluster
            embs_in_cluster = frame_embs[cluster_indices]
            
            # 4. Find the embedding in this cluster closest to the centroid
            distances = np.linalg.norm(embs_in_cluster - centroid, axis=1)
            closest_local_idx = np.argmin(distances) # Index *within* embs_in_cluster
            
            # 5. Get the global index of that embedding
            closest_global_idx = cluster_indices[closest_local_idx]
            
            # 6. REUSE the existing embedding. NO new model call.
            centroid_embeddings.append(frame_embs[closest_global_idx])
            
        return np.array(centroid_embeddings, dtype=np.float32)
    
    def encode(self, frames):
        try:
            frame_embs = self._embed_frames_clip(frames)
            km = cluster_frames(frame_embs, self.max_temporal_segments)
            keyframe_embedding = self._embed_image_clusters(frame_embs, km)

            k = choose_k(len(frames), self.max_temporal_segments)
            if k <= 1 or len(frames) < 8: # If scene is too short, treat as one chunk
                temporal_chunks = [frames] 
            else:
                # This splits the frames into k *temporally contiguous* groups
                temporal_chunks = np.array_split(frames, k, axis=0)
            
            temporal_embedding = self._embed_temporal_segments(temporal_chunks)
            representative_frames = self._get_representative_frames(frames, frame_embs, km)

            return {
                "image": keyframe_embedding.detach().cpu(),
                "video": torch.tensor(temporal_embedding, device="cpu", dtype=torch.float32),
                "keyframes": representative_frames,
            }
        finally:
            if self.device == "cuda":
                torch.cuda.empty_cache()