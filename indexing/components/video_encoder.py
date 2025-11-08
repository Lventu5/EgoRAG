import numpy as np
import torch
from transformers import (
    XCLIPModel,
    XCLIPProcessor,
    CLIPVisionModel,
    CLIPImageProcessor,
    AutoModel,
    AutoImageProcessor
)
import logging
from sklearn.cluster import KMeans

from .base_encoder import BaseEncoder
from indexing.utils.clustering import choose_k, cluster_frames
from configuration.config import CONFIG

class VideoEncoder(BaseEncoder):
    """
    Encodes video frames into two hierarchical representations:
    1.  Temporal "Bag-of-Actions": Spatiotemporal embeddings from video model
        (XCLIP or InternVideo2) of contiguous segments.
    2.  Static "Bag-of-Keyframes": Image embeddings (CLIP) of the
        most representative cluster centroids.
    """
    def __init__(
        self,
        device: str = "cuda",
    ):
        super().__init__(device)
        self.model_name = CONFIG.indexing.video.model_name
        max_frames_per_scene = CONFIG.indexing.video.max_frames_per_scene
        self.max_temporal_segments = CONFIG.indexing.video.max_temporal_segments
        
        # Set default max_frames_per_scene based on model
        if max_frames_per_scene is None:
            self.max_frames_per_scene = 8 if self.model_name == "xclip" else None
        else:
            self.max_frames_per_scene = max_frames_per_scene
                    
        # Models to be loaded
        self.video_model = None
        self.video_processor = None
        self.image_model: CLIPVisionModel = None
        self.image_processor: CLIPImageProcessor = None

    def load_models(self):
        logging.info(f"[{self.__class__.__name__}] Loading models with {self.model_name}...")
        
        # Load video model based on model_name
        if self.model_name == "xclip":
            self.model_id = CONFIG.indexing.video.xclip_id
            self.video_model = XCLIPModel.from_pretrained(self.model_path).to(self.device).eval()
            self.video_processor = XCLIPProcessor.from_pretrained(self.model_path)
        elif self.model_name == "internvideo2":
            # InternVideo2-1B model
            self.model_id = CONFIG.indexing.video.internvideo2_id
            self.video_model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True
            ).to(self.device).eval()
            self.video_processor = AutoImageProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}. Choose 'xclip' or 'internvideo2'.")
        
        # Load CLIP-Vision (Static) - same for all models
        self.image_model_id = CONFIG.indexing.video.clip_id
        self.image_model = CLIPVisionModel.from_pretrained(self.image_model_path).to(self.device).eval()
        self.image_processor = CLIPImageProcessor.from_pretrained(self.image_model_path)
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
        Embeds a list of temporally contiguous frame chunks using the video model.
        Each chunk is treated as a mini-clip and its embedding is calculated.
        The final embedding is the mean of all mini-clip embeddings.
        """
        segment_embeddings = []
        
        for segment_frames in temporal_chunks:
            if len(segment_frames) == 0:
                continue
            
            n = len(segment_frames)

            # If max_frames_per_scene is None, use all frames in the segment
            if self.max_frames_per_scene is None:
                segment_frames_processed = segment_frames
            else:
                # Subsample or pad to max_frames_per_scene
                if n > self.max_frames_per_scene:
                    idxs = np.linspace(0, n - 1, self.max_frames_per_scene, dtype=int)
                    segment_frames_processed = segment_frames[idxs]
                elif n < self.max_frames_per_scene:
                    padding = [segment_frames[-1]] * (self.max_frames_per_scene - n)
                    segment_frames_processed = np.concatenate((segment_frames, padding), axis=0)
                else:
                    segment_frames_processed = segment_frames

            # Process based on model type
            if self.model_name == "xclip":
                video_inputs = self.video_processor(images=list(segment_frames_processed), return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    text_inputs = self.video_processor(text="", return_tensors="pt").to(self.device)
                    outputs = self.video_model(
                        pixel_values=video_inputs["pixel_values"],
                        input_ids=text_inputs["input_ids"],
                        attention_mask=text_inputs["attention_mask"],
                    )
                    video_embedding = outputs.video_embeds.squeeze(0).detach().cpu()
                    segment_embeddings.append(video_embedding)
                    del video_inputs, text_inputs, outputs
                    
            elif self.model_name == "internvideo2":
                # InternVideo2 expects frames in shape (T, H, W, C)
                video_inputs = self.video_processor(images=list(segment_frames_processed), return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    outputs = self.video_model(video_inputs["pixel_values"])
                    # InternVideo2 outputs last_hidden_state, take mean pooling
                    video_embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu()
                    segment_embeddings.append(video_embedding)
                    del video_inputs, outputs
                    
            torch.cuda.empty_cache()

        if not segment_embeddings:
            # Default embedding size (adjust based on model)
            embed_dim = 768 if self.model_name == "xclip" else 1024
            return np.zeros(embed_dim, dtype=np.float32) 

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
            # If max_frames_per_scene is set and scene is too short, treat as one chunk
            min_frames_required = self.max_frames_per_scene if self.max_frames_per_scene else 1
            if k <= 1 or len(frames) < min_frames_required:
                temporal_chunks = [frames] 
            else:
                temporal_chunks = np.array_split(frames, k, axis=0)
            
            temporal_embedding = self._embed_temporal_segments(temporal_chunks)
            representative_frames = self._get_representative_frames(frames, frame_embs, km)

            return {
                "image": torch.tensor(keyframe_embedding, device="cpu", dtype=torch.float32) if isinstance(keyframe_embedding, np.ndarray) else keyframe_embedding,
                "video": torch.tensor(temporal_embedding, device="cpu", dtype=torch.float32),
                "keyframes": representative_frames,
            }
        finally:
            if self.device == "cuda":
                torch.cuda.empty_cache()