import os
import logging
import subprocess
import tempfile
import uuid
import shutil

import numpy as np
import torch
from transformers import (
    XCLIPModel,
    XCLIPProcessor,
    CLIPVisionModel,
    CLIPImageProcessor,
    AutoProcessor,
    LlavaNextVideoForConditionalGeneration,
)
from PIL import Image
from sklearn.cluster import KMeans
from decord import VideoReader, cpu

from .base_encoder import BaseEncoder
from indexing.utils.clustering import choose_k, cluster_frames
from configuration.config import CONFIG

class VideoEncoder(BaseEncoder):
    """
    Encodes video frames into two hierarchical representations:
    1.  Temporal "Bag-of-Actions": Spatiotemporal embeddings from video model
        (XCLIP or LLaVA Video) of contiguous segments.
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
        self.use_video_clips = CONFIG.indexing.video.use_video_clips
        
        # Logic for max_frames_per_scene and use_video_clips:
        # - If max_frames_per_scene is None:
        #   - llava-video: use video clips (use_video_clips will be overridden to True)
        #   - xclip: use 8 frames
        # - If max_frames_per_scene is a number: use that many frames (no clips)
        
        if max_frames_per_scene is None:
            if self.model_name == "xclip":
                self.max_frames_per_scene = 8
                self.use_video_clips = False
            elif self.model_name == "llava-video":
                self.max_frames_per_scene = 16  # Default for frame mode
                # use_video_clips stays as configured (can be True or False)
            else:
                self.max_frames_per_scene = 8
                self.use_video_clips = False
        else:
            # If user specifies a number, always use frame extraction (no clips)
            if self.model_name == "xclip":
                self.max_frames_per_scene = 8 # for xclip always 8
                self.use_video_clips = False
            elif self.model_name == "llava-video":
                self.max_frames_per_scene = max_frames_per_scene
                self.use_video_clips = False
        
        logging.info(f"VideoEncoder: model={self.model_name}, max_frames={self.max_frames_per_scene}, use_clips={self.use_video_clips}")
                    
        self.video_model = None
        self.video_processor = None
        self.image_model: CLIPVisionModel = None
        self.image_processor: CLIPImageProcessor = None

    def load_models(self):
        logging.info(f"[{self.__class__.__name__}] Loading models with {self.model_name}...")
        
        if self.model_name == "xclip":
            self.model_id = CONFIG.indexing.video.xclip_id
            self.video_model = XCLIPModel.from_pretrained(self.model_id).to(self.device).eval()
            self.video_processor = XCLIPProcessor.from_pretrained(self.model_id)
            
        elif self.model_name == "llava-video":
            # LLaVA Video model
            self.model_id = CONFIG.indexing.video.llava_video_id
            
            logging.info(f"Loading LLaVA Video model: {self.model_id}...")
            
            self.video_processor = AutoProcessor.from_pretrained(self.model_id)
            self.video_model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map=self.device,
                low_cpu_mem_usage=True,
            )
            self.video_model.eval()
            
            logging.info(f"LLaVA Video model loaded successfully")
            
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}. Choose 'xclip' or 'llava-video'.")
        
        # Load CLIP-Vision (Static) - same for all models
        self.image_model_id = CONFIG.indexing.video.clip_id
        self.image_model = CLIPVisionModel.from_pretrained(self.image_model_id).to(self.device).eval()
        self.image_processor = CLIPImageProcessor.from_pretrained(self.image_model_id)
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
                    
            elif self.model_name == "llava-video":
                # LLaVA Video - extract vision features
                with torch.inference_mode():
                    pil_frames = [Image.fromarray(frame.astype(np.uint8)) for frame in segment_frames_processed]
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "video"},
                                {"type": "text", "text": "Describe this video."},
                            ],
                        },
                    ]
                    
                    prompt = self.video_processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = self.video_processor(
                        text=prompt,
                        videos=[pil_frames],
                        return_tensors="pt",
                        padding=True,
                    ).to(self.device, torch.float16)
                    
                    # Extract vision embeddings from the vision tower
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        # pixel_values_videos shape: (batch, num_frames, channels, height, width)
                        # Reshape to (batch * num_frames, channels, height, width) for vision tower
                        pixel_values = inputs["pixel_values_videos"]
                        batch_size, num_frames, channels, height, width = pixel_values.shape
                        pixel_values_flat = pixel_values.reshape(batch_size * num_frames, channels, height, width)
                        
                        vision_outputs = self.video_model.vision_tower(
                            pixel_values_flat,
                            output_hidden_states=True
                        )
                        
                        # Pool the vision features
                        # last_hidden_state: (batch*num_frames, num_patches, hidden_dim)
                        vision_features = vision_outputs.last_hidden_state
                        
                        # Reshape back: (batch*num_frames, num_patches, hidden_dim) → (batch, num_frames, num_patches, hidden_dim)
                        num_patches = vision_features.shape[1]
                        hidden_dim = vision_features.shape[2]
                        vision_features = vision_features.reshape(batch_size, num_frames, num_patches, hidden_dim)
                        
                        # Pool over patches and frames: mean over num_patches and num_frames
                        pooled_features = vision_features.mean(dim=(1, 2))  # (batch, hidden_dim)
                        
                        video_embedding = pooled_features.squeeze(0).detach().cpu()
                        segment_embeddings.append(video_embedding)
                    
                    del inputs, vision_outputs
            else:
                raise ValueError(f"Unknown model {self.model_name}")
                    
            torch.cuda.empty_cache()

        if not segment_embeddings:
            # Default embedding size (adjust based on model)
            if self.model_name == "xclip":
                embed_dim = 768
            elif self.model_name == "llava-video":
                embed_dim = 1024  # LLaVA Video vision tower hidden size
            else:
                embed_dim = 768
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
            return np.zeros((0, self.image_model.config.hidden_size), dtype=np.float32)

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
    
    def _extract_scene_clip(self, video_path: str, start_time: float, end_time: float, tmp_dir: str) -> str:
        """
        Taglia [start_time, end_time] in un file mp4 temporaneo.
        Ritorna il path della mini-clip.
        """
        os.makedirs(tmp_dir, exist_ok=True)
        out_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.mp4")

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_time:.3f}",
            "-to", f"{end_time:.3f}",
            "-i", video_path,
            "-c", "copy",
            out_path
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            # Fallback: re-encode if copy fails
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_time:.3f}",
                "-to", f"{end_time:.3f}",
                "-i", video_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-ac", "2", "-b:a", "128k",
                out_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return out_path

    def _extract_frames_from_clip(self, clip_path: str, max_frames: int) -> np.ndarray:
        """
        Estrae frames da una clip video usando Decord.
        """
        vr = VideoReader(clip_path, ctx=cpu(0))
        total_frames = len(vr)
        
        if total_frames == 0:
            return np.array([])
        
        if max_frames is None or total_frames <= max_frames:
            indices = np.arange(total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        frames = vr.get_batch(indices).asnumpy()
        return frames

    def _extract_frames_direct(self, video_path: str, start_frame: int, end_frame: int, max_frames: int) -> np.ndarray:
        """
        Estrae frames direttamente dal video senza creare clip intermedia.
        Più efficiente quando non serve la clip completa.
        """
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames_in_scene = end_frame - start_frame
        
        if num_frames_in_scene == 0:
            return np.array([])
        
        if max_frames is None or num_frames_in_scene <= max_frames:
            indices = np.arange(start_frame, end_frame)
        else:
            indices = np.linspace(start_frame, end_frame - 1, max_frames, dtype=int)
        
        frames = vr.get_batch(indices).asnumpy()
        return frames

    def encode(self, frames=None, video_path=None, scene=None):
        """
        Encode frames in three modes:
        1. Direct frames: encode(frames=np.ndarray) - when frames are pre-extracted
        2. Video clip mode: encode(video_path=str, scene=Scene) with use_video_clips=True
           - Extracts video clip first, then processes entire clip
        3. Frame extraction mode: encode(video_path=str, scene=Scene) with use_video_clips=False
           - Directly extracts specific frames without intermediate clip
        
        Args:
            frames: Pre-extracted numpy array of frames (H, W, C)
            video_path: Path to video file (alternative to frames)
            scene: Scene object with start_time/end_time (required if video_path is provided)
        """
        tmp_root = None
        clip_path = None
        
        try:
            # Mode 1: Direct frames (existing behavior)
            if frames is not None:
                input_frames = frames
            
            # Mode 2 & 3: Extract from video
            elif video_path is not None and scene is not None:
                if self.use_video_clips:
                    # Mode 2: Extract video clip first, then frames
                    tmp_root = tempfile.mkdtemp(prefix="videnc_")
                    clip_path = self._extract_scene_clip(video_path, scene.start_time, scene.end_time, tmp_root)
                    input_frames = self._extract_frames_from_clip(clip_path, self.max_frames_per_scene * self.max_temporal_segments)
                else:
                    # Mode 3: Direct frame extraction (more efficient, no intermediate clip)
                    input_frames = self._extract_frames_direct(video_path, scene.start_frame, scene.end_frame, 
                                                               self.max_frames_per_scene * self.max_temporal_segments)
                
                if len(input_frames) == 0:
                    logging.warning(f"No frames extracted from {video_path} [{scene.start_time}s - {scene.end_time}s]")
                    return None
            else:
                raise ValueError("Must provide either 'frames' OR both 'video_path' and 'scene'")
            
            # Standard encoding pipeline
            frame_embs = self._embed_frames_clip(input_frames)
            km = cluster_frames(frame_embs, self.max_temporal_segments)
            keyframe_embedding = self._embed_image_clusters(frame_embs, km)

            k = choose_k(len(input_frames), self.max_temporal_segments)
            # If max_frames_per_scene is set and scene is too short, treat as one chunk
            min_frames_required = self.max_frames_per_scene if self.max_frames_per_scene else 1
            if k <= 1 or len(input_frames) < min_frames_required:
                temporal_chunks = [input_frames] 
            else:
                temporal_chunks = np.array_split(input_frames, k, axis=0)
            
            temporal_embedding = self._embed_temporal_segments(temporal_chunks)
            representative_frames = self._get_representative_frames(input_frames, frame_embs, km)

            return {
                "image": torch.tensor(keyframe_embedding, device="cpu", dtype=torch.float32) if isinstance(keyframe_embedding, np.ndarray) else keyframe_embedding,
                "video": torch.tensor(temporal_embedding, device="cpu", dtype=torch.float32),
                "keyframes": representative_frames,
            }
        finally:
            # Cleanup temporary files
            if tmp_root is not None:
                try:
                    shutil.rmtree(tmp_root, ignore_errors=True)
                except Exception as e:
                    logging.warning(f"Failed to cleanup temp directory {tmp_root}: {e}")
            
            if self.device == "cuda":
                torch.cuda.empty_cache()