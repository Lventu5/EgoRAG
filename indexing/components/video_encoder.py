import os
import logging
import subprocess
import tempfile
import uuid
import shutil
import sys
from pathlib import Path
sys.path.append(os.path.join(Path(os.getcwd()), 'external/InternVideo/InternVideo2'))
sys.path.append(os.path.join(Path(os.getcwd()), 'external/InternVideo/InternVideo2/multi_modality'))
import interface

import numpy as np
import torch
from transformers import (
    XCLIPModel,
    XCLIPProcessor,
    CLIPVisionModel,
    CLIPImageProcessor,
    AutoProcessor,
    AutoModel,
    Qwen2VLForConditionalGeneration,
)
# from qwen_vl_utils import process_vision_info
from PIL import Image
from sklearn.cluster import KMeans
from decord import VideoReader, cpu

from .base_encoder import BaseEncoder
from data.video_dataset import Scene
from indexing.utils.clustering import cluster_frames
from configuration.config import CONFIG
from utils.modeling_internvideo2 import vid2tensor

# Silence verbose torchcodec logging from qwen-vl-utils
logging.getLogger("torchcodec").setLevel(logging.WARNING)
logging.getLogger("qwen_vl_utils").setLevel(logging.WARNING)

class VideoEncoder(BaseEncoder):
    
    """
    Encodes video frames using video models:
    - XCLIP: 8 frames uniformly sampled
    - Qwen2-VL: extracts single embedding per scene from vision tower
    """
    def __init__(
        self,
        device: str = "cuda",
    ):
        super().__init__(device)
        self.model_name = CONFIG.indexing.video.model_name
        
        # Simple logic:
        # Configurazione basata sul modello
        # - XCLIP: 8 frames selezionati tramite clustering
        # - InternVideo2: vid2tensor automatically samples 8 frames uniformly
        if self.model_name == "xclip":
            self.max_frames = 8  # XCLIP usa esattamente 8 frame selezionati tramite clustering
        elif self.model_name == "internvideo2":
            self.max_frames = 8  # InternVideo2 uses 8 frames
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        logging.info(f"VideoEncoder: model={self.model_name}, max_frames={self.max_frames}")
                    
        self.video_model = None
        self.video_processor = None

    def _compute_adaptive_fps(self, video_path: str, max_frames_allowed: int = 1024, margin: int = 32) -> float:
        """Compute an adaptive fps so that the number of frames sent to the model
        does not exceed max_frames_allowed (with a safety margin).
        Returns a fps value >= 0.1.
        """
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            total_frames = len(vr)
            fps_orig = vr.get_avg_fps() or 1.0
            del vr
        except Exception:
            return 1.0

        target = max(1, min(total_frames, max_frames_allowed - margin))
        if total_frames <= target:
            return min(1.0, fps_orig)
        sampling_rate = total_frames / float(target)
        fps_new = max(0.1, float(fps_orig) / sampling_rate)
        logging.info(f"[VideoEncoder] adaptive fps for {video_path}: {fps_new:.3f} (total_frames={total_frames}, target={target})")
        return float(fps_new)

    def load_models(self):
        logging.info(f"[{self.__class__.__name__}] Loading models with {self.model_name}...")
        
        if self.model_name == "xclip":
            self.model_id = CONFIG.indexing.video.xclip_id
            self.video_model = XCLIPModel.from_pretrained(self.model_id).to(self.device).eval()
            self.video_processor = XCLIPProcessor.from_pretrained(self.model_id)
            
        elif self.model_name == "internvideo2":
            # InternVideo2 6B model
            # self.model_id = CONFIG.indexing.video.internvideo2_id
            # logging.info(f"Loading InternVideo2 model: {self.model_id}...")
            # self.video_model = AutoModel.from_pretrained(
            #     self.model_id,
            #     trust_remote_code=True
            # ).to(self.device).eval()
            # logging.info(f"InternVideo2 model loaded successfully")

            # Internvideo 1B
            config_path = "external/InternVideo/InternVideo2/multi_modality/demo/internvideo2_stage2_config.py"
            model_path = "external/InternVideo/InternVideo2/checkpoints/InternVideo2-stage2_1b-224p-f4.pt"

            model = interface.load_model(config_path, model_path)
            print("CHECK pos_embed:", model.vision_encoder.pos_embed.shape)

            self.video_model = model.to(self.device).eval()
            
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}. Choose 'xclip', 'qwen2-vl', or 'internvideo2'.")
        
        # Load CLIP Vision for XCLIP frame clustering
        if self.model_name == "xclip":
            self.image_model_id = CONFIG.indexing.video.clip_id
            # Keep CLIP vision on the same device as the video model (usually GPU),
            # but ensure we free intermediate tensors after clustering to avoid accumulation.
            self.image_model = CLIPVisionModel.from_pretrained(self.image_model_id).to(self.device).eval()
            self.image_processor = CLIPImageProcessor.from_pretrained(self.image_model_id)
            # Track the device used for the image model (match self.device)
            self.image_model_device = self.device
            logging.info(f"[{self.__class__.__name__}] CLIP Vision loaded on {self.image_model_device} for clustering.")
        
        logging.info(f"[{self.__class__.__name__}] Models loaded.")

    def _embed_frames_clip(self, frames: np.ndarray) -> np.ndarray:
        """Embed frames with CLIP for clustering (XCLIP only)"""
        # Run CLIP image model on the configured image model device (usually GPU).
        device = getattr(self, "image_model_device", self.device)
        inputs = self.image_processor(images=list(frames), return_tensors="pt")
        # Move processor tensors to the correct device
        if isinstance(inputs, dict):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
        else:
            inputs = inputs.to(device)

        outputs = None
        img_embs_cpu = None
        try:
            with torch.no_grad():
                outputs = self.image_model(**inputs)
                img_embs = outputs.pooler_output

                # Move embeddings to CPU numpy immediately to avoid keeping GPU tensors
                if isinstance(img_embs, torch.Tensor):
                    img_embs_cpu = img_embs.detach().cpu().numpy()
                else:
                    img_embs_cpu = np.asarray(img_embs)

            return img_embs_cpu
        finally:
            # Free intermediate tensors and clear CUDA cache if needed
            try:
                del inputs
            except Exception:
                pass
            try:
                del outputs
            except Exception:
                pass
            try:
                del img_embs
            except Exception:
                pass
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _select_frames_by_clustering(self, frames: np.ndarray, k: int = 8) -> np.ndarray:
        """Select k representative frames using clustering (for XCLIP)"""
        frame_embs = self._embed_frames_clip(frames)
        clusters = cluster_frames(frame_embs, max_temporal_segments=k)
        
        # Select frame closest to each centroid
        selected_frames = []
        for i in range(clusters.n_clusters):
            centroid = clusters.cluster_centers_[i]
            cluster_indices = np.where(clusters.labels_ == i)[0]
            embs_in_cluster = frame_embs[cluster_indices]
            distances = np.linalg.norm(embs_in_cluster - centroid, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            selected_frames.append(frames[closest_idx])
        
        return np.array(selected_frames)
    

    def _embed_frames(self, frames: np.ndarray = None, video_path: str = None, adaptive_fps: float = None) -> np.ndarray:
        """
        Embeds frames using the video model.
        - XCLIP: expects exactly 8 frames (uses frames parameter)
        - InternVideo2: uses video_path, vid2tensor samples frames automatically
        """

        # XCLIP: select 8 frames via clustering
        if self.model_name == "xclip":
            if len(frames) > 8:
                # Use clustering to select 8 representative frames
                frames = self._select_frames_by_clustering(frames, k=8)
            elif len(frames) < 8:
                # Pad to 8 if less
                padding = [frames[-1]] * (8 - len(frames))
                frames = np.concatenate((frames, padding), axis=0)
            
            video_inputs = self.video_processor(images=list(frames), return_tensors="pt").to(self.device)
            with torch.inference_mode():
                text_inputs = self.video_processor(text="", return_tensors="pt").to(self.device)
                outputs = self.video_model(
                    pixel_values=video_inputs["pixel_values"],
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"],
                )
                video_embedding = outputs.video_embeds.squeeze(0).detach().cpu()
                del video_inputs, text_inputs, outputs
                torch.cuda.empty_cache()
                return video_embedding.numpy()
                    
        elif self.model_name == "internvideo2":
            # InternVideo2 - uses vid2tensor which handles frame sampling automatically
            if video_path is None:
                raise ValueError("InternVideo2 requires video_path parameter")
            
            # vid2tensor automatically samples frames uniformly
            # frames_tensor = vid2tensor(
            #     video_path, 
            #     fnum=8,
            #     device=self.device
            # )
            # with torch.inference_mode():
            #     # Get video features using InternVideo2
            #     video_feat = self.video_model.get_vid_feat(frames_tensor)
            #     # video_feat shape: [1, 512]
            #     video_embedding = video_feat.squeeze(0).detach().cpu().to(torch.float32)
            # del frames_tensor
            # torch.cuda.empty_cache()
            # return video_embedding.numpy()

            # Internvideo 1B
            feat_dict = interface.extract_video_features([video_path], self.video_model, fn=4)
            video_emb = next(iter(feat_dict.values()))
            print("DEBUG internvideo shape:", video_emb.shape, "mean:", video_emb.mean(), "std:", video_emb.std())
            return video_emb.squeeze()        
        else:
            raise ValueError(f"Unknown model {self.model_name}")


    def _extract_frames_from_scene(self, video_path: str, scene: Scene) -> np.ndarray:
        """
        Estrae tutti i frame della scena dal video.
        Usato da XCLIP per poi fare clustering.
        """
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        num_frames_in_scene = scene.end_frame - scene.start_frame
        
        if num_frames_in_scene == 0:
            return np.array([])
        
        # Estrae tutti i frame della scena
        indices = np.arange(scene.start_frame, scene.end_frame)
        frames = vr.get_batch(indices).asnumpy()
        
        del vr
        return frames
    
    def _create_clip_with_decord(self, video_path: str, start_time: float, end_time: float, tmp_dir: str) -> str:
        """
        Crea clip video usando ffmpeg di imageio-ffmpeg (ha libx264!)
        """
        import imageio_ffmpeg
        os.makedirs(tmp_dir, exist_ok=True)
        out_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.mp4")
        
        # Get ffmpeg binary from imageio-ffmpeg (has libx264 enabled!)
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        
        # Calculate duration instead of end_time to avoid seeking issues
        duration = end_time - start_time
        
        # Use libx264 re-encoding for clean clips with timestamp fixes
        cmd = [
            ffmpeg_bin, "-y",
            "-ss", f"{start_time:.3f}",
            "-i", video_path,
            "-t", f"{duration:.3f}",  # Use duration instead of -to
            "-c:v", "libx264",
            "-preset", "ultrafast",  # Faster encoding
            "-crf", "28",  # Lower quality for speed
            "-avoid_negative_ts", "make_zero",  # Fix timestamp issues
            "-fflags", "+genpts",  # Generate timestamps
            "-an",
            out_path
        ]
        # Run ffmpeg and capture stderr for debugging; don't hide failures silently
        proc = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            stderr = proc.stderr.decode(errors="ignore") if proc.stderr is not None else ""
            logging.error(
                "ffmpeg create_clip failed (rc=%s, signal=%s) cmd=%s stderr=%s",
                proc.returncode,
                -proc.returncode if proc.returncode < 0 else None,
                " ".join(cmd),
                stderr.strip(),
            )
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=None, stderr=proc.stderr)
        else:
            logging.debug(
                "ffmpeg create_clip ok (rc=0) cmd=%s output=%s",
                " ".join(cmd),
                out_path,
            )
        return out_path
    
    def _extract_scene_clip(self, video_path: str, start_time: float, end_time: float, tmp_dir: str) -> str:
        """
        Crea una clip temporanea della scena usando ffmpeg.
        Ritorna il path della clip temporanea.
        """
        os.makedirs(tmp_dir, exist_ok=True)
        out_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.mp4")

        # Re-encode with VP9 (source codec) - stream copy causes pts/seeking issues
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_time:.3f}",
            "-to", f"{end_time:.3f}",
            "-i", video_path,
            "-c:v", "libvpx-vp9",
            "-crf", "30",  
            "-b:v", "0",
            "-an",
            out_path
        ]
        # Run ffmpeg and capture stderr for debugging; raise informative error
        proc = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            stderr = proc.stderr.decode(errors="ignore") if proc.stderr is not None else ""
            logging.error(
                "ffmpeg extract_scene failed (rc=%s, signal=%s) cmd=%s stderr=%s",
                proc.returncode,
                -proc.returncode if proc.returncode < 0 else None,
                " ".join(cmd),
                stderr.strip(),
            )
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=None, stderr=proc.stderr)
        else:
            logging.debug(
                "ffmpeg extract_scene ok (rc=0) cmd=%s output=%s",
                " ".join(cmd),
                out_path,
            )

        return out_path

    def encode(self, video_path: str, scene: Scene) -> dict:
        """
        Encode video scene into embeddings.
        Each encoder handles frame/clip extraction internally:
        - XCLIP: extracts all frames, clusters to 8, encodes
        - InternVideo2: creates scene clip, vid2tensor samples 8 frames automatically
        
        Args:
            video_path: Path to video file
            scene: Scene object with timing information
        
        Returns:
            dict with keys:
                - "video": torch.Tensor (embedding)
                - "keyframes": np.ndarray (primi 8 frame per captioner1, se necessario)
        """
        if self.model_name == "xclip":
            # XCLIP: estrae tutti frame, fa clustering a 8, embedda
            frames = self._extract_frames_from_scene(video_path, scene)
            if len(frames) == 0:
                return None
            
            video_emb = self._embed_frames(frames)
            
            # Keyframes: primi 8 per captioner1
            keyframes = frames[:8] if len(frames) >= 8 else frames
            
            return {
                "video": torch.from_numpy(video_emb),
                "keyframes": keyframes
            }
        
        elif self.model_name == "internvideo2":
            # InternVideo2: create scene clip and let vid2tensor handle frame sampling
            tmp_dir = tempfile.mkdtemp(prefix="internvideo2_scene_")
            try:
                # Create scene clip - vid2tensor will sample frames automatically
                clip_path = self._create_clip_with_decord(
                    video_path, scene.start_time, scene.end_time, tmp_dir
                )
                
                # vid2tensor handles frame sampling automatically
                video_emb = self._embed_frames(video_path=clip_path)
                
                # Extract keyframes for captioner
                vr = VideoReader(clip_path, ctx=cpu(0))
                total_frames = len(vr)
                if total_frames > 0:
                    keyframe_indices = np.linspace(0, total_frames - 1, min(8, total_frames), dtype=int)
                    keyframes = vr.get_batch(keyframe_indices).asnumpy()
                else:
                    keyframes = np.array([])
                del vr
                
                return {
                    "video": torch.from_numpy(video_emb),
                    "keyframes": keyframes
                }
            finally:
                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir)
        
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def encode_full_video(self, video_path: str) -> dict:
        """
        Encode the entire video using Qwen2-VL's automatic sampler (no subsampling, no clip creation).
        Only for Qwen2-VL. Passes the video_path directly to the model.
        Returns:
            dict with keys:
                - "video": torch.Tensor (embedding)
                - "keyframes": np.ndarray (empty, not used)
        """
        if self.model_name != "qwen2-vl" and self.model_name != "internvideo2":
            raise ValueError("encode_full_video is only supported for Qwen2-VL and InternVideo2")

        if self.model_name == "internvideo2":
            logging.info("[VideoEncoder] embedding the whole video")
            video_emb = self._embed_frames(video_path=video_path)
            return {
                "video": torch.from_numpy(video_emb),
                "keyframes": np.array([])
            }
