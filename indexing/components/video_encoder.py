import os
import logging
import subprocess
import tempfile
import uuid
import shutil
import sys
from pathlib import Path
sys.path.append(os.path.join(Path(os.getcwd()), 'external/InternVideo/InternVideo2'))
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
from qwen_vl_utils import process_vision_info
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
        # - Qwen2-VL: estrae embedding diretto dalla vision tower
        # - InternVideo2: vid2tensor automatically samples 8 frames uniformly
        if self.model_name == "xclip":
            self.max_frames = 8  # XCLIP usa esattamente 8 frame selezionati tramite clustering
        elif self.model_name == "qwen2-vl":
            self.max_frames = None  # Qwen2-VL gestisce internamente i frame
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
            
        elif self.model_name == "qwen2-vl":
            # Qwen2-VL model
            self.model_id = CONFIG.indexing.video.qwen2_vl_id
            
            logging.info(f"Loading Qwen2-VL model: {self.model_id}...")
            
            # HuggingFace automatically uses HF_HOME/TRANSFORMERS_CACHE if set
            cache_dir = os.environ.get("HF_HOME", os.environ.get("TRANSFORMERS_CACHE"))
            if cache_dir:
                logging.info(f"Using cache directory: {cache_dir}")
            
            self.video_processor = AutoProcessor.from_pretrained(
                self.model_id,
                min_pixels=256*28*28,
                max_pixels=1280*28*28
            )
            
            self.video_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                dtype=torch.bfloat16,
                device_map=self.device,
            )
            
            self.video_model.eval()
            
            logging.info(f"Qwen2-VL model loaded successfully")
        
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
            model_path = "external/InternVideo/InternVideo2/checkpoints/internvideo2_stage2_1b.pth"

            model = interface.load_model(config_path, model_path)
            self.video_model = model.to(self.device).eval()
            
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}. Choose 'xclip', 'qwen2-vl', or 'internvideo2'.")
        
        # Load CLIP Vision for XCLIP frame clustering
        if self.model_name == "xclip":
            self.image_model_id = CONFIG.indexing.video.clip_id
            self.image_model = CLIPVisionModel.from_pretrained(self.image_model_id).to(self.device).eval()
            self.image_processor = CLIPImageProcessor.from_pretrained(self.image_model_id)
            logging.info(f"[{self.__class__.__name__}] CLIP Vision loaded for clustering.")
        
        logging.info(f"[{self.__class__.__name__}] Models loaded.")

    def _embed_frames_clip(self, frames: np.ndarray) -> np.ndarray:
        """Embed frames with CLIP for clustering (XCLIP only)"""
        inputs = self.image_processor(images=list(frames), return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.image_model(**inputs)
            img_embs = outputs.pooler_output
        return img_embs.cpu().numpy()

    def _select_frames_by_clustering(self, frames: np.ndarray, k: int = 8) -> np.ndarray:
        """Select k representative frames using clustering (for XCLIP)"""
        frame_embs = self._embed_frames_clip(frames)
        clusters = cluster_frames(frame_embs, k=k)
        
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
        - Qwen2-VL: processes entire video scene (uses video_path parameter)
        - InternVideo2: uses video_path, vid2tensor samples frames automatically
        """
        if self.model_name == "qwen2-vl" and video_path is None:
            raise ValueError("Qwen2-VL requires video_path parameter")
        
        if self.model_name != "qwen2-vl" and (frames is None or len(frames) == 0):
            # Return zero embedding
            if self.model_name == "xclip":
                embed_dim = 768
            elif self.model_name == "internvideo2":
                embed_dim = 512
            else:
                embed_dim = 768
            return np.zeros(embed_dim, dtype=np.float32)
        
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
                    
        elif self.model_name == "qwen2-vl":
            # Qwen2-VL - passa direttamente il video file path (leggero, no preload)
            with torch.inference_mode():
                prompt = "Describe this video scene."
                
                # Minimal approach: let processor load the video directly
                # Use adaptive_fps only when provided; otherwise keep fps=1.0 for scene clips
                fps_to_use = adaptive_fps if adaptive_fps is not None else 1.0
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": fps_to_use},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                
                # Apply chat template
                text = self.video_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                # Let process_vision_info load and process the video efficiently
                image_inputs, video_inputs = process_vision_info(messages)
                
                # Verify video was loaded
                assert video_inputs is not None and len(video_inputs) > 0, \
                    f"process_vision_info failed to load video: {video_path}, video_inputs={video_inputs}"
                
                logging.debug(f"[Qwen2-VL] video_inputs type: {type(video_inputs)}, len: {len(video_inputs)}")
                
                # Process inputs - processor handles video loading
                inputs = self.video_processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.device)

                # DEBUG: log keys and shapes of tensors passed to the visual encoder
                shapes = []
                for k, v in inputs.items():
                    try:
                        if isinstance(v, torch.Tensor):
                            shapes.append(f"{k}:{tuple(v.shape)}")
                        elif isinstance(v, (list, tuple)):
                            shapes.append(f"{k}:list(len={len(v)})")
                        else:
                            shapes.append(f"{k}:{type(v).__name__}")
                    except Exception:
                        shapes.append(f"{k}:<uninspectable>")

                if "video_grid_thw" in inputs:
                    vg = inputs["video_grid_thw"].detach().cpu().tolist()
                    # vg is typically [[T, H_grid, W_grid]]
                    if isinstance(vg, list) and len(vg) > 0 and len(vg[0]) >= 3:
                        t, h_grid, w_grid = int(vg[0][0]), int(vg[0][1]), int(vg[0][2])
            
                # Verify pixel_values were created
                assert "pixel_values_videos" in inputs or "pixel_values" in inputs, \
                    f"No pixel_values in processed inputs. Keys: {inputs.keys()}"
                
                # Extract vision embeddings from visual encoder
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    # Qwen2-VL visual encoder expects pixel_values as first positional arg
                    # See: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
                    if "pixel_values_videos" in inputs:
                        # Video input - call visual encoder with pixel_values directly
                        vision_outputs = self.video_model.visual(
                            inputs["pixel_values_videos"],  # First positional argument (hidden_states/pixel_values)
                            grid_thw=inputs["video_grid_thw"]
                        )
                    elif "pixel_values" in inputs:
                        # Image input (fallback)
                        vision_outputs = self.video_model.visual(
                            inputs["pixel_values"],  # First positional argument
                            grid_thw=inputs["image_grid_thw"]
                        )
                    else:
                        raise ValueError(f"No pixel_values found in inputs. Keys: {inputs.keys()}")
                    
                    # Check vision_outputs validity
                    assert vision_outputs is not None, \
                        f"vision_outputs is None - video processing failed for {video_path}"
                    
                    assert isinstance(vision_outputs, torch.Tensor), \
                        f"vision_outputs is not a tensor: {type(vision_outputs)}"
                        
                    # Handle both 2D [seq_len, hidden_dim] and 3D [batch, seq_len, hidden_dim] shapes
                    if vision_outputs.dim() == 2:
                        # Shape: [seq_len, hidden_dim] - use first token directly
                        video_embedding = vision_outputs[0, :].detach().cpu().to(torch.float32)
                    elif vision_outputs.dim() == 3:
                        # Shape: [batch, seq_len, hidden_dim] - use first token of first batch
                        video_embedding = vision_outputs[0, 0, :].detach().cpu().to(torch.float32)
                    else:
                        raise ValueError(f"Unexpected vision_outputs shape: {vision_outputs.shape}")
                
                del inputs, vision_outputs, image_inputs, video_inputs
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
            return interface.extract_video_features([video_path], self.video_model, )        
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
        - Qwen2-VL: extracts frames from scene, gets single embedding from vision tower
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
        
        elif self.model_name == "qwen2-vl":
            # Qwen2-VL: crea clip usando decord (no ffmpeg, no encoding issues)
            tmp_dir = tempfile.mkdtemp(prefix="qwen2vl_scene_")
            try:
                # Extract frames con decord e salva come video temp
                clip_path = self._create_clip_with_decord(
                    video_path, scene.start_time, scene.end_time, tmp_dir
                )
                video_emb = self._embed_frames(video_path=clip_path)
                
                # Extract keyframes from clip
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

        if self.model_name == "qwen2-vl":
            fps_adaptive = self._compute_adaptive_fps(video_path, max_frames_allowed=42, margin=0)
            logging.info(f"[VideoEncoder] adaptive fps (full video) for {video_path}: {fps_adaptive:.3f}")
            video_emb = self._embed_frames(video_path=video_path, adaptive_fps=fps_adaptive)
            return {
                "video": torch.from_numpy(video_emb),
                "keyframes": np.array([])
            }
        elif self.model_name == "internvideo2":
            logging.info("[VideoEncoder] embedding the whole video")
            video_emb = self._embed_frames(video_path=video_path)
            return {
                "video": torch.from_numpy(video_emb),
                "keyframes": np.array([])
            }
