import torch
from decord import VideoReader, cpu
import gc
import numpy as np
import logging
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import subprocess, tempfile, uuid, os, shutil
from configuration.config import CONFIG

from .base_encoder import BaseEncoder

class VisualCaptioner(BaseEncoder):
    """
    Generates textual captions from video scenes using Qwen2-VL.
    """
    def __init__(self, device: str = "cuda", max_k_clusters: int = 5):
        super().__init__(device)
        self.max_k_clusters = max_k_clusters
        self.model_id = CONFIG.indexing.caption.caption2_model_id
        
        # Models to be loaded
        self.processor = None
        self.model = None

    def load_models(self):
        logging.info(f"[{self.__class__.__name__}] Loading Qwen2-VL models...")
        
        # HuggingFace automatically uses HF_HOME/TRANSFORMERS_CACHE if set
        cache_dir = os.environ.get("HF_HOME", os.environ.get("TRANSFORMERS_CACHE"))
        if cache_dir:
            logging.info(f"Using cache directory: {cache_dir}")
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=256*28*28,
            max_pixels=1280*28*28
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        logging.info(f"[{self.__class__.__name__}] Models loaded.")


    def _extract_scene_clip(self, video_path: str, start_time: float, end_time: float, tmp_dir: str) -> str:
        """
        Crea una clip temporanea della scena usando ffmpeg di imageio-ffmpeg (ha libx264!)
        """
        import imageio_ffmpeg
        os.makedirs(tmp_dir, exist_ok=True)
        out_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.mp4")

        # Get ffmpeg binary from imageio-ffmpeg (has libx264 enabled!)
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        
        # Calculate duration instead of end_time to avoid seeking issues
        duration = end_time - start_time
        
        # Validate times
        if end_time <= start_time or duration <= 0.01:
            logging.error(f"Invalid scene times: start={start_time}, end={end_time}, duration={duration}")
            raise ValueError("Invalid scene times for clip extraction")

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

        proc = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            stderr = proc.stderr.decode(errors="ignore") if proc.stderr is not None else ""
            logging.error(f"ffmpeg failed when creating caption clip: {stderr}")
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=None, stderr=proc.stderr)

        return out_path

    def encode(self, video_path: str, scene, prompt: str = "Describe this scene briefly.", fps: float | None = None) -> str:
        """
        Genera una caption per una singola scena partendo dal video path usando Qwen2-VL.
        'scene' è un oggetto Scene con start_time/end_time (in secondi).
        """
        # Verify model and processor are loaded
        if self.model is None:
            error_msg = f"[Caption][{scene.scene_id if scene else 'unknown'}] Model is None! Call load_models() or share model first."
            logging.error(error_msg)
            return ""
        
        if self.processor is None:
            error_msg = f"[Caption][{scene.scene_id if scene else 'unknown'}] Processor is None! Call load_models() first."
            logging.error(error_msg)
            return ""
        
        if scene is None:
            logging.warning("encode_scene: scena None.")
            return ""

        tmp_dir = tempfile.mkdtemp(prefix="qwen2vl_caption_")
        try:
            # Crea clip temporanea dell'intera scena
            clip_path = self._extract_scene_clip(video_path, scene.start_time, scene.end_time, tmp_dir)
            
            # Decide fps: use provided fps (from caller) or default to 1.0 for scene captioning
            fps_to_use = fps if fps is not None else 1.0

            # Minimal approach: let processor load the video directly with chosen fps
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": clip_path, "max_pixels": 360 * 420, "fps": fps_to_use},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            with torch.inference_mode():
                # Apply chat template
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                # Let process_vision_info load and process the video efficiently
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.device)

                # DEBUG: log keys and shapes of tensors passed to the model to diagnose large allocations
                try:
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

                    # Also log grid tensors if present (useful for Qwen2-VL)
                    extra = []
                    if hasattr(inputs, "get"):
                        try:
                            if "video_grid_thw" in inputs:
                                extra.append(f"video_grid_thw:{inputs['video_grid_thw']}")
                        except Exception:
                            pass
                        try:
                            if "image_grid_thw" in inputs:
                                extra.append(f"image_grid_thw:{inputs['image_grid_thw']}")
                        except Exception:
                            pass

                    logging.info(f"[Caption][DEBUG] processed inputs: {', '.join(shapes + extra)}")
                except Exception:
                    logging.exception("[Caption][DEBUG] failed to log processed input shapes")
                
                # Generate caption
                try:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        generated_ids = self.model.generate(**inputs, max_new_tokens=64)

                    # Trim input tokens from generated output
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]

                    caption = self.processor.batch_decode(
                        generated_ids_trimmed, 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=False
                    )[0].strip()
                except TypeError as e:
                    # Known potential mismatch in underlying visual forward signature
                    # (e.g. missing hidden_states). Log full error and return empty caption.
                    logging.error(f"Model.generate failed with TypeError: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
                    return ""
                
                # Explicit cleanup to free memory
                del inputs, generated_ids, generated_ids_trimmed, image_inputs, video_inputs
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            return caption
            
        except Exception as e:
            logging.error(f"[Caption][{scene.scene_id}] failed: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return ""
        finally:
            # Cleanup temp directory
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

## --- TEMPORARY TEST --- ##

if __name__ == "__main__":
    import glob
    import os
    from tqdm import tqdm
    from data.video_dataset import Scene

    logging.basicConfig(level=logging.INFO)

    video_dir = "../ego4d_data/v2/full_scale"
    video_ids = glob.glob(os.path.join(video_dir, "*.mp4"))
    print(f"Found {len(video_ids)} videos")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = "Describe this scene briefly."

    for video in tqdm(video_ids[9:]):
        print("-" * 50)
        print(f"Encoding video {video}")
        print("-" * 50)

        captioner = VisualCaptioner(device=device)
        captioner.load_models()

        # Esempio: scena 0–15s
        scene = Scene(scene_id="scene_0", start_time=0.0, end_time=15.0)

        caption = captioner.encode(video, scene, prompt=prompt)
        print("\n=== Generated Caption ===")
        print(caption)
        print("=========================\n")

        del captioner
        if device == "cuda":
            torch.cuda.empty_cache()