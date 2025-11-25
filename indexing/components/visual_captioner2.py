import torch
import numpy as np
import logging
import re
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from indexing.utils.clustering import cluster_frames
import subprocess, tempfile, uuid, os, shutil
import os
import torchvision
import av

os.environ.setdefault("TORCHVISION_DISABLE_TORCHCODEC", "1")
try:
    torchvision.set_video_backend("pyav")
except Exception:
    pass

def read_video_pyav(path, num_frames=16):
    container = av.open(path)
    stream = container.streams.video[0]

    frames = []
    indices = set(np.linspace(0, max(stream.frames-1, 0), num_frames, dtype=int))

    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) == num_frames:
            break

    container.close()
    return np.stack(frames)

from .base_encoder import BaseEncoder

class VisualCaptioner(BaseEncoder):
    """
    Generates textual captions from video frames using BLIP.
    It clusters frames to find key visual moments and captions them.
    """
    def __init__(self, device: str = "cuda", max_k_clusters: int = 5, model_id: str = "llava-hf/LLaVA-NeXT-Video-7B-hf"):
        super().__init__(device)
        self.max_k_clusters = max_k_clusters
        self.model_id = model_id
        
        # Models to be loaded
        self.processor: LlavaNextVideoProcessor = None
        self.model: LlavaNextVideoForConditionalGeneration = None

    def load_models(self):
        logging.info(f"[{self.__class__.__name__}] Loading LLaVA models...")
        self.processor = LlavaNextVideoProcessor.from_pretrained(self.model_id)
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(self.model_id, torch_dtype=torch.float16 if self.device=="cuda" else torch.float32).to(self.device)
        try:
            self.processor.tokenizer.padding_side = "left"
        except Exception:
            pass
        logging.info(f"[{self.__class__.__name__}] Models loaded.")


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

    def encode(self, video_path: str, scene, prompt: str = "Describe this video clip briefly.") -> str:
        """
        Genera una caption per una singola scena partendo dal video path.
        'scene' è un oggetto Scene con start_time/end_time (in secondi).
        """
        if scene is None:
            logging.warning("encode_scene: scena None.")
            return ""

        tmp_root = tempfile.mkdtemp(prefix="vidcap_")
        clip_path = None
        try:
            clip_path = self._extract_scene_clip(video_path, scene.start_time, scene.end_time, tmp_root)

            video_np = read_video_pyav(clip_path, num_frames=16)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video"},   # ← NOTE: no path here
                ],
            }]

            with torch.inference_mode():
                prompt_text = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                )

                proc_inputs = self.processor(
                    text=prompt_text,
                    videos=video_np,
                    return_tensors="pt",
                    num_frames=16,
                )

                proc_inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in proc_inputs.items()}

                out = self.model.generate(**proc_inputs, max_new_tokens=512)
                text = self.processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()

                # Extract only the assistant reply if the processor/model returns a
                # chat-style string containing USER: ... ASSISTANT: ...
                assistant_text = text
                m = re.search(r"assistant:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
                if m:
                    assistant_text = m.group(1).strip()
                else:
                    # fallback: split on assistant marker and take last chunk
                    parts = re.split(r"assistant:\s*", text, flags=re.IGNORECASE)
                    if parts and len(parts) > 1:
                        assistant_text = parts[-1].strip()

                # Remove trailing visual separators (lines of dashes) if present
                assistant_text = re.sub(r"\n[-]{3,}.*$", "", assistant_text, flags=re.DOTALL).strip()

                print(f"{'-'*80}\n caption \n {assistant_text} \n {'-'*80}")
                return assistant_text
        except Exception as e:
            logging.error(f"[Caption][{scene.scene_id}] failed: {e}")
            return ""
        finally:
            try:
                shutil.rmtree(tmp_root, ignore_errors=True)
            except Exception:
                pass

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

    for video in tqdm(video_ids[6:7]):
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