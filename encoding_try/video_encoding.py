import scenedetect
import torch
import os.path as osp
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, XCLIPModel, XCLIPProcessor
import whisper
from sentence_transformers import SentenceTransformer
from moviepy.editor import VideoFileClip, concatenate_videoclips, ColorClip, AudioClip, AudioFileClip
from scenedetect import detect, ContentDetector
from decord import VideoReader, cpu
import logging
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import tempfile
import threading
from utils.logging_formatter import LevelAwareFormatter
import subprocess
import torch.nn.functional as F
from transformers import logging as hf_logging
from data.dataclass import SceneClip

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(LevelAwareFormatter())
logging.basicConfig(
    level=logging.INFO,
    handlers=[handler],
)
hf_logging.set_verbosity_error()  
hf_logging.disable_progress_bar() 

class MultiModalEncoder:
    def __init__(
        self,
        video_path: str = "../../data/",
        video_encoder: str = "microsoft/xclip-large-patch14",
        audio_encoder: str = "laion/clap-htsat-unfused",
        image_encoder: str = "openai/clip-vit-large-patch14",
        text_encoder: str = "all-MiniLM-L6-v2",
        device: str ='cuda'
    ):
        self.device = device
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            logging.warning("CUDA not available, switching to CPU.")
        else:
            logging.info(f"Using device: {self.device}")
        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.video_path = video_path
        self._load_videos()

    def _load_videos(self):
        video_files = []
        for file in Path(self.video_path).glob("*.mp4"):
            video_files.append(str(file))
        self.video_files = video_files

    def load_model(self):
        stages = [
            ("Video", "microsoft/xclip-large-patch14", XCLIPProcessor, XCLIPModel),
            ("Image", "openai/clip-vit-large-patch14", AutoProcessor, AutoModel),
            ("Audio", "laion/clap-htsat-unfused", AutoProcessor, AutoModel),
            ("Text (Whisper)", "whisper", None, whisper),
            ("Text (SentenceTransformer)", "all-MiniLM-L6-v2", None, SentenceTransformer),
        ]
        logging.info("Loading models:")
        for name, model_id, proc_class, model_class in stages:
            if name == "Video":
                self.video_processor = proc_class.from_pretrained(model_id)
                self.video_model = model_class.from_pretrained(model_id).to(self.device)
            elif name == "Image":
                self.token_processor = proc_class.from_pretrained(model_id)
                self.token_model = model_class.from_pretrained(model_id).to(self.device)
                self.token_model = self.token_model.to(dtype=torch.float16)
            elif name == "Audio":
                self.audio_processor = proc_class.from_pretrained(model_id)
                self.audio_model = model_class.from_pretrained(model_id).to(self.device)
            elif name == "Text (Whisper)":
                self.asr_model = model_class.load_model("base").to(self.device)
            elif name == "Text (SentenceTransformer)":
                self.text_embedder = model_class(model_id)

        logging.info("All models loaded successfully!")


    def _encode_videos(self):
        all_embeddings = []
        for video_path in tqdm(self.video_files, desc="Encoding videos"):
            scenes = self._get_scenes(video_path)
            corpus_data = []
            logging.info(f"Processing {len(scenes)} scenes in video: {video_path} on device: {self.device}")

            with ThreadPoolExecutor(max_workers = min(4, os.cpu_count())) as executor:
                futures = {
                    executor.submit(self.process_scene, video_path, start.get_seconds(), end.get_seconds()): i
                    for i, (start, end) in enumerate(scenes)
                }
                for f in tqdm(as_completed(futures), total = len(futures), desc = "Scenes processed"):
                    i = futures[f]
                    try:
                        res = f.result()
                        if res:
                            res["scene_id"] = f"scene_{i}"
                            corpus_data.append(res)
                        else:
                            logging.warning(f"[SKIP] Scene {i} didn't return anything")
                    except Exception as e:
                        logging.error(f"[ERROR] Scene {i} processing failed: {e}")
            all_embeddings.append(corpus_data)
        return all_embeddings

    def process_scene(self, video_path, start_time, end_time):
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = vr.get_avg_fps()
            start_frame, end_frame = int(start_time * fps), int(end_time * fps)
            frame_indices = np.linspace(start_frame, end_frame - 1, num=8, dtype=int)
            
            frames_batch = vr.get_batch(frame_indices).asnumpy()
            sampled_frames = [frame for frame in frames_batch]
            video_inputs = self.video_processor(images=sampled_frames, return_tensors="pt").to(self.device)

            with torch.inference_mode():
                dummy_text = ""
                text_inputs = self.video_processor(text=dummy_text, return_tensors="pt").to(self.device)

                outputs = self.video_model(
                    pixel_values=video_inputs['pixel_values'],
                    input_ids=text_inputs['input_ids'],
                    attention_mask=text_inputs['attention_mask']
                )

                pooled_video_embedding = outputs.video_embeds

            with VideoFileClip(video_path) as video:
                scene_clip = video.subclip(start_time, end_time)
                keyframe_indices = np.linspace(start_frame, end_frame - 1, num=3, dtype=int)
                keyframes = vr.get_batch(keyframe_indices).asnumpy()
                token_inputs = self.token_processor(images=list(keyframes), return_tensors="pt").to(self.device)
                token_inputs = {k: v.half().to(self.device) for k, v in token_inputs.items()}
                token_embeddings = self.token_model.get_image_features(**token_inputs)

                if scene_clip.duration > 0 and scene_clip.audio is not None:
                    if not hasattr(self, "_asr_lock"):
                        self._asr_lock = threading.Lock()
                    try:
                        os.makedirs("temp_audio", exist_ok=True)
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="temp_audio") as tmp:
                            temp_audio_path = tmp.name
                        scene_clip.audio.write_audiofile(temp_audio_path, fps=48000, logger=None)

                        audio_array, _ = librosa.load(temp_audio_path, sr=48000, mono=True)
                        audio_inputs = self.audio_processor(
                            audio=audio_array,
                            sampling_rate=48000,
                            return_tensors="pt"
                        ).to(self.device)
                        audio_embedding = self.audio_model.get_audio_features(**audio_inputs)

                        with self._asr_lock:
                            transcript = self.asr_model.transcribe(
                                temp_audio_path, fp16=True, language="en"
                            )["text"]

                        dense_text_embedding = self.text_embedder.encode(transcript)

                    except Exception as e:
                        logging.error(f"[AUDIO ERROR] scene {start_time:.2f}-{end_time:.2f}s: {e}")
                        audio_embedding = torch.zeros((1, 512))
                        transcript = ""
                        dense_text_embedding = self.text_embedder.encode("")
                    finally:
                        if "temp_audio_path" in locals() and os.path.exists(temp_audio_path):
                            try:
                                os.remove(temp_audio_path)
                            except Exception as e:
                                logging.warning(f"Could not delete {temp_audio_path}: {e}")

                else:
                    audio_embedding = torch.zeros((1, 512))
                    transcript = ""
                    dense_text_embedding = self.text_embedder.encode("")

            clip = SceneClip(
                video_path=video_path,
                start_time=start_time,
                end_time=end_time,
                embeds = {
                    "pooled_video": pooled_video_embedding.cpu().detach().numpy(),
                    "visual_tokens": token_embeddings.cpu().squeeze().detach().numpy(),
                    "audio": audio_embedding.cpu().detach().numpy(),
                    "transcript": transcript,
                    "dense_text": dense_text_embedding
                }
            )

            clip.save_to_pickle()

            return clip.embeds()

        except Exception as e:
            logging.error(f"\n--- ERROR processing scene from {start_time:.2f}s to {end_time:.2f}s ---")
            import traceback
            traceback.print_exc()
            return None
        finally:
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    def _get_scenes(self, video_path):
        return detect(video_path, ContentDetector())
    

if __name__ == "__main__":
    encoder = MultiModalEncoder()
    encoder.video_files = [f for f in encoder.video_files if "animals" not in f and "AI" not in f]
    logging.info(f"Found {len(encoder.video_files)} video(s) in '{encoder.video_path}': {encoder.video_files}")

    encoder.load_model()
    video_embeddings = encoder._encode_videos()

    pooled_video_embs = []
    video_names = []

    for vid_path, vid_data in zip(encoder.video_files, video_embeddings):
        if not vid_data:
            continue
        video_embs = [torch.tensor(scene["pooled_video"], dtype=torch.float32) for scene in vid_data]
        video_emb = torch.stack(video_embs).mean(dim=0)
        pooled_video_embs.append(video_emb)
        video_names.append(Path(vid_path).stem)

    pooled_video_embs = torch.stack(pooled_video_embs)
    pooled_video_embs = F.normalize(pooled_video_embs, dim=-1)

    queries = [
        "Who won the tennis match?",
        "What minute was the goal scored?",
        "Can Sinner hold the lead?",
        "Who was the best player of the football match?"
    ]

    for query in queries:
        text_inputs = encoder.video_processor(text=query, return_tensors="pt").to(encoder.device)

        with torch.no_grad():
            text_emb = encoder.video_model.get_text_features(**text_inputs).to(encoder.device)
            text_emb = F.normalize(text_emb, dim=-1).squeeze(0)

        sims = torch.matmul(pooled_video_embs.to("cpu"), text_emb.to("cpu"))
        best_idx = torch.argmax(sims).item()
        best_score = sims[best_idx].item()

        print(f"\n Query: {query}")
        print(f"Most similar video: {video_names[best_idx]}  (similarity = {best_score:.4f})")



if __name__ == "__main2__":
    encoder = MultiModalEncoder()
    # SKIP for now longest video
    encoder.video_files = [f for f in encoder.video_files if "animals" not in f]
    logging.info(f"Found {len(encoder.video_files)} video(s) in '{encoder.video_path}': {encoder.video_files}")
    encoder.load_model()
    logging.info("Models loaded successfully.")
    video_embeddings = encoder._encode_videos()
    
    if video_embeddings and video_embeddings[0]:
        logging.info("--- SUCCESS ---")
        logging.info("Pooled video embedding shape: %s", video_embeddings[0][0]["pooled_video"].shape)
        logging.info("Visual tokens embedding shape: %s", video_embeddings[0][0]["visual_tokens"].shape)
        logging.info("Audio embedding shape: %s", video_embeddings[0][0]["audio"].shape)
        logging.info("Transcript: '%s'", video_embeddings[0][0]["transcript"])
        logging.info("Dense text embedding shape: %s", video_embeddings[0][0]["dense_text"].shape)
        logging.info("="*50 + "\n")
    else:
        logging.error("--- ERROR: No embeddings were generated. Check the error messages above. ---")


# Code to test scene  detection
if __name__ == "__main3__":
    encoder = MultiModalEncoder()
    encoder.video_files = [f for f in encoder.video_files if "animals" not in f and "prova" not in f]
    if not encoder.video_files:
        logging.error("❌ Nessun video trovato nella cartella specificata.")
        exit(1)

    # 🔹 Prendi il primo video
    video_path = encoder.video_files[0]
    logging.info(f"🎬 Analizzo: {video_path}")

    # 🔹 Rileva le scene con la funzione della classe
    scenes = encoder._get_scenes(video_path)

    # 🔹 Stampa i risultati
    logging.info(f"📸 Scene trovate: {len(scenes)}")
    for i, (start, end) in enumerate(scenes):
        start_s, end_s = start.get_seconds(), end.get_seconds()
        logging.info(f"[{i:03d}] {start_s:.2f}s → {end_s:.2f}s  (durata: {end_s - start_s:.2f}s)")