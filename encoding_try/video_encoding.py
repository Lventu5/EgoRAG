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
from moviepy.editor import VideoFileClip
from scenedetect import detect, ContentDetector
from decord import VideoReader, cpu
import logging
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class MultiModalEncoder:
    def __init__(
        self,
        video_path: str = "../../data/",
        video_encoder: str = "microsoft/xclip-large-patch14",
        audio_encoder: str = "laion/clap-htsat-unfused",
        image_encoder: str = "openai/clip-vit-large-patch14",
        text_encoder: str = "all-MiniLM-L6-v2",
        device='cuda'
    ):
        self.device = device
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            logging.warning("CUDA not available, switching to CPU.")
        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.video_path = video_path

    def _load_videos(self):
        video_files = []
        for file in Path(self.video_path).glob("*.mp4"):
            video_files.append(str(file))
        self.video_files = video_files

    def _load_model(self):
        from datetime import datetime
        stages = [
            ("Video", "microsoft/xclip-large-patch14", XCLIPProcessor, XCLIPModel),
            ("Image", "openai/clip-vit-large-patch14", AutoProcessor, AutoModel),
            ("Audio", "laion/clap-htsat-unfused", AutoProcessor, AutoModel),
            ("Text (Whisper)", "whisper", None, whisper),
            ("Text (SentenceTransformer)", "all-MiniLM-L6-v2", None, SentenceTransformer),
        ]

        print("🔹 Loading models:")
        pbar = tqdm(stages, desc="Initializing encoders", ncols=100)

        for name, model_id, proc_class, model_class in pbar:
            start = time.time()
            pbar.set_description(f"Loading {name} model")
            if name == "Video":
                self.video_processor = proc_class.from_pretrained(model_id)
                self.video_model = model_class.from_pretrained(model_id).to(self.device)
            elif name == "Image":
                self.token_processor = proc_class.from_pretrained(model_id)
                self.token_model = model_class.from_pretrained(model_id).to(self.device)
            elif name == "Audio":
                self.audio_processor = proc_class.from_pretrained(model_id)
                self.audio_model = model_class.from_pretrained(model_id).to(self.device)
            elif name == "Text (Whisper)":
                self.asr_model = model_class.load_model("base").to(self.device)
            elif name == "Text (SentenceTransformer)":
                self.text_embedder = model_class(model_id)

            end = time.time()
            print(f"{name} loaded in {end - start:.2f}s at {datetime.now().strftime('%H:%M:%S')}")

        print("All models loaded successfully!")


    def _encode_video(self):
        all_embeddings = []
        for video_path in tqdm(self.video_files[0:1], desc="Encoding videos"):
            scenes = self._get_scenes(video_path)
            corpus_data = []
            logging.info(f"Processing {len(scenes)} scenes in video: {video_path} on device: {self.device}")

            with ThreadPoolExecutor(max_workers = min(8, os.cpu_count())) as executor:
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
                            logging.info(f"[COMPLETED] Scene {i}")
                        else:
                            logging.warning(f"[SKIP] Scene {i}")
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
                token_embeddings = self.token_model.get_image_features(**token_inputs)

                if scene_clip.duration > 0 and scene_clip.audio is not None:
                    temp_audio_path = "temp_audio.wav"
                    scene_clip.audio.write_audiofile(temp_audio_path, fps=48000, logger=None)
                    audio_array, _ = librosa.load(temp_audio_path, sr=48000, mono=True)
                    os.remove(temp_audio_path)
                    audio_inputs = self.audio_processor(audio=audio_array, sampling_rate=48000, return_tensors="pt").to(self.device)
                    audio_embedding = self.audio_model.get_audio_features(**audio_inputs)
                    temp_audio_path = "temp_audio.wav"
                    scene_clip.audio.write_audiofile(temp_audio_path, logger=None)
                    transcript = self.asr_model.transcribe(temp_audio_path, fp16 = True, language = "en", device = self.device)["text"]
                    dense_text_embedding = self.text_embedder.encode(transcript)
                    os.remove(temp_audio_path)
                else:
                    audio_embedding = torch.zeros((1, 512))
                    transcript = ""
                    dense_text_embedding = self.text_embedder.encode("")

            return {
                "pooled_video": pooled_video_embedding.cpu().detach().numpy(),
                "visual_tokens": token_embeddings.cpu().squeeze().detach().numpy(),
                "audio": audio_embedding.cpu().detach().numpy(),
                "transcript": transcript,
                "dense_text": dense_text_embedding
            }
        except Exception as e:
            logging.error(f"\n--- ERROR processing scene from {start_time:.2f}s to {end_time:.2f}s ---")
            import traceback
            traceback.print_exc()
            return None

    def _get_scenes(self, video_path):
        return detect(video_path, ContentDetector())

if __name__ == "__main__":
    encoder = MultiModalEncoder()
    encoder._load_videos()
    # SKIP for now longest video
    encoder.video_files = [f for f in encoder.video_files if "animals" not in f]
    logging.info(f"Found {len(encoder.video_files)} video(s) in '{encoder.video_path}': {encoder.video_files}")
    encoder._load_model()
    logging.info("Models loaded successfully.")
    video_embeddings = encoder._encode_video()
    
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