import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import librosa
import logging
import tempfile
from tqdm import tqdm
from decord import VideoReader, cpu
from sklearn.cluster import KMeans
from moviepy.editor import VideoFileClip
from transformers import AutoModel, AutoProcessor, XCLIPModel, XCLIPProcessor
from sentence_transformers import SentenceTransformer
from scenedetect import detect, ContentDetector
import whisper

from video_dataset import VideoDataset, VideoDataPoint, Scene
from utils.logging_formatter import LevelAwareFormatter
from transformers import logging as hf_logging
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        video_dataset: VideoDataset,
        device: str = "cuda",
        max_frames_per_scene: int = 96,
        max_temporal_segments: int = 8
    ):
        if video_dataset is None or len(video_dataset) == 0:
            raise ValueError("Video dataset is empty or not provided.")
        self.dataset = video_dataset
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_frames_per_scene = max_frames_per_scene
        self.max_temporal_segments = max_temporal_segments
        logging.info(f"[DEVICE] Using: {self.device}")


    def load_models(self):
        logging.info("[MODELS] Loading...")

        self.video_processor = XCLIPProcessor.from_pretrained("microsoft/xclip-large-patch14", local_files_only=True)
        self.video_model = XCLIPModel.from_pretrained("microsoft/xclip-large-patch14", local_files_only=True).to(self.device)

        self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336", local_files_only=True)
        self.image_model = AutoModel.from_pretrained("openai/clip-vit-large-patch14-336", local_files_only=True).to(self.device)

        self.audio_processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused", local_files_only=True)
        self.audio_model = AutoModel.from_pretrained("laion/clap-htsat-unfused", local_files_only=True).to(self.device)

        self.asr_model = whisper.load_model("base").to(self.device)
        self.text_embedder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True).to(self.device)

        logging.info("[MODELS] Loaded successfully.")


    def encode_videos(self):
        for dp in tqdm(self.dataset.video_datapoints, desc="Encoding videos"):
            logging.info(f"\n=== VIDEO: {dp.video_name} ===")

            self._extract_scenes(dp)
            if not dp.scenes:
                logging.warning(f"[WARN] Nessuna scena trovata in {dp.video_name}, skip.")
                continue
            else:
                logging.info(f"[INFO] {len(dp.scenes)} scene trovate in {dp.video_name}.")

            scene_embeds_video, scene_embeds_audio, scene_embeds_text = [], [], []

            max_workers = min(4, os.cpu_count() or 2)
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for i, scene in enumerate(dp.scenes):
                    logging.info(f"[SCENE] Encoding scena {i}: frames {scene.start_frame}-{scene.end_frame}, time {scene.start_time:.2f}-{scene.end_time:.2f}s")
                    sid = f"scene_{i}"
                    futures[executor.submit(self._encode_scene, dp.video_path, scene)] = sid

                for f in tqdm(as_completed(futures), total=len(futures), desc=f"Scenes ({dp.video_name})"):
                    sid = futures[f]
                    try:
                        scene_out = f.result()
                        if scene_out:
                            dp.scene_embeddings[sid] = scene_out
                            scene_embeds_video.append(scene_out["video"])
                            scene_embeds_audio.append(scene_out["audio"])
                            scene_embeds_text.append(scene_out["text"])
                        else:
                            logging.warning(f"[SKIP] scena {sid} vuota.")
                    except Exception as e:
                        logging.error(f"[ERROR] scena {sid} fallita: {e}")

            if scene_embeds_video:
                dp.global_embeddings["video"] = torch.stack(scene_embeds_video).mean(dim=0)
            if scene_embeds_audio:
                dp.global_embeddings["audio"] = torch.stack(scene_embeds_audio).mean(dim=0)
            if scene_embeds_text:
                dp.global_embeddings["text"] = torch.stack(scene_embeds_text).mean(dim=0)

        return self.dataset


    def _extract_scenes(self, dp: VideoDataPoint):
        scene_list = detect(dp.video_path, ContentDetector())
        dp.scenes = []
        for start, end in scene_list:
            dp.scenes.append(
                Scene(
                    start_time=start.get_seconds(),
                    end_time=end.get_seconds(),
                    start_frame=start.get_frames(),
                    end_frame=end.get_frames()
                )
            )
        logging.info(f"[SCENE DETECTION] Found {len(dp.scenes)} scenes.")


    def _encode_scene(self, video_path: str, scene: Scene):
        frames, frame_idxs = self._extract_frames(video_path, scene.start_frame, scene.end_frame, self.max_frames_per_scene)
        
        if len(frames) == 0:
            logging.warning(f"No frames extracted for scene starting at {scene.start_time:.2f}s. Skipping.")
            return None

        frame_embs = self._embed_frames_clip(frames)
        clusters = self._cluster_frames(frame_embs)
        
        scene_video_emb = self._embed_scene_video_with_representatives(frames, clusters)

        image_clusters = self._embed_image_clusters(frames, frame_embs, clusters)
        audio_emb, transcript, text_emb = self._encode_audio_and_text(video_path, scene.start_time, scene.end_time)

        return {
            "video": torch.tensor(scene_video_emb, dtype=torch.float32),
            "audio": torch.tensor(audio_emb, dtype=torch.float32),
            "text": torch.tensor(text_emb, dtype=torch.float32),
            "transcript": transcript,
            "image": image_clusters,
            "meta": {
                "num_segments": len(clusters)
            }
        }


    def _extract_frames(self, video_path, start_frame=None, end_frame=None, max_frames=48):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)

        if start_frame is None:
            start_frame = 0
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames - 1

        num_frames_in_scene = end_frame - start_frame
        if num_frames_in_scene <= 0:
            return np.array([]), np.array([])

        num = min(max_frames, num_frames_in_scene)
        idxs = np.linspace(start_frame, end_frame - 1, num=num, dtype=int)

        frames = vr.get_batch(idxs).asnumpy()
        return frames, idxs


    def _embed_frames_clip(self, frames: np.ndarray) -> np.ndarray:
        inputs = self.image_processor(images=list(frames), return_tensors="pt").to(self.device)
        with torch.no_grad():
            img_embs = self.image_model.get_image_features(**inputs)
        return img_embs.cpu().numpy()


    def _choose_k(self, n_frames: int) -> int:
        k = (n_frames // 8) + 1
        return min(k, self.max_temporal_segments)
    

    def _cluster_frames(self, frame_embs: np.ndarray):
        n = len(frame_embs)
        if n < 2:
            return {0: list(range(n))}

        k = self._choose_k(n)
        if k <= 1:
            return {0: list(range(n))}

        km = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(frame_embs)
        clusters = {}
        for i, lab in enumerate(km.labels_):
            clusters.setdefault(int(lab), []).append(i)
        return clusters


    def _embed_scene_video_with_representatives(self, frames: np.ndarray, clusters: dict[int, list[int]]) -> np.ndarray:
        segment_embeddings = []
        
        sorted_cids = sorted(clusters.keys())

        for cid in sorted_cids:
            local_indices = sorted(clusters[cid])
            if not local_indices:
                continue

            segment_frames = frames[local_indices]
            
            n_frames_expected = 8
            current_n_frames = len(segment_frames)
            
            if current_n_frames == 0:
                continue

            if current_n_frames < n_frames_expected:
                last_frame = segment_frames[-1]
                padding = [last_frame] * (n_frames_expected - current_n_frames)
                final_frames = np.concatenate([segment_frames, padding], axis=0)
            elif current_n_frames > n_frames_expected:
                indices = np.linspace(0, current_n_frames - 1, num=n_frames_expected, dtype=int)
                final_frames = segment_frames[indices]
            else:
                final_frames = segment_frames
            
            resized_frames = [cv2.resize(f, (224, 224)) for f in final_frames]

            inputs = self.video_processor(images=resized_frames, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.video_model(pixel_values=inputs["pixel_values"])
                segment_emb = out.video_embeds.squeeze(0)
                segment_embeddings.append(segment_emb)
        
        if not segment_embeddings:
            logging.warning("[VIDEO] No segments could be processed, returning zeros.")
            return np.zeros(512, dtype=np.float32)

        final_embedding = torch.stack(segment_embeddings).mean(dim=0)
        return final_embedding.cpu().numpy()


    def _embed_image_clusters(self, frames: np.ndarray, frame_embs: np.ndarray, clusters: dict[int, list[int]]):
        result = {}
        for cid, idxs in clusters.items():
            if not idxs:
                continue
            
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
                img_emb = self.image_model.get_image_features(**inputs).cpu().numpy().squeeze()

            result[f"cluster_{cid}"] = {
                "frame_idx_local": int(best_local),
                "frame_embedding": torch.tensor(img_emb, dtype=torch.float32),
            }
        return result


    def _encode_audio_and_text(self, video_path, start_t, end_t):
        try:
            with VideoFileClip(video_path) as vid:
                if start_t >= vid.duration or start_t >= end_t:
                    raise ValueError("Invalid time range for audio extraction.")
                
                clip = vid.subclip(start_t, end_t)
                
                if clip.audio is None:
                    raise ValueError("No audio found in the subclip.")

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    temp_path = tmp.name
                
                clip.audio.write_audiofile(temp_path, fps=48000, logger=None)

                audio_array, _ = librosa.load(temp_path, sr=48000, mono=True)
                if audio_array.size == 0:
                    raise ValueError("Empty audio array after loading.")
                
                inputs_audio = self.audio_processor(audio=audio_array, sampling_rate=48000, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    audio_emb = self.audio_model.get_audio_features(**inputs_audio).cpu().numpy().squeeze()

                transcript_result = self.asr_model.transcribe(temp_path)
                transcript = transcript_result["text"]
                text_emb = self.text_embedder.encode(transcript)

                os.remove(temp_path)
                return audio_emb, transcript, text_emb

        except Exception as e:
            logging.error(f"[AUDIO/TEXT] Failed for clip {os.path.basename(video_path)} ({start_t:.2f}s-{end_t:.2f}s) with error: {e}")
            audio_dim = self.audio_model.config.hidden_size
            text_dim = self.text_embedder.get_sentence_embedding_dimension()
            return np.zeros(audio_dim, dtype=np.float32), "", np.zeros(text_dim, dtype=np.float32)


if __name__ == "__main__":
    import sys

    logging.info("🚀 Starting MultiModalEncoder test run...")
    
    # Assicurati che le directory di supporto esistano
    if not os.path.exists("video_dataset.py") or not os.path.exists("utils/logging_formatter.py"):
         print("❌ Assicurati che i file 'video_dataset.py' e 'utils/logging_formatter.py' siano presenti.")
         sys.exit(1)

    data_dir = "../../data"
    if not os.path.exists(data_dir):
        logging.error(f"❌ Directory {os.path.abspath(data_dir)} non trovata.")
        sys.exit(1)

    video_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi"))
    ]

    if not video_files:
        logging.error(f"❌ Nessun video trovato in {data_dir}.")
        sys.exit(1)

    logging.info(f"📁 Trovati {len(video_files)} video:\n" + "\n".join(f" - {f}" for f in video_files))

    dataset = VideoDataset(video_files)
    logging.info(f"✅ Creato VideoDataset con {len(dataset)} elementi.")

    encoder = MultiModalEncoder(video_dataset=dataset, device="cuda")
    encoder.load_models()

    encoded_dataset = encoder.encode_videos()

    if not encoded_dataset.video_datapoints:
        logging.warning("Nessun video è stato processato.")
        sys.exit(0)
        
    first_dp = encoded_dataset.video_datapoints[0]
    if first_dp:
        logging.info("\n=== ✅ RISULTATI PRIMO VIDEO ===")
        logging.info(f"Video path: {first_dp.video_path}")
        logging.info(f"Numero scene: {len(first_dp.scenes)}")

        if first_dp.global_embeddings:
            logging.info(f"Chiavi global embeddings: {list(first_dp.global_embeddings.keys())}")
        
        if first_dp.scene_embeddings:
            scene_keys = list(first_dp.scene_embeddings.keys())
            logging.info(f"Chiavi scene embeddings: {scene_keys[:3]} ...")
            
            if scene_keys:
                first_scene_key = scene_keys[0]
                first_scene = first_dp.scene_embeddings[first_scene_key]
                logging.info(f"--- Esempio scena: {first_scene_key} ---")
                logging.info(f"Video emb shape: {first_scene['video'].shape}")
                logging.info(f"Audio emb shape: {first_scene['audio'].shape}")
                logging.info(f"Text emb shape:  {first_scene['text'].shape}")
                logging.info(f"Transcript: '{first_scene['transcript'][:80]}...'")

    logging.info("✅ Encoding completato con successo!")