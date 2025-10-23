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
from transformers import (
    AutoModel, 
    AutoProcessor, 
    XCLIPModel,
    XCLIPProcessor, 
    CLIPVisionModel,
    WhisperProcessor,
    WhisperForConditionalGeneration
)
from sentence_transformers import SentenceTransformer
from scenedetect import detect, ContentDetector

from data.video_dataset import VideoDataset, VideoDataPoint, Scene
from indexing.utils.logging_formatter import LevelAwareFormatter
from indexing.utils.clustering import choose_k, cluster_frames
from lori.EgoRAG.indexing.components.text_encoder_old import CaptionEncoder
from transformers import logging as hf_logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

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
        max_temporal_segments: int = 8,
        max_workers: int = 2,
    ):
        if video_dataset is None or len(video_dataset) == 0:
            raise ValueError("Video dataset is empty or not provided.")
        self.dataset = video_dataset
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_frames_per_scene = max_frames_per_scene
        self.max_temporal_segments = max_temporal_segments
        self.max_workers = max_workers
        logging.info(f"[DEVICE] Using: {self.device}")


    def load_models(self): # FIXME: Move to __init__
        logging.info("[MODELS] Loading...")
        self.text_encoder = CaptionEncoder(self.device, self.max_frames_per_scene, self.max_temporal_segments)

        self.video_processor = XCLIPProcessor.from_pretrained("microsoft/xclip-large-patch14", local_files_only=True)
        self.video_model = XCLIPModel.from_pretrained("microsoft/xclip-large-patch14", local_files_only=True).to(self.device)

        self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336", local_files_only=True)
        self.image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336", local_files_only=True).to(self.device)

        self.audio_processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused", local_files_only=True)
        self.audio_model = AutoModel.from_pretrained("laion/clap-htsat-unfused", local_files_only=True).to(self.device)
        self.audio_sr = 48000

        self.asr_processor = WhisperProcessor.from_pretrained(
            "openai/whisper-base", 
            local_files_only=True
        )
        self.asr_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-base", 
            local_files_only=True
        ).to(self.device)
        
        # Come da tuo esempio, per assicurarci che faccia la trascrizione
        self.asr_model.config.forced_decoder_ids = None
        self.asr_sr = 16000
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

            futures = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for i, scene in enumerate(dp.scenes):
                    sid = f"scene_{i}"
                    futures[executor.submit(self._encode_scene, dp.video_path, scene)] = sid

                for f in tqdm(as_completed(futures), total=len(futures), desc=f"Scenes ({dp.video_name})"):
                    sid = futures[f]
                    try:
                        scene_out = f.result()
                        if scene_out:
                            dp.scene_embeddings[sid] = scene_out
                            scene_embeds_video.append(scene_out["video"])
                            if scene_out["audio"] is not None:
                                scene_embeds_audio.append(scene_out["audio"])
                            if scene_out["text"] is not None:
                                scene_embeds_text.append(scene_out["text"])
                        else:
                            logging.warning(f"[SKIP] scena {sid} vuota.")
                    except Exception as e:
                        logging.error(f"[ERROR] scena {sid} fallita: {e}")

            if scene_embeds_video:
                dp.global_embeddings["video"] = torch.stack(scene_embeds_video).mean(dim=0)
           
                dims = [e.shape[0] for e in scene_embeds_audio]
                unique_dims = sorted(set(dims))

                if len(unique_dims) > 1:
                    logging.warning(f"[WARN] Dimension mismatch detected in audio embeddings: {unique_dims}")

            if scene_embeds_audio:
                dims = [e.shape[0] for e in scene_embeds_audio]
                unique_dims = sorted(set(dims))
                if len(unique_dims) > 1:
                    logging.warning(f"[WARN] Dimension mismatch detected in audio embeddings: {unique_dims}")
                elif unique_dims and unique_dims[0] != 512:
                     logging.warning(f"[WARN] Audio embedding dimension is {unique_dims[0]}, expected 512.")
                dp.global_embeddings["audio"] = torch.stack(scene_embeds_audio).mean(dim=0)
            else:
                logging.warning(f"[WARN] No valid audio embeddings found for {dp.video_name}. Global audio embedding will be None.")
                dp.global_embeddings["audio"] = None

            if scene_embeds_text:
                dp.global_embeddings["text"] = torch.stack(scene_embeds_text).mean(dim=0)
            else:
                logging.warning(f"[WARN] No valid text embeddings found for {dp.video_name}. Global text embedding will be None.")
                dp.global_embeddings["text"] = None 
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

    def _encode_scene(self, video_path: str, scene: Scene):
        try:
            frames, _ = self._extract_frames(video_path, scene.start_frame, scene.end_frame, self.max_frames_per_scene)
            
            if len(frames) == 0:
                logging.warning(f"No frames extracted for scene starting at {scene.start_time:.2f}s. Skipping.")
                return None

            frame_embs = self._embed_frames_clip(frames)
            clusters = cluster_frames(frame_embs, self.max_temporal_segments)
            k = choose_k(len(frames), self.max_temporal_segments)
            if k <= 1 or len(frames) < 8: # If scene is too short, treat as one chunk
                temporal_chunks = [frames] 
            else:
                # This splits the frames into k *temporally contiguous* groups
                temporal_chunks = np.array_split(frames, k, axis=0)
            
            scene_video_emb = self._embed_temporal_segments(temporal_chunks)

            image_clusters = self._embed_image_clusters(frames, frame_embs, clusters)
            audio_emb, transcript, text_emb = self._encode_audio_and_text(video_path, scene.start_time, scene.end_time)

            audio_tensor = torch.tensor(audio_emb, dtype=torch.float32) if audio_emb is not None else None
            text_tensor = torch.tensor(text_emb, dtype=torch.float32) if text_emb is not None else None

            return {
                "video": torch.tensor(scene_video_emb, dtype=torch.float32),
                "audio": audio_tensor,
                "text": text_tensor,
                "transcript": transcript,
                "image": image_clusters,
                "meta": {
                    "num_segments": len(clusters)
                }
            }
        finally:
            if self.device == "cuda":
                torch.cuda.empty_cache()


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
            outputs = self.image_model(**inputs)
            img_embs = outputs.pooler_output
            
        return img_embs.cpu().numpy()


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
                video_embedding = outputs.video_embeds.squeeze(0)
                segment_embeddings.append(video_embedding)

        if not segment_embeddings:
            # Use the dimension of your XCLIP model, e.g., 768
            return np.zeros(768, dtype=np.float32) 

        video_emb = torch.stack(segment_embeddings).mean(dim=0)
        return video_emb.cpu().numpy()


    def _embed_image_clusters(self, frames: np.ndarray, frame_embs: np.ndarray, clusters: dict[int, list[int]]):
        result = {}
        for cid, idxs in clusters.items():
            if not idxs:
                continue
            
            # ... (Questa parte del codice per trovare l'immagine migliore non cambia)
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
                outputs = self.image_model(**inputs)

                img_emb = outputs.pooler_output.cpu().numpy().squeeze()

            result[f"cluster_{cid}"] = {
                "frame_idx_local": int(best_local),
                "frame_embedding": torch.tensor(img_emb, dtype=torch.float32),
            }
        return result

    def _encode_audio_and_text(self, video_path, start_t, end_t):
        try:
            audio_array_48k = None
            
            # --- INIZIO MODIFICA ---
            # 1. Estrai l'audio IN-MEMORY (usando il WORKAROUND per il bug di moviepy)
            with VideoFileClip(video_path) as vid:
                if start_t >= vid.duration or start_t >= end_t:
                    raise ValueError("Invalid time range for audio extraction.")
                
                clip = vid.subclip(start_t, end_t)
                
                if clip.audio is None:
                    raise ValueError("No audio found in the subclip.")

                # NON USIAMO .to_soundarray() perché è bacato
                # audio_array_48k = clip.audio.to_soundarray(fps=self.audio_sr)
                
                # USIAMO .iter_frames() e lo convertiamo in array numpy
                # Questo bypassa il bug di moviepy/numpy
                audio_frames = list(clip.audio.iter_frames(fps=self.audio_sr))
                audio_array_48k = np.array(audio_frames)

            # --- FINE MODIFICA ---

            if audio_array_48k.ndim == 2:
                # audio_array_48k ora è [num_campioni, 2] (stereo)
                # Convertiamo in mono
                audio_array_48k = audio_array_48k.mean(axis=1)

            # 2. Controlla il silenzio (per CLAP)
            audio_trimmed_48k, _ = librosa.effects.trim(audio_array_48k, top_db=20)
            if audio_trimmed_48k.size == 0:
                raise ValueError("Audio clip is silent or too short after trimming (for CLAP).")

            # 3. Codifica l'audio per CLAP (a 48kHz)
            inputs_audio = self.audio_processor(audio=audio_trimmed_48k, sampling_rate=self.audio_sr, return_tensors="pt").to(self.device)
            with torch.no_grad():
                audio_emb = self.audio_model.get_audio_features(**inputs_audio).cpu().numpy().squeeze()

            # 4. Prepara l'audio per Whisper (a 16kHz)
            if self.audio_sr != self.asr_sr:
                audio_array_16k = librosa.resample(audio_array_48k, orig_sr=self.audio_sr, target_sr=self.asr_sr)
            else:
                audio_array_16k = audio_array_48k

            # 5. CONTROLLO FINALE per Whisper
            if audio_array_16k.size == 0:
                logging.warning(f"[ASR SKIP] Audio too short for ASR after resampling for clip {os.path.basename(video_path)} ({start_t:.2f}s-{end_t:.2f}s).")
                transcript = ""
                text_dim = self.text_embedder.get_sentence_embedding_dimension()
                text_emb = np.zeros(text_dim, dtype=np.float32)
                
                return audio_emb, transcript, text_emb 

            # 6. Trascrivi IN-MEMORY (con il metodo Processor+Model)
            inputs = self.asr_processor(
                audio_array_16k, 
                sampling_rate=self.asr_sr, 
                return_tensors="pt"
            )
            input_features = inputs.input_features.to(self.device) 

            with torch.no_grad():
                predicted_ids = self.asr_model.generate(input_features)

            transcript_list = self.asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)
            transcript = transcript_list[0].strip() if transcript_list else ""
            
            # 7. Codifica il testo
            if transcript:
                text_emb = self.text_embedder.encode(transcript)
            else:
                text_dim = self.text_embedder.get_sentence_embedding_dimension()
                text_emb = np.zeros(text_dim, dtype=np.float32)
                transcript = ""

            return audio_emb, transcript, text_emb

        except Exception as e:
            logging.error(f"[AUDIO/TEXT] Failed for clip {os.path.basename(video_path)} ({start_t:.2f}s-{end_t:.2f}s) with error: {e}")
            return None, "", None


if __name__ == "__main1__":
    import sys

    logging.info("Starting MultiModalEncoder test run...")
    
    # Assicurati che le directory di supporto esistano
    if not os.path.exists("video_dataset.py") or not os.path.exists("utils/logging_formatter.py"):
         print("Assicurati che i file 'video_dataset.py' e 'utils/logging_formatter.py' siano presenti.")
         sys.exit(1)

    data_dir = "../../../data" # FIXME: aggiorna il percorso alla directory dei dati
    if not os.path.exists(data_dir):
        logging.error(f"Directory {os.path.abspath(data_dir)} non trovata.")
        sys.exit(1)

    video_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi"))
        and "animal" not in f.lower()  # Esempio di filtro per escludere certi video
        and "ai" not in f.lower()
    ]

    if not video_files:
        logging.error(f"Nessun video trovato in {data_dir}.")
        sys.exit(1)

    logging.info(f"Trovati {len(video_files)} video:\n" + "\n".join(f" - {f}" for f in video_files))

    dataset = VideoDataset(video_files)
    logging.info(f"Creato VideoDataset con {len(dataset)} elementi.")

    encoder = MultiModalEncoder(video_dataset=dataset, device="cuda")
    encoder.load_models()

    encoded_dataset = encoder.encode_videos()

    if not encoded_dataset.video_datapoints:
        logging.warning("Nessun video è stato processato.")
        sys.exit(0)
        
    first_dp = encoded_dataset.video_datapoints[0]
    if first_dp:
        logging.info("\n=== RISULTATI PRIMO VIDEO ===")
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

    logging.info("Encoding completato con successo!")