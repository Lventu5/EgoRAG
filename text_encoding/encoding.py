import os
import cv2
import torch
import numpy as np
import librosa
import logging
import tempfile
from tqdm import tqdm
from decord import VideoReader, cpu
from sklearn.cluster import KMeans
from moviepy.editor import VideoFileClip
from transformers import AutoModel, AutoProcessor, XCLIPModel, XCLIPProcessor, BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from scenedetect import detect, ContentDetector
import whisper

from encoding_try.video_dataset import VideoDataset, VideoDataPoint, Scene
from utils.logging_formatter import LevelAwareFormatter
from transformers import logging as hf_logging
from concurrent.futures import ThreadPoolExecutor, as_completed

class CaptionEncoder:
    """
    Pipeline per:
      - Scene detection automatica
      - Frame sampling per scena
      - Scene captioning
      - Caption embedding (CLIP text model)
    """

    def __init__(
        self,
        video_dataset: VideoDataset,
        device: str = "cuda",
        max_frames_per_scene: int = 48,
        max_k_clusters: int = 5
    ):
        if video_dataset is None or len(video_dataset) == 0:
            raise ValueError("Video dataset is empty or not provided.")
        self.dataset = video_dataset
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_frames_per_scene = max_frames_per_scene
        self.max_k_clusters = max_k_clusters
        logging.info(f"[DEVICE] Using: {self.device}")


    def load_models(self):
        logging.info("[MODELS] Loading...")

        # CAPTIONING
        try:
            self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", local_files_only=True)
            self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", local_files_only=True).to(self.device)
        except Exception as e:
            logging.warning(f"[CAPTION] BLIP local not found. Set local_files_only=False on first run if needed. Error: {e}")
            self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", local_files_only=False)
            self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", local_files_only=False).to(self.device)
        
        # TEXT
        self.asr_model = whisper.load_model("base").to(self.device)
        self.text_embedder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True).to(self.device)

        logging.info("[MODELS] Loaded successfully.")
    
    def _caption_image(self, img_np: np.ndarray, max_new_tokens: int = 25) -> str:
        """
        Caption singola immagine (np.ndarray RGB HxWx3) con BLIP.
        """
        self.caption_model.eval()
        with torch.no_grad():
            inputs = self.caption_processor(images=img_np, return_tensors="pt").to(self.device)
            out = self.caption_model.generate(**inputs, max_new_tokens=max_new_tokens)
            cap = self.caption_processor.decode(out[0], skip_special_tokens=True).strip()
        return cap

    def _embed_text(self, texts: list[str]) -> np.ndarray:
        """
        Restituisce (N, D) con gli embedding SentenceTransformer.
        """
        if not texts:
            return np.zeros((0, self.text_embedder.get_sentence_embedding_dimension()), dtype=np.float32)
        embs = self.text_embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False, device=self.device)
        return embs.astype(np.float32)

    def _summarize_scene_captions(self, caps: list[str], transcript: str | None = None) -> str:
        """
        Dato un insieme di caption per i frame di una scena, produce una singola descrizione testuale della scena, prendendo le caption più significative per lunghezza e varietà.
        Opzionalmente usa anche il transcript audio per arricchire la descrizione.
        Ritorna una stringa.
        """
        if not caps and not transcript:
            return ""

        # Deduplica (case insensitive, strip)
        uniq = []
        seen = set()
        for c in caps:
            k = c.lower().strip()
            if k and k not in seen:
                uniq.append(c.strip())
                seen.add(k)

        # tieni max 3 frasi più descrittive
        uniq.sort(key=len, reverse=True)
        base = ". ".join(uniq[:3]).strip()

        if transcript:
            # prendi la prima frase del transcript (grezzo)
            snippet = transcript.strip().split(".")[0][:180].strip()
            if snippet:
                return (base + (". " if base else "") + f"Audio mentions: {snippet}.").strip()
        return base
    
    def _embed_image_clusters(self, frames: np.ndarray, frame_embs: np.ndarray, clusters: dict[int, list[int]]):
        """
        Per ogni cluster:
        - seleziona il frame medoid (più vicino al centroide)
        - genera la caption del frame rappresentativo (BLIP)
        - calcola l'embedding della caption (SentenceTransformer)
        """
        result = {}
        for cid, idxs in clusters.items():
            embs = frame_embs[idxs]
            centroid = embs.mean(axis=0, keepdims=True) # centroide
            sims = (embs @ centroid.T) / (np.linalg.norm(embs, axis=1, keepdims=True) * np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-8) # cosine similarity
            best_local = idxs[int(np.argmax(sims))]
            img = frames[best_local]

            # Caption del frame rappresentativo
            cap = self._caption_image(img)
            if cap:
                cap_emb = self._embed_text([cap]).squeeze()
            else:
                cap_emb = np.zeros(self.text_embedder.get_sentence_embedding_dimension(), dtype=np.float32)

            result[f"cluster_{cid}"] = {
                "frame_idx_local": int(best_local),
                "caption": cap,
                "caption_embedding": torch.tensor(cap_emb, dtype=torch.float32),
            }

        return result

    def encode_videos(self):
        """
        Loop sui video del dataset con parallelizzazione a livello di scene.
        """
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
        print(f"[DEBUG] Estratti {len(frames)} frame da {scene.start_frame} a {scene.end_frame}")

        frame_embs = self._embed_frames_clip(frames)
        print(f"[DEBUG] frame_embs shape = {frame_embs.shape}")

        clusters = self._cluster_frames(frame_embs)
        print(f"[DEBUG] cluster sizes = { {k: len(v) for k,v in clusters.items()} }")

        rep_indices = self._select_representative_indices(frame_embs, clusters)  # indici locali rispetto a 'frames'
        print(f"[DEBUG] rep_indices = {rep_indices}")


        rep_indices_sorted = sorted(rep_indices, key=lambda i: frame_idxs[i])
        rep_frames = [frames[i] for i in rep_indices_sorted]
        print(f"[DEBUG] Numero frame rappresentativi = {len(rep_frames)}")
        print(f"[DEBUG] Passo al video model {len(rep_frames)} frame "
                  f"di shape {[f.shape for f in rep_frames[:3]]} ...")


        scene_video_emb = self._embed_scene_video_with_representatives(rep_frames)  # numpy (D,)
        print(f"[DEBUG] scene_video_emb shape = {scene_video_emb.shape}")


        image_clusters = self._embed_image_clusters(frames, frame_embs, clusters)

        audio_emb, transcript, text_emb = self._encode_audio_and_text(video_path, scene.start_time, scene.end_time)

        return {
            "video": torch.tensor(scene_video_emb, dtype=torch.float32),
            "audio": torch.tensor(audio_emb, dtype=torch.float32),
            "text": torch.tensor(text_emb, dtype=torch.float32),
            "transcript": transcript,
            "image": image_clusters,
            "meta": {
                "selected_frame_global_idxs": [int(frame_idxs[i]) for i in rep_indices_sorted],
                "selected_frame_local_idxs": [int(i) for i in rep_indices_sorted],
                "num_clusters": len(clusters)
            }
        }


    def _extract_frames(self, video_path, start_frame=None, end_frame=None, max_frames=48):
        """
        Estrae al massimo 'max_frames' frame equispaziati tra start_frame ed end_frame.
        Ritorna sia le immagini (np arrays) sia gli indici globali dei frame.
        """
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)

        if start_frame is None:
            start_frame = 0
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames - 1

        # evitiamo degenerazioni quando la scena è molto corta
        num = min(max_frames, max(2, end_frame - start_frame))
        idxs = np.linspace(start_frame, end_frame - 1, num=num, dtype=int)

        frames = vr.get_batch(idxs).asnumpy()  # (N, H, W, 3) in RGB
        return frames, idxs


    def _choose_k(self, n_frames: int) -> int:
        """ euristica veloce per il numero di cluster """
        if n_frames <= 6:
            return 1
        if n_frames <= 20:
            return 2
        if n_frames <= 40:
            return 3
        if n_frames <= 80:
            return 4
        return min(self.max_k_clusters, 5)
    

    def _cluster_frames(self, frame_embs: np.ndarray):
        """
        Raggruppa i frame in K cluster (su embedding CLIP).
        Ritorna: dict cid -> [idx locali dei frame]
        """
        n = len(frame_embs)
        k = self._choose_k(n)
        if k <= 1:
            return {0: list(range(n))}

        km = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(frame_embs)
        clusters = {}
        for i, lab in enumerate(km.labels_):
            clusters.setdefault(int(lab), []).append(i)
        return clusters

    def _select_representative_indices(self, frame_embs: np.ndarray, clusters: dict[int, list[int]]) -> list[int]:
        """
        Per ogni cluster seleziona il frame 'medoid' (più vicino al centroide).
        Ritorna la lista degli indici locali dei frame rappresentativi.
        """
        reps = []
        for cid, idxs in clusters.items():
            embs = frame_embs[idxs]                       # (m, D)
            centroid = embs.mean(axis=0, keepdims=True)   # (1, D)
            sims = (embs @ centroid.T) / (
                np.linalg.norm(embs, axis=1, keepdims=True) * np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-8
            )  # (m, 1)
            best_local = idxs[int(np.argmax(sims))]
            reps.append(best_local)
        return reps

    def _embed_scene_video_with_representatives(self, rep_frames: list[np.ndarray]) -> np.ndarray:
        """
        Calcola l'embedding video della scena usando SOLO i frame rappresentativi in ordine temporale.
        """
        if len(rep_frames) == 0:
            # fallback (non dovrebbe accadere): ritorna zero
            logging.warning("[VIDEO] No representative frames, returning zeros.")
            return np.zeros(512, dtype=np.float32)
    
        resized_frames = [cv2.resize(f, (224, 224)) for f in rep_frames] # TODO: Make it dynamically adjust with the selected model
        print(f"[DEBUG] resized_frames shapes = {[f.shape for f in resized_frames[:3]]} ...")

        inputs = self.video_processor(images=list(resized_frames), return_tensors="pt").to(self.device)  # type: ignore
        with torch.no_grad():
            out = self.video_model(pixel_values=inputs["pixel_values"])
            vid = out.video_embeds.squeeze(0).detach().cpu().numpy()  # (D,)
        return vid

    def _embed_image_clusters(self, frames: np.ndarray, frame_embs: np.ndarray, clusters: dict[int, list[int]]):
        """
        Per ogni cluster:
          - seleziona il frame medoid (come sopra)
          - calcola l'embedding immagine (CLIP) del frame rappresentativo
        """
        result = {}
        for cid, idxs in clusters.items():
            embs = frame_embs[idxs]
            centroid = embs.mean(axis=0, keepdims=True)
            sims = (embs @ centroid.T) / (
                np.linalg.norm(embs, axis=1, keepdims=True) * np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-8
            )
            best_local = idxs[int(np.argmax(sims))]
            img = frames[best_local]

            inputs = self.image_processor(images=[img], return_tensors="pt").to(self.device)
            with torch.no_grad():
                img_emb = self.image_model.get_image_features(**inputs).cpu().numpy().squeeze()

            result[f"cluster_{cid}"] = {
                "frame_idx_local": int(best_local),
                "frame_embedding": torch.tensor(img_emb, dtype=torch.float32),
            }
        return result