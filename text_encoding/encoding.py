from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # cartella "EgoRAG"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import os
import cv2
import torch
import torch.nn.functional as F
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
import numpy as np
import logging
import tempfile
from tqdm import tqdm
from decord import VideoReader, cpu
from sklearn.cluster import KMeans
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from scenedetect import detect, ContentDetector

from encoding_try.video_dataset import VideoDataset, VideoDataPoint, Scene
from encoding_try.utils.logging_formatter import LevelAwareFormatter
from transformers import logging as hf_logging
from concurrent.futures import ThreadPoolExecutor, as_completed


# --- Setup logging ---
handler = logging.StreamHandler()
handler.setFormatter(LevelAwareFormatter())
logging.basicConfig(
    level=logging.INFO,
    handlers=[handler],
)
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()


class CaptionEncoder:
    """ Class for encoding video scenes into textual captions and embeddings."""
    def __init__(self, video_dataset: VideoDataset, device: str = "cuda", max_frames_per_scene: int = 48, max_k_clusters: int = 5):
        """
        Initialize the CaptionEncoder with the given video dataset and device.
        Args:
            video_dataset (VideoDataset): The dataset containing videos to be processed.
            device (str): The device to run the models on ('cuda' or 'cpu').
            max_frames_per_scene (int): Maximum number of frames to extract per scene.
            max_k_clusters (int): Maximum number of clusters for frame clustering.
        """

        if video_dataset is None or len(video_dataset) == 0:
            raise ValueError("Video dataset is empty or not provided.")
        self.dataset = video_dataset
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_frames_per_scene = max_frames_per_scene
        self.max_k_clusters = max_k_clusters
        logging.info(f"[DEVICE] Using: {self.device}")

    def load_models(self):
        """
        Load necessary models: BLIP for captioning, CLIP for image embeddings, and SentenceTransformer for text embeddings.
        """
        logging.info("[MODELS] Loading...")

        # CAPTIONING
        try:
            self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", local_files_only=True)
            self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", local_files_only=True).to(self.device)
        except Exception as e:
            logging.warning(f"[CAPTION] BLIP local not found. Set local_files_only=False on first run if needed. Error: {e}")
            self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", local_files_only=False)
            self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", local_files_only=False).to(self.device)
        
        # CLIP (image encoder for frame embedding)
        try:
            # Uso modello piu leggero per velocità con impatto minimo su clustering
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(self.device)

            # self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336", local_files_only=True)
            # self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", local_files_only=True).to(self.device)
        except Exception as e:
            logging.warning(f"[CLIP] local not found. {e}")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)

            # self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
            # self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(self.device)

        # TEXT
        self.text_embedder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True).to(self.device)
        logging.info("[MODELS] Loaded successfully.")

        if self.device == "cuda":
            self.caption_model.half()
            self.clip_model.half()
    
    def _caption_image(self, img_np: np.ndarray, max_new_tokens: int = 25) -> str:
        """
        Caption an image using BLIP model.
        Args:
            img_np (np.ndarray): Input image as a numpy array (H, W, 3) in RGB format.
            max_new_tokens (int): Maximum number of tokens to generate for the caption.
        Returns:
            str: Generated caption for the image.
        """
        self.caption_model.eval()
        with torch.no_grad():
            inputs = self.caption_processor(images=img_np, return_tensors="pt").to(self.device)
            out = self.caption_model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
            cap = self.caption_processor.decode(out[0], skip_special_tokens=True).strip()
        return cap

    def _caption_images_batch(self, imgs: list[np.ndarray], max_new_tokens:int=16) -> list[str]:
        """
        Caption a batch of images using BLIP model.
        Args:
            imgs (list[np.ndarray]): List of input images as numpy arrays (H, W, 3) in RGB format.
            max_new_tokens (int): Maximum number of tokens to generate for each caption.
        Returns:
            list[str]: List of generated captions for the images.
        """
        if not imgs:
            return []
        self.caption_model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
            inputs = self.caption_processor(images=imgs, return_tensors="pt").to(self.device)
            out = self.caption_model.generate(
                **inputs, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False, early_stopping=True
            )
        caps = [self.caption_processor.decode(o, skip_special_tokens=True).strip() for o in out]
        return caps

    def _embed_text(self, texts: list[str]) -> np.ndarray:
        """
        Embeds a list of texts using SentenceTransformer.
        Args:
            texts (list[str]): List of input texts to embed.
        Returns:
            np.ndarray: Array of shape (N, D) with text embeddings.
        """
        if not texts:
            return np.zeros((0, self.text_embedder.get_sentence_embedding_dimension()), dtype=np.float32)
        embs = self.text_embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False, device=self.device)
        return embs.astype(np.float32)

    def _summarize_scene_captions(self, caps: list[str], transcript: str | None = None) -> str:
        """
        Generate a concise summary of scene captions and optional transcript snippet (from audio).
        Args:
            caps (list[str]): List of captions from representative frames.
            transcript (str | None): Optional transcript text from audio.
        Returns:
            str: Summarized scene caption.
        """
        if not caps and not transcript:
            return ""

        # Removes duplicates (case insensitive, trimmed)
        uniq = []
        seen = set()
        for c in caps:
            k = c.lower().strip()
            if k and k not in seen:
                uniq.append(c.strip())
                seen.add(k)

        # keep the longest 3 captions as summary base
        uniq.sort(key=len, reverse=True)
        base = ". ".join(uniq[:3]).strip()

        # Optional transcript snippet 
        if transcript:
            snippet = transcript.strip().split(".")[0][:180].strip()
            if snippet:
                return (base + (". " if base else "") + f"Audio mentions: {snippet}.").strip()
            
        return base
    
    def _embed_image_clusters(self, frames: np.ndarray, frame_embs: np.ndarray, clusters: dict[int, list[int]]):
        """
        For each cluster of frames, select the representative frame (closest to centroid), caption it, and embed the caption.
        Args:
            frames (np.ndarray): Array of frames (N, H, W, 3).
            frame_embs (np.ndarray): Array of frame embeddings (N, D).
            clusters (dict[int, list[int]]): Dictionary mapping cluster IDs to lists of local frame indices.
        Returns:
            dict: Mapping from cluster ID to dict with 'frame_idx_local', 'caption', and 'caption_embedding'.
        """
        result = {}
        for cid, idxs in clusters.items():
            embs = frame_embs[idxs]
            centroid = embs.mean(axis=0, keepdims=True) # centroid
            sims = (embs @ centroid.T) / (np.linalg.norm(embs, axis=1, keepdims=True) * np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-8) # cosine similarity
            best_local = idxs[int(np.argmax(sims))]
            img = frames[best_local]

            # Captioning and embedding of the representative frame
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

    def _caption_scene_from_reps(self, rep_frames: list[np.ndarray], transcript: str | None = None) -> tuple[str, np.ndarray]:
        """
        Caption a scene based on its representative frames.
        Args:
            rep_frames (list[np.ndarray]): List of representative frame images (H, W, 3).
            transcript (str | None): Optional transcript text from audio.
        Returns:
            tuple[str, np.ndarray]: Scene caption and its embedding.
        """
        try:
            caps = self._caption_images_batch(rep_frames, max_new_tokens=16)
        except Exception as e:
            logging.warning(f"[CAPTION] batch failed: {e}")
            caps = []
            for img in rep_frames:
                try:
                    caps.append(self._caption_image(img, max_new_tokens=16))
                except Exception as e2:
                    logging.warning(f"[CAPTION] single frame failed: {e2}")
        scene_caption = self._summarize_scene_captions(caps, transcript)
        scene_caption_emb = self._embed_text([scene_caption]).squeeze() if scene_caption else \
            np.zeros(self.text_embedder.get_sentence_embedding_dimension(), dtype=np.float32)
        return scene_caption, scene_caption_emb

    def encode_video_scenes(self):
        """
        Encode all videos in the dataset: detect scenes, extract frames, caption scenes, and compute embeddings.
        Returns:
            VideoDataset: The dataset with updated scene captions and embeddings.
        """
        for dp in tqdm(self.dataset.video_datapoints, desc="Encoding videos"): # looping on videos
            logging.info(f"\n=== VIDEO: {dp.video_name} ===")
            dp.scene_embeddings = {}
            self._extract_scenes(dp)
            if not dp.scenes:
                logging.warning(f"[WARN] Nessuna scena trovata in {dp.video_name}, skip.")
                continue
            else:
                logging.info(f"[INFO] {len(dp.scenes)} scene trovate in {dp.video_name}.")

            '''
            os.environ.setdefault("DECORD_NUM_THREADS", "1")
            try:
                vr = VideoReader(dp.video_path, ctx=cpu(0), num_threads=1)  # ← UN SOLO READER, thread interno disattivato
            except Exception as e:
                logging.error(f"[DECORD] cannot open {dp.video_path}: {e}")
                continue
            '''
            max_workers = 1 if "cuda" in self.device else min(4, os.cpu_count() or 2)
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # futures = {executor.submit(self._encode_scene, vr, dp.video_path, scene): f"scene_{i}"
                        # for i, scene in enumerate(dp.scenes)}
                futures = {
                    executor.submit(self._encode_scene, dp.video_path, scene): f"scene_{i}"
                    for i, scene in enumerate(dp.scenes)
                }
                for f in tqdm(as_completed(futures), total=len(futures), desc=f"Scenes ({dp.video_name})"):
                    sid = futures[f]
                    try:
                        scene_out = f.result()
                        if scene_out:
                            dp.scene_embeddings[sid] = scene_out   # contains caption, embedding
                        else:
                            logging.warning(f"[SKIP] scena {sid} vuota.")
                    except Exception as e:
                        logging.error(f"[ERROR] scena {sid} fallita: {e}")
            
            try:
                vid_cap, vid_cap_emb = self._caption_video_global(dp)
                dp.global_embeddings["caption"] = torch.tensor(vid_cap_emb, dtype=torch.float32)
                logging.info(f"[VIDEO CAPTION] {dp.video_name}: {vid_cap[:300]}{'...' if len(vid_cap) > 300 else ''}")
            except Exception as e:
                logging.warning(f"[VIDEO CAPTION] failed for {dp.video_name}: {e}")

        return self.dataset

    def _caption_video_global(self, dp: VideoDataPoint) -> tuple[str, np.ndarray]:
        """
        Generate a global caption for the entire video based on scene captions.
        Args:
            dp (VideoDataPoint): The video data point containing scene embeddings and captions.
        Returns:
            tuple[str, np.ndarray]: Global video caption and its embedding.
        """
        caps = [s.get("caption", "").strip() for s in dp.scene_embeddings.values() if s.get("caption")]

        # Deduplicate (case insensitive)
        uniq = []
        seen = set()
        for c in caps:
            k = c.lower()
            if k and k not in seen:
                uniq.append(c)
                seen.add(k)

        uniq.sort(key=len, reverse=True)
        text = ". ".join(uniq[:5]).strip()
        emb = self._embed_text([text]).squeeze() if text else \
            np.zeros(self.text_embedder.get_sentence_embedding_dimension(), dtype=np.float32)
        return text, emb

    def _extract_scenes(self, dp: VideoDataPoint):
        """
        Extract scenes from the video using content-based scene detection.
        Updates the dp.scenes list with detected scenes.
        Args:
            dp (VideoDataPoint): The video data point to process.
        """
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
    
    def _embed_frames_clip(self, frames: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """
        Embed frames using CLIP image encoder.
        Args:
            frames (np.ndarray): Array of frames (N, H, W, 3).
            batch_size (int): Batch size for processing frames.
        Returns:
            np.ndarray: Array of frame embeddings (N, D).
        """
        if frames is None or len(frames) == 0:
            return np.zeros((0, 768), dtype=np.float32)  # D=768 for ViT-L/14-336

        self.clip_model.eval()
        all_embs = []
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i+batch_size]
                inputs = self.clip_processor(images=list(batch), return_tensors="pt").to(self.device)
                feats = self.clip_model.get_image_features(**inputs)
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8) # normalize
                all_embs.append(feats.detach().cpu().float())
        embs = torch.cat(all_embs, dim=0).numpy()
        return embs

    # def _encode_scene(self, vr: VideoReader | None, video_path: str, scene: Scene):
    def _encode_scene(self, video_path: str, scene: Scene):
        """
        Encode a single scene: extract frames, embed, cluster, caption representatives.
        Args:
            video_path (str): Path to the video file.
            scene (Scene): The scene to encode.
        Returns:
            dict: Scene caption, embedding, and metadata.
        """
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        frames, frame_idxs = self._extract_frames(vr, video_path, scene.start_frame, scene.end_frame, self.max_frames_per_scene)
        frame_embs = self._embed_frames_clip(frames)

        # Clustering of frames, selection of representatives, and captioning
        clusters = self._cluster_frames(frame_embs)
        rep_indices = self._select_representative_indices(frame_embs, clusters)
        rep_indices_sorted = sorted(rep_indices, key=lambda i: frame_idxs[i])
        rep_frames = [frames[i] for i in rep_indices_sorted]
        scene_caption, scene_caption_emb = self._caption_scene_from_reps(rep_frames, transcript=None)

        return {
            # ← il dataset ha già il campo "caption" a livello scena
            "caption": scene_caption,  # stringa

            # ← usa "text" per l'embedding testuale della caption di scena
            "text": torch.tensor(scene_caption_emb, dtype=torch.float32),

            # gli altri campi previsti dal dataset li lasci vuoti/opzionali
            "video": None,
            "audio": None,
            "transcript": "",   # o lo lasci vuoto; lo userai quando aggiungi ASR
            "image": {},

            "meta": {
                "selected_frame_global_idxs": [int(frame_idxs[i]) for i in rep_indices_sorted],
                "selected_frame_local_idxs": [int(i) for i in rep_indices_sorted],
                "num_clusters": len(clusters),

                # 👇 aggiunta minimal
                "time": {"start": float(scene.start_time), "end": float(scene.end_time)},
                "frames": {"start": int(scene.start_frame), "end": int(scene.end_frame)},
            },
        }

    def _extract_frames(self, vr: VideoReader, video_path, start_frame=None, end_frame=None, max_frames=48):
        """
        Extract frames from a video between start_frame and end_frame.
        Args:
            video_path (str): Path to the video file.
            start_frame (int | None): Starting frame index.
            end_frame (int | None): Ending frame index.
            max_frames (int): Maximum number of frames to extract.
        Returns:
            tuple[np.ndarray, np.ndarray]: Extracted frames (N, H, W, 3) and their indices.
        """
        # vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        start = 0 if start_frame is None else max(0, min(int(start_frame), total_frames-1))
        end   = (total_frames-1) if end_frame is None else max(0, min(int(end_frame), total_frames-1))
        if end < start:
            end = start
        span = end - start + 1
        num = min(max_frames, span)
        idxs = np.linspace(start, end, num=num, dtype=int)
        idxs = np.unique(idxs)
        frames = vr.get_batch(idxs).asnumpy()
        return frames, idxs


    def _choose_k(self, n_frames: int) -> int:
        """
        Choose the number of clusters K based on the number of frames.
        Args:
            n_frames (int): Number of frames.
        Returns:
            int: Chosen number of clusters K.
        """
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
        Cluster frame embeddings using KMeans.
        Args:
            frame_embs (np.ndarray): Array of frame embeddings (N, D).
        Returns:
            dict[int, list[int]]: Mapping from cluster ID to list of local frame indices.
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
        Select representative frame indices for each cluster based on proximity to centroid.
        Args:
            frame_embs (np.ndarray): Array of frame embeddings (N, D).
            clusters (dict[int, list[int]]): Mapping from cluster ID to list of local frame indices.
        Returns:
            list[int]: List of representative frame indices.
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
    
    # --- Queries and Retrieval methods can be added here ---

    def queries_embedder(self, queries: list[str]) -> torch.Tensor:
        text_embeds = self._embed_text(queries)
        return torch.tensor(text_embeds, dtype=torch.float32)
    
    def _gather_video_index(self, dataset: VideoDataset): 
        embs, idxs, names = [], [], []
        for i, dp in enumerate(dataset.video_datapoints):
            caption_embedding = dp.global_embeddings.get("caption", None)
            if isinstance(caption_embedding, torch.Tensor) and caption_embedding.numel() > 0:
                e = caption_embedding.detach().float().view(1, -1)
                embs.append(e)
                idxs.append(i)
                names.append(dp.video_name)
        if not embs:
            return None, [], []
        embs = torch.cat(embs, dim=0)                 # (N, D)
        embs = F.normalize(embs, p=2, dim=1)          # L2 normalize to have cosine similarity equal to dot product
        return embs, idxs, names
    
    def search_videos(self, dataset: VideoDataset, query: str, top_k: int = 5):
        vid_embs, vid_idxs, vid_names = self._gather_video_index(dataset)
        if vid_embs is None:
            return []

        q = self.queries_embedder([query])               # (1, D)
        q = F.normalize(q, p=2, dim=1)                # L2 normalize to have cosine similarity equal to dot product
        scores = (q @ vid_embs.T).squeeze(0)          # (N,)
        topk = torch.topk(scores, k=min(top_k, scores.numel()))
        out = [(float(scores[i]), vid_idxs[i], vid_names[i]) for i in topk.indices.tolist()]
        return out

    def search_scenes(self, dp: VideoDataPoint, query: str, top_k: int = 5):
        keys, embs, caps = [], [], []
        for k, s in dp.scene_embeddings.items():
            e = s.get("text", None)
            if isinstance(e, torch.Tensor) and e.numel() > 0:
                keys.append(k) # append scene key -> scene_{i}
                caps.append(s.get("caption", ""))
                embs.append(e.detach().float().view(1, -1))
        if not embs:
            return []

        E = torch.cat(embs, dim=0)            # (S, D), S = num scenes - D = dim embedding
        E = F.normalize(E, p=2, dim=1) # L2 normalize to have cosine similarity equal to dot product

        q = self.queries_embedder([query])       # (1, D)
        q = F.normalize(q, p=2, dim=1)

        scores = (q @ E.T).squeeze(0)         # (S,)
        topk = torch.topk(scores, k=min(top_k, scores.numel())).indices.tolist()
        results = []
        for i in topk:
            key = keys[i]
            cap = caps[i]
            sdict = dp.scene_embeddings[key]

            t = sdict.get("meta", {}).get("time")
            if t is not None:
                start_t, end_t = float(t.get("start", 0.0)), float(t.get("end", 0.0))
            else:
                try:
                    idx_num = int(key.split("_")[1])
                    start_t, end_t = dp.scenes[idx_num].start_time, dp.scenes[idx_num].end_time
                except Exception:
                    start_t, end_t = 0.0, 0.0

            results.append((float(scores[i]), key, cap, start_t, end_t))
        return results

if __name__ == "__main__":

    logging.info("Starting CaptionEncoder test run...")
    data_dir = "../../../ego4d_data/v2/full_scale"
    if not os.path.exists(data_dir):
        logging.error(f"Directory {os.path.abspath(data_dir)} not found.")
        sys.exit(1)

    video_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi"))
    ]

    print(video_files)

    if not video_files:
        logging.error(f"No videos found in {data_dir}.")
        sys.exit(1)

    logging.info(f"Found {len(video_files)} videos:\n" + "\n".join(f" - {f}" for f in video_files))

    dataset = VideoDataset(video_files[9:10]) # test with first 3 videos
    logging.info(f"VideoDataset was created with {len(dataset)} elements.")

    encoder = CaptionEncoder(video_dataset=dataset, device="cuda")
    encoder.load_models()
    encoded_dataset = encoder.encode_video_scenes()

    if not encoded_dataset.video_datapoints:
        logging.warning("No video has been processed.")
        sys.exit(0)
        
    first_dp = encoded_dataset.video_datapoints[0]
    if first_dp:
        logging.info("\n=== FIRST VIDEO RESULTS ===")
        logging.info(f"Video path: {first_dp.video_path}")
        logging.info(f"Number of scenes: {len(first_dp.scenes)}")

        if first_dp.global_embeddings:
            logging.info(f"Chiavi global embeddings: {list(first_dp.global_embeddings.keys())}")
            # Embedding della caption globale (se presente)
            cap_emb = first_dp.global_embeddings.get("caption")
            if isinstance(cap_emb, torch.Tensor):
                logging.info(f"Global caption emb shape: {tuple(cap_emb.shape)}")

        if first_dp.scene_embeddings:
            scene_keys = list(first_dp.scene_embeddings.keys())
            logging.info(f"Chiavi scene embeddings: {scene_keys[:3]} ...")

            if scene_keys:
                first_scene_key = scene_keys[0]
                first_scene = first_dp.scene_embeddings[first_scene_key]
                logging.info(f"--- Esempio scena: {first_scene_key} ---")

                # Caption testuale della scena
                logging.info(f"Caption: {first_scene.get('caption', '')!r}")

                # Embedding testuale della caption (torch.Tensor) se presente
                text_emb = first_scene.get('text')
                if isinstance(text_emb, torch.Tensor):
                    logging.info(f"Caption emb shape: {tuple(text_emb.shape)}")

                # Transcript (se in futuro userai ASR)
                logging.info(f"Transcript: {first_scene.get('transcript', '')[:80]!r}")

    logging.info("Encoding completato con successo!")

    # --- Retrieval Test ---
    test_query = "Where was the onion before I picked it?"
    logging.info(f"\n=== RETRIEVAL TEST for query: {test_query!r} ===")
    video_results = encoder.search_videos(encoded_dataset, test_query, top_k=3)
    for score, vid_idx, vid_name in video_results:
        logging.info(f"[VIDEO] Score: {score:.4f}, Index: {vid_idx}, Name: {vid_name}")

        dp = encoded_dataset.video_datapoints[vid_idx]
        scene_results = encoder.search_scenes(dp, test_query, top_k=2)
        for s_score, s_key, s_cap, t0, t1 in scene_results:
            logging.info(
                f"    [SCENE] {s_key} | {t0:.2f}s–{t1:.2f}s | Score: {s_score:.4f} | Caption: {s_cap[:100]!r}"
            )