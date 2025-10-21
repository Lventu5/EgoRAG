import os
import cv2
import torch
import sys
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
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336", local_files_only=True)
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", local_files_only=True).to(self.device)
        except Exception as e:
            logging.warning(f"[CLIP] local not found. {e}")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(self.device)

        # TEXT
        self.text_embedder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True).to(self.device)
        logging.info("[MODELS] Loaded successfully.")
    
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
            out = self.caption_model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=3, do_sample=False)
            cap = self.caption_processor.decode(out[0], skip_special_tokens=True).strip()
        return cap

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
        caps = []
        for img in rep_frames:
            try:
                caps.append(self._caption_image(img))
            except Exception as e:
                logging.warning(f"[CAPTION] Frame caption failed: {e}")
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

            os.environ.setdefault("DECORD_NUM_THREADS", "1")
            try:
                vr = VideoReader(dp.video_path, ctx=cpu(0), num_threads=1)  # ← UN SOLO READER, thread interno disattivato
            except Exception as e:
                logging.error(f"[DECORD] cannot open {dp.video_path}: {e}")
                continue
            max_workers = 1 if "cuda" in self.device else min(4, os.cpu_count() or 2)
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._encode_scene, vr, dp.video_path, scene): f"scene_{i}"
                        for i, scene in enumerate(dp.scenes)}
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
                logging.info(f"[VIDEO CAPTION] {dp.video_name}: {vid_cap[:160]}{'...' if len(vid_cap) > 160 else ''}")
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
        caps = [s.get("caption", "") for s in dp.scene_embeddings.values() if s.get("caption")]
        caps.sort(key=len, reverse=True)
        text = " ".join(caps[:5]).strip() # concatenate top 5 longest captions
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
    
    def _embed_frames_clip(self, frames: np.ndarray, batch_size: int = 64) -> np.ndarray:
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
        with torch.no_grad():
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i+batch_size]
                inputs = self.clip_processor(images=list(batch), return_tensors="pt").to(self.device)
                feats = self.clip_model.get_image_features(**inputs)
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8) # normalize
                all_embs.append(feats.detach().cpu().float())
        embs = torch.cat(all_embs, dim=0).numpy()
        return embs

    def _encode_scene(self, vr: VideoReader, video_path: str, scene: Scene):
        """
        Encode a single scene: extract frames, embed, cluster, caption representatives.
        Args:
            video_path (str): Path to the video file.
            scene (Scene): The scene to encode.
        Returns:
            dict: Scene caption, embedding, and metadata.
        """
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

        if start_frame is None:
            start_frame = 0
        if end_frame is None or end_frame >= total_frames:
            end_frame = total_frames - 1
        if end_frame < start_frame:
            end_frame = start_frame

        span = end_frame - start_frame + 1
        num = min(max_frames, span)

        idxs = np.linspace(start_frame, end_frame, num=num, dtype=int)
        frames = vr.get_batch(idxs).asnumpy()  # (N, H, W, 3) RGB uint8
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

    if not video_files:
        logging.error(f"No videos found in {data_dir}.")
        sys.exit(1)

    logging.info(f"Found {len(video_files)} videos:\n" + "\n".join(f" - {f}" for f in video_files))

    dataset = VideoDataset(video_files)
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