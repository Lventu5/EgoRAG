import logging
import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Dict, Union
from transformers import CLIPProcessor, CLIPModel
from indexing.components.tags_new import TAGS_CLIP
from configuration.config import CONFIG

class VisionTagger:
    """Tagger that predicts which tags (from TAGS_CLIP) are present in a video
    based on visual analysis of frames using CLIP.

    Usage:
        tagger = VisionTagger(device="cuda")
        tagger.load_model()
        tagger.tag_datapoint(dp)

    The implementation extracts frames from the video (globally or per scene),
    encodes them using CLIP, and compares them against the pre-computed embeddings
    of the TAGS_CLIP categories. Tags exceeding a confidence threshold are selected.
    """

    def __init__(
        self, 
        device: str = "cuda", 
        batch_size: int = 4, 
        confidence_threshold: float = 0.25,
        frames_per_segment: int = 5
    ):
        self.model_name = getattr(CONFIG.indexing.tag, "vision_model_id", "openai/clip-vit-base-patch32")
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.frames_per_segment = frames_per_segment

        self.processor: Optional[CLIPProcessor] = None
        self.model: Optional[CLIPModel] = None
        self.text_features: Optional[torch.Tensor] = None
        self.tag_keys: List[str] = []  # Category keys (e.g., "obj_kitchenware")
        self.tag_prompts: List[str] = []  # All text prompts flattened

    def load_model(self):
        logging.info(f"[VisionTagger] Loading model {self.model_name} on {self.device}")
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        logging.info("[VisionTagger] Pre-computing text embeddings for tags...")
        self._precompute_tags()

    def _precompute_tags(self):
        """Encodes all tag descriptions from TAGS_CLIP into vectors.
        
        Strategy: For each category, encode all its descriptions separately,
        then average them to get a single embedding per category.
        This allows multiple perspectives to contribute to each tag.
        """
        self.tag_keys = []
        category_embeddings = []
        
        for category_key, descriptions in TAGS_CLIP.items():
            self.tag_keys.append(category_key)
            
            # Encode all descriptions for this category
            inputs = self.processor(text=descriptions, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                desc_features = self.model.get_text_features(**inputs)
                # Normalize each description embedding
                desc_features = desc_features / desc_features.norm(p=2, dim=-1, keepdim=True)
                # Average all description embeddings for this category
                category_embedding = desc_features.mean(dim=0, keepdim=True)
                # Normalize the averaged embedding
                category_embedding = category_embedding / category_embedding.norm(p=2, dim=-1, keepdim=True)
                category_embeddings.append(category_embedding)
        
        # Stack all category embeddings: shape [num_categories, embedding_dim]
        self.text_features = torch.cat(category_embeddings, dim=0)
        
        logging.info(f"[VisionTagger] Precomputed embeddings for {len(self.tag_keys)} tag categories")

    def _extract_frames(self, video_path: str, start_time: float = None, end_time: float = None, num_frames: int = 5) -> List[Image.Image]:
        """Extracts evenly spaced frames from a video file within a time window."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logging.error(f"[VisionTagger] Could not open video: {video_path}")
            return []

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Determine start/end frames
        start_frame = int(start_time * fps) if start_time is not None else 0
        end_frame = int(end_time * fps) if end_time is not None and end_time < duration else total_frames
        
        if end_frame <= start_frame:
            end_frame = total_frames

        # Select indices
        frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR (OpenCV) to RGB (PIL)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        
        cap.release()
        return frames

    def _predict_tags(self, images: List[Image.Image]) -> List[str]:
        """Runs CLIP inference on a list of images and returns aggregated tags."""
        if not images or self.text_features is None:
            return []

        # Prepare images
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalize
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            # Calculate Similarity (Cosine similarity via dot product)
            # Shape: [num_images, num_categories]
            similarity = image_features @ self.text_features.T
            
            # Aggregate scores across all frames using max
            # Max is better than mean for detecting presence of objects/actions
            max_scores = similarity.max(dim=0).values  # Shape: [num_categories]

        # Thresholding
        found_indices = (max_scores > self.confidence_threshold).nonzero(as_tuple=True)[0]
        
        # Sort by score for relevance
        results = []
        for idx in found_indices:
            score = max_scores[idx].item()
            tag_key = self.tag_keys[idx.item()]
            results.append((tag_key, score))
        
        # Sort descending by score and extract category keys
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results]

    def tag_datapoint(self, dp, tag_scenes: bool = True) -> List[str]:
        """Tag a single VideoDataPoint using its visual modality.
        
        Expects `dp.video_path` to be valid.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        video_path = getattr(dp, "video_path", None)
        if not video_path:
            logging.warning(f"[VisionTagger] No video_path found for datapoint {dp}")
            return []

        # --- Global Tagging ---
        # Extract frames from the whole video
        global_frames = self._extract_frames(video_path, num_frames=self.frames_per_segment)
        global_tags = self._predict_tags(global_frames)
        
        # Initialize global embeddings if missing
        if not hasattr(dp, "global_embeddings") or dp.global_embeddings is None:
            dp.global_embeddings = {}
        
        dp.global_embeddings["tags"] = global_tags

        # --- Scene Tagging ---
        if tag_scenes:
            scenes = getattr(dp, "scenes", {}) or {}
            scene_embeddings = getattr(dp, "scene_embeddings", {}) or {}
            
            if not scenes:
                logging.debug(f"[VisionTagger] No scenes found for {video_path}")
            
            for sid, scene in scenes.items():
                # Get timing from Scene object (not from scene_embeddings)
                start = getattr(scene, "start_time", None)
                end = getattr(scene, "end_time", None)
                
                if start is not None and end is not None:
                    logging.debug(f"[VisionTagger] Tagging scene {sid} [{start}-{end}]")
                    scene_frames = self._extract_frames(
                        video_path, 
                        start_time=float(start), 
                        end_time=float(end), 
                        num_frames=self.frames_per_segment
                    )
                    scene_tags = self._predict_tags(scene_frames)
                    
                    # Store tags in scene_embeddings dict
                    if sid not in scene_embeddings:
                        scene_embeddings[sid] = {}
                    scene_embeddings[sid]["tags"] = scene_tags
                    logging.debug(f"[VisionTagger] Scene {sid} tags: {scene_tags}")
                else:
                    logging.warning(f"[VisionTagger] Scene {sid} missing start/end time in Scene object")

        return global_tags

    def tag_dataset(self, dataset) -> None:
        """Tag all datapoints in a VideoDataset (in-place)."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        for i, dp in enumerate(dataset.video_datapoints):
            try:
                # Default behavior: tag scenes if they exist
                has_scenes = bool(getattr(dp, "scene_embeddings", {}))
                self.tag_datapoint(dp, tag_scenes=has_scenes)
                
                if i % 10 == 0:
                    logging.info(f"[VisionTagger] Processed {i}/{len(dataset.video_datapoints)} videos.")
            except Exception as e:
                logging.error(f"[VisionTagger] Failed to tag datapoint {getattr(dp, 'video_name', '<unknown>')}: {e}")

    def infer_tags_from_frames(self, frames: List[Image.Image]) -> List[str]:
        """Infer tags from a list of loaded PIL Images."""
        if not frames:
            return []
        if self.model is None:
            try:
                self.load_model()
            except Exception as e:
                logging.warning(f"[VisionTagger] Could not load model: {e}")
                return []
        return self._predict_tags(frames)

    def _truncate(self, text: str, max_len: int = 200, suffix: str = "...") -> str:
        if not text:
            return ""
        t = str(text)
        if len(t) <= max_len:
            return t
        return t[:max_len].rstrip() + suffix

    def pretty_print_datapoint(
        self,
        dp,
        max_text_len: int = 200,
        show_scenes: bool = True,
        truncate_suffix: str = "...",
        color: bool = False,
    ) -> None:
        """Print a VideoDataPoint's path and visual tags."""
        GREEN = "\033[92m" if color else ""
        YELLOW = "\033[93m" if color else ""
        RESET = "\033[0m" if color else ""

        name = getattr(dp, "video_name", getattr(dp, "video_path", "<unknown>"))
        print("=" * 80)
        print(f"Video (Vision Analysis): {name}")

        ge = getattr(dp, "global_embeddings", {}) or {}
        tags = ge.get("tags") or []
        tags_str = ", ".join(tags) if tags else "(none)"

        print(f" Global tags: {GREEN}{tags_str}{RESET}")

        if show_scenes:
            scenes = getattr(dp, "scene_embeddings", {}) or {}
            if not scenes:
                print(" No scene embeddings.")
            else:
                print(" Scenes:")
                for sid, sd in sorted(scenes.items()):
                    if not isinstance(sd, dict):
                        continue
                    
                    # Print time range if available
                    start = sd.get("start_time", "?")
                    end = sd.get("end_time", "?")
                    time_info = f"[{start}-{end}s]"
                    
                    stag = sd.get("tags") or []
                    stag_str = ", ".join(stag) if stag else "(none)"
                    print(f"  - {sid} {time_info}")
                    print(f"     tags: {GREEN}{stag_str}{RESET}")

        print()

    def pretty_print_dataset(
        self,
        dataset,
        max_text_len: int = 200,
        show_scenes: bool = True,
        truncate_suffix: str = "...",
        color: bool = False,
    ) -> None:
        """Pretty-print all datapoints in a VideoDataset."""
        for dp in dataset.video_datapoints:
            self.pretty_print_datapoint(
                dp,
                max_text_len=max_text_len,
                show_scenes=show_scenes,
                truncate_suffix=truncate_suffix,
                color=color,
            )
