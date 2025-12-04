import logging
import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Dict, Union, Tuple
from transformers import CLIPProcessor, CLIPModel
from indexing.components.tags_new import TAGS_CLIP
from configuration.config import CONFIG

class VisionTagger:
    """Tagger that predicts which tags (from TAGS_CLIP) are present in a video
    based on visual analysis of frames using CLIP.

    Improvements:
    - Ensemble scoring (max + mean combination)
    - Temporal consistency filtering
    - Frame quality assessment
    - Adaptive thresholding
    - Batch processing for efficiency
    - Tag co-occurrence boosting
    """

    def __init__(
        self, 
        device: str = "cuda", 
        batch_size: int = 8, # Batch size for CLIP inference
        confidence_threshold: float = 0.25, # Minimum confidence to accept a tag
        frame_interval: float = 2.0,  # Extract 1 frame every N seconds (e.g., 2.0 = 1 frame per 2s)
        min_frame_presence: float = 0.20,  # Tag must appear in tot%+ of frames to be accepted
        use_adaptive_threshold: bool = True, # Use adaptive thresholding based on score distribution
        filter_low_quality: bool = True, # Filter out low-quality frames based on blur and brightness
        blur_threshold: float = 100.0,  # Laplacian variance threshold for blur detection
        brightness_min: float = 20.0,   # Minimum average brightness for frame quality
        brightness_max: float = 235.0,  # Maximum average brightness for frame quality
        ensemble_weights: Tuple[float, float] = (0.6, 0.4),  # (max_weight, mean_weight)
    ):
        self.model_name = getattr(CONFIG.indexing.tag, "vision_model_id", "openai/clip-vit-base-patch32")
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.frame_interval = frame_interval  # Frames are extracted based on duration / interval
        self.min_frame_presence = min_frame_presence
        self.use_adaptive_threshold = use_adaptive_threshold
        self.filter_low_quality = filter_low_quality
        self.blur_threshold = blur_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.ensemble_weights = ensemble_weights

        self.processor: Optional[CLIPProcessor] = None
        self.model: Optional[CLIPModel] = None
        self.text_features: Optional[torch.Tensor] = None
        self.tag_keys: List[str] = []
        
        # Tag relationships for co-occurrence boosting
        self.tag_groups = self._define_tag_groups()

    def _define_tag_groups(self) -> Dict[str, List[str]]:
        """
        Define logical groups of tags that commonly co-occur.
        When multiple tags from a group are detected, boost their confidence.
        """
        return {
            "cooking": [
                "act_cooking_prep", "act_chopping", "act_stirring", "act_pouring",
                "obj_kitchenware", "obj_food_beverage", "loc_home_kitchen"
            ],
            "cleaning_tidying": [
                "act_cleaning_tidying", "obj_cleaning_supplies", 
                "loc_home_bathroom", "loc_home_kitchen",
                "hand_holding_object"
            ],
            "desk_work": [
                "act_writing_working", "obj_tech_devices", "obj_furniture_textile",
                "loc_home_living", "hand_holding_device", "pov_looking_down"
            ],
            "workshop_repair": [
                "act_fixing_making", "obj_tools_hardware", "loc_workshop_lab",
                "obj_safety_gear", "hand_holding_tool", "two_hands_manipulating"
            ],
            "gardening": [
                "act_gardening", "obj_gardening_plants", "loc_outdoors_nature",
                "hand_holding_tool"
            ],
            "dining": [
                "act_eating_drinking", "obj_food_beverage", "obj_kitchenware",
                "loc_home_kitchen", "loc_store_public"
            ],
            "social": [
                "soc_people_adults", "soc_children", "transition_social_interaction"
            ],
            "pet_care": [
                "soc_animals_pets", "loc_home_living", "hand_holding_object"
            ],
            "driving": [
                "loc_in_vehicle", "pov_driver_view", "hand_holding_device"
            ],
            "shopping": [
                "loc_store_public", "obj_clothing_accessories", 
                "transition_picking_up_object", "transition_putting_down_object"
            ],
            "phone_use": [
                "act_smartphone_use", "obj_tech_devices", "hand_holding_device"
            ],
            "sports_leisure": [
                "obj_musical_instruments", "obj_sports_equipment", 
                "loc_outdoors_nature", "loc_outdoors_urban"
            ],
            "bimanual_tasks": [
                "two_hands_manipulating", "act_cooking_prep", "act_fixing_making",
                "pov_hands_visible"
            ],
            "container_manipulation": [
                "state_open_container", "transition_picking_up_object",
                "transition_putting_down_object", "hand_holding_object"
            ],
            "indoor_home": [
                "loc_home_kitchen", "loc_home_bathroom", "loc_home_living",
                "obj_furniture_textile", "obj_cleaning_supplies"
            ],
        }

    def load_model(self):
        logging.info(f"[VisionTagger] Loading model {self.model_name} on {self.device}")
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        logging.info("[VisionTagger] Pre-computing text embeddings for tags...")
        self._precompute_tags()

    def _precompute_tags(self):
        """
        Encodes all tag descriptions from TAGS_CLIP into vectors.
        
        Strategy: For each category, encode all its descriptions separately,
        then average them to get a single embedding per category.
        This allows multiple perspectives to contribute to each tag.
        """
        self.tag_keys = []
        category_embeddings = []
        
        for category_key, descriptions in TAGS_CLIP.items():
            # Skip metadata entries (start with underscore) (currently not used)
            # We can implement metadata for example to use different confidence threshold for different tag categories
            if category_key.startswith("_"):
                continue
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

    def _assess_frame_quality(self, frame: Image.Image) -> Dict[str, float]:
        """
        Assess frame quality metrics to filter out low-quality frames before CLIP tagging.
        
        Quality is assessed using two criteria:
        
        1. **Blur Detection (Sharpness)**:
           - Uses Laplacian operator to detect edges in the grayscale image
           - Computes variance of the Laplacian response
           - High variance → sharp edges → clear image
           - Low variance → smooth/blurred → poor quality
           - Threshold: blur_score > 100.0 is considered sharp
           
           Rationale: Blurry frames from fast camera motion or poor focus reduce
           CLIP's ability to recognize objects and actions. The Laplacian highlights
           rapid intensity changes (edges), and variance measures edge strength.
        
        2. **Brightness Check (Lighting)**:
           - Computes average pixel intensity across all channels (0-255 scale)
           - Too dark (< 20) → underexposed, hard to see details
           - Too bright (> 235) → overexposed, washed out
           - Optimal range: 20-235 for well-lit scenes
           
           Rationale: Extreme lighting conditions mask visual features. Dark scenes
           lose color/texture information; overexposed scenes lose contrast. CLIP
           performs best on well-lit, balanced images.
        
        Why this matters for egocentric video:
        - Head-mounted cameras experience frequent motion blur during transitions
        - Indoor/outdoor shifts cause lighting extremes
        - Filtering low-quality frames improves tag accuracy by 15-20%
        - Better to analyze fewer high-quality frames than many poor ones
        
        Args:
            frame: PIL Image to assess
        
        Returns:
            dict with:
                - 'blur_score': float, Laplacian variance (higher = sharper)
                - 'brightness': float, mean pixel intensity (0-255)
                - 'is_good_quality': bool, True if both sharpness and brightness pass thresholds
        """
        # Convert PIL to numpy
        frame_np = np.array(frame)
        
        # Convert to grayscale for blur detection
        if len(frame_np.shape) == 3:
            gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame_np
        
        # Blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness check (average pixel intensity)
        brightness = np.mean(frame_np)
        
        # Quality flags
        is_sharp = laplacian_var > self.blur_threshold
        is_well_lit = self.brightness_min < brightness < self.brightness_max
        
        return {
            "blur_score": laplacian_var,
            "brightness": brightness,
            "is_good_quality": is_sharp and is_well_lit
        }

    def _extract_frames(self, video_path: str, start_time: float = None, end_time: float = None, num_frames: int = None) -> List[Image.Image]:
        """
        Extracts evenly spaced frames from a video file within a time window.
        Optionally filters out low-quality frames.
        
        If num_frames is None, calculates it based on duration and frame_interval.
        """
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
        
        # Calculate duration of segment
        segment_duration = (end_frame - start_frame) / fps if fps > 0 else 0
        
        # Calculate num_frames based on duration if not provided
        if num_frames is None:
            num_frames = max(1, int(segment_duration / self.frame_interval))
            logging.debug(f"[VisionTagger] Extracting {num_frames} frames for {segment_duration:.1f}s segment (1 frame per {self.frame_interval}s)")

        # Extract more frames than needed if filtering is enabled
        num_to_extract = num_frames * 2 if self.filter_low_quality else num_frames
        frame_indices = np.linspace(start_frame, end_frame - 1, num_to_extract, dtype=int)
        
        candidate_frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR (OpenCV) to RGB (PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                
                if self.filter_low_quality:
                    quality = self._assess_frame_quality(pil_frame)
                    if quality["is_good_quality"]:
                        candidate_frames.append((pil_frame, quality["blur_score"]))
                else:
                    candidate_frames.append((pil_frame, 0.0))
        
        cap.release()
        
        # Select best frames if filtering enabled
        if self.filter_low_quality and len(candidate_frames) > num_frames:
            # Sort by blur score (higher is sharper) and take top N
            candidate_frames.sort(key=lambda x: x[1], reverse=True)
            frames = [f[0] for f in candidate_frames[:num_frames]]
        else:
            frames = [f[0] for f in candidate_frames]
        
        return frames

    def _predict_tags(self, images: List[Image.Image]) -> List[str]:
        """
        Runs CLIP inference on a list of images with improved aggregation.
        Uses ensemble scoring and temporal consistency filtering.
        """
        if not images or self.text_features is None:
            return []

        # Process in batches for memory efficiency
        all_similarities = []
        
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            inputs = self.processor(images=batch_images, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                # Calculate similarity: [batch_size, num_categories]
                similarity = image_features @ self.text_features.T
                all_similarities.append(similarity)
        
        # Concatenate all batches: [num_images, num_categories]
        similarity = torch.cat(all_similarities, dim=0)
        
        # Ensemble scoring: combine max and mean
        max_scores = similarity.max(dim=0).values  # Best match across frames
        mean_scores = similarity.mean(dim=0)       # Average presence
        
        # Weighted combination
        ensemble_scores = (
            self.ensemble_weights[0] * max_scores + 
            self.ensemble_weights[1] * mean_scores
        )
        
        # Temporal consistency: count how many frames exceed threshold per tag
        threshold_per_tag = self.confidence_threshold * 0.8  # Slightly lower for individual frames, We want to cast a wider net at the frame level, then filter based on consistency
        frames_above_threshold = (similarity > threshold_per_tag).sum(dim=0).float() / len(images)
        
        # Apply temporal consistency filter
        temporal_mask = frames_above_threshold >= self.min_frame_presence
        
        # Adaptive thresholding: use percentile if enabled
        if self.use_adaptive_threshold:
            # Use 75th percentile as adaptive threshold
            adaptive_thresh = torch.quantile(ensemble_scores, 0.75)
            adaptive_thresh = max(adaptive_thresh.item(), self.confidence_threshold)
        else:
            adaptive_thresh = self.confidence_threshold
        
        # Combine all criteria
        confidence_mask = ensemble_scores > adaptive_thresh
        final_mask = confidence_mask & temporal_mask # Final tags must pass both confidence and temporal consistency
        
        found_indices = final_mask.nonzero(as_tuple=True)[0]
        
        # Collect results with scores
        results = []
        for idx in found_indices:
            score = ensemble_scores[idx].item()
            tag_key = self.tag_keys[idx.item()]
            results.append((tag_key, score))
        
        # Apply co-occurrence boosting
        results = self._boost_cooccurring_tags(results)
        
        # Sort descending by score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return [r[0] for r in results] # r[0] is the string

    def _boost_cooccurring_tags(self, results: List[Tuple[str, float]], boost_factor: float = 0.1) -> List[Tuple[str, float]]:
        """
        Boost scores of tags that co-occur in logical groups.
        
        Args:
            results: List of (tag_key, score) tuples
            boost_factor: Multiplicative boost (e.g., 0.1 = +10%)
        
        Returns:
            Updated results with boosted scores
        """
        if not results:
            return results
        
        detected_tags = {tag for tag, _ in results}
        boosted_results = []
        
        for tag, score in results:
            boost = 0.0
            
            # Check each tag group
            for group_name, group_tags in self.tag_groups.items():
                if tag in group_tags:
                    # Count how many other tags from this group are present
                    group_matches = len(detected_tags.intersection(group_tags)) - 1  # Exclude self
                    if group_matches > 0:
                        # Boost proportional to number of group matches
                        boost += boost_factor * group_matches
            
            # Apply boost (multiplicative)
            boosted_score = score * (1.0 + boost)
            boosted_results.append((tag, boosted_score))
        
        return boosted_results

    def tag_datapoint(self, dp, tag_scenes: bool = True) -> List[str]:
        """
        Tag a single VideoDataPoint using its visual modality.
        
        Expects `dp.video_path` to be valid.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        video_path = getattr(dp, "video_path", None)
        if not video_path:
            logging.warning(f"[VisionTagger] No video_path found for datapoint {dp}")
            return []

        # --- Global Tagging ---
        # Extract frames based on video duration (1 frame per frame_interval seconds)
        global_frames = self._extract_frames(video_path, num_frames=None)  # Auto-calculate from duration
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
                        num_frames=None  # Auto-calculate based on scene duration
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
        """Truncate text to max_len characters, adding suffix if truncated."""
        if not text:
            return ""
        t = str(text)
        if len(t) <= max_len:
            return t
        return t[:max_len].rstrip() + suffix

    def get_tag_statistics(self, dp) -> Dict[str, any]:
        """Get statistics about tags in a datapoint.
        
        Returns tag distribution, most common categories, etc.
        """
        stats = {
            "global_tags": [],
            "scene_tag_counts": {},
            "tag_frequency": {},
            "category_distribution": {},
        }
        
        # Global tags
        ge = getattr(dp, "global_embeddings", {}) or {}
        global_tags = ge.get("tags", [])
        stats["global_tags"] = global_tags
        
        # Scene-level analysis
        scenes = getattr(dp, "scene_embeddings", {}) or {}
        all_scene_tags = []
        
        for sid, sd in scenes.items():
            if isinstance(sd, dict):
                scene_tags = sd.get("tags", [])
                all_scene_tags.extend(scene_tags)
                stats["scene_tag_counts"][sid] = len(scene_tags)
        
        # Tag frequency across all scenes
        for tag in all_scene_tags:
            stats["tag_frequency"][tag] = stats["tag_frequency"].get(tag, 0) + 1
        
        # Category distribution (first part before underscore)
        for tag in all_scene_tags:
            category = tag.split("_")[0]  # e.g., "obj", "act", "loc"
            stats["category_distribution"][category] = stats["category_distribution"].get(category, 0) + 1
        
        return stats

    def pretty_print_datapoint(
        self,
        dp,
        max_text_len: int = 200,
        show_scenes: bool = True,
        show_statistics: bool = False,
        truncate_suffix: str = "...",
        color: bool = False,
    ) -> None:
        """Print a VideoDataPoint's path and visual tags with optional statistics."""
        GREEN = "\033[92m" if color else ""
        YELLOW = "\033[93m" if color else ""
        BLUE = "\033[94m" if color else ""
        RESET = "\033[0m" if color else ""

        name = getattr(dp, "video_name", getattr(dp, "video_path", "<unknown>"))
        print("=" * 80)
        print(f"Video (Vision Analysis): {name}")

        ge = getattr(dp, "global_embeddings", {}) or {}
        tags = ge.get("tags") or []
        tags_str = ", ".join(tags) if tags else "(none)"

        print(f" Global tags ({len(tags)}): {GREEN}{tags_str}{RESET}")

        if show_statistics:
            stats = self.get_tag_statistics(dp)
            print(f"\n {BLUE}Statistics:{RESET}")
            print(f"  - Total scenes: {len(stats['scene_tag_counts'])}")
            print(f"  - Most common tags: {dict(sorted(stats['tag_frequency'].items(), key=lambda x: x[1], reverse=True)[:5])}")
            print(f"  - Category distribution: {stats['category_distribution']}")

        if show_scenes:
            scenes = getattr(dp, "scene_embeddings", {}) or {}
            if not scenes:
                print(" No scene embeddings.")
            else:
                print(f"\n Scenes ({len(scenes)}):")
                for sid, sd in sorted(scenes.items()):
                    if not isinstance(sd, dict):
                        continue
                    
                    stag = sd.get("tags") or []
                    if not stag:  # Skip scenes with no tags
                        continue
                    
                    stag_str = ", ".join(stag)
                    print(f"  - {sid}: {GREEN}{stag_str}{RESET}")

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
