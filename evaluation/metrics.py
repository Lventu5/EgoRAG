"""
Retrieval Metrics for Ego4D NLQ / Moment Retrieval Evaluation.

This module implements the standard metrics for temporal moment retrieval:
1. tIoU (Temporal Intersection over Union) - base helper function
2. Mean tIoU (Overlap) - average of best tIoU for Top-1 prediction per query
3. Recall@K with IoU Thresholds - hit rate at different K and IoU thresholds

Ground truth format supports multiple valid windows per query.
"""

from abc import ABC, abstractmethod
import numpy as np
import bert_score
from data.video_dataset import Scene
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_temporal_iou(pred_start: float, pred_end: float, 
                           gt_start: float, gt_end: float, use_gt_len: bool = False) -> float:
    """
    Calculate Temporal Intersection over Union (tIoU) between two time intervals.
    
    Args:
        pred_start: Predicted segment start time in seconds
        pred_end: Predicted segment end time in seconds
        gt_start: Ground truth segment start time in seconds
        gt_end: Ground truth segment end time in seconds
    
    Returns:
        tIoU value between 0.0 and 1.0
        Returns 0.0 if either segment has zero or negative length
    """
    # Validate segment lengths
    pred_duration = pred_end - pred_start
    gt_duration = gt_end - gt_start
    
    if pred_duration <= 0 or gt_duration <= 0:
        return 0.0
    
    # Calculate intersection
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0.0, intersection_end - intersection_start)
    
    # Calculate union (Base IoU metric)
    if use_gt_len:
        union = gt_duration
    else:
        union = pred_duration + gt_duration - intersection
    
    return intersection / union if union > 0 else 0.0


def scene_to_interval(scene: Scene) -> tuple[float, float]:
    """Extract start and end times from a Scene object."""
    if scene is None:
        return (0.0, 0.0)
    return (scene.start_time, scene.end_time)


def compute_best_iou_against_ground_truths(
    pred_scene: Scene,
    ground_truths: list[tuple[float, float]]
) -> float:
    """
    Compute the maximum tIoU between a predicted scene and multiple ground truth windows.
    
    Args:
        pred_scene: The predicted Scene object
        ground_truths: List of (start_time, end_time) tuples for valid GT windows
    
    Returns:
        Maximum tIoU found among all ground truth windows (0.0 if no valid GT)
    """
    if pred_scene is None or not ground_truths:
        return 0.0
    
    pred_start, pred_end = scene_to_interval(pred_scene)
    
    best_iou = 0.0
    for gt_start, gt_end in ground_truths:
        iou = calculate_temporal_iou(pred_start, pred_end, gt_start, gt_end)
        best_iou = max(best_iou, iou)
    
    return best_iou


# =============================================================================
# Base Metric Class
# =============================================================================

class Metric(ABC):
    """Abstract base class for all metrics."""
    
    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def compute(self, **kwargs) -> float:
        pass

    def __call__(self, **kwargs) -> float:
        return self.compute(**kwargs)
    
    def __repr__(self) -> str:
        return f"Metric: {self.name}"


# =============================================================================
# Retrieval Metrics
# =============================================================================

class TemporalIoU(Metric):
    """
    Plain tIoU metric.
    
    For each query, computes the tIoU between the Top-1 predicted scene and
    the ground truth window. Returns the average tIoU across all queries.
    
    Note: This assumes a single ground truth window per query. For multiple
    ground truth windows, use MeanTemporalIoU instead.
    """
    
    def __init__(self, name: str | None = None):
        super().__init__(name=name or "tIoU")
    
    def compute(
        self,
        pred: list[list[tuple[str, Scene]]],
        true: list[tuple[str, float, float]]
    ) -> float:
        """
        Compute average tIoU for Top-1 predictions.
        
        Args:
            pred: List of predictions per query. Each prediction is a list of
                  (video_name, Scene) tuples, ordered by relevance (best first).
            true: List of ground truths per query as (video_uid, start_sec, end_sec).
        
        Returns:
            Average tIoU across all queries.
        """
        assert len(pred) == len(true), "Predictions and ground truths must have the same length"
        
        n = len(pred)
        if n == 0:
            return 0.0
        
        total_iou = 0.0
        
        for i in range(n):
            predictions = pred[i]
            gt_video, gt_start, gt_end = true[i]
            print("Query", i, "Len GT:", gt_end-gt_start)
            
            # Handle empty predictions
            if not predictions:
                total_iou += 0.0
                continue
            
            # Get Top-1 prediction
            top1_video, top1_scene = predictions[0]
            
            # Only compute IoU if video matches
            if top1_video != gt_video:
                total_iou += 0.0
                continue
            
            # Compute tIoU
            pred_start, pred_end = scene_to_interval(top1_scene)
            iou = calculate_temporal_iou(pred_start, pred_end, float(gt_start), float(gt_end))
            total_iou += iou
        
        return total_iou / n


class MeanTemporalIoU(Metric):
    """
    Mean Temporal IoU (mIoU) metric.
    
    For each query:
    1. Get the Top-1 predicted scene.
    2. Compute tIoU between Top-1 scene and the ground truth window.
    3. If no predictions or video doesn't match, assign 0.0.
    
    Final metric is the average tIoU across all queries.
    """
    
    def __init__(self, name: str | None = None):
        super().__init__(name=name or "mIoU")
    
    def compute(
        self,
        pred: list[list[tuple[str, Scene]]],
        true: list[tuple[str, float, float]]
    ) -> float:
        """
        Compute Mean tIoU for Top-1 predictions.
        
        Args:
            pred: List of predictions per query. Each prediction is a list of
                  (video_name, Scene) tuples, ordered by relevance (best first).
            true: List of ground truths per query as (video_uid, start_sec, end_sec).
        
        Returns:
            Mean tIoU across all queries.
        """
        assert len(pred) == len(true), "Predictions and ground truths must have the same length"
        
        n = len(pred)
        if n == 0:
            return 0.0
        
        total_iou = 0.0
        
        for i in range(n):
            predictions = pred[i]
            gt_video, gt_start, gt_end = true[i]
            
            # Handle empty predictions
            if not predictions:
                total_iou += 0.0
                continue
            
            # Get Top-1 prediction
            top1_video, top1_scene = predictions[0]
            
            # Only compute IoU if video matches
            if top1_video != gt_video:
                total_iou += 0.0
                continue
            
            # Compute tIoU
            pred_start, pred_end = scene_to_interval(top1_scene)
            iou = calculate_temporal_iou(pred_start, pred_end, float(gt_start), float(gt_end))
            total_iou += iou
        
        return total_iou / n


class Overlap(Metric):
    """
    Overlap metric - ratio of predictions that overlap with the ground truth.
    
    For each query:
    1. Count how many of the predicted scenes have ANY overlap (tIoU > 0) with GT.
    2. Divide by total number of predictions.
    
    Final metric is the average overlap ratio across all queries.
    
    Note: A scene "overlaps" if it has tIoU > 0 with the ground truth.
    """
    
    def __init__(self, name: str | None = None):
        super().__init__(name=name or "Overlap")
    
    def compute(
        self,
        pred: list[list[tuple[str, Scene]]],
        true: list[tuple[str, float, float]]
    ) -> float:
        """
        Compute Overlap ratio.
        
        Args:
            pred: List of predictions per query. Each prediction is a list of
                  (video_name, Scene) tuples, ordered by relevance (best first).
            true: List of ground truths per query as (video_uid, start_sec, end_sec).
        
        Returns:
            Average ratio of predictions that overlap with GT across all queries.
        """
        assert len(pred) == len(true), "Predictions and ground truths must have the same length"
        
        n = len(pred)
        if n == 0:
            return 0.0
        
        total_overlap_ratio = 0.0
        
        for i in range(n):
            predictions = pred[i]
            gt_video, gt_start, gt_end = true[i]
            
            # Handle empty predictions
            if not predictions:
                total_overlap_ratio += 0.0
                continue
            
            # Count predictions that overlap with GT
            overlapping_count = 0
            for pred_video, pred_scene in predictions:
                # Video must match
                if pred_video != gt_video:
                    continue
                
                # Check if there's any overlap (tIoU > 0)
                pred_start, pred_end = scene_to_interval(pred_scene)
                iou = calculate_temporal_iou(pred_start, pred_end, float(gt_start), float(gt_end))
                if iou > 0:
                    overlapping_count += 1
            
            # Compute ratio for this query
            overlap_ratio = overlapping_count / len(predictions)
            total_overlap_ratio += overlap_ratio
        
        return total_overlap_ratio / n


class RecallAtKIoU(Metric):
    """
    Recall@K with IoU Threshold metric.
    
    For each query:
    1. Slice predictions to keep only Top-K scenes.
    2. For each scene in Top-K, compute tIoU against the ground truth.
    3. If any scene achieves tIoU >= threshold, count as Hit (1), else Miss (0).
    
    Final metric is the hit rate across all queries.
    
    Standard configurations:
    - R@1, IoU=0.3: Top-1 scene, threshold 0.3
    - R@1, IoU=0.5: Top-1 scene, threshold 0.5
    - R@K, IoU=0.3: Top-K scenes, threshold 0.3
    - R@K, IoU=0.5: Top-K scenes, threshold 0.5
    """
    
    def __init__(self, k: int = 1, iou_threshold: float = 0.5, name: str | None = None):
        self.k = k
        self.iou_threshold = iou_threshold
        super().__init__(name=name or f"R@{k}_IoU={iou_threshold}")
    
    def compute(
        self,
        pred: list[list[tuple[str, Scene]]],
        true: list[tuple[str, float, float]]
    ) -> float:
        """
        Compute Recall@K with IoU threshold.
        
        Args:
            pred: List of predictions per query. Each prediction is a list of
                  (video_name, Scene) tuples, ordered by relevance (best first).
            true: List of ground truths per query as (video_uid, start_sec, end_sec).
        
        Returns:
            Recall@K (hit rate) with the specified IoU threshold.
        """
        assert len(pred) == len(true), "Predictions and ground truths must have the same length"
        
        n = len(pred)
        if n == 0:
            return 0.0
        
        hits = 0
        
        for i in range(n):
            predictions = pred[i][:self.k]  # Top-K only
            gt_video, gt_start, gt_end = true[i]
            
            # Check if any prediction in Top-K is a hit
            is_hit = self._check_hit(predictions, gt_video, float(gt_start), float(gt_end))
            if is_hit:
                hits += 1
        
        return hits / n
    
    def _check_hit(
        self,
        predictions: list[tuple[str, Scene]],
        gt_video: str,
        gt_start: float,
        gt_end: float
    ) -> bool:
        """Check if any prediction achieves IoU >= threshold with the GT."""
        for pred_video, pred_scene in predictions:
            # Video must match
            if pred_video != gt_video:
                continue
            
            # Check IoU against GT
            pred_start, pred_end = scene_to_interval(pred_scene)
            iou = calculate_temporal_iou(pred_start, pred_end, gt_start, gt_end)
            if iou >= self.iou_threshold:
                return True
        
        return False

class SimpleRecallAtK(Metric):
    
    def __init__(self, k: int = 1, iou_threshold: float = 0.5, name: str | None = None):
        self.k = k
        super().__init__(name=name or f"SimpleRecall@{k}")
    
    def compute(
        self,
        pred: list[list[tuple[str, Scene]]],
        true: list[tuple[str, float, float]]
    ) -> float:
        """
        Compute Recall@K with IoU threshold.
        
        Args:
            pred: List of predictions per query. Each prediction is a list of
                  (video_name, Scene) tuples, ordered by relevance (best first).
            true: List of ground truths per query as (video_uid, start_sec, end_sec).
        
        Returns:
            Recall@K: Average coverage of the ground truth scene
        """
        assert len(pred) == len(true), "Predictions and ground truths must have the same length"
        
        n = len(pred)
        if n == 0:
            return 0.0
        
        overlap = 0
        
        for i in range(n):
            predictions = pred[i][:self.k]  # Top-K only
            gt_video, gt_start, gt_end = true[i]
            
            # Check if any prediction in Top-K is a hit
            overlap += self._compute_overlappings(predictions, gt_video, float(gt_start), float(gt_end))

        return overlap / n
    
    def _compute_overlappings(
        self,
        predictions: list[tuple[str, Scene]],
        gt_video: str,
        gt_start: float,
        gt_end: float
    ) -> bool:
        """compute total_overlapping of the query to the top-k predictions"""
        total_overlapping = 0
        for pred_video, pred_scene in predictions:
            # Video must match
            if pred_video != gt_video:
                continue
            
            # Check IoU against GT
            pred_start, pred_end = scene_to_interval(pred_scene)
            iou = calculate_temporal_iou(pred_start, pred_end, gt_start, gt_end, use_gt_len=True)
            total_overlapping += iou

        return total_overlapping


# =============================================================================
# Aggregated Evaluation Function
# =============================================================================

def evaluate_retrieval(
    predictions: list[list[tuple[str, Scene]]],
    ground_truths: list[tuple[str, float, float]],
    k_values: list[int] = [1, 3, 5, 10],
    iou_thresholds: list[float] = [0.3, 0.5]
) -> dict[str, float]:
    """
    Evaluate retrieval performance with all standard metrics.
    
    Computes:
    - mIoU (Mean tIoU): Average tIoU for Top-1 predictions
    - tIoU: Plain temporal IoU for Top-1 (same as mIoU)
    - Overlap: Ratio of predictions that overlap with GT
    - R@K, IoU=τ: Recall at K with various IoU thresholds
    
    Args:
        predictions: List of predictions per query. Each is a list of
                     (video_name, Scene) tuples, ordered by relevance.
        ground_truths: List of ground truths per query as (video_uid, start_sec, end_sec).
        k_values: List of K values for Recall@K (default: [1, 5, 10])
        iou_thresholds: List of IoU thresholds (default: [0.3, 0.5])
    
    Returns:
        Dictionary with metric names as keys and values as floats.
        Example: {"mIoU": 0.45, "Overlap": 0.60, "R@1_IoU=0.3": 0.55, ...}
    """
    results = {}
    
    # Compute mIoU
    miou_metric = MeanTemporalIoU()
    results["mIoU"] = miou_metric.compute(pred=predictions, true=ground_truths)
    
    # Compute plain tIoU (same as mIoU for single GT)
    tiou_metric = TemporalIoU()
    results["tIoU"] = tiou_metric.compute(pred=predictions, true=ground_truths)
    
    # Compute Overlap (ratio of predictions that overlap with GT)
    overlap_metric = Overlap()
    results["Overlap"] = overlap_metric.compute(pred=predictions, true=ground_truths)
    
    # Compute Recall@K with IoU thresholds
    for k in k_values:
        for iou_thresh in iou_thresholds:
            recall_metric = RecallAtKIoU(k=k, iou_threshold=iou_thresh)
            metric_name = f"R@{k}_IoU={iou_thresh}"
            results[metric_name] = recall_metric.compute(pred=predictions, true=ground_truths)
        simple_recall = SimpleRecallAtK(k = k)
        results[f"SimpleRecall@{k}"] = simple_recall.compute(pred=predictions, true=ground_truths)
    
    return results