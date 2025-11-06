"""
Evaluation Metrics: Metrics for retrieval and QA evaluation.

This module provides standard metrics for evaluating video retrieval
and question answering performance.
"""

from typing import List, Tuple
from data.video_dataset import Scene
from indexing.utils.logging import get_logger

logger = get_logger(__name__)


def compute_iou(interval1: Tuple[float, float], interval2: Tuple[float, float]) -> float:
    """
    Compute Intersection over Union (IoU) for two time intervals.
    
    Args:
        interval1: (start, end) tuple
        interval2: (start, end) tuple
        
    Returns:
        IoU score in [0, 1]
    """
    start1, end1 = interval1
    start2, end2 = interval2
    
    # Compute intersection
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    intersection = max(0, inter_end - inter_start)
    
    # Compute union
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = union_end - union_start
    
    if union == 0:
        return 0.0
    
    return intersection / union


def recall_at_k_scene(
    gt_intervals: List[Tuple[float, float]],
    pred_scenes: List[Scene],
    iou_threshold: float = 0.5,
    k: int = 5
) -> float:
    """
    Compute Recall@K for scene retrieval.
    
    A predicted scene is considered correct if its IoU with any ground truth
    interval exceeds the threshold.
    
    Args:
        gt_intervals: List of (start_sec, end_sec) ground truth intervals
        pred_scenes: List of predicted Scene objects
        iou_threshold: Minimum IoU to consider a match
        k: Number of top predictions to consider
        
    Returns:
        Recall@K score in [0, 1]
    """
    if not gt_intervals:
        return 0.0
    
    # Consider only top-k predictions
    top_k_preds = pred_scenes[:k]
    
    # Track which ground truth intervals have been matched
    matched_gt = set()
    
    for pred_scene in top_k_preds:
        pred_interval = (pred_scene.start_time, pred_scene.end_time)
        
        for gt_idx, gt_interval in enumerate(gt_intervals):
            if gt_idx in matched_gt:
                continue
            
            iou = compute_iou(pred_interval, gt_interval)
            if iou >= iou_threshold:
                matched_gt.add(gt_idx)
                break  # Move to next prediction
    
    recall = len(matched_gt) / len(gt_intervals)
    return recall


def mean_reciprocal_rank(ranked_list: List[str], gt_video: str) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) for video retrieval.
    
    Args:
        ranked_list: List of video names in ranked order
        gt_video: Ground truth video name
        
    Returns:
        MRR score (1/rank if found, 0 otherwise)
    """
    try:
        rank = ranked_list.index(gt_video) + 1  # 1-indexed
        return 1.0 / rank
    except ValueError:
        return 0.0


def qa_exact_match(pred: str, gold: str) -> float:
    """
    Compute exact match score for QA.
    
    Performs case-insensitive comparison after normalizing whitespace.
    
    Args:
        pred: Predicted answer
        gold: Gold answer
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    # Normalize: lowercase, strip, collapse whitespace
    pred_norm = " ".join(pred.lower().strip().split())
    gold_norm = " ".join(gold.lower().strip().split())
    
    return 1.0 if pred_norm == gold_norm else 0.0


def qa_f1(pred: str, gold: str) -> float:
    """
    Compute token-level F1 score for QA.
    
    Args:
        pred: Predicted answer
        gold: Gold answer
        
    Returns:
        F1 score in [0, 1]
    """
    # Tokenize: lowercase, split on whitespace
    pred_tokens = set(pred.lower().strip().split())
    gold_tokens = set(gold.lower().strip().split())
    
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    # Compute precision and recall
    common = pred_tokens & gold_tokens
    
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    
    # Compute F1
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def precision_at_k(relevant_items: List[str], retrieved_items: List[str], k: int) -> float:
    """
    Compute Precision@K.
    
    Args:
        relevant_items: List of relevant item identifiers
        retrieved_items: List of retrieved item identifiers (ranked)
        k: Number of top items to consider
        
    Returns:
        Precision@K in [0, 1]
    """
    if k == 0 or not retrieved_items:
        return 0.0
    
    relevant_set = set(relevant_items)
    top_k = retrieved_items[:k]
    
    num_relevant = sum(1 for item in top_k if item in relevant_set)
    
    return num_relevant / k


def average_precision(relevant_items: List[str], retrieved_items: List[str]) -> float:
    """
    Compute Average Precision (AP).
    
    Args:
        relevant_items: List of relevant item identifiers
        retrieved_items: List of retrieved item identifiers (ranked)
        
    Returns:
        AP score in [0, 1]
    """
    if not relevant_items or not retrieved_items:
        return 0.0
    
    relevant_set = set(relevant_items)
    
    precision_sum = 0.0
    num_relevant_retrieved = 0
    
    for k, item in enumerate(retrieved_items, start=1):
        if item in relevant_set:
            num_relevant_retrieved += 1
            precision_sum += num_relevant_retrieved / k
    
    if num_relevant_retrieved == 0:
        return 0.0
    
    return precision_sum / len(relevant_items)


def mean_average_precision(
    all_relevant: List[List[str]],
    all_retrieved: List[List[str]]
) -> float:
    """
    Compute Mean Average Precision (MAP) across multiple queries.
    
    Args:
        all_relevant: List of relevant item lists (one per query)
        all_retrieved: List of retrieved item lists (one per query)
        
    Returns:
        MAP score in [0, 1]
    """
    if not all_relevant or len(all_relevant) != len(all_retrieved):
        return 0.0
    
    ap_scores = [
        average_precision(relevant, retrieved)
        for relevant, retrieved in zip(all_relevant, all_retrieved)
    ]
    
    return sum(ap_scores) / len(ap_scores)
