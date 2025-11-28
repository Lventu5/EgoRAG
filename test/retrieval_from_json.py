from __future__ import annotations

import torch
import os
import sys
import logging
import warnings

from data.dataset import Ego4DDataset
from retrieval.hierarchical_retriever import HierarchicalRetriever
from evaluation.evaluator import RetrievalEvaluator

from configuration.config import CONFIG

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["HF_HOME"] = os.environ.get("TRANSFORMERS_CACHE", "")

def convert_and_evaluate(retrieval_results: dict, queries, evaluator: RetrievalEvaluator):
    """Convert hierarchical retriever output to metric inputs and run evaluator.

    Args:
        retrieval_results: output of HierarchicalRetriever.retrieve_hierarchically()
        queries: iterable of Query objects (order must match retrieval_results keys expected)
        evaluator: an instance of RetrievalEvaluator

    Returns:
        dict of metric name -> value
    """
    preds = []
    trues = []

    for q in queries:
        qid = q.qid
        # `retrieval_results` can be a plain dict, a RetrievalResults wrapper,
        # or the detailed_results list returned by the retriever.
        entry = retrieval_results.get(qid, {})

        # entry can be either a dict with a 'fused' key, or a list of fused tuples
        if isinstance(entry, dict):
            fused_list = entry.get("fused", [])
        elif isinstance(entry, list):
            fused_list = entry
        else:
            fused_list = []

        # Flatten scenes with their scores: produce list of (video_name, Scene, score)
        # We need to track scores for global ranking
        query_preds_with_scores = []
        for video_name, global_score, scene_ranking in fused_list:
            if scene_ranking is None:
                continue
            for scene_item in scene_ranking:
                # scene_item may be (Scene, score) or Scene
                if isinstance(scene_item, (list, tuple)) and len(scene_item) >= 2:
                    scene_obj = scene_item[0]
                    scene_score = scene_item[1]
                elif isinstance(scene_item, (list, tuple)) and len(scene_item) == 1:
                    scene_obj = scene_item[0]
                    scene_score = global_score  # fallback to video score
                else:
                    scene_obj = scene_item
                    scene_score = global_score  # fallback to video score
                
                if scene_obj is None:
                    continue
                    
                query_preds_with_scores.append((video_name, scene_obj, scene_score))

        # Sort by scene score (descending) for proper global ranking
        query_preds_with_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Extract (video_name, Scene) tuples for evaluation
        query_preds = [(v, s) for v, s, _ in query_preds_with_scores]

        preds.append(query_preds)

        # Ground truth: extract start_sec and end_sec
        gt_video = q.video_uid
        gt_moment_start = None
        gt_moment_end = None
        
        # Safely extract GT
        if getattr(q, "gt", None):
            gt_moment_start = q.gt.get("start_sec")
            gt_moment_end = q.gt.get("end_sec")
        
        # Validate GT times: must be non-negative real numbers
        # If invalid, log a warning as IoU metrics will be meaningless for this query
        has_valid_gt = True
        if gt_moment_start is None or gt_moment_start < 0:
            has_valid_gt = False
            gt_moment_start = 0.0
        if gt_moment_end is None or gt_moment_end < 0:
            has_valid_gt = False
            gt_moment_end = gt_moment_start  # At least make it non-zero if start is valid
        
        if not has_valid_gt:
            logging.debug(f"Query {q.qid} has invalid GT times (start={q.gt.get('start_sec') if q.gt else None}, "
                         f"end={q.gt.get('end_sec') if q.gt else None}). IoU metrics will be 0.")

        trues.append((gt_video, float(gt_moment_start), float(gt_moment_end)))

    # Log statistics about the evaluation data
    num_empty_preds = sum(1 for p in preds if len(p) == 0)
    num_invalid_gt = sum(1 for t in trues if t[1] >= t[2])  # start >= end means invalid
    avg_preds_per_query = sum(len(p) for p in preds) / len(preds) if preds else 0
    
    logging.info(f"Evaluation stats: {len(preds)} queries, {num_empty_preds} with empty predictions, "
                 f"{num_invalid_gt} with invalid GT times, avg {avg_preds_per_query:.1f} predictions/query")

    return evaluator.forward_pass(pred=preds, true=trues)


def main(
        video_pickle: str, 
        annotations: str, 
        modalities: list[str], 
        topk_videos: int = 3, 
        topk_scenes: int = 1, 
        device: str = "cuda",
        skip_video_retrieval: bool = False
    ):
    logging.basicConfig(level=logging.INFO)
    if not torch.cuda.is_available() and device == "cuda":
        logging.warning("CUDA not available, switching to CPU")
        device = "cpu"

    # Validate paths
    if not os.path.exists(video_pickle):
        logging.error(f"Video pickle not found: {video_pickle}")
        sys.exit(1)
    if not os.path.exists(annotations):
        logging.error(f"Annotations file not found: {annotations}")
        sys.exit(1)

    logging.info("Loading Data and Queries...")

    dataset = Ego4DDataset(video_pickle, annotations)
    video_dataset = dataset.load_videos(is_pickle=True)
    video_ids = video_dataset.get_uids()
    annotations_dict = dataset.load_annotations(video_ids)
    query_dataset = dataset.load_queries(video_ids)

    logging.info(f"Loaded {len(video_dataset.video_datapoints)} videos from pickle")
    logging.info(f"Loaded {len(query_dataset.queries)} queries matching the pickled videos")

    # Instantiate retriever
    retriever = HierarchicalRetriever(video_dataset=video_dataset, device=device)

    if skip_video_retrieval:
        logging.info("Running scene-only retrieval (using ground truth videos)...")
    else:
        logging.info("Running hierarchical retrieval...")
    
    retrieval_results = retriever.retrieve_hierarchically(
        queries=query_dataset,
        modalities=modalities,
        top_k_videos=topk_videos,
        top_k_scenes=topk_scenes,
        skip_video_retrieval=skip_video_retrieval,
    )

    logging.info("Running evaluation...")
    evaluator = RetrievalEvaluator()
    metrics = convert_and_evaluate(retrieval_results, query_dataset, evaluator)

    print("Evaluation results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    video_pickle = CONFIG.data.video_dataset
    annotations = CONFIG.data.annotation_path
    modalities = CONFIG.retrieval.modalities
    topk_videos = CONFIG.retrieval.top_k_videos
    topk_scenes = CONFIG.retrieval.top_k_scenes
    device = CONFIG.device
    skip_video_retrieval = getattr(CONFIG.retrieval, 'skip_video_retrieval', False)

    main(video_pickle, annotations, modalities, topk_videos, topk_scenes, device, skip_video_retrieval)
