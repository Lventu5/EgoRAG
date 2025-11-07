from __future__ import annotations

import torch
import os
import sys
import logging

from data.dataset import Ego4DDataset
from retrieval.hierarchical_retriever import HierarchicalRetriever
from evaluation.evaluator import RetrievalEvaluator


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

    for q in queries[:1]:
        qid = q.qid
        # `retrieval_results` can be a plain dict, a RetrievalResults wrapper,
        # or the detailed_results list returned by the retriever.
        if hasattr(retrieval_results, "get"):
            entry = retrieval_results.get(qid, {})
        else:
            try:
                entry = retrieval_results[qid]
            except Exception:
                entry = {}

        # entry can be either a dict with a 'fused' key, or a list of fused tuples
        if isinstance(entry, dict):
            fused_list = entry.get("fused", [])
        elif isinstance(entry, list):
            fused_list = entry
        else:
            fused_list = []

        # flatten scenes: produce list of (video_name, Scene)
        query_preds = []
        for video_name, global_score, scene_ranking in fused_list:
            if scene_ranking is None:
                continue
            for scene_item in scene_ranking:
                # scene_item may be (Scene, score) or Scene
                if isinstance(scene_item, (list, tuple)) and len(scene_item) >= 1:
                    scene_obj = scene_item[0]
                else:
                    scene_obj = scene_item
                query_preds.append((video_name, scene_obj))

        preds.append(query_preds)

        # Ground truth: try start_sec, else end_sec, else None
        gt_video = q.video_uid
        if getattr(q, "gt", None):
            if q.gt.get("start_sec") is not None and q.gt.get("start_sec") >= 0:
                gt_moment_start = q.gt.get("start_sec")
            if q.gt.get("end_sec") is not None and q.gt.get("end_sec") >= 0:
                gt_moment_end = q.gt.get("end_sec")

        trues.append((gt_video, gt_moment_start, gt_moment_end))

    return evaluator.forward_pass(pred=preds, true=trues)


def main(
        video_pickle: str, 
        annotations: str, 
        modalities: list[str], 
        topk_videos: int = 3, 
        topk_scenes: int = 1, 
        device: str = "cuda"
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

    logging.info("Running hierarchical retrieval...")
    retrieval_results = retriever.retrieve_hierarchically(
        queries=query_dataset,
        modalities=modalities,
        top_k_videos=topk_videos,
        top_k_scenes=topk_scenes,
    )

    logging.info("Running evaluation...")
    evaluator = RetrievalEvaluator()
    metrics = convert_and_evaluate(retrieval_results, query_dataset, evaluator)

    print("Evaluation results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    video_pickle = "../ego4d_data/v2/noframe_encoded_videos/2b5569df-5deb-4ebd-8a45-dd6524330eb8_encoded.pkl"
    annotations = "../ego4d_data/v2/annotations/nlq_train.json"
    modalities = ["video", "caption", "text"]
    topk_videos = 1
    topk_scenes = 10
    device = "cuda"

    main(video_pickle, annotations, modalities, topk_videos, topk_scenes, device)
