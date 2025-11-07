"""
Evaluation Harness: End-to-end evaluation orchestration.

This module provides the main evaluation pipeline that encodes videos,
builds indices, runs retrieval and QA, and computes metrics.
"""

import os
import json
import argparse
import pickle
from datetime import datetime
from typing import Dict, List, Any
import yaml

from data.dataset import DatasetFactory
from data.video_dataset import VideoDataset
from data.query import QueryDataset
from indexing.multimodal_encoder import MultiModalEncoder
from indexing.memory.memory_bank import SceneMemoryBank
from indexing.memory.pruner import prune_dataset
from indexing.qa.pipeline import QAPipeline
from indexing.qa.generator import QAGenerator
from retrieval.hierarchical_retriever import HierarchicalRetriever
from retrieval.chain_retriever import ChainRetriever
from retrieval.rewriter import QueryRewriterLLM
from retrieval.reranker import Reranker
from train.metrics import *
from indexing.utils.logging import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate(args):
    """
    Main evaluation function.
    
    Steps:
    1. Load dataset
    2. Encode videos (if needed)
    3. Prune scenes
    4. Build memory indices
    5. Run retrieval and QA
    6. Compute metrics
    7. Write report
    """
    logger.info("="*80)
    logger.info("Starting EgoRAG Evaluation")
    logger.info("="*80)
    
    # Load configuration
    cfg = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.save_run_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Run directory: {run_dir}")
    
    # Save config to run directory
    with open(os.path.join(run_dir, "config.yaml"), 'w') as f:
        yaml.dump(cfg, f)
    
    # Step 1: Load dataset
    logger.info("\n" + "="*80)
    logger.info("Step 1: Loading dataset")
    logger.info("="*80)
    
    dataset = DatasetFactory.create(
        dataset_type=args.dataset_type,
        video_path=args.video_path,
        annotation_path=args.annotation_path
    )
    
    logger.info(f"Loaded {len(dataset.video_datapoints)} videos")
    logger.info(f"Loaded {len(dataset.query_dataset)} queries")
    
    # Step 2: Encode videos (if needed)
    logger.info("\n" + "="*80)
    logger.info("Step 2: Encoding videos")
    logger.info("="*80)
    
    if args.load_encoded and os.path.exists(args.load_encoded):
        logger.info(f"Loading encoded dataset from {args.load_encoded}")
        with open(args.load_encoded, 'rb') as f:
            dataset.video_dataset = pickle.load(f)
    else:
        logger.info("Encoding videos...")
        encoder = MultiModalEncoder(
            video_dataset=dataset.video_dataset,
            device=args.device,
            max_frames_per_scene=cfg.get("encoder", {}).get("max_frames_per_scene", 96),
            max_temporal_segments=cfg.get("encoder", {}).get("max_temporal_segments", 8)
        )
        encoder.load_models()
        dataset.video_dataset = encoder.encode_videos()
        encoder.unload_models()
        
        # Save encoded dataset
        if args.save_encoded:
            encoded_path = os.path.join(run_dir, "encoded_dataset.pkl")
            with open(encoded_path, 'wb') as f:
                pickle.dump(dataset.video_dataset, f)
            logger.info(f"Saved encoded dataset to {encoded_path}")
    
    # Step 3: Prune scenes
    logger.info("\n" + "="*80)
    logger.info("Step 3: Pruning scenes")
    logger.info("="*80)
    
    keep_ratio = cfg.get("memory", {}).get("prune", {}).get("keep_ratio", 0.75)
    prune_dataset(dataset.video_dataset, keep_ratio, cfg)
    
    # Step 4: Build memory indices
    logger.info("\n" + "="*80)
    logger.info("Step 4: Building memory indices")
    logger.info("="*80)
    
    memory_bank = SceneMemoryBank(dataset.video_dataset)
    memory_bank.build_indices()
    
    # Save memory bank
    memory_path = os.path.join(run_dir, "memory_bank.pkl")
    memory_bank.serialize(memory_path)
    
    # Step 5: Initialize retrieval components
    logger.info("\n" + "="*80)
    logger.info("Step 5: Initializing retrieval pipeline")
    logger.info("="*80)
    
    # Initialize rewriter
    rewriter = QueryRewriterLLM(
        model_name=cfg.get("retrieval", {}).get("rewriter_model", "google/gemma-2-9b"),
        device=args.device
    )
    
    # Initialize hierarchical retriever with MemoryBank
    hierarchical_retriever = HierarchicalRetriever(
        video_dataset=dataset.video_dataset,
        memory_bank=memory_bank,  # Use memory bank for efficient retrieval
        fuser=None,  # Will use default RRF fuser
        device=args.device,
        text_model_name=cfg.get("encoder", {}).get("text_model", "all-MiniLM-L6-v2"),
        video_model_name=cfg.get("encoder", {}).get("video_model", "microsoft/xclip-base-patch16"),
        audio_model_name=cfg.get("encoder", {}).get("audio_model", "laion/clap-htsat-unfused"),
        caption_model_name=cfg.get("encoder", {}).get("caption_model", "all-MiniLM-L6-v2"),
        rewriter_name=cfg.get("retrieval", {}).get("rewriter_model", "google/gemma-2-9b")
    )
    
    # Initialize chain retriever
    chain_retriever = ChainRetriever(
        base_retriever=hierarchical_retriever,
        rewriter=rewriter,
        memory_bank=memory_bank,
        cfg=cfg
    )
    
    # Initialize reranker
    reranker = Reranker(
        alpha=cfg.get("reranker", {}).get("alpha", 0.70),
        beta=cfg.get("reranker", {}).get("beta", 0.20),
        gamma=cfg.get("reranker", {}).get("gamma", 0.10),
        half_life_sec=cfg.get("reranker", {}).get("temporal_decay_half_life_sec", 86400)
    )
    
    # Initialize QA generator
    qa_generator = QAGenerator(
        llm_name=cfg.get("qa", {}).get("llm_model", "placeholder-local-llm"),
        device=args.device
    )
    
    # Initialize QA pipeline
    qa_pipeline = QAPipeline(
        chain_retriever=chain_retriever,
        reranker=reranker,
        generator=qa_generator,
        video_dataset=dataset.video_dataset,
        cfg=cfg
    )
    
    # Step 6: Run retrieval and QA
    logger.info("\n" + "="*80)
    logger.info("Step 6: Running retrieval and QA")
    logger.info("="*80)
    
    modalities = args.modalities.split(',') if isinstance(args.modalities, str) else args.modalities
    
    all_results = []
    for query in dataset.query_dataset:
        logger.info(f"\nProcessing query {query.qid}: {query.query_text}")
        
        result = qa_pipeline.answer(query, modalities=modalities)
        all_results.append(result)
    
    # Step 7: Compute metrics
    logger.info("\n" + "="*80)
    logger.info("Step 7: Computing metrics")
    logger.info("="*80)
    
    metrics_summary = compute_metrics(all_results, dataset.query_dataset)
    
    # Step 8: Write report
    logger.info("\n" + "="*80)
    logger.info("Step 8: Writing report")
    logger.info("="*80)
    
    report = {
        "config": cfg,
        "args": vars(args),
        "metrics": metrics_summary,
        "results": all_results
    }
    
    report_path = os.path.join(run_dir, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report saved to {report_path}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    for metric_name, value in metrics_summary.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    logger.info("\nEvaluation complete!")
    
    return report


def compute_metrics(results: List[Dict], queries: QueryDataset) -> Dict[str, float]:
    """Compute aggregate metrics from results."""
    metrics = {}
    
    # QA metrics (if gold answers available)
    # This is a placeholder - actual implementation depends on dataset format
    
    # Retrieval metrics
    mrr_scores = []
    recall_scores = []
    
    for result in results:
        query_id = result["query_id"]
        
        # Find corresponding query
        query = None
        for q in queries:
            if q.qid == query_id:
                query = q
                break
        
        if query and query.video_uid:
            # Compute MRR
            retrieved_videos = [v[0] for v in result["candidates"]["videos"]]
            mrr = mean_reciprocal_rank(retrieved_videos, query.video_uid)
            mrr_scores.append(mrr)
        
        # Compute Recall@K for scenes (if ground truth available)
        if query and query.gt and query.gt.get("start_sec") is not None:
            gt_intervals = [(query.gt["start_sec"], query.gt["end_sec"])]
            pred_scenes = [
                Scene(
                    scene_id=s["scene_id"],
                    start_time=s["start_time"],
                    end_time=s["end_time"],
                    start_frame=0,
                    end_frame=0
                )
                for s in result["candidates"]["scenes"]
            ]
            recall = recall_at_k_scene(gt_intervals, pred_scenes, iou_threshold=0.5, k=5)
            recall_scores.append(recall)
    
    # Aggregate
    if mrr_scores:
        metrics["MRR"] = sum(mrr_scores) / len(mrr_scores)
    if recall_scores:
        metrics["Recall@5"] = sum(recall_scores) / len(recall_scores)
    
    metrics["num_queries"] = len(results)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="EgoRAG Evaluation Harness")
    
    # Dataset arguments
    parser.add_argument("--dataset_type", type=str, required=True,
                       help="Dataset type (e.g., ego4d, egoschema)")
    parser.add_argument("--video_path", type=str, required=True,
                       help="Path to video files")
    parser.add_argument("--annotation_path", type=str, required=True,
                       help="Path to annotation file")
    
    # Retrieval arguments
    parser.add_argument("--modalities", type=str, default="video,text,audio",
                       help="Comma-separated list of modalities")
    parser.add_argument("--top_k_videos", type=int, default=3,
                       help="Number of videos to retrieve")
    parser.add_argument("--top_k_scenes", type=int, default=2,
                       help="Number of scenes per video")
    parser.add_argument("--max_hops", type=int, default=2,
                       help="Number of retrieval hops")
    
    # System arguments
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    
    # I/O arguments
    parser.add_argument("--save_run_dir", type=str, default="runs",
                       help="Directory to save run outputs")
    parser.add_argument("--load_encoded", type=str, default=None,
                       help="Path to pre-encoded dataset pickle")
    parser.add_argument("--save_encoded", action="store_true",
                       help="Save encoded dataset")
    
    args = parser.parse_args()
    
    evaluate(args)


if __name__ == "__main__":
    main()
