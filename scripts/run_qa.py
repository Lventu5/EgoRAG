#!/usr/bin/env python3
"""
Run QA Script: Runs QAPipeline on queries and writes outputs.

Usage:
    python scripts/run_qa.py --dataset encoded_dataset.pkl --memory_bank memory_bank.pkl \
        --queries queries.json --output results.json
"""

import argparse
import pickle
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.query import Query, QueryDataset
from indexing.memory.memory_bank import SceneMemoryBank
from indexing.qa.pipeline import QAPipeline
from indexing.qa.generator import QAGenerator
from retrieval.hierarchical_retriever import HierarchicalRetriever
from retrieval.chain_retriever import ChainRetriever
from retrieval.rewriter import QueryRewriterLLM
from retrieval.reranker import Reranker
from train.metrics import *
from indexing.utils.logging import get_logger
import yaml

logger = get_logger(__name__)


def load_queries(query_path: str) -> QueryDataset:
    """Load queries from JSON file."""
    with open(query_path, 'r') as f:
        data = json.load(f)
    
    queries = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                queries.append(Query.from_dict(item))
            else:
                queries.append(Query(qid=len(queries), query_text=str(item)))
    else:
        raise ValueError("Query file must contain a JSON list")
    
    return QueryDataset(queries)


def main():
    parser = argparse.ArgumentParser(description="Run QA pipeline")
    
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to encoded VideoDataset pickle")
    parser.add_argument("--memory_bank", type=str, required=True,
                       help="Path to memory bank pickle")
    parser.add_argument("--queries", type=str, required=True,
                       help="Path to queries JSON file")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for results JSON")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--modalities", type=str, default="video,text,audio",
                       help="Comma-separated list of modalities")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("Run QA Pipeline")
    logger.info("="*80)
    
    # Load configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded config from {args.config}")
    
    # Load encoded dataset
    logger.info(f"Loading dataset from {args.dataset}")
    with open(args.dataset, 'rb') as f:
        video_dataset = pickle.load(f)
    logger.info(f"Loaded {len(video_dataset.video_datapoints)} videos")
    
    # Load memory bank
    logger.info(f"Loading memory bank from {args.memory_bank}")
    memory_bank = SceneMemoryBank(video_dataset)
    memory_bank.load(args.memory_bank)
    
    # Load queries
    logger.info(f"Loading queries from {args.queries}")
    query_dataset = load_queries(args.queries)
    logger.info(f"Loaded {len(query_dataset)} queries")
    
    # Initialize components
    logger.info("Initializing retrieval pipeline...")
    
    rewriter = QueryRewriterLLM(
        model_name=cfg.get("retrieval", {}).get("rewriter_model", "google/gemma-2-9b"),
        device=args.device
    )
    
    hierarchical_retriever = HierarchicalRetriever(
        video_dataset=video_dataset,
        query_dataset=query_dataset,
        rewriter=rewriter,
        device=args.device
    )
    
    chain_retriever = ChainRetriever(
        base_retriever=hierarchical_retriever,
        rewriter=rewriter,
        memory_bank=memory_bank,
        cfg=cfg
    )
    
    reranker = Reranker(
        alpha=cfg.get("reranker", {}).get("alpha", 0.70),
        beta=cfg.get("reranker", {}).get("beta", 0.20),
        gamma=cfg.get("reranker", {}).get("gamma", 0.10),
        half_life_sec=cfg.get("reranker", {}).get("temporal_decay_half_life_sec", 86400)
    )
    
    qa_generator = QAGenerator(
        llm_name=cfg.get("qa", {}).get("llm_model", "placeholder-local-llm"),
        device=args.device
    )
    
    qa_pipeline = QAPipeline(
        chain_retriever=chain_retriever,
        reranker=reranker,
        generator=qa_generator,
        video_dataset=video_dataset,
        cfg=cfg
    )
    
    # Run QA
    logger.info("Running QA pipeline...")
    modalities = args.modalities.split(',')
    
    results = []
    for query in query_dataset:
        logger.info(f"Processing query {query.qid}: {query.query_text}")
        result = qa_pipeline.answer(query, modalities=modalities)
        results.append(result)
    
    # Compute metrics if ground truth available
    logger.info("Computing metrics...")
    metrics = {}
    
    mrr_scores = []
    for result in results:
        query_id = result["query_id"]
        query = None
        for q in query_dataset:
            if q.qid == query_id:
                query = q
                break
        
        if query and query.video_uid:
            retrieved_videos = [v[0] for v in result["candidates"]["videos"]]
            mrr = mean_reciprocal_rank(retrieved_videos, query.video_uid)
            mrr_scores.append(mrr)
    
    if mrr_scores:
        metrics["MRR"] = sum(mrr_scores) / len(mrr_scores)
    
    # Save results
    logger.info(f"Saving results to {args.output}")
    output_data = {
        "metrics": metrics,
        "results": results,
        "config": cfg
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info("QA pipeline complete!")
    logger.info(f"Results saved to: {args.output}")
    
    # Print summary
    if metrics:
        logger.info("\nMetrics Summary:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
