#!/usr/bin/env python3
"""
Build Indices Script: Builds SceneMemoryBank indices from encoded dataset.

Usage:
    python scripts/build_indices.py --input encoded_dataset.pkl --output memory_bank.pkl
"""

import argparse
import pickle
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexing.memory.memory_bank import SceneMemoryBank
from indexing.memory.pruner import prune_dataset
from indexing.utils.logging import get_logger
import yaml

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build memory bank indices")
    
    parser.add_argument("--input", type=str, required=True,
                       help="Path to encoded VideoDataset pickle")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for memory bank pickle")
    parser.add_argument("--prune", action="store_true",
                       help="Apply scene pruning before building indices")
    parser.add_argument("--keep_ratio", type=float, default=0.75,
                       help="Ratio of scenes to keep when pruning")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("Build Memory Bank Indices")
    logger.info("="*80)
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        logger.info(f"Loaded config from {args.config}")
    else:
        cfg = {}
        logger.warning(f"Config file not found: {args.config}, using defaults")
    
    # Load encoded dataset
    logger.info(f"Loading encoded dataset from {args.input}")
    with open(args.input, 'rb') as f:
        video_dataset = pickle.load(f)
    
    logger.info(f"Loaded {len(video_dataset.video_datapoints)} videos")
    
    # Apply pruning if requested
    if args.prune:
        logger.info(f"Pruning scenes with keep_ratio={args.keep_ratio}")
        prune_dataset(video_dataset, args.keep_ratio, cfg)
    
    # Build memory bank
    logger.info("Building memory bank indices...")
    memory_bank = SceneMemoryBank(video_dataset)
    memory_bank.build_indices()
    
    # Save memory bank
    logger.info(f"Saving memory bank to {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    memory_bank.serialize(args.output)
    
    logger.info("Memory bank build complete!")
    logger.info(f"Memory bank saved to: {args.output}")


if __name__ == "__main__":
    main()
