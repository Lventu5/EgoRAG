#!/usr/bin/env python3
"""
Encode Videos Script: Encodes videos using MultiModalEncoder and saves dataset.

Usage:
    python scripts/encode_videos.py --dataset_type ego4d --video_path /path/to/videos \
        --annotation_path /path/to/annotations.json --output encoded_dataset.pkl
"""

import argparse
import pickle
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import DatasetFactory
from indexing.multimodal_encoder import MultiModalEncoder
from indexing.analytics.tagging import tag_all_scenes
from indexing.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Encode videos with MultiModalEncoder")
    
    parser.add_argument("--dataset_type", type=str, required=True,
                       help="Dataset type (e.g., ego4d, egoschema)")
    parser.add_argument("--video_path", type=str, required=True,
                       help="Path to video files or directory")
    parser.add_argument("--annotation_path", type=str, required=True,
                       help="Path to annotation file")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for pickled VideoDataset")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--max_frames_per_scene", type=int, default=96,
                       help="Maximum frames per scene")
    parser.add_argument("--max_temporal_segments", type=int, default=8,
                       help="Maximum temporal segments")
    parser.add_argument("--apply_tagging", action="store_true",
                       help="Apply scene tagging after encoding")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("Video Encoding Script")
    logger.info("="*80)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_type}")
    dataset = DatasetFactory.create(
        dataset_type=args.dataset_type,
        video_path=args.video_path,
        annotation_path=args.annotation_path
    )
    
    logger.info(f"Loaded {len(dataset.video_datapoints)} videos")
    
    # Initialize encoder
    logger.info("Initializing MultiModalEncoder...")
    encoder = MultiModalEncoder(
        video_dataset=dataset.video_dataset,
        device=args.device,
        max_frames_per_scene=args.max_frames_per_scene,
        max_temporal_segments=args.max_temporal_segments
    )
    
    # Load models
    logger.info("Loading encoder models...")
    encoder.load_models()
    
    # Encode videos
    logger.info("Encoding videos...")
    encoded_dataset = encoder.encode_videos()
    
    # Unload models
    encoder.unload_models()
    
    # Apply tagging if requested
    if args.apply_tagging:
        logger.info("Applying scene tagging...")
        for dp in encoded_dataset.video_datapoints:
            tag_all_scenes(dp)
    
    # Save encoded dataset
    logger.info(f"Saving encoded dataset to {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(encoded_dataset, f)
    
    logger.info("Encoding complete!")
    logger.info(f"Encoded dataset saved to: {args.output}")


if __name__ == "__main__":
    main()
