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
    parser.add_argument("--max_workers", type=int, default=1,
                       help="Maximum number of worker threads (IMPORTANT: >1 may cause segfaults due to VideoReader thread-safety issues)")
    parser.add_argument("--apply_tagging", action="store_true",
                       help="Apply scene tagging during encoding")
    
    args = parser.parse_args()
    
    # Validate max_workers
    if args.max_workers > 1:
        logger.warning("="*80)
        logger.warning("WARNING: max_workers > 1 may cause segmentation faults!")
        logger.warning("VideoReader (decord) is NOT thread-safe when shared across threads.")
        logger.warning("Recommended: Use --max_workers 1 for stable execution.")
        logger.warning("="*80)
    
    logger.info("="*80)
    logger.info("Video Encoding Script")
    logger.info("="*80)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_type}")
    dataset = DatasetFactory.get_dataset(
        dataset_type=args.dataset_type,
        video_path=args.video_path,
        annotation_path=args.annotation_path
    )
    
    logger.info(f"Loaded {len(dataset)} videos")
    
    # Initialize encoder
    logger.info("Initializing MultiModalEncoder...")
    encoder = MultiModalEncoder(
        video_dataset=dataset.load_videos(is_pickle=False),
        device=args.device,
        max_frames_per_scene=args.max_frames_per_scene,
        max_temporal_segments=args.max_temporal_segments,
        max_workers=args.max_workers,
        apply_tagging=args.apply_tagging  # Tagging integrated into pipeline
    )
    
    # NOTE: Models are loaded lazily via ModelRegistry - no explicit load_models() needed
    
    # Encode videos (models will be loaded on-demand)
    logger.info("Encoding videos (models loaded lazily via ModelRegistry)...")
    encoded_dataset = encoder.encode_videos()
    
    # Unload models to free GPU memory
    logger.info("Unloading models...")
    encoder.unload_models()
    
    # Note: Scene tagging is now integrated into encode_videos() when apply_tagging=True
    # No need for separate tagging step
    
    # Save encoded dataset
    logger.info(f"Saving encoded dataset to {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(encoded_dataset, f)
    
    logger.info("Encoding complete!")
    logger.info(f"Encoded dataset saved to: {args.output}")


if __name__ == "__main__":
    main()
