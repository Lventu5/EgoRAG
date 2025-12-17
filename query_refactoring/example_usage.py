"""
Example usage of QueryRefactorer class
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_refactoring.refactorer import QueryRefactorer
from data.video_dataset import VideoDataset
from data.query import Query, QueryDataset
from configuration.config import CONFIG
import logging
import pickle
import json

logging.basicConfig(level=logging.INFO)


def load_video_nlq_gt(nlq_json_path: str, video_uid: str):
    """Load NLQ entries for a specific video_uid"""
    with open(nlq_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    videos = data["videos"] if isinstance(data, dict) and "videos" in data else []
    for v in videos:
        if v.get("video_uid") != video_uid:
            continue

        for clip in v.get("clips", []):
            for ann in clip.get("annotations", []):
                for lq in ann.get("language_queries", []):
                    q = (lq.get("query") or "").strip()
                    vs = lq.get("video_start_sec")
                    ve = lq.get("video_end_sec")
                    fs = lq.get("video_start_frame")
                    fe = lq.get("video_end_frame")
                    if q and vs is not None and ve is not None:
                        results.append({
                            "query": q,
                            "start_sec": float(vs),
                            "end_sec": float(ve),
                            "start_frame": int(fs) if fs is not None else None,
                            "end_frame": int(fe) if fe is not None else None,
                        })
    return results


def example_with_video_dataset():
    """Example: Load VideoDataset and refactor only relevant queries"""
    print("="*80)
    print("Refactor queries from VideoDataset")
    print("="*80)
    
    video_dataset_path = CONFIG.data.video_dataset
    nlq_annotations_path = CONFIG.data.annotation_path
    video_dir = CONFIG.data.video_path
    
    logging.info(f"Loading video dataset from: {video_dataset_path}")
    with open(video_dataset_path, 'rb') as f:
        video_dataset = pickle.load(f)
    
    logging.info(f"Loaded {len(video_dataset)} videos into the dataset.")
    
    logging.info(f"Loading NLQ annotations from: {nlq_annotations_path}")
    
    all_queries = []
    for dp in video_dataset.video_datapoints:
        video_uid = getattr(dp, "video_uid", None)
        if not video_uid:
            continue
            
        nlq_entries = load_video_nlq_gt(nlq_annotations_path, video_uid)
        dp.queries = []
        
        for i, entry in enumerate(nlq_entries):
            q = Query(
                qid=f"{video_uid}_{i}",
                query_text=entry["query"],
                video_uid=video_uid,
                gt={
                    "start_sec": entry["start_sec"],
                    "end_sec": entry["end_sec"],
                    "start_frame": entry.get("start_frame"),
                    "end_frame": entry.get("end_frame")
                }
            )
            dp.queries.append(q)
            all_queries.append(q)
    
    query_dataset = QueryDataset(all_queries)
    logging.info(f"Loaded {len(query_dataset)} queries for the video dataset.")
    
    refactorer = QueryRefactorer(
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        model_type="qwen",
        device="cuda",
        verbose=True
    )
    
    refactored_queries = refactorer.refactor_query_dataset(
        query_dataset=query_dataset,
        video_dir=video_dir
    )
    
    print(f"\nRefactored {len(refactored_queries)} queries")
    
    for i, rq in enumerate(refactored_queries[:5]):
        print(f"\n{i+1}. Video: {rq.video_uid}")
        print(f"   Original:   {rq.original_query}")
        print(f"   Refactored: {rq.refactored_query}")
    
    stats = refactorer.get_statistics()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    output_path = "/cluster/project/cvg/students/tnanni/EgoRAG/results/nlq_train_edited.json"
    refactorer.save_to_ego4d_format(
        output_path=output_path,
        base_annotation_path=nlq_annotations_path
    )
    
    simple_output = "/cluster/project/cvg/students/tnanni/EgoRAG/results/refactored_queries_simple.json"
    refactorer.save_simple_json(simple_output)
    
    refactorer.unload_vllm()


def example_single_query():
    """Example: Refactor a single query"""
    print("="*80)
    print("Example 1: Refactor a single query")
    print("="*80)
    
    refactorer = QueryRefactorer(
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        model_type="qwen",
        device="cuda",
        verbose=True
    )
    
    query = "did I close the door?"
    video_path = "/path/to/video.mp4"
    ground_truth = {
        "video_start_sec": 10.0,
        "video_end_sec": 15.0,
        "video_start_frame": 300,
        "video_end_frame": 450,
        "clip_start_sec": 0.0,
        "clip_end_sec": 5.0
    }
    
    refactored = refactorer.refactor_query(
        query=query,
        video_path=video_path,
        ground_truth=ground_truth,
        video_uid="test_video_001",
        clip_uid="test_clip_001",
        query_idx=0
    )
    
    print(f"\nOriginal:   {refactored.original_query}")
    print(f"Refactored: {refactored.refactored_query}")
    
    refactorer.unload_vllm()


def example_from_ego4d_annotations():
    """Example: Refactor queries from Ego4D NLQ annotations"""
    print("\n" + "="*80)
    print("Example 2: Refactor from Ego4D annotations")
    print("="*80)
    
    refactorer = QueryRefactorer(
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        model_type="qwen",
        device="cuda",
        verbose=True
    )
    
    annotation_path = "/cluster/project/cvg/students/tnanni/ego4d_data/v2/annotations/nlq_train.json"
    video_dir = "/cluster/project/cvg/students/tnanni/ego4d_data/v2/full_scale/"
    
    refactored_queries = refactorer.refactor_from_ego4d_annotations(
        annotation_path=annotation_path,
        video_dir=video_dir,
        max_queries=10,  # Process only 10 queries for testing
        video_uids=None  # Process all videos, or pass specific UIDs
    )
    
    print(f"\nRefactored {len(refactored_queries)} queries")
    
    for i, rq in enumerate(refactored_queries[:3]):
        print(f"\n{i+1}. Video: {rq.video_uid}")
        print(f"   Original:   {rq.original_query}")
        print(f"   Refactored: {rq.refactored_query}")
    
    stats = refactorer.get_statistics()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    output_path = "/cluster/project/cvg/students/tnanni/EgoRAG/results/nlq_train_edited.json"
    refactorer.save_to_ego4d_format(
        output_path=output_path,
        base_annotation_path=annotation_path
    )
    
    simple_output = "/cluster/project/cvg/students/tnanni/EgoRAG/results/nlq_train_edited_simple.json"
    refactorer.save_simple_json(simple_output)
    
    refactorer.unload_vllm()


def example_from_query_dataset():
    """Example: Refactor queries from QueryDataset"""
    print("\n" + "="*80)
    print("Example 3: Refactor from QueryDataset")
    print("="*80)
    
    queries = [
        Query(
            qid="q1",
            query_text="where did I put the book?",
            video_uid="video_001",
            gt={
                "start_sec": 5.0,
                "end_sec": 10.0,
                "start_frame": 150,
                "end_frame": 300
            }
        ),
        Query(
            qid="q2",
            query_text="what tool did I use?",
            video_uid="video_001",
            gt={
                "start_sec": 20.0,
                "end_sec": 25.0,
                "start_frame": 600,
                "end_frame": 750
            }
        )
    ]
    
    query_dataset = QueryDataset(queries)
    
    refactorer = QueryRefactorer(
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        model_type="qwen",
        device="cuda",
        verbose=True
    )
    
    video_dir = "/path/to/videos/"
    refactored_queries = refactorer.refactor_query_dataset(
        query_dataset=query_dataset,
        video_dir=video_dir
    )
    
    for rq in refactored_queries:
        print(f"\nOriginal:   {rq.original_query}")
        print(f"Refactored: {rq.refactored_query}")
    
    refactorer.unload_vllm()


if __name__ == "__main__":
    example_with_video_dataset()
    # example_single_query()
    # example_from_ego4d_annotations()
    # example_from_query_dataset()
