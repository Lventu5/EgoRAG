"""
Example usage of QueryRefactorer class
"""

from query_refactoring.refactorer import QueryRefactorer
import logging

logging.basicConfig(level=logging.INFO)


def example_single_query():
    """Example: Refactor a single query"""
    print("="*80)
    print("Example 1: Refactor a single query")
    print("="*80)
    
    refactorer = QueryRefactorer(
        model_id="lmms-lab/LLaVA-Video-7B-Qwen2",
        model_type="llava",
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
        model_id="lmms-lab/LLaVA-Video-7B-Qwen2",
        model_type="llava",
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
    
    output_path = "/cluster/project/cvg/students/tnanni/EgoRAG/results/refactored_queries.json"
    refactorer.save_to_ego4d_format(
        output_path=output_path,
        base_annotation_path=annotation_path
    )
    
    simple_output = "/cluster/project/cvg/students/tnanni/EgoRAG/results/refactored_queries_simple.json"
    refactorer.save_simple_json(simple_output)
    
    refactorer.unload_vllm()


def example_from_query_dataset():
    """Example: Refactor queries from QueryDataset"""
    print("\n" + "="*80)
    print("Example 3: Refactor from QueryDataset")
    print("="*80)
    
    from data.query import Query, QueryDataset
    
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
    
    refamodel_id="lmms-lab/LLaVA-Video-7B-Qwen2",
        model_type="llava
        vllm_model_name="LLaVAVideo",
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
    # example_single_query()
    example_from_ego4d_annotations()
    # example_from_query_dataset()
