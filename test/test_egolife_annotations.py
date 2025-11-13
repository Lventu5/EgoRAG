"""
Test script for EgoLife dataset annotation and query loading.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import EgoLifeDataset

def test_annotation_loading():
    """Test annotation and query loading from JSON file."""
    
    video_path = "/cluster/scratch/lventuroli/hub/datasets--lmms-lab--EgoLife/snapshots/143fb319be7aa5ae210c936bf4f0f3a86092afb0"
    annotation_path = os.path.join(video_path, "EgoLifeQA/EgoLifeQA_A1_JAKE.json")
    
    print("=" * 80)
    print("Testing EgoLife Annotation & Query Loading")
    print("=" * 80)
    
    # Initialize dataset
    print("\n1. Initializing dataset with annotations...")
    dataset = EgoLifeDataset(video_path=video_path, annotation_path=annotation_path)
    print(f"   ✓ Dataset initialized")
    print(f"   - Annotation file: {annotation_path}")
    
    # Load videos
    print("\n2. Loading videos...")
    videos = dataset.load_videos(is_pickle=False)
    print(f"   ✓ Loaded {len(videos)} videos")
    
    # Get video IDs
    video_ids = [vd.video_path for vd in videos.video_datapoints]
    
    # Test annotation loading
    print("\n3. Testing annotation loading...")
    annotations = dataset.load_annotations(video_ids)
    
    # Count annotations per video
    video_ann_counts = {vid: len(anns) for vid, anns in annotations.items() if anns}
    total_annotations = sum(len(anns) for anns in annotations.values())
    
    print(f"   ✓ Loaded {total_annotations} total annotations")
    print(f"   ✓ Distributed across {len(video_ann_counts)} videos")
    
    # Show which videos have annotations
    print("\n   Videos with annotations:")
    for vid, count in sorted(video_ann_counts.items())[:5]:
        vid_name = os.path.basename(vid)
        print(f"   - {vid_name}: {count} annotations")
    
    # Show detailed annotation info
    print("\n4. Detailed annotation examples:")
    for vid, anns in list(annotations.items())[:2]:
        if anns:
            vid_name = os.path.basename(vid)
            print(f"\n   Video: {vid_name}")
            print(f"   Annotations: {len(anns)}")
            
            for i, ann in enumerate(anns[:2]):
                print(f"\n   Annotation {i+1}:")
                print(f"   - ID: {ann.get('id')}")
                print(f"   - Type: {ann.get('type')}")
                print(f"   - Trigger: {ann.get('trigger')}")
                print(f"   - Question: {ann.get('question')}")
                print(f"   - Choices:")
                print(f"     A: {ann.get('choice_a')}")
                print(f"     B: {ann.get('choice_b')}")
                print(f"     C: {ann.get('choice_c')}")
                print(f"     D: {ann.get('choice_d')}")
                print(f"   - Answer: {ann.get('answer')}")
                print(f"   - Query time: {ann.get('query_time')}")
                print(f"   - Target time: {ann.get('target_time')}")
                print(f"   - Keywords: {ann.get('keywords')}")
                print(f"   - Need audio: {ann.get('need_audio')}")
                print(f"   - Need name: {ann.get('need_name')}")
            
            if len(anns) > 2:
                print(f"\n   ... and {len(anns) - 2} more annotations")
            break
    
    # Test query loading
    print("\n5. Testing query loading...")
    queries = dataset.load_queries(video_ids)
    print(f"   ✓ Loaded {len(queries.queries)} queries")
    
    # Show detailed query info
    print("\n6. Detailed query examples:")
    for i, query in enumerate(queries.queries[:3]):
        print(f"\n   Query {i+1}:")
        print(f"   - QID: {query.qid}")
        print(f"   - Video UID: {query.video_uid}")
        print(f"   - Query text: {query.query_text[:100]}...")
        print(f"   - Ground truth start: {query.gt['start_sec']:.0f}s")
        print(f"   - Ground truth end: {query.gt['end_sec']:.0f}s")
        
        # Show metadata
        metadata = query.decomposed.get('metadata', {})
        print(f"   - Metadata:")
        print(f"     • Type: {metadata.get('type')}")
        print(f"     • Query time: {metadata.get('query_time_sec')}s ({metadata.get('query_date')})")
        print(f"     • Target time: {metadata.get('target_time_sec')}s ({metadata.get('target_date')})")
        print(f"     • Answer: {metadata.get('answer')}")
        print(f"     • Keywords: {metadata.get('keywords')}")
        
        choices = metadata.get('choices', {})
        print(f"     • Choices:")
        for choice_key in ['A', 'B', 'C', 'D']:
            print(f"       {choice_key}: {choices.get(choice_key)}")
    
    if len(queries.queries) > 3:
        print(f"\n   ... and {len(queries.queries) - 3} more queries")
    
    # Test query timestamp parsing
    print("\n7. Verifying timestamp parsing:")
    for query in queries.queries[:3]:
        metadata = query.decomposed.get('metadata', {})
        query_sec = metadata.get('query_time_sec')
        target_sec = metadata.get('target_time_sec')
        
        query_hms = f"{int(query_sec // 3600):02d}:{int((query_sec % 3600) // 60):02d}:{int(query_sec % 60):02d}"
        target_hms = f"{int(target_sec // 3600):02d}:{int((target_sec % 3600) // 60):02d}:{int(target_sec % 60):02d}"
        
        print(f"   - Query {query.qid}:")
        print(f"     Query time: {query_sec}s ({query_hms})")
        print(f"     Target time: {target_sec}s ({target_hms})")
    
    print("\n" + "=" * 80)
    print("All annotation/query tests completed!")
    print("=" * 80)

if __name__ == "__main__":
    test_annotation_loading()
