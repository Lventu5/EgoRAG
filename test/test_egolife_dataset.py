"""
Test script for EgoLife dataset wrapper.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import EgoLifeDataset, DatasetFactory

def test_egolife_dataset():
    """Test the EgoLife dataset loading and structure."""
    
    # Update this path to your actual EgoLife dataset location
    video_path = "/cluster/scratch/lventuroli/hub/datasets--lmms-lab--EgoLife/snapshots/143fb319be7aa5ae210c936bf4f0f3a86092afb0" 
    # Use sample annotations for testing, or set to your actual annotation file
    annotation_path = os.path.join(os.path.dirname(__file__), "sample_egolife_annotations.json")
    # Set to None to skip annotation tests
    # annotation_path = None
    
    print("=" * 80)
    print("Testing EgoLife Dataset Wrapper")
    print("=" * 80)
    
    # Test 1: Initialize dataset
    print("\n1. Initializing EgoLife dataset...")
    dataset = EgoLifeDataset(video_path=video_path, annotation_path=annotation_path)
    print(f"   ✓ Dataset initialized")
    print(f"   - Expected videos: {len(dataset)} (6 people × 7 days = 42)")
    
    # Test 2: Get person-day folders
    print("\n2. Finding person-day folders...")
    person_day_folders = dataset._get_person_day_folders()
    print(f"   ✓ Found {len(person_day_folders)} person-day combinations")
    
    for i, (folder_path, person_id, day_id) in enumerate(person_day_folders[:3]):
        person_name = dataset.PERSON_NAMES.get(person_id, "UNKNOWN")
        print(f"   - Example {i+1}: A{person_id}_{person_name}/DAY{day_id}")
    
    if len(person_day_folders) > 3:
        print(f"   ... and {len(person_day_folders) - 3} more")
    
    # Test 3: Load videos (without pickle)
    print("\n3. Loading videos...")
    try:
        videos = dataset.load_videos(is_pickle=False)
        print(f"   ✓ Loaded {len(videos)} videos")
        
        # Show some video details
        print("\n4. Video details (first 3):")
        for i, video_dp in enumerate(videos.video_datapoints[:3]):
            print(f"\n   Video {i+1}: {video_dp.video_name}")
            print(f"   - Path: {video_dp.video_path}")
            print(f"   - Number of scenes/clips: {len(video_dp.scenes)}")
            
            # Show clip details
            for j, (scene_id, scene) in enumerate(list(video_dp.scenes.items())[:3]):
                print(f"     • {scene_id}: {scene.start_time:.0f}s - {scene.end_time:.0f}s "
                      f"(duration: {scene.end_time - scene.start_time:.1f}s)")
            
            if len(video_dp.scenes) > 3:
                print(f"     ... and {len(video_dp.scenes) - 3} more clips")
        
        if len(videos.video_datapoints) > 3:
            print(f"\n   ... and {len(videos.video_datapoints) - 3} more videos")
        
        # Test 5: Test annotations loading
        if annotation_path:
            print("\n5. Testing annotation loading...")
            video_ids = [vd.video_path for vd in videos.video_datapoints]
            annotations = dataset.load_annotations(video_ids)
            
            total_annotations = sum(len(anns) for anns in annotations.values())
            print(f"   ✓ Loaded {total_annotations} total annotations")
            
            # Show sample annotations
            for vid, anns in list(annotations.items())[:2]:
                if anns:
                    vid_name = os.path.basename(vid)
                    print(f"\n   Video: {vid_name}")
                    print(f"   - Annotations: {len(anns)}")
                    for ann in anns[:2]:
                        print(f"     • ID {ann.get('id')}: {ann.get('question')[:60]}...")
                        print(f"       Type: {ann.get('type')}, Answer: {ann.get('answer')}")
                    if len(anns) > 2:
                        print(f"     ... and {len(anns) - 2} more")
            
            # Test 6: Test query loading
            print("\n6. Testing query loading...")
            queries = dataset.load_queries(video_ids)
            print(f"   ✓ Loaded {len(queries.queries)} queries")
            
            # Show sample queries
            print("\n   Sample queries:")
            for i, query in enumerate(queries.queries[:3]):
                print(f"\n   Query {i+1}:")
                print(f"   - ID: {query.qid}")
                print(f"   - Video: {query.video_uid}")
                print(f"   - Question: {query.decomposed['text'][:80]}...")
                print(f"   - Target time: {query.gt['start_sec']:.0f}s")
                metadata = query.decomposed.get('metadata', {})
                print(f"   - Type: {metadata.get('type')}")
                print(f"   - Answer: {metadata.get('answer')}")
                if metadata.get('choices'):
                    choices = metadata['choices']
                    print(f"   - Choices: A={choices.get('A')}, B={choices.get('B')}, "
                          f"C={choices.get('C')}, D={choices.get('D')}")
            
            if len(queries.queries) > 3:
                print(f"\n   ... and {len(queries.queries) - 3} more queries")
        else:
            print("\n5-6. Skipping annotation/query tests (no annotation_path provided)")
            
    except Exception as e:
        print(f"   ✗ Error loading videos: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 7: Test clip timestamp parsing
    print("\n7. Testing timestamp parsing...")
    test_filenames = [
        "DAY1_A1_JAKE_11300000.mp4",
        "DAY3_A2_ALICE_14453000.mp4",
        "DAY7_A6_SHURE_22200000.mp4",
    ]
    
    for filename in test_filenames:
        timestamp = dataset._parse_clip_timestamp(filename)
        hours = timestamp // 3600
        minutes = (timestamp % 3600) // 60
        seconds = timestamp % 60
        print(f"   - {filename} → {timestamp}s ({hours}:{minutes}:{seconds})")
    
    # Test 8: Test query timestamp parsing
    print("\n8. Testing query timestamp parsing...")
    test_queries = [
        ("DAY1", "11210217"),
        ("DAY3", "14453000"),
        ("DAY7", "22200000"),
    ]
    
    for date, time_str in test_queries:
        timestamp = dataset._parse_query_timestamp(date, time_str)
        hours = timestamp // 3600
        minutes = (timestamp % 3600) // 60
        seconds = timestamp % 60
        print(f"   - {date} @ {time_str} → {timestamp}s ({hours}:{minutes}:{seconds})")
    
    # Test 9: DatasetFactory
    print("\n9. Testing DatasetFactory...")
    dataset2 = DatasetFactory.get_dataset("egolife", video_path, annotation_path)
    print(f"   ✓ Factory created EgoLife dataset: {type(dataset2).__name__}")
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)

def test_clip_ordering():
    """Test that clips are properly ordered by timestamp."""
    print("\n" + "=" * 80)
    print("Testing Clip Ordering")
    print("=" * 80)
    
    # Simulate some clip filenames
    from data.dataset import EgoLifeDataset
    dataset = EgoLifeDataset("/cluster/scratch/lventuroli/hub/datasets--lmms-lab--EgoLife/snapshots/143fb319be7aa5ae210c936bf4f0f3a86092afb0", None)
    
    test_clips = [
        "DAY1_A1_JAKE_11300000.mp4",  # 08:30:00
        "DAY1_A1_JAKE_11303000.mp4",  # 08:30:30
        "DAY1_A1_JAKE_11310000.mp4",  # 08:29:59
        "DAY1_A1_JAKE_09000000.mp4",  # 09:00:00
    ]
    
    print("\nOriginal order:")
    for clip in test_clips:
        ts = dataset._parse_clip_timestamp(clip)
        print(f"  {clip} → {ts}s")
    
    # Sort by timestamp
    sorted_clips = sorted(test_clips, key=lambda x: dataset._parse_clip_timestamp(x))
    
    print("\nSorted order:")
    for clip in sorted_clips:
        ts = dataset._parse_clip_timestamp(clip)
        print(f"  {clip} → {ts}s")
    
    print("\n✓ Clips are properly sortable by timestamp")

if __name__ == "__main__":
    # Run clip ordering test (doesn't require actual data)
    test_clip_ordering()
    
    # Uncomment and update path to run full test
    test_egolife_dataset()
    
    print("\nNote: Update video_path in test_egolife_dataset() to run full tests")
