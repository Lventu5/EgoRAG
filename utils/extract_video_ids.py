"""
Utility to extract all unique video IDs from Ego4D NLQ validation annotations
"""

import json
import sys

def extract_video_ids(nlq_json_path: str):
    """Extract all unique video UIDs from NLQ annotation file"""
    with open(nlq_json_path, 'r') as f:
        data = json.load(f)
    
    video_ids = []
    videos = data.get("videos", [])
    
    for video_entry in videos:
        video_uid = video_entry.get("video_uid")
        if video_uid:
            video_ids.append(video_uid)
    
    return video_ids


if __name__ == "__main__":
    nlq_val_path = "/cluster/project/cvg/students/tnanni/ego4d_data/v2/annotations/nlq_val.json"
    output_path = "/cluster/project/cvg/students/tnanni/EgoRAG/val_uids.txt"
    
    video_ids = extract_video_ids(nlq_val_path)
    
    print(f"Found {len(video_ids)} videos in validation set")
    
    with open(output_path, 'w') as f:
        for vid in video_ids:
            f.write(f"{vid}\n")
    
    print(f"Saved video IDs to {output_path}")
