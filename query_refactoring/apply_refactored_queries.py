#!/usr/bin/env python3
"""
Script to replace original queries with refactored queries in the Ego4D NLQ JSON file
"""

import json
import sys

def apply_refactored_queries(
    refactored_simple_path: str,
    nlq_original_path: str,
    nlq_output_path: str
):
    """
    Replace queries in the original NLQ JSON with refactored queries
    
    Args:
        refactored_simple_path: Path to the refactored_queries_simple.json
        nlq_original_path: Path to the original nlq_train.json
        nlq_output_path: Path to save the edited nlq_train_edited.json
    """
    print(f"Loading refactored queries from: {refactored_simple_path}")
    with open(refactored_simple_path, 'r') as f:
        refactored_list = json.load(f)
    
    print(f"Loaded {len(refactored_list)} refactored queries")
    
    # Create lookup map: (video_uid, original_query) -> refactored_query
    refactored_map = {}
    for item in refactored_list:
        video_uid = item['video_uid']
        original_query = item['original_query']
        refactored_query = item['refactored_query']
        key = (video_uid, original_query)
        refactored_map[key] = refactored_query
    
    print(f"Loading original NLQ annotations from: {nlq_original_path}")
    with open(nlq_original_path, 'r') as f:
        nlq_data = json.load(f)
    
    # Replace queries in the structure
    replaced_count = 0
    total_queries = 0
    
    for video_entry in nlq_data.get("videos", []):
        video_uid = video_entry.get("video_uid", "")
        
        for clip in video_entry.get("clips", []):
            for annotation_group in clip.get("annotations", []):
                lang_queries = annotation_group.get("language_queries", [])
                
                for lang_query in lang_queries:
                    total_queries += 1
                    original_query = lang_query.get("query", "")
                    key = (video_uid, original_query)
                    
                    if key in refactored_map:
                        lang_query["query"] = refactored_map[key]
                        replaced_count += 1
    
    print(f"\nReplaced {replaced_count} out of {total_queries} total queries")
    
    print(f"Saving edited NLQ file to: {nlq_output_path}")
    with open(nlq_output_path, 'w') as f:
        json.dump(nlq_data, f, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    refactored_simple = "/cluster/project/cvg/students/tnanni/EgoRAG/results/refactored_queries_simple.json"
    nlq_original = "/cluster/project/cvg/students/tnanni/ego4d_data/v2/annotations/nlq_train.json"
    nlq_output = "/cluster/project/cvg/students/tnanni/EgoRAG/results/nlq_train_edited.json"
    
    apply_refactored_queries(refactored_simple, nlq_original, nlq_output)
