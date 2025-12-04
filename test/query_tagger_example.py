"""
Example usage of QueryTagger with QueryRewriterLLM for multimodal retrieval.

This demonstrates how to:
1. Decompose a natural language query into modalities (text, audio, video)
2. Use the 'video' modality to predict visual tags with CLIP
3. Match query tags against video tags from VisionTagger for filtering

Integration pattern:
- QueryRewriterLLM: Decomposes query → {"text": ..., "audio": ..., "video": ...}
- QueryTagger: Tags decomposed["video"] → ["obj_kitchenware", "act_cooking_prep", ...]
- Retrieval: Filter videos by tag overlap between query and indexed videos
"""

import logging
import gc
import torch
from retrieval.query_tagger import QueryTagger
from retrieval.rewriter import QueryRewriterLLM
from configuration.config import CONFIG

logging.basicConfig(level=logging.INFO)


def main():
    # Initialize components
    print("=" * 80)
    print("Step 1: Initializing QueryRewriter (LLM)...")
    print("=" * 80)
    
    rewriter = QueryRewriterLLM(
        model_name=CONFIG.retrieval.rewriter_model_id,
        device="cuda"
    )
    
    # Example queries
    queries = [
        "Where did I put my coffee mug in the kitchen?",
        "What was I cooking when I chopped the vegetables?",
        "Did I turn off the stove after boiling water?",
        "Who was I talking to while driving?",
        "Where are my tools in the workshop?",
    ]
    
    print("\nStep 2: Rewriting queries with LLM...\n")
    
    # First, decompose all queries
    decomposed_queries = []
    for query in queries:
        decomposed = rewriter.decompose_to_json(query)
        decomposed_queries.append({
            "original": query,
            "decomposed": decomposed
        })
        print(f"Query: {query}")
        print(f"  → Video: {decomposed['video']}")
        print()
    
    # Step 3: Unload LLM to free memory
    print("=" * 80)
    print("Step 3: Unloading LLM to free GPU memory...")
    print("=" * 80)
    del rewriter.language_model
    del rewriter.tokenizer
    del rewriter.pipeline
    del rewriter
    gc.collect()
    torch.cuda.empty_cache()
    print("✓ LLM unloaded\n")
    
    # Step 4: Load CLIP tagger
    print("=" * 80)
    print("Step 4: Loading QueryTagger (CLIP)...")
    print("=" * 80)
    tagger = QueryTagger(device="cuda")
    tagger.load_model()
    print("✓ QueryTagger loaded\n")
    
    # Step 5: Tag all queries
    print("=" * 80)
    print("Step 5: Tagging queries with CLIP...")
    print("=" * 80)
    
    for item in decomposed_queries:
        query = item["original"]
        decomposed = item["decomposed"]
        
        print("\n" + "=" * 80)
        print(f"Original Query: {query}")
        print("-" * 80)
        print(f"Decomposed:")
        print(f"  - Text:  {decomposed['text']}")
        print(f"  - Audio: {decomposed['audio']}")
        print(f"  - Video: {decomposed['video']}")
        
        # Tag the video modality with CLIP
        video_query = decomposed["video"]
        tags = tagger.tag_query(video_query)
        
        print(f"\nPredicted Visual Tags ({len(tags)}):")
        if tags:
            print(f"  {', '.join(tags)}")
        else:
            print("  (no tags above threshold)")
        
        # Show how to use tags for filtering
        print("\nRetrieval Strategy:")
        print(f"  - Use 'video' embedding for semantic search")
        print(f"  - Filter candidates by tag overlap: {tags[:3]}...")
        print(f"  - Boost scores for videos with matching tags")
        print()


def example_tag_matching():
    """Example of tag-based filtering between query and video."""
    print("\n" + "=" * 80)
    print("Example: Tag Matching for Retrieval")
    print("=" * 80)
    
    # Simulated query tags (from QueryTagger)
    query_tags = ["obj_kitchenware", "act_cooking_prep", "act_chopping", "loc_home_kitchen"]
    
    # Simulated video tags (from VisionTagger during indexing)
    video_candidates = [
        {
            "video_id": "video_001",
            "tags": ["obj_kitchenware", "act_cooking_prep", "obj_food_beverage", "loc_home_kitchen"],
            "score": 0.85,
        },
        {
            "video_id": "video_002", 
            "tags": ["obj_tech_devices", "act_smartphone_use", "loc_home_living"],
            "score": 0.82,
        },
        {
            "video_id": "video_003",
            "tags": ["obj_kitchenware", "act_chopping", "act_stirring", "loc_home_kitchen"],
            "score": 0.78,
        },
    ]
    
    print(f"\nQuery Tags: {query_tags}\n")
    print("Candidate Videos:")
    
    for video in video_candidates:
        # Calculate tag overlap
        overlap = set(query_tags) & set(video["tags"])
        overlap_count = len(overlap)
        overlap_ratio = overlap_count / len(query_tags) if query_tags else 0
        
        # Boost score based on tag match
        boosted_score = video["score"] * (1.0 + 0.1 * overlap_count)  # +10% per matching tag
        
        print(f"\n  {video['video_id']}:")
        print(f"    - Base score: {video['score']:.3f}")
        print(f"    - Tags: {video['tags']}")
        print(f"    - Overlap: {list(overlap)} ({overlap_count}/{len(query_tags)})")
        print(f"    - Boosted score: {boosted_score:.3f}")
    
    print("\n" + "=" * 80)
    print("Result: video_001 and video_003 ranked higher due to tag overlap")
    print("=" * 80)


if __name__ == "__main__":
    main()
    example_tag_matching()
