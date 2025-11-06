from indexing.multimodal_encoder import MultiModalEncoder
from retrieval.hierarchical_retriever import HierarchicalRetriever
from data.video_dataset import VideoDataset
from data.query import QueryDataset, Query
import torch
import logging
from transformers import logging as hf_logging
from indexing.utils.logging import LevelAwareFormatter
import os

handler = logging.StreamHandler()
handler.setFormatter(LevelAwareFormatter())
logging.basicConfig(
    level=logging.INFO,
    handlers=[handler],
)
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()


def main(
    data_directory: str = "../../data",
    pickle_file: str = "../../data/video_dataset.pkl"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(data_directory):
        raise ValueError(f"Data directory {data_directory} does not exist.")

    if os.path.exists(pickle_file):
        logging.info(f"Loading video dataset from pickle file: {pickle_file}")
        video_dataset = VideoDataset.load_from_pickle(pickle_file)
    else:
        logging.info(f"Loading video files from directory: {data_directory}")
        video_files = [
            os.path.join(data_directory, f)
            for f in os.listdir(data_directory)
            if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi"))
        and "animal" not in f.lower() 
        and "ai" not in f.lower()
        ]
        video_dataset = VideoDataset(video_files)

        logging.info("Initializing MultiModalEncoder...")
        encoder = MultiModalEncoder(
            video_dataset=video_dataset,
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_workers=2
        )
        encoder.load_models()
        encoder.encode_videos()

        video_dataset = encoder.dataset
        video_dataset.save_to_pickle(pickle_file)
        logging.info(f"[SAVE] Video dataset saved to pickle file: {pickle_file}")
        del encoder
        torch.cuda.empty_cache()

    retrieval = HierarchicalRetriever(video_dataset, device=device)

    if not video_dataset.video_datapoints:
        raise RuntimeError("No encoded video datapoints found in the dataset.")
    
    queries = [
        "When is the first goal from Bologna scored?", # 00:30
        "When is the player from Pisa receiving a red card?", # 00:52
        "Who jumps over the ad board to celebrate?", # 01:14
        "When do the second half highlights start?", # 01:38
        "Who is the highlights sposnsor?", # 00:01
        "When does the ball bounce on the net?", # 00:41
        "When does the player without the cap wins the first set?", # 01:24
        "How many people with a cap on the stands are celebrating the winning point?", # 01:32
        "Who wins the tennis match?", # 04:33
        "Does the commentary rhythm suggest an important moment?",
        "Are there changes in pace or intensity in the commentary?",
        "Does it look like a break or preparation moment in the game?",
        "Does the video show strong emotions from the players or crowd?",
    ]

    queries = QueryDataset(queries)

    modalities = ["video", "audio", "text", "caption"]

    hierarchical_results = retrieval.retrieve_hierarchically(
        queries=queries,
        modalities=modalities,
        top_k_videos=1,
        top_k_scenes=5
    )

    for query, results in hierarchical_results.items():
        print(f"\n===== QUERY: '{query}' → {queries[int(query[-1])].get_query()} =====")

        fused_videos = results["fused"]
        if not fused_videos:
            print("  No fused results found.")
            continue

        print("\n--- FUSED MULTIMODAL RANKING ---")

        for rank, (video_name, global_score, fused_scenes) in enumerate(fused_videos, start=1):
            print(f"  [Rank {rank}] Video: '{video_name}'  |  Score: {global_score:.4f}")

            if fused_scenes:
                for scene_rank, (scene_id, scene_score) in enumerate(fused_scenes, start=1):
                    print(f"      → Scene {scene_rank}: {scene_id}  |  Score: {scene_score:.4f}")
            else:
                print("      → No relevant scenes found.")


if __name__ == "__main__":    
    main()