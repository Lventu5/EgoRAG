from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional, Any
import json
import os
import gc
import logging
import torch
from data.query import Query, QueryDataset
from data.video_dataset import VideoDataset, VideoDataPoint

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

class Ego4D_NLQ_Runner:
    """
    Runner for Ego4D NLQ retrieval and question answering using LLMs.
    """
    def __init__(
        self,
        nlq_annotations_path: str,
        dataset: VideoDataset,
        retriever: Any = None,
        llm_generate: Callable[[str], str] = None,
        llm_rewrite: Optional[Callable[[str], str]] = None,
        llm_decompose: Optional[Callable[[str], Dict[str, str]]] = None,
        topk_videos: int = 3,
        topk_scenes: int = 5,
    ):
        """
        Args:
            nlq_annotations_path: path to NLQ annotations (json with "videos" list)
            dataset: VideoDataset instance
            retriever: Retriever instance with search_videos and search_scenes methods
            llm_generate: function that takes a prompt string and returns an answer string
            llm_rewrite: optional function that rewrites a query string
            llm_decompose: optional function that decomposes a query string into modalities
            topk_videos: number of top videos to retrieve
            topk_scenes: number of top scenes to retrieve per video
        """
        logging.info("Initializing Ego4D_NLQ_Runner...")
        self.nlq_annotations_path = nlq_annotations_path
        self.dataset = dataset
        self.retriever = retriever
        self.llm_generate = llm_generate
        self.llm_rewrite = llm_rewrite
        self.llm_decompose = llm_decompose
        self.topk_videos = topk_videos
        self.topk_scenes = topk_scenes

    # ---------- NLQ loading ----------

    def load_video_nlq_gt(self, nlq_json_path: str, video_uid: str):
        """
        Load NLQ entries and GT temporal answers for a given video_uid from the NLQ annotations.
        Args:
            nlq_json_path: path to NLQ annotations (json with "videos" list)
            video_uid: target video UID to filter NLQ entries
        Returns:
            List of dicts with keys: "query", "start_sec", "end_sec"
        """
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
                                "start_frame": int(fs),
                                "end_frame": int(fe),
                            })
        return results
    
    def load_queries_for_dataset(self) -> QueryDataset:
        """
        Load NLQ entries and GT temporal answers for all videos in the dataset.
        Populates each VideoDataPoint's `queries` attribute,
        and returns a global QueryDataset containing all queries.
        """
        all_queries = []

        for dp in self.dataset.video_datapoints:
            video_uid = getattr(dp, "video_uid", None)
            if not video_uid:
                continue

            nlq_entries = self.load_video_nlq_gt(self.nlq_annotations_path, video_uid)
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

        query_dataset = QueryDataset([q.to_dict() for q in all_queries])
        return query_dataset
    
    # ---------- Retrieval ----------

    def retrieve_for_query(self, nlq: Query, use_decomposition: bool = True) -> Dict[str, Any]:
        """
        Run retrieval for a single query:
          - optional rewrite
          - optional multimodal decomposition
        Returns a dict with retrieval results and (if available) GT segment.
        """
        raw_q = nlq.query
        q_rewritten = self.llm_rewrite(raw_q) if self.llm_rewrite else raw_q

        text_q = q_rewritten
        audio_q = ""
        video_q = ""

        if use_decomposition and self.llm_decompose:
            dec = self.llm_decompose(raw_q)
            text_q = dec.get("text_query", text_q) or text_q
            audio_q = dec.get("audio_query", "")
            video_q = dec.get("video_query", "")

        # --- Video shortlist by text only (simple & fast) ---
        ### TODO: ADD RETRIEVER

        pass

    def _scene_time_bounds(self, dp: "VideoDataPoint", scene_key: str) -> Tuple[float, float]:
        """
        Look up start/end seconds for a scene key like 'scene_12' from dp.scenes.
        If not found, returns (0.0, 0.0).
        """
        try:
            idx = int(scene_key.split("_")[-1])
            if 0 <= idx < len(dp.scenes):
                sc = dp.scenes[idx]
                return float(sc.start_time), float(sc.end_time)
        except Exception:
            pass
        return (0.0, 0.0)

    # ---------- GPU memory ----------

    @staticmethod
    def free_gpu():
        """Aggressive GPU cleanup."""
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    # ---------- LLM context ----------

    def build_llm_context_prompt(self, nlq: Query, retrieval: Dict[str, Any], max_scenes: int = 8) -> str:
        """
        Build a compact prompt for the answering LLM, with top-scoring scenes.
        """
        lines = []
        lines.append("You are given a user query and candidate video scenes.")
        lines.append("Use the evidence to answer the user's query. Be concise.")
        lines.append("")
        lines.append(f"User Query: {nlq.query}")
        lines.append("")

        ### TODO: ADD SCENES TO CONTEXT BASED ON RETRIEVAL RESULTS

        lines.append("Answer:")
        return "\n".join(lines)

    def answer_with_llm(self, nlq: Query, retrieval: Dict[str, Any]) -> str:
        """
        Create context from retrieval and produce an answer using the provided LLM generator.
        """
        prompt = self.build_llm_context_prompt(nlq, retrieval)
        return self.llm_generate(prompt)

if __name__ == "__main__":
    data_directory = "../../ego4d_data/v2/full_scale"
    logging.info(f"Loading video files from directory: {data_directory}")
    video_files = [
        os.path.join(data_directory, f)
        for f in os.listdir(data_directory)
        if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi"))
        and "animal" not in f.lower() 
        and "ai" not in f.lower()
    ]
    video_dataset = VideoDataset(video_files)
    print(f"Loaded {len(video_dataset)} videos into the dataset.")
    print("Sample video datapoint:", video_dataset.video_datapoints[0] if video_dataset.video_datapoints else "No datapoints found.")

    nlq_annotations_path = "../../ego4d_data/v2/annotations/nlq_train.json"
    logging.info(f"Loading NLQ annotations from: {nlq_annotations_path}")
    runner = Ego4D_NLQ_Runner(
        nlq_annotations_path=nlq_annotations_path,
        dataset=video_dataset,
        retriever=None,  # TODO: provide a Retriever instance
        llm_generate=lambda prompt: "This is a placeholder answer.",  # TODO: replace with actual LLM call
        llm_rewrite=None,
        llm_decompose=None,
    )
    logging.info("Loading NLQ entries for the dataset...")
    runner.load_queries_for_dataset()
    logging.info("NLQ entries loaded for dataset.")

    if video_dataset.video_datapoints and video_dataset.video_datapoints[0].queries:
        first_dp = video_dataset.video_datapoints[0]
        nlq = first_dp.queries
        for i, query in enumerate(nlq):
            logging.info(f"NLQ {i+1}/{len(nlq)}: {query.query}")
            start_frame = query.start_frame if query.start_frame is not None else "N/A"
            end_frame = query.end_frame if query.end_frame is not None else "N/A"
            logging.info(f"GT Segment: {query.start_sec}s to {query.end_sec}s (frames {start_frame} to {end_frame})")

