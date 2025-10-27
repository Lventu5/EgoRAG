from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional, Any
import json
import os
import gc
import logging
import torch
from copy import deepcopy
from data.query import Query, QueryDataset
from data.video_dataset import VideoDataset, VideoDataPoint
from indexing.multimodal_encoder import MultiModalEncoder
from retrieval.hierarchical_retriever import HierarchicalRetriever

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
        encoder: MultiModalEncoder = None,
        retriever: Optional[HierarchicalRetriever] = None,
        llm_generate: Callable[[str], str] = None,
        llm_rewrite: Optional[Callable[[str], str]] = None,
        llm_decompose: Optional[Callable[[str], Dict[str, str]]] = None,
        topk_videos: int = 3,
        topk_scenes: int = 5,
        device: str = "cpu",
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
        self.encoder = encoder
        self.retriever = retriever
        self.llm_generate = llm_generate
        self.llm_rewrite = llm_rewrite
        self.llm_decompose = llm_decompose
        self.topk_videos = topk_videos
        self.topk_scenes = topk_scenes
        self.device = device

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
                                "start_frame": int(fs) if fs is not None else None,
                                "end_frame": int(fe) if fe is not None else None,
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
        self.dataset.query_dataset = query_dataset
        return query_dataset
    
    # ---------- Retrieval ----------

    def _ensure_query_dataset(self) -> QueryDataset:
        """Ensure the VideoDataset has a QueryDataset loaded."""
        qd = getattr(self.dataset, "query_dataset", None)
        if qd is None or len(qd) == 0:
            logging.info("QueryDataset not found on VideoDataset — loading from NLQ annotations...")
            qd = self.load_queries_for_dataset()
        return qd

    def run_retrieval(self, modalities: list[str] | str = ("text", "audio", "video"), top_k_videos: int = 3, top_k_scenes: int = 1) -> Dict[str, Dict[str, list[tuple]]]:
        """
        Run retrieval for all queries in the dataset's QueryDataset.
        Args:
            modalities: list of modalities to use for retrieval
            top_k_videos: number of top videos to retrieve per query
            top_k_scenes: number of top scenes to retrieve per video
        Returns:
            Dict mapping query IDs to retrieval results per modality.
        """
        # Check if encoding is needed
        needs_encoding = (
            len(self.dataset.video_datapoints) == 0 or
            any((not getattr(dp, "global_embeddings", None)) for dp in self.dataset.video_datapoints)
        )

        if needs_encoding:
            local_encoder = self.encoder or MultiModalEncoder(
                video_dataset=self.dataset,
                device=self.device,
                max_workers=1,
            )
            try:
                logging.info("Encoding videos in the dataset (on-demand)...")
                local_encoder.load_models()
                local_encoder.encode_videos()

                self.dataset = local_encoder.dataset
                if self.retriever is not None:
                    self.retriever.video_dataset = self.dataset
            finally:
                # free GPU memory
                try:
                    local_encoder.unload_models()
                except Exception as e:
                    logging.warning(f"Could not unload encoder models: {e}")
                del local_encoder
                torch.cuda.empty_cache()
                gc.collect()

        if self.retriever is None:
            from retrieval.hierarchical_retriever import HierarchicalRetriever
            logging.info("No retriever found — initializing a default HierarchicalRetriever.")
            self.retriever = HierarchicalRetriever(self.dataset, device=self.device)

        queries = self._ensure_query_dataset()
        results = self.retriever.retrieve_hierarchically(
            queries=queries,
            modalities=modalities,
            top_k_videos=top_k_videos,
            top_k_scenes=top_k_scenes
        )

        return results

    def run_retrieval_for_video(self, video_uid: str, modalities: list[str] | str = ("text", "audio", "video"), top_k_videos: int = 3, top_k_scenes: int = 1) -> Dict[str, Dict[str, list[tuple]]]:
        """
        Run retrieval for all queries associated with a specific video_uid.
        Args:
            video_uid: target video UID to filter queries
            modalities: list of modalities to use for retrieval
            top_k_videos: number of top videos to retrieve per query
            top_k_scenes: number of top scenes to retrieve per video
        Returns:
            Dict mapping query IDs to retrieval results per modality.
        """
        target_dp = None
        for dp in getattr(self.dataset, "video_datapoints", []):
            if getattr(dp, "video_uid", None) == video_uid:
                target_dp = dp
                break

        if target_dp is None:
            logging.warning(f"Video uid '{video_uid}' not found.")
            return {}

        needs_encoding = not getattr(target_dp, "global_embeddings", None)

        if needs_encoding:
            tmp_ds = VideoDataset([target_dp.video_path])
            tmp_dp = tmp_ds.video_datapoints[0]
            tmp_dp.video_uid = target_dp.video_uid
            tmp_dp.video_name = getattr(target_dp, "video_name", os.path.basename(target_dp.video_path))

            local_encoder = self.encoder or MultiModalEncoder(
                video_dataset=tmp_ds,
                device=self.device,
                max_workers=1,
            )
            try:
                logging.info(f"Encoding on-demand only of video: '{video_uid}'...")
                local_encoder.load_models()
                local_encoder.encode_videos()

                enc_dp = local_encoder.dataset.video_datapoints[0]
                target_dp.scenes = enc_dp.scenes
                target_dp.scene_embeddings = enc_dp.scene_embeddings
                target_dp.global_embeddings = enc_dp.global_embeddings

            finally:
                # Free GPU/CPU memory of the temporary encoder
                try:
                    local_encoder.unload_models()
                except Exception as e:
                    logging.warning(f"Could not unload encoder models: {e}")
                del local_encoder
                torch.cuda.empty_cache()
                gc.collect()

        if self.retriever is None:
            from retrieval.hierarchical_retriever import HierarchicalRetriever
            logging.info("No retriever found — initializing a default HierarchicalRetriever.")
            self.retriever = HierarchicalRetriever(self.dataset, device=self.device)
        else:
            self.retriever.video_dataset = self.dataset

        qd = self._ensure_query_dataset()
        filtered = QueryDataset([q.to_dict() for q in qd
                                if getattr(q, "video_uid", None) == video_uid])

        if len(filtered) == 0:
            logging.warning(f"No queries found for video_uid='{video_uid}'.")
            return {}

        results = self.retriever.retrieve_hierarchically(
            queries=filtered,
            modalities=modalities,
            top_k_videos=top_k_videos,
            top_k_scenes=top_k_scenes
        )
        return results

    @staticmethod
    def pretty_print_retrieval(results: Dict[str, Dict[str, list[tuple]]], max_videos: int = 3, max_scenes: int = 1):
        """
        Pretty-print retrieval results.
        Args:
            results: Dict mapping query IDs to retrieval results per modality.
            max_videos: maximum number of videos to print per query
            max_scenes: maximum number of scenes to print per video
        """
        for qid, per_mod in results.items():
            print(f"\n=== Query {qid} ===")
            for mod, items in per_mod.items():
                print(f"  [{mod}]")
                for vi, (video_name, global_score, scenes) in enumerate(items[:max_videos], 1):
                    print(f"    {vi}. {video_name} (video score: {global_score:.4f})")
                    for si, (scene_obj, s_score) in enumerate(scenes[:max_scenes], 1):
                        if hasattr(scene_obj, "start_time"):
                            span = f"{scene_obj.start_time:.2f}s–{scene_obj.end_time:.2f}s"
                        else:
                            span = str(scene_obj)
                        print(f"       - scene {si}: {span} (score: {s_score:.4f})")

    # ---------- GPU memory ----------

    @staticmethod
    def free_gpu():
        """Aggressive GPU cleanup."""
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
    
    def unload_encoder_models(self):
        """Unload heavy model weights from GPU but keep embeddings in memory."""
        if self.encoder is not None:
            try:
                if hasattr(self.encoder, "unload_models"):
                    self.encoder.unload_models()
                else:
                    del self.encoder.model
            except Exception as e:
                logging.warning(f"Could not unload encoder models: {e}")
        torch.cuda.empty_cache()
        gc.collect()

    # ---------- LLM context ----------

    def build_llm_context_prompt(self, nlq: Query, retrieval: Dict[str, Any], max_scenes: int = 8) -> str:
        """
        Build a compact prompt for the answering LLM, with top-scoring scenes.
        """
        pass

    def answer_with_llm(self, nlq: Query, retrieval: Dict[str, Any]) -> str:
        """
        Create context from retrieval and produce an answer using the provided LLM generator.
        """
        prompt = self.build_llm_context_prompt(nlq, retrieval)
        return self.llm_generate(prompt)

if __name__ == "__main__":

    logging.info("Starting Ego4D NLQ Runner...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_directory = "../../ego4d_data/v2/full_scale"
    pickle_file: str = "../../ego4d_data/video_dataset.pkl"

    if os.path.exists(pickle_file):
        logging.info(f"Loading video dataset from pickle file: {pickle_file}")
        video_dataset = VideoDataset.load_from_pickle(pickle_file)
    else:
        logging.info(f"Loading video files from directory: {data_directory}")
        video_files = [
            os.path.join(data_directory, f)
            for f in os.listdir(data_directory)
            if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi"))
        ]
        video_dataset = VideoDataset([video_files[0]])

        logging.info("Initializing MultiModalEncoder...")
        encoder = MultiModalEncoder(
            video_dataset=video_dataset,
            device=device,
            max_frames_per_scene=32,
            max_workers=1
        )
        encoder.load_models()
        encoder.encode_videos()

        video_dataset = encoder.dataset
        # video_dataset.save_to_pickle(pickle_file)
        # logging.info(f"[SAVE] Video dataset saved to pickle file: {pickle_file}")
        del encoder
        torch.cuda.empty_cache()

    print(f"Loaded {len(video_dataset)} videos into the dataset.")
    print("Sample video datapoint:", video_dataset.video_datapoints[0] if video_dataset.video_datapoints else "No datapoints found.")

    nlq_annotations_path = "../../ego4d_data/v2/annotations/nlq_train.json"
    logging.info(f"Loading NLQ annotations from: {nlq_annotations_path}")

    # encoder = MultiModalEncoder(video_dataset=video_dataset, device=device, max_workers=1)
    retriever = HierarchicalRetriever(video_dataset, device=device)
    runner = Ego4D_NLQ_Runner(
        nlq_annotations_path=nlq_annotations_path,
        dataset=video_dataset,
        retriever=retriever,
        llm_generate=lambda prompt: "This is a placeholder answer.",  # TODO: replace with actual LLM call
        llm_rewrite=None,
        llm_decompose=None,
    )

    logging.info("Loading NLQ entries for the dataset...")
    queries = runner.load_queries_for_dataset()
    logging.info(f"Loaded a total of {len(queries)} NLQ entries across the dataset.")

    for i, query in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}]")
        print(f"Video UID: {query.video_uid}")
        print(f"Query: {query.query_text}")
        
        if hasattr(query, "gt") and query.gt:
            gt = query.gt
            print(f"GT Segment: {gt['start_sec']}s → {gt['end_sec']}s "
                f"(frames {gt.get('start_frame', 'N/A')}–{gt.get('end_frame', 'N/A')})")

    logging.info("Running retrieval for all queries in the dataset...")
    modalities = ["text", "caption", "video", "audio"]
    results = runner.run_retrieval(
        modalities=modalities,
        top_k_videos=3,
        top_k_scenes=2
    )
    runner.pretty_print_retrieval(results, max_videos=3, max_scenes=2)