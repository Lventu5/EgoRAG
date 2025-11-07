from typing import Any

import torch

from indexing.multimodal_encoder import MultiModalEncoder
from data.dataset import DatasetFactory
from indexing.utils.logging import pretty_print_retrieval
from retrieval.hierarchical_retriever import HierarchicalRetriever
from evaluation.evaluator import RetrievalEvaluator, GenerationEvaluator
from generation.generator import Generator
from data.query import QueryDataset

class Launcher:
    def __init__(
        self,
        dataset_type: str,
        video_path: str,
        nlq_annotations_path: str,
        generator: str = "InternVideo",
        modalities: list[str] = ["video", "audio", "text", "caption"],
        topk_videos: int = 3,
        topk_scenes: int = 5,
        is_pickle: bool = False,
        save_dir: str = None,
        save_encoded: bool = False,
        workers: int = 2,
    ):
        """
        Initialize the Launcher class that will manage the entire RAG pipeline.
        Args:
            dataset_type (str): Type of dataset to use (e.g., "Ego4D").
            video_path (str): Path to the video data.
            nlq_annotations_path (str): Path to the NLQ annotations.
            encoder (MultiModalEncoder): The multimodal encoder instance.
            retriever (HierarchicalRetriever): The hierarchical retriever instance.
            generator (Any): The LLM generator instance.
        """
        self.video_path = video_path
        self.nlq_annotations_path = nlq_annotations_path
        self.dataset = DatasetFactory.get_dataset(dataset_type, video_path, nlq_annotations_path)
        self.generator = generator
        self.modalities = modalities
        self.topk_videos = topk_videos
        self.topk_scenes = topk_scenes
        self.save_dir = save_dir
        self.save_encoded = save_encoded

        self.video_dataset = self.dataset.load_videos(is_pickle=is_pickle)
        self.query_datasets = self.dataset.load_queries(self.video_dataset.get_uids())

        self.encoder = MultiModalEncoder(
            video_dataset=self.video_dataset,
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_workers=workers
        )

        self.retriever = HierarchicalRetriever(self.video_dataset, device=self.encoder.device)

        self.generator = Generator(model_name=generator)

    def encode(self):
        """
        Encode the videos using the multimodal encoder.
        """
        self.video_dataset = self.encoder.encode_videos()
        if self.save_encoded and self.save_dir is not None:
            pickle_path = f"{self.save_dir}/encoded_videos.pkl"
            self.video_dataset.save_to_pickle(pickle_path)
        torch.cuda.empty_cache()

    def retrieve(self):
        """
        Retrieve relevant video segments for each query using the hierarchical retriever.
        """
        results = self.retriever.retrieve_hierarchically(
            queries=self.query_datasets,
            modalities=self.modalities,
            top_k_videos=self.topk_videos,
            top_k_scenes=self.topk_scenes
        )
        return results
    
    def generate_answers(self, retrieval_results: dict, queries: QueryDataset):
        """
        Generate answers for the retrieved video segments using the LLM generator.
        Args:
            retrieval_results (dict): The retrieval results from the retriever.
            queries (QueryDataset): The original queries.
        Returns:
            dict: Generated answers for each query.
        """
        return self.generator.generate_answers(retrieval_results, queries)

    def run(self):
        """
        Run the entire RAG pipeline: encoding, retrieval, and generation.
        """
        if not self.video_dataset.encoded:
            self.encode()

        retrieval_results = self.retrieve()
        pretty_print_retrieval(retrieval_results, max_videos=self.topk_videos, max_scenes=self.topk_scenes)
        # answers = self.generate_answers(retrieval_results)
        return retrieval_results
    
    

    def evaluate(
        self,
        retrieval_results: dict,
        responses: dict | list[str] | None = None,
        run_retrieval: bool = True,
        run_generation: bool = False,
    ) -> dict:
        """
        Evaluate retrieval and generation results.

        Args:
            retrieval_results: output of `HierarchicalRetriever.retrieve_hierarchically`.
            responses: either a dict mapping qid -> generated string, or a list of generated strings
                aligned with `self.query_datasets` order. If None and run_generation=True an error is raised.
            run_retrieval: whether to compute retrieval metrics (requires retrieval_results).
            run_generation: whether to compute generation metrics (requires responses).

        Returns:
            dict with optional keys "retrieval" and "generation" containing metric dicts.
        """
        results = {}

        # Build ordered list of queries to preserve alignment.
        # QueryDataset stores queries in .queries as a list[Query]
        queries = getattr(self.query_datasets, "queries", list(self.query_datasets))

        if run_retrieval:
            if retrieval_results is None:
                raise ValueError("retrieval_results must be provided when run_retrieval is True")

            # Convert hierarchical retriever output to the format expected by RetrievalEvaluator:
            # pred: list (per query) of list of (video_name, Scene) tuples
            preds = []
            trues = []

            for query in queries:
                qid = query.qid
                entry = retrieval_results.get(qid, {})
                fused_list = entry.get("fused", [])

                # Flatten: for each top video, append its top scenes (scene objects)
                query_preds: list[tuple[str, Any]] = []
                for video_name, global_score, scene_ranking in fused_list:
                    # scene_ranking is expected to be a list of tuples (Scene, score)
                    for scene_item in scene_ranking:
                        # scene_item may be (Scene, score) or Scene depending on fuser
                        if isinstance(scene_item, tuple) and len(scene_item) >= 1:
                            scene_obj = scene_item[0]
                        else:
                            scene_obj = scene_item
                        query_preds.append((video_name, scene_obj))

                preds.append(query_preds)

                # Ground truth: try to get a representative timestamp from the query gt
                gt_video = query.video_uid
                gt_moment = None
                if query.gt and query.gt.get("start_sec") is not None:
                    gt_moment = query.gt.get("start_sec")
                elif query.gt and query.gt.get("end_sec") is not None:
                    gt_moment = query.gt.get("end_sec")

                trues.append((gt_video, gt_moment))

            retrieval_evaluator = RetrievalEvaluator()
            results["retrieval"] = retrieval_evaluator.forward_pass(pred=preds, true=trues)

        if run_generation:
            if responses is None:
                raise ValueError("responses must be provided when run_generation is True")

            # Normalize responses into ordered list aligned with queries
            if isinstance(responses, dict):
                gen_preds = [responses.get(q.qid, "") for q in queries]
            else:
                # assume list aligned with queries
                gen_preds = list(responses)

            # Attempt to extract ground truth textual answers from the query dataset if present
            # Fallback: empty strings
            gen_trues = []
            for q in queries:
                # look for a 'answer' field in query.gt or elsewhere; default to empty
                gt_answer = None
                if q.gt and isinstance(q.gt.get("answer"), str):
                    gt_answer = q.gt.get("answer")
                gen_trues.append(gt_answer or "")

            generation_evaluator = GenerationEvaluator()
            results["generation"] = generation_evaluator.forward_pass(pred=gen_preds, true=gen_trues)

        return results
