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
        self.encoder.load_models()
        self.video_dataset = self.encoder.encode_videos()
        if self.save_encoded and self.save_dir is not None:
            pickle_path = f"{self.save_dir}/encoded_videos.pkl"
            self.video_dataset.save_to_pickle(pickle_path)
        self.encoder.unload_models()
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
    
    def evaluate(self):
        """
        Runs the evaluation using the metrics created
        """
        retrieval_evaluator = RetrievalEvaluator()
        generation_evaluator = GenerationEvaluator()
        return 
