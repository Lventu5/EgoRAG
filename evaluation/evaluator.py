"""
Evaluators for Retrieval and Generation tasks.

This module provides evaluator classes that aggregate multiple metrics
for convenient evaluation of retrieval and generation performance.
"""

from abc import ABC, abstractmethod
from data.video_dataset import Scene
from .metrics import (
    # Retrieval metrics
    TemporalIoU,
    MeanTemporalIoU,
    Overlap,
    RecallAtKIoU,
    SimpleRecallAtK,
    evaluate_retrieval,
)


class Evaluator(ABC):
    """Abstract base class for all evaluators."""
    
    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def forward_pass(self, **kwargs) -> dict:
        pass

    def __call__(self, **kwargs) -> dict:
        return self.forward_pass(**kwargs)
    
    def __repr__(self) -> str:
        return f"Evaluator: {self.name}"


class RetrievalEvaluator(Evaluator):
    """
    Evaluator for retrieval tasks.
    
    Computes the following metrics:
    - mIoU (Mean tIoU / Overlap): Average best tIoU for Top-1 predictions
    - tIoU: Plain temporal IoU for Top-1
    - R@K, IoU=τ: Recall at K with IoU thresholds (0.3 and 0.5)
    
    Standard K values: 1, 5, 10
    Standard IoU thresholds: 0.3, 0.5
    """
    
    def __init__(
        self, 
        name: str = "Retrieval Evaluator",
        k_values: list[int] = [1, 5, 10],
        iou_thresholds: list[float] = [0.3, 0.5]
    ):
        super().__init__(name=name)
        self.k_values = k_values
        self.iou_thresholds = iou_thresholds
        
        # Initialize metrics
        self.tiou = TemporalIoU(name="tIoU")
        self.miou = MeanTemporalIoU(name="mIoU")
        self.overlap = Overlap(name="Overlap")
        
        # Initialize Recall@K metrics for all combinations
        self.recall_metrics = {}
        self.simple_recalls = {}
        for k in k_values:
            for iou_thresh in iou_thresholds:
                metric_name = f"R@{k}_IoU={iou_thresh}"
                self.recall_metrics[metric_name] = RecallAtKIoU(
                    k=k, 
                    iou_threshold=iou_thresh, 
                    name=metric_name
                )
            metric_name = f"SimpleRecall@{k}"
            self.simple_recalls[metric_name] = SimpleRecallAtK(k = k)    
            

    def forward_pass(
        self,
        pred: list[list[tuple[str, Scene]]], 
        true: list[tuple[str, float, float]]
    ) -> dict:
        """
        Compute all retrieval metrics.
        
        Args:
            pred: List of predictions per query. Each prediction is a list of
                  (video_name, Scene) tuples, ordered by relevance (best first).
            true: List of ground truths per query as (video_uid, start_sec, end_sec).
        
        Returns:
            Dictionary with metric names as keys and computed values as floats.
        """
        results = {}
        
        # Compute tIoU (plain)
        results["tIoU"] = self.tiou.compute(pred=pred, true=true)
        
        # Compute mIoU
        results["mIoU"] = self.miou.compute(pred=pred, true=true)
        
        # Compute Overlap (ratio of predictions that overlap with GT)
        results["Overlap"] = self.overlap.compute(pred=pred, true=true)
        
        # Compute Recall@K with IoU thresholds
        for metric_name, metric in self.recall_metrics.items():
            results[metric_name] = metric.compute(pred=pred, true=true)
        for metric_name, simple_recall in self.simple_recalls.items():
            results[metric_name] = simple_recall.compute(pred=pred, true=true)

        return results


class GenerationEvaluator(Evaluator):
    """
    Evaluator for text generation tasks.
    
    Computes the following metrics:
    - BLEU Score
    - ROUGE-L Score
    - METEOR Score
    - BERT Score
    """
    
    def __init__(self, name: str = "Generation Evaluator"):
        super().__init__(name=name)
        self.bleu_score = BLEUScore()
        self.rouge_score = ROUGEScore()
        self.meteor = METEOR()
        self.bert_score = BERTScore()

    def forward_pass(
        self,
        pred: list[str], 
        true: list[str],
    ) -> dict:
        """
        Compute all generation metrics.
        
        Args:
            pred: List of generated text responses.
            true: List of ground truth text responses.
        
        Returns:
            Dictionary with metric names as keys and computed values as floats.
        """
        results = {}
        results["bleu_score"] = self.bleu_score.compute(pred=pred, true=true)
        results["rouge_score"] = self.rouge_score.compute(pred=pred, true=true)
        results["meteor"] = self.meteor.compute(pred=pred, true=true)
        results["bert_score"] = self.bert_score.compute(pred=pred, true=true)
        
        return results


def evaluate(
    responses: list[str] | None = None,
    retrieved_scenes: list[list[tuple[str, Scene]]] | None = None,
    true_responses: list[str] | None = None,
    true_scenes: list[tuple[str, float, float]] | None = None,
    run_retrieval: bool = True,
    run_generation: bool = True,
) -> dict:
    """
    Convenience function that runs the appropriate evaluators over the provided
    predictions and ground-truths and returns a dictionary with computed metrics.

    Args:
        responses: List of generated textual responses (one per query). 
                   If None, generation metrics are skipped.
        retrieved_scenes: List where each element is the top-k list of
                          (video_name, Scene) tuples retrieved for a query. 
                          If None, retrieval metrics are skipped.
        true_responses: List of ground-truth textual responses (one per query).
        true_scenes: List of ground-truth tuples as (video_uid, start_sec, end_sec).
        run_retrieval: Whether to run retrieval metrics (requires retrieved_scenes
                       and true_scenes).
        run_generation: Whether to run generation metrics (requires responses and
                        true_responses).

    Returns:
        A dict with two optional keys: "retrieval" and "generation", each
        mapping to another dict with the computed metric values.
    """
    results: dict = {}

    if run_retrieval:
        if retrieved_scenes is None or true_scenes is None:
            raise ValueError(
                "To run retrieval evaluation you must provide both "
                "'retrieved_scenes' and 'true_scenes'."
            )
        retrieval_eval = RetrievalEvaluator()
        results["retrieval"] = retrieval_eval.forward_pass(
            pred=retrieved_scenes, 
            true=true_scenes
        )

    if run_generation:
        if responses is None or true_responses is None:
            raise ValueError(
                "To run generation evaluation you must provide both "
                "'responses' and 'true_responses'."
            )
        generation_eval = GenerationEvaluator()
        results["generation"] = generation_eval.forward_pass(
            pred=responses, 
            true=true_responses
        )

    return results

