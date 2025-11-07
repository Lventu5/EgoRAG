from abc import ABC, abstractmethod
from collections import defaultdict
from data.video_dataset import Scene
from .metrics import(
    topKAccuracyScene,
    topKAccuracyVideo,
    MeanReciprocalRank,
    topKPrecisionVideo,
    MeanRank,
    BLEUScore,
    ROUGEScore,
    METEOR,
    BERTScore,
    BLEURTScore
)

class Evaluator(ABC):
    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def forward_pass(self, **kwargs)-> dict:
        pass

    def __call__(self, **kwargs)-> dict:
        return self.forward_pass(**kwargs)
    
    def __repr__(self)-> str:
        return f"Evaluator: {self.name}"

class RetrievalEvaluator(Evaluator):
    def __init__(
        self, 
        name: str = "Retrieval Evaluator",
    ):
        super().__init__(name = name)
        self.topk_acc_scene = topKAccuracyScene()
        self.topk_acc_video = topKAccuracyVideo()
        self.mean_reciprocal_rank = MeanReciprocalRank()
        self.topk_prec_video = topKPrecisionVideo()
        self.mean_rank = MeanRank()

    def forward_pass(
        self,
        pred: list[list[tuple[str, Scene]]], 
        true: list[tuple[str, float, float]]
    )-> dict:
        results = defaultdict(float)
        results["topk_acc_scene"] = self.topk_acc_scene(pred=pred, true=true)
        results["topk_acc_video"] = self.topk_acc_video(pred=pred, true=true)
        results["mean_reciprocal_rank"] = self.mean_reciprocal_rank(pred=pred, true=true)
        results["topk_prec_video"] = self.topk_prec_video(pred=pred, true=true)
        results["mean_rank"] = self.mean_rank(pred=pred, true=true)

        return results


class GenerationEvaluator(Evaluator):
    def __init__(
        self, 
        name: str = "Generation Evaluator",
    ):
        super().__init__(name = name)
        self.bleu_score = BLEUScore()
        self.rouge_score = ROUGEScore()
        self.meteor = METEOR()
        self.berts_score = BERTScore()
        self.bleurt_score = BLEURTScore()

    def forward_pass(
        self,
        pred: list[str], 
        true: list[str],
    )-> dict:
        results = defaultdict(float)
        results["bleu_score"] = self.bleu_score(pred = pred, true = true)
        results["rouge_score"] = self.rouge_score(pred=pred, true=true)
        results["meteor"] = self.meteor(pred=pred, true=true)
        results["berts_score"] = self.berts_score(pred=pred, true=true)
        results["bleurt_score"] = self.bleurt_score(pred=pred, true=true)

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
        responses: list of generated textual responses (one per query). If None,
            generation metrics are skipped.
        retrieved_scenes: list where each element is the top-k list of
            (video_name, Scene) tuples retrieved for a query. If None,
            retrieval metrics are skipped.
        true_responses: list of ground-truth textual responses (one per query).
        true_scenes: list of ground-truth (video_name, moment) tuples where
            moment is a float timestamp (seconds) indicating the ground-truth
            time to check against retrieved scenes.
        run_retrieval: whether to run retrieval metrics (requires retrieved_scenes
            and true_scenes).
        run_generation: whether to run generation metrics (requires responses and
            true_responses).

    Returns:
        A dict with two optional keys: "retrieval" and "generation", each
        mapping to another dict with the computed metric values.
    """
    results: dict = {}

    if run_retrieval:
        if retrieved_scenes is None or true_scenes is None:
            raise ValueError("To run retrieval evaluation you must provide both 'retrieved_scenes' and 'true_scenes'.")
        retrieval_eval = RetrievalEvaluator()
        results["retrieval"] = retrieval_eval.forward_pass(pred=retrieved_scenes, true=true_scenes)

    if run_generation:
        if responses is None or true_responses is None:
            raise ValueError("To run generation evaluation you must provide both 'responses' and 'true_responses'.")
        generation_eval = GenerationEvaluator()
        results["generation"] = generation_eval.forward_pass(pred=responses, true=true_responses)

    return results

