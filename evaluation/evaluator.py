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
        results["topk_acc_scene"] = self.topk_acc_scene(pred, true)
        results["topk_acc_video"] = self.topk_acc_video(pred, true)
        results["mean_reciprocal_rank"] = self.mean_reciprocal_rank(pred, true)
        results["topk_prec_video"] = self.topk_prec_video(pred, true)
        results["mean_rank"] = self.mean_rank(pred, true)

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
        results["bleu_score"] = self.bleu_score(pred, true)
        results["rouge_score"] = self.rouge_score(pred, true)
        results["meteor"] = self.meteor(pred, true)
        results["berts_score"] = self.berts_score(pred, true)
        results["bleurt_score"] = self.bleurt_score(pred, true)

        return results

