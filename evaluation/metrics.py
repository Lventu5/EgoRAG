from abc import ABC, abstractmethod
import numpy as np
import bert_score
from data.video_dataset import Scene
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bleurt import score as bleurt_score

class Metric(ABC):
    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def compute(self, **kwargs) -> float:
        pass

    def __call__(self, **kwargs)-> float:
        return self.compute(**kwargs)
    
    def __repr__(self) -> str:
        return f"Metric: {self.name}"
    

class topKAccuracyScene(Metric):
    """
    How many times have we retrieved the correct video and the correct scene in the video
    """
    def __init__(self, name = "topKAccuracyScene"):
        super().__init__(name = name)

    def compute(self, pred: list[list[tuple[str, Scene]]], true: list[tuple[str, float, float]]) -> float:
        """
        pred: list where each element is a list with the top k (video, scene) tuples
        true: list of (video-moments) of the ground truth
        """
        correct = 0
        assert len(pred) == len(true), "The predictions and the ground truths must have the same length"
        n = len(pred)
        for i in range(n):
            top_k = pred[i]
            gt_video, gt_moment = true[i]
            for video_name, prediction in top_k:
                if (video_name == gt_video 
                    and prediction.start_time < gt_moment 
                    and prediction.end_time > gt_moment
                ):
                    correct += 1
        return correct/n
    

class topKAccuracyVideo(Metric):
    """
    How many times have we retrieved the correct video at least once (not caring about retrieving the correct scene)
    """
    def __init__(self, name = "topKAccuracyVideo"):
        super().__init__(name)
    
    def compute(self, pred: list[list[tuple[str, Scene]]], true: list[tuple[str, float, float]]) -> float:
        """
        pred: list where each element is a list with the top k (video, scene) tuples
        true: list of (video-moments) of the ground truth
        """
        correct = 0
        assert len(pred) == len(true), "The predictions and the ground truths must have the same length"
        n = len(pred)
        for i in range(n):
            top_k = pred[i]
            gt_video, _ = true[i]
            for video_name, _ in top_k:
                if video_name == gt_video:
                    correct += 1
                    break
        return correct/n
        
class MeanReciprocalRank(Metric):
    """
    In which position is (on average) the correct scene among the topK
    """
    def __init__(self, name = "Mean Reciprocal Rank"):
        super().__init__(name)

    def compute(self, pred: list[list[tuple[str, Scene]]], true: list[tuple[str, float, float]]):
        """
        pred: list where each element is a list with the top k (video, scene) tuples
        true: list of (video-moments) of the ground truth
        """
        cum_sum = 0
        assert len(pred) == len(true), "The predictions and the ground truths must have the same length"
        n = len(pred)
        for i in range(n):
            top_k = pred[i]
            gt_video, gt_moment = true[i]
            for pos, (video_name, prediction) in enumerate(top_k):
                if (video_name == gt_video 
                    and prediction.start_time < gt_moment 
                    and prediction.end_time > gt_moment
                ):
                    cum_sum += 1/(pos + 1)
                    break
        return cum_sum/n
    

class topKPrecisionVideo(Metric):
    """
    How many of the retrieved clips are from the correct video?
    """
    def __init__(self, name = "topKPrecisionVideo"):
        super().__init__(name)

    def compute(self, pred: list[list[tuple[str, Scene]]], true: list[tuple[str, float, float]]) -> float:
        """
        pred: list where each element is a list with the top k (video, scene) tuples
        true: list of (video-moments) of the ground truth
        """
        correct = 0
        total = 0
        assert len(pred) == len(true), "The predictions and the ground truths must have the same length"
        for i in range(len(pred)):
            top_k = pred[i]
            gt_video, _ = true[i]
            for video_name, _ in top_k:
                if video_name == gt_video:
                    correct += 1
                total += 1
        return correct/total
    
class MeanRank(Metric):
    """
    Average rank of the correctly retrieved scene
    """
    def __init__(self, name="MeanRank"):
        super().__init__(name)
    
    def compute(self, pred: list[list[tuple[str, Scene]]], true: list[tuple[str, float, float]]):
        """
        pred: list where each element is a list with the top k (video, scene) tuples
        true: list of (video-moments) of the ground truth
        """
        assert len(pred) == len(true)
        ranks = []
        n = len(pred)
        for i in range(n):
            top_k = pred[i]
            gt_video, gt_moment = true[i]
            found_rank = None
            for pos, (video_name, prediction) in enumerate(top_k):
                if (video_name == gt_video 
                    and prediction.start_time < gt_moment 
                    and prediction.end_time > gt_moment):
                    found_rank = pos + 1
                    break
            ranks.append(found_rank if found_rank is not None else len(top_k) + 1)
        return float(np.mean(ranks))


class BLEUScore(Metric):
    """
    BLEU Score between the generated response and the ground truth respons
    """
    def __init__(self, name = "BLEU Score"):
        super().__init__(name)

    def compute(self, pred: list[str], true: list[str])-> float:
        """
        pred: list with the LLM-generated responses for each of the queries
        true: Ground truth completion for each of the queries
        """
        assert len(pred) == len(true), "The number of predicted responses and the groud truth must be the same"
        smoothie = SmoothingFunction().method4
        scores = []
        for p, t in zip(pred, true):
            t_tokens = [t.split()]
            p_tokens = p.split()
            scores.append(sentence_bleu(t_tokens, p_tokens, smoothing_function = smoothie))
        return float(np.mean(scores))
    

class ROUGEScore(Metric):
    """
    Rouge-N = # n-grams overlapped / # n-grams in the target
    """
    def __init__(self, name="ROUGE-L"):
        super().__init__(name)
        self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def compute(self, pred: list[str], true: list[str]) -> float:
        """
        pred: list with the LLM-generated responses for each of the queries
        true: Ground truth completion for each of the queries
        """
        assert len(pred) == len(true), "The number of predicted responses and the groud truth must be the same"
        scores = []
        for p, t in zip(pred, true):
            result = self._scorer.score(t, p)["rougeL"].fmeasure
            scores.append(result)
        return float(np.mean(scores))
    

class METEOR(Metric):
    """
    Similar to BLEU, more semantical
    """
    def __init__(self, name = "METEOR Score"):
        super().__init__(name)

    def compute(self, pred: list[str], true: list[str])-> float:
        """
        pred: list with the LLM-generated responses for each of the queries
        true: Ground truth completion for each of the queries
        """
        assert len(pred) == len(true), "The number of predicted responses and the groud truth must be the same"
        scores = [meteor_score([t], p) for p, t in zip(pred, true)]
        return float(np.mean(scores))
    

class BERTScore(Metric):
    """
    Similarity using BERT embeddings
    """
    def __init__(
            self, 
            name = "BERT Score",
            model_name: str = "All-MiniLM-L6-v2",
    ):
        super().__init__(name)
        self.model_name = model_name

    def compute(self, pred: list[str], true: list[str]):
        """
        pred: list with the LLM-generated responses for each of the queries
        true: Ground truth completion for each of the queries
        """
        assert len(pred) == len(true), "The number of predicted responses and the groud truth must be the same"
        P, R, F1 = bert_score.score(pred, true, model_type = self.model_name, verbose = False)
        return float(F1.mean())
    
class BLEURTScore(Metric):
    """
    Requires Google Bleurt-20 model
    """
    def __init__(
            self, 
            name = "BLEURT Score",
            bleurt_model_path = "",    # FIXME!! 
        ):
        super().__init__(name)
        self.scorer = bleurt_score.BleurtScorer(bleurt_model_path)
    
    def compute(self, pred: list[str], true: list[str])-> float:
        """
        pred: list with the LLM-generated responses for each of the queries
        true: Ground truth completion for each of the queries
        """
        assert len(pred) == len(true), "The number of predicted responses and the groud truth must be the same"
        scores = self.scorer.score(references = true, candidates = pred)
        return float(np.mean(scores))