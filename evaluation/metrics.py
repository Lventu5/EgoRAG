from abc import ABC, abstractmethod
import numpy as np
import bert_score
from data.video_dataset import Scene
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
# from bleurt import score as bleurt_score # FIXME, to install

def iou(pred_scene, gt_start, gt_end):
    p_start, p_end = pred_scene.start_time, pred_scene.end_time
    intersection = max(0.0, min(p_end, gt_end) - max(p_start, gt_start))
    
    area_pred = p_end - p_start
    area_gt = gt_end - gt_start
    union = area_pred + area_gt - intersection
    
    return intersection / union if union > 0 else 0.0

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

        assert len(pred) == len(true), "The predictions and the ground truths must have the same length"
        n = len(pred)
        total_score = 0.0
        for i in range(n):
            top_k = pred[i]
            gt_video, gt_start, gt_end = true[i]
            best_iou = 0.0
            for el in top_k:
                video_name = el[0]
                prediction = el[1]
                if video_name != gt_video:
                    continue
                best_iou = max(best_iou, iou(prediction, gt_start, gt_end))
            total_score += best_iou
        return total_score / n
    

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
            gt_video = true[i][0]
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
        Modified Mean Reciprocal Rank: incorporate IoU overlap and rank position.
        For each query we compute max(overlap / (pos+1)) across predictions and
        average those scores.
        """
        def _unpack_true(t):
            if len(t) == 2:
                vid, moment = t
                return vid, float(moment) - 0.5, float(moment) + 0.5
            elif len(t) >= 3:
                vid, start, end = t[0], t[1], t[2]
                return vid, float(start) if start is not None else 0.0, float(end) if end is not None else float(start or 0.0)
            else:
                raise ValueError("Unsupported true format")

        cum_sum = 0.0
        assert len(pred) == len(true), "The predictions and the ground truths must have the same length"
        n = len(pred)
        for i in range(n):
            top_k = pred[i]
            gt_video, gt_start, gt_end = _unpack_true(true[i])
            best_score = 0.0
            for pos, (video_name, prediction) in enumerate(top_k):
                if video_name != gt_video:
                    continue
                overlap = iou(prediction, gt_start, gt_end)
                best_score = max(best_score, overlap / (pos + 1))
            cum_sum += best_score
        return cum_sum / n
    

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
            gt_video = true[i][0]
            for el in top_k:
                video_name = el[0]
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
        MeanRank adjusted to account for overlap: for each query we find the
        predicted scene with maximum IoU and compute an effective rank which is
        reduced toward 1 proportionally to the overlap. If no overlap is found
        for a query, we return len(top_k)+1 for that query.
        """
        def _unpack_true(t):
            if len(t) == 2:
                vid, moment = t
                return vid, float(moment) - 0.5, float(moment) + 0.5
            elif len(t) >= 3:
                vid, start, end = t[0], t[1], t[2]
                return vid, float(start) if start is not None else 0.0, float(end) if end is not None else float(start or 0.0)
            else:
                raise ValueError("Unsupported true format")

        assert len(pred) == len(true)
        ranks = []
        n = len(pred)
        for i in range(n):
            top_k = pred[i]
            gt_video, gt_start, gt_end = _unpack_true(true[i])
            best_pos = None
            best_iou = 0.0
            for pos, (video_name, prediction) in enumerate(top_k):
                if video_name != gt_video:
                    continue
                iou = iou(prediction, gt_start, gt_end)
                if iou > best_iou:
                    best_iou = iou
                    best_pos = pos

            if best_pos is None:
                ranks.append(len(top_k) + 1)
            else:
                # reduce the rank towards 1 proportionally to the overlap
                effective_rank = (best_pos + 1) - best_iou * best_pos
                ranks.append(effective_rank)

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


class RecallAtK(Metric):
    """
    Recall@K (solo sul video):
    Per ogni query controlla se il video corretto compare almeno una volta
    nei top-K retrieved.
    
    Risultato:
        (# query per cui il video corretto è nelle top-K) / (# query totali)
    """
    def __init__(self, k: int = 1, name: str | None = None):
        self.k = k
        if name is None:
            name = f"Recall@{k}"
        super().__init__(name)

    def compute(self, pred: list[list[tuple[str, Scene]]],
                true: list[tuple[str, float, float]]) -> float:
        assert len(pred) == len(true), "The predictions and the ground truths must have the same length"
        n = len(pred)
        hits = 0

        for i in range(n):
            top_k = pred[i][:self.k]
            gt_video = true[i][0]
            # hit se il video corretto appare almeno una volta nelle top-K
            if any(video_name == gt_video for video_name, _ in top_k):
                hits += 1

        return hits / n if n > 0 else 0.0

class IoUAtThreshold(Metric):
    """
    IoU@τ:
    
    Per ogni query:
      - prendo TUTTE le predizioni (o le top-K che gli passiamo in pred[i])
        che appartengono al video corretto
      - considero la best IoU tra predizioni e ground truth
      - conto la query come "successo" se best_iou >= iou_threshold

    Risultato:
        (# query con best IoU >= τ) / (# query totali)
    """
    def __init__(self, iou_threshold: float = 0.5, name: str | None = None):
        self.iou_threshold = iou_threshold
        if name is None:
            name = f"IoU@{iou_threshold}"
        super().__init__(name)

    def compute(self, pred: list[list[tuple[str, Scene]]],
                true: list[tuple[str, float, float]]) -> float:
        def _unpack_true(t):
            if len(t) == 2:
                vid, moment = t
                return vid, float(moment) - 0.5, float(moment) + 0.5
            elif len(t) >= 3:
                vid, start, end = t[0], t[1], t[2]
                gt_start = float(start) if start is not None else 0.0
                if end is not None:
                    gt_end = float(end)
                else:
                    gt_end = gt_start
                return vid, gt_start, gt_end
            else:
                raise ValueError("Unsupported true format")

        assert len(pred) == len(true), "The predictions and the ground truths must have the same length"

        n = len(pred)
        hits = 0

        for i in range(n):
            candidates = pred[i]
            gt_video, gt_start, gt_end = _unpack_true(true[i])

            best_iou = 0.0
            for video_name, prediction in candidates:
                if video_name != gt_video:
                    continue
                best_iou = max(best_iou, iou(prediction, gt_start, gt_end))

            if best_iou >= self.iou_threshold:
                hits += 1

        return hits / n if n > 0 else 0.0


class CumulativeIoUAtThreshold(Metric):
    """
    IoU@τ:
    
    Per ogni query:
      - prendo TUTTE le predizioni (o le top-K che gli passiamo in pred[i])
        che appartengono al video corretto
      - considero la best IoU tra predizioni e ground truth
      - conto la query come "successo" se best_iou >= iou_threshold

    Risultato:
        (# query con best IoU >= τ) / (# query totali)
    """
    def __init__(self, iou_threshold: float = 0.5, name: str | None = None):
        self.iou_threshold = iou_threshold
        if name is None:
            name = f"IoU@{iou_threshold}"
        super().__init__(name)

    def compute(self, pred: list[list[tuple[str, Scene]]],
                true: list[tuple[str, float, float]]) -> float:
        def _unpack_true(t):
            if len(t) == 2:
                vid, moment = t
                return vid, float(moment) - 0.5, float(moment) + 0.5
            elif len(t) >= 3:
                vid, start, end = t[0], t[1], t[2]
                gt_start = float(start) if start is not None else 0.0
                if end is not None:
                    gt_end = float(end)
                else:
                    gt_end = gt_start
                return vid, gt_start, gt_end
            else:
                raise ValueError("Unsupported true format")

        assert len(pred) == len(true), "The predictions and the ground truths must have the same length"

        n = len(pred)
        hits = 0

        for i in range(n):
            candidates = pred[i]
            gt_video, gt_start, gt_end = _unpack_true(true[i])

            cumul_iou = 0.0
            for video_name, prediction in candidates:
                if video_name != gt_video:
                    continue
                cumul_iou += iou(prediction, gt_start, gt_end)
            if cumul_iou >= self.iou_threshold:
                hits += 1

        return hits / n if n > 0 else 0.0



class RecallAtKIoU(Metric):
    """
    Recall@K con requisito IoU >= τ (metrica standard nei paper di moment retrieval).

    Per ogni query:
      - guardo solo le top-K predizioni
      - cerco una predizione con video corretto e IoU >= soglia
      - se esiste, la query è un "hit"

    Risultato:
        (# query con almeno una predizione nelle top-K
         che ha video corretto e IoU >= τ) / (# query totali)
    """
    def __init__(self, k: int = 1, iou_threshold: float = 0.5, name: str | None = None):
        self.k = k
        self.iou_threshold = iou_threshold
        if name is None:
            name = f"Recall@{k}_IoU@{iou_threshold}"
        super().__init__(name)

    def compute(self, pred: list[list[tuple[str, Scene]]],
                true: list[tuple[str, float, float]]) -> float:
        def _unpack_true(t):
            if len(t) == 2:
                vid, moment = t
                return vid, float(moment) - 0.5, float(moment) + 0.5
            elif len(t) >= 3:
                vid, start, end = t[0], t[1], t[2]
                gt_start = float(start) if start is not None else 0.0
                if end is not None:
                    gt_end = float(end)
                else:
                    gt_end = gt_start
                return vid, gt_start, gt_end
            else:
                raise ValueError("Unsupported true format")

        assert len(pred) == len(true), "The predictions and the ground truths must have the same length"

        n = len(pred)
        hits = 0

        for i in range(n):
            top_k = pred[i][:self.k]
            gt_video, gt_start, gt_end = _unpack_true(true[i])

            hit = False
            for video_name, prediction in top_k:
                if video_name != gt_video:
                    continue
                iou = iou(prediction, gt_start, gt_end)
                if iou >= self.iou_threshold:
                    hit = True
                    break

            if hit:
                hits += 1

        return hits / n if n > 0 else 0.0