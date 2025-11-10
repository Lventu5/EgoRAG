from abc import ABC, abstractmethod
import logging

class BaseFuser(ABC):
    """Interfaccia base per tutte le tecniche di fusion ranking."""
    
    @abstractmethod
    def fuse(self, rankings: dict[str, list[tuple[str, float]]]):
        """
        Fuses rankings from multiple modalities into a single ranking.
        
        Args:
            rankings: { modality: [(item_id, score), ...] }
                     Each modality provides a ranked list of items with scores.
                     Some items may appear in some modalities but not others.
        
        Returns:
            list[tuple[str, float]]: Fused ranking as [(item_id, score), ...]
        """
        pass


class FuserRRF(BaseFuser):
    """
    Reciprocal Rank Fusion (RRF).
    
    Original behavior: Only considers items that appear in at least one modality.
    Items missing from a modality are treated as not ranked (effectively low score).
    """
    def __init__(self, k=60):
        self.k = k

    def fuse(self, rankings: dict[str, list[tuple[str, float]]]):
        scores = {}

        for _, ranking in rankings.items():
            for rank, (item_id, _) in enumerate(ranking):
                scores[item_id] = scores.get(item_id, 0) + 1 / (self.k + rank + 1)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class FuserMeanImputation(BaseFuser):
    """
    Mean Imputation Fusion.
    
    For each item, compute the average score across modalities where it appears.
    Missing modality scores are imputed with the mean of available modality scores
    for that item.
    
    This ensures items are not penalized for missing modalities - instead, 
    missing scores are replaced with the item's average performance on other modalities.
    
    Example:
        Item A appears in 3/4 modalities with scores [0.8, 0.7, 0.9]
        → Missing modality imputed as mean(0.8, 0.7, 0.9) = 0.8
        → Final score: (0.8 + 0.7 + 0.9 + 0.8) / 4 = 0.8
    """
    
    def __init__(self, normalize_scores: bool = True):
        """
        Args:
            normalize_scores: If True, normalize scores to [0, 1] before fusion
        """
        self.normalize_scores = normalize_scores
    
    def fuse(self, rankings: dict[str, list[tuple[str, float]]]):
        # Collect all unique items across all modalities
        all_items = set()
        for ranking in rankings.values():
            for item_id, _ in ranking:
                all_items.add(item_id)
        
        # Build a score matrix: {item_id: {modality: score}}
        item_scores = {item: {} for item in all_items}
        
        for modality, ranking in rankings.items():
            for item_id, score in ranking:
                item_scores[item_id][modality] = score
        
        # Optional: Normalize scores per modality to [0, 1]
        if self.normalize_scores:
            item_scores = self._normalize_scores(item_scores, rankings.keys())
        
        # Compute final scores with mean imputation
        final_scores = {}
        for item_id, modality_scores in item_scores.items():
            available_scores = list(modality_scores.values())
            
            if not available_scores:
                final_scores[item_id] = 0.0
                continue
            
            # Compute mean of available modalities
            mean_score = sum(available_scores) / len(available_scores)
            
            # Impute missing modalities with the mean
            num_modalities = len(rankings)
            num_missing = num_modalities - len(available_scores)
            
            # Final score: (sum of available + mean * num_missing) / total_modalities
            total_score = sum(available_scores) + mean_score * num_missing
            final_scores[item_id] = total_score / num_modalities
        
        return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    
    def _normalize_scores(self, item_scores: dict, modalities):
        """Normalize scores per modality to [0, 1] range."""
        # Find min/max per modality
        modality_ranges = {mod: {'min': float('inf'), 'max': float('-inf')} 
                          for mod in modalities}
        
        for item_id, modality_scores in item_scores.items():
            for modality, score in modality_scores.items():
                modality_ranges[modality]['min'] = min(modality_ranges[modality]['min'], score)
                modality_ranges[modality]['max'] = max(modality_ranges[modality]['max'], score)
        
        # Normalize
        normalized_scores = {item: {} for item in item_scores.keys()}
        for item_id, modality_scores in item_scores.items():
            for modality, score in modality_scores.items():
                min_val = modality_ranges[modality]['min']
                max_val = modality_ranges[modality]['max']
                
                if max_val > min_val:
                    normalized = (score - min_val) / (max_val - min_val)
                else:
                    normalized = 1.0  # All scores are the same
                
                normalized_scores[item_id][modality] = normalized
        
        return normalized_scores


class FuserExcludeMissing(BaseFuser):
    """
    Exclude Missing Modality Fusion.
    
    Treats items missing from a modality as if they are NOT in the top-k for that modality.
    Only fuses scores from modalities where the item actually appears.
    
    This is the fairest approach when some videos lack certain modalities (e.g., no audio):
    - Videos are only scored on modalities they have
    - No penalty for missing modalities
    - No imputation needed
    
    Example:
        Video A (has audio): appears in all 4 modalities
        → Final score: average of 4 modality scores
        
        Video B (no audio): appears in 3 modalities (text, video, caption)
        → Final score: average of 3 modality scores
        → NOT penalized for missing audio
    """
    
    def __init__(self, normalize_scores: bool = True, aggregation: str = "mean"):
        """
        Args:
            normalize_scores: If True, normalize scores to [0, 1] before fusion
            aggregation: How to aggregate available modality scores
                        - "mean": Average of available modalities (default)
                        - "sum": Sum of available modalities
                        - "max": Maximum score across available modalities
        """
        self.normalize_scores = normalize_scores
        self.aggregation = aggregation.lower()
        
        if self.aggregation not in ["mean", "sum", "max"]:
            raise ValueError(f"Unknown aggregation: {aggregation}. Use 'mean', 'sum', or 'max'")
    
    def fuse(self, rankings: dict[str, list[tuple[str, float]]]):
        # Collect all unique items
        all_items = set()
        for ranking in rankings.values():
            for item_id, _ in ranking:
                all_items.add(item_id)
        
        # Build score matrix: {item_id: {modality: score}}
        item_scores = {item: {} for item in all_items}
        
        for modality, ranking in rankings.items():
            for item_id, score in ranking:
                item_scores[item_id][modality] = score
        
        # Optional: Normalize scores per modality
        if self.normalize_scores:
            item_scores = self._normalize_scores(item_scores, rankings.keys())
        
        # Compute final scores using only available modalities
        final_scores = {}
        for item_id, modality_scores in item_scores.items():
            available_scores = list(modality_scores.values())
            
            if not available_scores:
                final_scores[item_id] = 0.0
                continue
            
            # Aggregate available scores based on strategy
            if self.aggregation == "mean":
                final_scores[item_id] = sum(available_scores) / len(available_scores)
            elif self.aggregation == "sum":
                final_scores[item_id] = sum(available_scores)
            elif self.aggregation == "max":
                final_scores[item_id] = max(available_scores)
        
        return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    
    def _normalize_scores(self, item_scores: dict, modalities):
        """Normalize scores per modality to [0, 1] range."""
        # Find min/max per modality
        modality_ranges = {mod: {'min': float('inf'), 'max': float('-inf')} 
                          for mod in modalities}
        
        for item_id, modality_scores in item_scores.items():
            for modality, score in modality_scores.items():
                modality_ranges[modality]['min'] = min(modality_ranges[modality]['min'], score)
                modality_ranges[modality]['max'] = max(modality_ranges[modality]['max'], score)
        
        # Normalize
        normalized_scores = {item: {} for item in item_scores.keys()}
        for item_id, modality_scores in item_scores.items():
            for modality, score in modality_scores.items():
                min_val = modality_ranges[modality]['min']
                max_val = modality_ranges[modality]['max']
                
                if max_val > min_val:
                    normalized = (score - min_val) / (max_val - min_val)
                else:
                    normalized = 1.0  # All scores are the same
                
                normalized_scores[item_id][modality] = normalized
        
        return normalized_scores


class Fuser:
    """
    Wrapper generico: permette di usare diverse tecniche di fusione.
    
    Available methods:
    - "rrf": Reciprocal Rank Fusion (default)
    - "mean_imputation": Impute missing modality scores with mean of available ones
    - "exclude_missing": Only use available modalities (fairest for videos without audio)
    """
    
    def __init__(self, method: str = "rrf", **kwargs):
        self.method_name = method.lower()

        if self.method_name == "rrf":
            self.method = FuserRRF(**kwargs)
        elif self.method_name == "mean_imputation":
            self.method = FuserMeanImputation(**kwargs)
        elif self.method_name == "exclude_missing":
            self.method = FuserExcludeMissing(**kwargs)
        else:
            raise ValueError(f"Unknown fusion method: {method}. "
                           f"Use 'rrf', 'mean_imputation', or 'exclude_missing'")

    def fuse(self, rankings: dict[str, list[tuple[str, float]]]):
        return self.method.fuse(rankings)
