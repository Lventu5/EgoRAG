from abc import ABC, abstractmethod

class BaseFuser(ABC):
    """Interfaccia base per tutte le tecniche di fusion ranking."""
    
    @abstractmethod
    def fuse(self, rankings: dict[str, list[tuple[str, float]]]):
        """rankings: { modality: [(item_id, score), ...] }"""
        pass


class FuserRRF(BaseFuser):
    def __init__(self, k=60):
        self.k = k

    def fuse(self, rankings: dict[str, list[tuple[str, float]]]):
        scores = {}

        for _, ranking in rankings.items():
            for rank, (item_id, _) in enumerate(ranking):
                scores[item_id] = scores.get(item_id, 0) + 1 / (self.k + rank + 1)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class Fuser:
    """Wrapper generico: permette di usare diverse tecniche di fusione."""
    
    def __init__(self, method: str = "rrf", **kwargs):
        self.method_name = method.lower()

        if self.method_name == "rrf":
            self.method = FuserRRF(**kwargs)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

    def fuse(self, rankings: dict[str, list[tuple[str, float]]]):
        return self.method.fuse(rankings)
