import numpy as np
from sklearn.cluster import KMeans

def choose_k(n_frames: int, max_temporal_segments: int) -> int:
        k = (n_frames // 8) + 1
        return min(k, max_temporal_segments)

def cluster_frames(frame_embs: np.ndarray, max_temporal_segments: int) -> dict[int, list[int]]:
        n = len(frame_embs)
        if n < 2:
            return {0: list(range(n))}

        k = choose_k(n, max_temporal_segments)
        if k <= 1:
            return {0: list(range(n))}

        km = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(frame_embs)
        return km