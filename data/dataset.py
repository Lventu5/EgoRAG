from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import os
import json
import glob

from .video_dataset import VideoDataset
from .query import QueryDataset, Query

class BaseDataset(Dataset):
    def __init__(self, video_path: str, annotation_path: str):
        self.video_path = video_path
        self.annotation_path = annotation_path

    def __len__(self):
        raise NotImplementedError("Subclasses should implement this method.")  
    
    def __getitem__(self, vid):
        raise NotImplementedError("Subclasses should implement this method.")

    def load_videos(self, is_pickle: bool) -> VideoDataset:
        raise NotImplementedError("Subclasses should implement this method.")

    def load_annotations(self, video_ids: List[str]) -> List[dict]:
        raise NotImplementedError("Subclasses should implement this method.")
    
    def load_queries(self, video_ids: List[str]) -> QueryDataset:
        raise NotImplementedError("Subclasses should implement this method.")


class Ego4DDataset(BaseDataset):
    def __init__(self, video_path: str, annotation_path: str):
        super().__init__(video_path, annotation_path)
        self.video_dataset = None
        self._load_data()

    def __len__(self):
        return len(self.load_videos(False))

    def __getitem__(self, vid) -> Tuple[List[dict], QueryDataset]:
        annotations = self.load_annotations(vid)
        return {vid : annotations}
    
    def _load_data(self):
        with open(self.annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.video_data = data["videos"]

    def get_videos(self) -> VideoDataset:
        if self.video_dataset is None:
            self.video_dataset = self.load_videos(is_pickle=False)
        return self.video_dataset

    ## FIXME: implement for various video formats
    def load_videos(self, is_pickle: bool) -> VideoDataset:
        if self.video_dataset is not None:
            return self.video_dataset
        if is_pickle:
            self.video_dataset = VideoDataset.load_from_pickle(self.video_path)
        else:
            video_ids = glob.glob(os.path.join(self.video_path, "*.mp4"))
            self.video_dataset = VideoDataset(video_ids)
        return self.video_dataset

    def load_annotations(self, video_ids: List[str]) -> Dict[str, List[dict]]:
        annotations = {}
        for v in self.video_data:
            if v["video_uid"] not in video_ids:
                continue
            
            annotations[v["video_uid"]] = [
                ann 
                for clip in v.get("clips", []) 
                for ann in clip.get("annotations", [])
            ]

        return annotations

    def load_queries(self, video_ids: List[str]) -> QueryDataset:
        annotations = self.load_annotations(video_ids)
        queries = [
            Query(
                qid=f"{vid}_{j}_{i}",
                query_text=lq.get("query", ""),
                video_uid=vid,
                gt={
                    "start_sec": float(lq.get("video_start_sec", -1.0)),
                    "end_sec": float(lq.get("video_end_sec", -1.0)),
                    "start_frame": int(lq.get("video_start_frame", -1)),
                    "end_frame": int(lq.get("video_end_frame", -1)),
                }
            )
            for vid, ann in annotations.items()
            for j, a in enumerate(ann)
            for i, lq in enumerate(a.get("language_queries", []))
        ]
        return QueryDataset(queries)
    
class DatasetFactory:
    @staticmethod
    def get_dataset(dataset_type: str, video_path: str, annotation_path: str) -> BaseDataset:
        if dataset_type.lower() == "ego4d":
            return Ego4DDataset(video_path, annotation_path)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

def main():
    dataset = Ego4DDataset(
        video_path="../../ego4d_data/v2/full_scale/",
        annotation_path="../../ego4d_data/v2/annotations/nlq_train.json"
    )
    videos = dataset.load_videos(is_pickle=False)
    for video in videos.video_files: 
        print(video)
    queries = dataset.load_queries(videos.get_uids())
    print(len(queries))
    for q in queries.queries[:50]:
        print(q)
    # Check that no duplicate qids exist
    qid_set = set()
    for q in queries.queries:
        if q.qid in qid_set:
            print(f"\n\nDuplicate qid found: {q.qid}\n\n")
        qid_set.add(q.qid)

    # Test pickle loading
    dataset.video_path="../../data/video_dataset.pkl"
    videos_pickle = dataset.load_videos(is_pickle=True)
    for video in videos_pickle.video_files: 
        print(video)

if __name__ == "__main__":
    main()