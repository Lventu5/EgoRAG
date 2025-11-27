import pickle
import os
import torch
from typing import Optional, Dict, Union

class Query:
    def __init__(
            self, qid: str | int, query_text: str, 
            video_uid: Optional[str] = None, 
            decomposed: Optional[dict] = None, 
            embeds: Optional[dict] = None, 
            tags: Optional[list[str]] = None,
            gt: Optional[Dict[str, Optional[float | int]]] = None
    ):
        self.qid = qid if isinstance(qid, str) else f"query_{qid}"
        self.query_text = query_text
        self.video_uid = video_uid
        self.decomposed = decomposed if decomposed is not None else {"text": query_text, "audio": query_text, "video": query_text}
        self.embeddings = embeds if embeds is not None else {"text": None, "audio": None, "video": None, "caption": None}
        self.gt = {
            "start_sec": None,
            "end_sec": None,
            "start_frame": None,
            "end_frame": None,
        }
        if gt:
            self.gt.update(gt)
        self.tags = tags if tags is not None else None
            
    def __repr__(self):
        return f"Query(qid={self.qid}, query_text={self.query_text.strip()})"

    def to_dict(self) -> Dict:
        return {
            "qid": self.qid,
            "query_text": self.query_text,
            "video_uid": self.video_uid,
            "decomposed": self.decomposed,
            "embeddings": self.embeddings,
            "gt": self.gt,
        }
    
    def get_query(self, modality: str | None = None) -> str:
        if modality is None:
            return self.query_text
        return self.decomposed.get(modality, self.query_text)

    def get_embedding(self, modality: str = "video") -> Optional[torch.Tensor]:
        return self.embeddings.get(modality, None)
    

    @staticmethod
    def from_dict(data: Dict) -> "Query":
        return Query(
            qid=data["qid"],
            query_text=data["query_text"],
            video_uid=data.get("video_uid"),
            decomposed=data.get("decomposed"),
            embeds=data.get("embeddings"),
            gt=data.get("gt"),
        )

    # def save_to_pickle(self, file_path: str):
    #     os.makedirs(os.path.dirname(file_path), exist_ok=True)
    #     with open(file_path, 'wb') as f:
    #         pickle.dump(self, f)

    # @staticmethod
    # def load_from_pickle(file_path: str) -> "Query":
    #     with open(file_path, 'rb') as f:
    #         return pickle.load(f)

    #     self.embeds = embeds

class QueryDataset:
    def __init__(self, queries: Optional[list[str] | list[Query]] = None):
        # self.queries = [Query(qid=i, query_text=q, ) for i, q in enumerate(queries)] if queries is not None else []
        if queries and isinstance(queries[0], Query):
            self.queries = queries  # type: ignore
        else:
            self.queries = [
                Query(
                    qid=i,
                    query_text=q["query_text"] if isinstance(q, dict) else q,
                    video_uid=q.get("video_uid") if isinstance(q, dict) else None,
                    gt=q.get("gt") if isinstance(q, dict) else None,
                )
                for i, q in enumerate(queries)
            ] if queries is not None else []

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx: int) -> Query:
        return self.queries[idx]

    def __iter__(self):
        return iter(self.queries)

    def group_by_modality(self, modality: Optional[str] = None) -> Union[Dict[str, list[str]], list[str]]:
        if modality is not None:
            return [query.get_query(modality) for query in self.queries]
        grouped = {}
        for query in self.queries:
            for mod in query.decomposed.keys():
                if modality is None or modality == mod:
                    if mod not in grouped:
                        grouped[mod] = []
                    grouped[mod].append(query.get_query(mod))
        return grouped
    
    def embeddings_by_modality(self, modality: str) -> torch.Tensor:
        return torch.stack(
            [query.get_embedding(modality) 
             for query in self.queries 
             if query.get_embedding(modality) is not None]
        )

def main():
    queries = [
        "When is the first goal from Bologna scored?", # 00:30
        "When is the player from Pisa receiving a red card?", # 00:52
        "Who jumps over the ad board to celebrate?", # 01:14
        "When do the second half highlights start?", # 01:38
        "Who is the highlights sposnsor?", # 00:01
        "When does the ball bounce on the net?", # 00:41
        "When does the player without the cap wins the first set?", # 01:24
        "How many people with a cap on the stands are celebrating the winning point?", # 01:32
        "Who wins the tennis match?", # 04:33
        "Does the commentary rhythm suggest an important moment?",
        "Are there changes in pace or intensity in the commentary?",
        "Does it look like a break or preparation moment in the game?",
        "Does the video show strong emotions from the players or crowd?",
    ]
    query_dataset = QueryDataset(queries)
    for query in query_dataset:
        print(query)
    grouped = query_dataset.group_by_modality()
    print(grouped)

if __name__ == "__main__":
    main()