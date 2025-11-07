type TopKVideosPerModality = list[list[tuple[str, float]]]
type TopKVideosPerQuery = dict[str, dict[str, TopKVideosPerModality]]
type TopKScenes = list[tuple[str, float]]
type DetailedResults = dict[str, tuple[str, float, TopKScenes]]

class RetrievalResults:
    def __init__(self, results = None):
        if results is None:
            self.results = {}
        else:
            self.results = results

    def __getitem__(self, query_qid: str):
        return self.results[query_qid]

    def get(self, query_qid: str, default=None):
        """Dict-like get method for compatibility with callers expecting a mapping.

        Returns the stored entry for `query_qid` or `default` if not present or
        if the underlying results is not a mapping.
        """
        if isinstance(self.results, dict):
            return self.results.get(query_qid, default)
        return default

    def add_top_level(self, top_level_results: TopKVideosPerQuery):
        self.results = top_level_results

    def add_detailed_results(self, results: DetailedResults):
        self.results = results