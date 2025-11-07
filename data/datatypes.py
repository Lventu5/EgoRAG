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

    def add_top_level(self, top_level_results: TopKVideosPerQuery):
        self.results = top_level_results

    def add_detailed_results(self, results: DetailedResults):
        self.results = results