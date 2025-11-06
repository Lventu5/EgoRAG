from .wrapper import Wrapper, create_wrapper

class Generator:
    def __init__(self, model_name: str):
        self.wrapper = create_wrapper(model_name)(model_name)

    def _prepare_input(self, retrieved_results: dict, queries: dict) -> tuple[str, str]:
        pass  # Implement input preparation logic here

    def generate(self, retrieved_results: dict, queries: dict) -> str:
        video_path, prompt = self._prepare_input(retrieved_results, queries)
        return self.wrapper.generate(video_path, prompt)
    
