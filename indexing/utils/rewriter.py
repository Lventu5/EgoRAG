from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json, re, logging
from typing import Dict, Any, Optional, Union
from indexing.utils.json_parser import JSONParser

class QueryRewriterLLM:
    """
    Rewrites the query before using it for retrieval
    """
    def __init__(self, model_name: str = "google/gemma-2-9b", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._load_model()
        logging.info(f"Initialized {self.__class__.__name__} on {self.device}")

    def _load_model(self):
        """
        Loads the models and initializes the rewriting pipeline
        """
        try:
            logging.info(f"[LLM] Loading {self.model_name} locally...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self.language_model = AutoModelForCausalLM.from_pretrained(self.model_name, local_files_only=True, dtype="auto")
        except Exception as e:
            logging.warning(f"[LLM] Local weights not found, fetching online... ({e})")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.language_model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype="auto")

        if self.device == "cuda":
            self.language_model = self.language_model.to("cuda")
            pipe_device = 0
        else:
            pipe_device = -1

        self.pipeline = pipeline("text-generation", model=self.language_model, tokenizer=self.tokenizer, device=pipe_device)
        logging.info("[LLM] Model loaded successfully.")

    def __call__(self, query: str, modality: str = "default") -> Union[str, Dict[str, str]]:
        """
        Rewrites the query according to the chosen modality
        """
        if modality == "default":
            return self.rewrite(query)
        elif modality == "decompose":
            return self.decompose_to_json(query)
        else:
            raise ValueError("Unknown modality. Use 'default' or 'decompose'.")

    def generate(self, prompt: str) -> str:
        """
        Generate a text completion for a given prompt.
        """
        out = self.pipeline(prompt,
            max_new_tokens=64,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            return_full_text=True)[0]["generated_text"]
        logging.info(f"Rewriting completed. Result:\n {out}")
        return out

    def build_prompt(self, query: str) -> str:
        """
        Prompt to rewrite the query.
        """
        return (
            "Rewrite the following query in a concise and natural way, keeping its meaning, key words and key information. "
            "Do not explain or justify. Output only the rewritten query.\n\n"
            f"Query: {query}\n"
            "Rewritten query:"
        )

    def build_decompose_prompt(self, query: str) -> str:
        """
        Prompt to decompose the query into its multimodal components (text, audio, video).
        Requests a minimal JSON output.
        """
        return (
            "Decompose this query for a MULTIMODAL video search. "
            "Return ONLY a valid JSON object with EXACTLY these keys:\n"
            '  "text_query": string,\n'
            '  "audio_query": string (comma-separated words/onomatopoeias),\n'
            '  "video_query": string (short visual description)\n'
            "Do not add explanations, notes, code fences or extra text. \n\n "
            f'Query: "{query}"\n'
            "JSON:"
        )
    
    def rewrite(self, query: str) -> str:
        """
        Rewrites the query in a concise format by focusing on keywords
        """
        prompt = self.build_prompt(query)
        raw = self.generate(prompt)
        return self._strip_prompt_echo(raw, after="Rewritten query:")

    def decompose_to_json(self, query: str) -> Dict[str, str]:
        """
        Decompose the query into multimodal components. Returns a dict:
        {text_query, audio_query, video_query}.
        """
        prompt = self.build_decompose_prompt(query)
        raw = self.generate(prompt)

        data = JSONParser.parse_with_defaults(
            raw,
            default_keys={"text_query": "", "audio_query": "", "video_query": ""}
        )

        data["text_query"]  = " ".join(data["text_query"].split()).strip()
        data["audio_query"] = ",".join(
            [t.strip().lower() for t in data["audio_query"].split(",") if t.strip()]
        )
        data["video_query"] = " ".join(data["video_query"].split()).strip()

        return data