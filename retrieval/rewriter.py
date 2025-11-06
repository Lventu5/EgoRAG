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
        
    # @staticmethod
    def _strip_prompt_echo(self, text: str, after: Optional[str] = None) -> str:
        """
        Removes the prompt echo from the model’s output. If `after` is provided,
        keeps only the substring after its first occurrence.
        """
        if after and after in text:
            text = text.split(after, 1)[-1]
        return text.strip()

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
        # logging.info(f"Rewriting completed. Result:\n {out}")
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
            '  "text": string,\n'
            '  "audio": string (comma-separated words/sounds/onomatopoeias),\n'
            '  "video": string (short visual description)\n'
            "Do not add explanations, notes, code fences or extra text. \n\n "
            "Here are a few examples:\n\n"
            "1) Query: \"A dog barks as a car explodes\"\n"
            "JSON: {\"text\": \"A dog barks as a car explodes\", \"audio\": \"barking, explosion\", \"video\": \"A dog barking and an exploding car\"}\n"
            "2) Query: \"A plane flies over a city while sirens blare\"\n"
            "JSON: {\"text\": \"A plane flies over a city while sirens blare\", \"audio\": \"sirens blaring, plane noise\", \"video\": \"A plane flying over a city\"}\n\n"
            f'Query: "{query}"\n'
            "Generated JSON:"
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
        {text, audio, video}.
        """
        prompt = self.build_decompose_prompt(query)
        raw = self.generate(prompt)

        raw = self._strip_prompt_echo(raw, after="Generated JSON:")

        data = JSONParser.parse_with_defaults(
            raw,
            default_keys={"text": "", "audio": "", "video": ""}
        )

        data["text"]  = " ".join(data["text"].split()).strip()
        data["audio"] = ",".join(
            [t.strip().lower() for t in data["audio"].split(",") if t.strip()]
        )
        data["video"] = " ".join(data["video"].split()).strip()

        return data
    
    def decompose(self, query_text: str) -> Dict[str, str]:
        """
        Decompose a query into multimodal components.
        
        Alias for decompose_to_json for consistency with the new API.
        
        Args:
            query_text: Query text to decompose
            
        Returns:
            Dictionary with keys: text, audio, video
        """
        return self.decompose_to_json(query_text)
    
    def subquestions(self, query_text: str, num_hops: int = 2) -> list[str]:
        """
        Generate follow-up sub-questions for multi-hop retrieval.
        
        Args:
            query_text: Original query text
            num_hops: Number of sub-questions to generate
            
        Returns:
            List of sub-question strings
        """
        prompt = self._build_subquestions_prompt(query_text, num_hops)
        raw = self.generate(prompt)
        
        # Parse sub-questions from output
        raw = self._strip_prompt_echo(raw, after="Sub-questions:")
        
        # Extract numbered questions
        lines = raw.strip().split("\n")
        subqs = []
        for line in lines:
            # Match patterns like "1. ", "1) ", or just numbered items
            line = line.strip()
            if not line:
                continue
            # Remove leading numbers/bullets
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
            if cleaned and len(cleaned) > 5:  # Minimum length check
                subqs.append(cleaned)
        
        # Return up to num_hops questions
        return subqs[:num_hops]
    
    def _build_subquestions_prompt(self, query: str, num_hops: int) -> str:
        """
        Build prompt for generating sub-questions.
        
        Args:
            query: Original query
            num_hops: Number of sub-questions to generate
            
        Returns:
            Formatted prompt string
        """
        return (
            f"Given a query about video content, generate {num_hops} related follow-up questions "
            "that would help gather more specific information to answer the original query. "
            "Each sub-question should explore a different aspect or detail.\n\n"
            f"Original query: {query}\n\n"
            f"Generate exactly {num_hops} numbered sub-questions:\n"
            "Sub-questions:"
        )