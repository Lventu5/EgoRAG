from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json, re, logging
from typing import Dict, Any, Optional, Union, List
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

    def __call__(self, query: str, modality: str = "default") -> Union[str, Dict[str, str], List[Dict[str, str]]]:
        """
        Rewrites the query according to the chosen modality
        """
        if modality == "default":
            return self.rewrite(query)
        elif modality == "decompose":
            return self.decompose_to_json(query)
        elif modality == "sequence":
            return self.decompose_to_sequence(query)
        else:
            raise ValueError(f"Unknown modality {modality}. Use 'default', 'decompose', or 'sequence'.")
        
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
            "1) Query: \"Who was I talking to yesterday in the office?\"\n"
            "JSON: {\"text\": \"Someone talking to another person in an office\", \"audio\": \"office chatter, office noises\", \"video\": \"Two people having a conversation in an office\"}\n"
            "2) Query: \"What is happening when a plane flies over a city while sirens are sounding?\"\n"
            "JSON: {\"text\": \"A plane flying over a city while sirens are sounding\", \"audio\": \"sirens blaring, airplane noise\", \"video\": \"A plane flying over an urban skyline\"}\n\n"
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
        print(f"{'-'*50}\n original query {query} \n\n")
        for modality in ["text", "audio", "video"]:
            if modality in ["text", "video"]:
                ans = " ".join(data["text"].split()).strip()
                if ans == "" or len(ans.split()) <= 2:
                    data[modality] = query
                else:
                    data[modality]  = query # ans
            else:
                ans = ",".join(
                    [t.strip().lower() for t in data["audio"].split(",") if t.strip()]
                )
                if ans == "" or len(ans.split()) <= 2:
                    data[modality] = query
                else:
                    data[modality] = query # ans
        return data

    def build_sequence_prompt(self, query: str) -> str:
        """
        Prompt to decompose the query into a sequential execution plan for Chain of Retrieval.
        Returns an ordered list of sub-goals with temporal relations.
        """
        return (
            "Decompose this query into a SEQUENTIAL EXECUTION PLAN for video retrieval. "
            "Return ONLY a valid JSON array of objects, where each object represents a sub-goal.\n"
            "Each object must have EXACTLY these keys:\n"
            '  "query_text": string (the search query for this sub-goal),\n'
            '  "type": string (either "anchor" for reference events or "target" for the main goal),\n'
            '  "temporal_relation": string (one of: "none", "before", "after", "during", "near")\n\n'
            "Rules:\n"
            "- The first sub-goal should typically be the anchor (a known/easy-to-find event)\n"
            "- The target should have a temporal_relation relative to the anchor\n"
            "- Use 'none' for temporal_relation when there's no temporal dependency\n"
            "- Keep queries concise and searchable\n\n"
            "Examples:\n"
            '1) Query: "What did I do after I finished cooking dinner?"\n'
            'JSON: [{"query_text": "cooking dinner, finishing meal preparation", "type": "anchor", "temporal_relation": "none"}, '
            '{"query_text": "activity after cooking", "type": "target", "temporal_relation": "after"}]\n\n'
            '2) Query: "What was on the table before I cleaned it?"\n'
            'JSON: [{"query_text": "cleaning a table", "type": "anchor", "temporal_relation": "none"}, '
            '{"query_text": "items on table", "type": "target", "temporal_relation": "before"}]\n\n'
            '3) Query: "Who called me while I was working on the computer?"\n'
            'JSON: [{"query_text": "working on computer, using laptop", "type": "anchor", "temporal_relation": "none"}, '
            '{"query_text": "phone call, someone calling", "type": "target", "temporal_relation": "during"}]\n\n'
            f'Query: "{query}"\n'
            "Generated JSON:"
        )

    def decompose_to_sequence(self, query: str) -> List[Dict[str, str]]:
        """
        Decompose the query into a sequential execution plan for Chain of Retrieval.
        Returns a list of sub-goals, each with:
        - query_text: the search query for this sub-goal
        - type: "anchor" or "target"
        - temporal_relation: "none", "before", "after", "during", "near"
        """
        prompt = self.build_sequence_prompt(query)
        raw = self.generate(prompt)
        raw = self._strip_prompt_echo(raw, after="Generated JSON:")
        
        # Try to parse the JSON array
        try:
            # Clean up the raw output
            raw = raw.strip()
            # Find JSON array boundaries
            start_idx = raw.find('[')
            end_idx = raw.rfind(']')
            if start_idx != -1 and end_idx != -1:
                raw = raw[start_idx:end_idx + 1]
            
            data = json.loads(raw)
            
            # Validate the structure
            if not isinstance(data, list):
                raise ValueError("Expected a JSON array")
            
            valid_types = {"anchor", "target"}
            valid_relations = {"none", "before", "after", "during", "near"}
            
            validated_data = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                validated_item = {
                    "query_text": str(item.get("query_text", query)),
                    "type": item.get("type", "target") if item.get("type") in valid_types else "target",
                    "temporal_relation": item.get("temporal_relation", "none") if item.get("temporal_relation") in valid_relations else "none"
                }
                validated_data.append(validated_item)
            
            if not validated_data:
                # Fallback: single target query with no temporal relation
                validated_data = [{"query_text": query, "type": "target", "temporal_relation": "none"}]
            
            return validated_data
            
        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Failed to parse sequence JSON: {e}. Raw output: {raw}")
            # Fallback: return single target query
            return [{"query_text": query, "type": "target", "temporal_relation": "none"}]