# utils/json_sanitizer.py
import json
import re
import logging
from typing import Dict, Any, Optional

class JSONParser:
    """
    Robust utilities to extract and parse JSON emitted by LLMs.
    Handles:
      - code fences ```json ... ```
      - extra prose before/after the JSON
      - unbalanced/bracketed text (finds the first balanced {...})
      - smart quotes, trailing commas, inline comments
    """
    _CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    @classmethod
    def extract_json_block(cls, text: str) -> str:
        """
        Return the first JSON-looking block from `text`.
        Steps:
          1) strip code fences if present
          2) find the first balanced {...} block
        If nothing found, returns stripped `text`.
        """
        text = cls._strip_code_fences(text)
        return cls._find_balanced_json(text)

    @classmethod
    def parse_json(cls, text_or_json: str) -> Dict[str, Any]:
        """
        Parse a JSON string (or raw LLM output containing JSON).
        Tries json.loads; on failure sanitizes and retries.
        Returns {} on failure.
        """
        candidate = cls.extract_json_block(text_or_json)
        try:
            return json.loads(candidate)
        except Exception:
            candidate = cls._sanitize_json_string(candidate)
            try:
                return json.loads(candidate)
            except Exception:
                logging.warning("[LLM] Unable to parse JSON; returning empty dict.\nRAW (first 500 chars):\n%s",
                                text_or_json[:500])
                return {}

    @classmethod
    def parse_with_defaults(cls, text_or_json: str, default_keys: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Parse JSON and ensure specific keys exist with string values.
        Useful to enforce a fixed schema (e.g., text_query/audio_query/video_query).
        """
        data = cls.parse_json(text_or_json)
        default_keys = default_keys or {}
        out: Dict[str, str] = {}
        for k, v in default_keys.items():
            out[k] = str(data.get(k, v) or "").strip()
        for k, v in data.items():
            if k not in out and isinstance(v, str):
                out[k] = v.strip()
        return out

    @classmethod
    def _strip_code_fences(cls, text: str) -> str:
        """If ```...``` fences exist, return their inner content; else return original text."""
        m = cls._CODE_FENCE_RE.search(text)
        return m.group(1) if m else text

    @staticmethod
    def _find_balanced_json(text: str) -> str:
        """
        Find the FIRST balanced {...} block in `text`.
        If none is found, return stripped text.
        """
        start = text.find("{")
        if start == -1:
            return text.strip()
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1].strip()
        return text.strip()

    @staticmethod
    def _sanitize_json_string(s: str) -> str:
        """
        Light sanitization:
          - smart quotes -> normal
          - remove inline comments (// and #)
          - remove trailing commas before } or ]
          - strip whitespace
        """
        s = s.replace("“", '"').replace("”", '"').replace("’", "'")
        s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
        s = re.sub(r"#.*?$", "", s, flags=re.MULTILINE)
        s = re.sub(r",\s*([}\]])", r"\1", s)
        return s.strip()