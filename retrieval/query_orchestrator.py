from __future__ import annotations

import logging
import re
from copy import deepcopy
from typing import Any

from indexing.utils.json_parser import JSONParser

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except Exception:  # pragma: no cover - transformers is already a project dependency
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None


TIME_OF_DAY_RANGES: dict[str, list[list[float]]] = {
    "early_morning": [[5 * 3600, 9 * 3600]],
    "morning": [[5 * 3600, 12 * 3600]],
    "noon": [[11 * 3600, 14 * 3600]],
    "afternoon": [[12 * 3600, 18 * 3600]],
    "evening": [[18 * 3600, 22 * 3600]],
    "night": [[0.0, 5 * 3600], [21 * 3600, 24 * 3600]],
    "late_night": [[0.0, 3 * 3600], [22 * 3600, 24 * 3600]],
    "dawn": [[5 * 3600, 7 * 3600]],
    "sunrise": [[5 * 3600, 7 * 3600]],
    "breakfast": [[6 * 3600, 10 * 3600]],
    "lunch": [[11 * 3600, 14 * 3600]],
    "dinner": [[18 * 3600, 21 * 3600]],
}


class RetrievalOrchestratorLLM:
    """
    Build a structured retrieval plan for each query.

    The planner is intentionally conservative:
    - it always emits a valid plan dict
    - it uses an LLM only when available locally
    - it falls back to heuristics for temporal reasoning
    """

    AUDIO_KEYWORDS = {
        "say",
        "said",
        "hear",
        "heard",
        "sound",
        "sounds",
        "audio",
        "noise",
        "noises",
        "conversation",
        "talk",
        "talking",
        "voice",
        "voices",
        "music",
        "sing",
        "singing",
        "siren",
        "sirens",
        "shout",
        "shouting",
        "yell",
        "yelling",
        "ring",
        "ringing",
        "phone",
        "call",
    }

    VISUAL_KEYWORDS = {
        "wearing",
        "holding",
        "looking",
        "see",
        "seen",
        "look",
        "color",
        "colour",
        "where",
        "which",
        "who",
        "what",
    }

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        use_llm: bool = True,
        max_new_tokens: int = 192,
    ):
        self.model_name = model_name
        self.device = device
        self.use_llm = use_llm
        self.max_new_tokens = max_new_tokens
        self._pipeline = None
        self._llm_failed = False

    def plan_queries(
        self,
        queries,
        modalities: list[str],
        default_use_windows: bool,
        rewrite_queries: bool,
    ) -> None:
        for query in queries:
            query.retrieval_plan = self.plan_query(
                query=query,
                modalities=modalities,
                default_use_windows=default_use_windows,
                rewrite_queries=rewrite_queries,
            )
            logging.info(
                "[Planner] Query %s | text: %s\n  plan: %s",
                getattr(query, "qid", "?"),
                query.get_query(),
                query.retrieval_plan,
            )

    def plan_query(
        self,
        query,
        modalities: list[str],
        default_use_windows: bool,
        rewrite_queries: bool,
    ) -> dict[str, Any]:
        heuristic_plan = self._heuristic_plan(
            query=query,
            modalities=modalities,
            default_use_windows=default_use_windows,
            rewrite_queries=rewrite_queries,
        )

        if not self.use_llm:
            return heuristic_plan

        llm_plan = self._llm_plan(
            query=query,
            modalities=modalities,
            default_use_windows=default_use_windows,
            rewrite_queries=rewrite_queries,
        )
        if not llm_plan:
            return heuristic_plan

        merged = self._merge_plan(heuristic_plan, llm_plan)
        merged["planner"] = "llm"
        return merged

    def _ensure_pipeline(self) -> bool:
        if self._pipeline is not None:
            return True
        if self._llm_failed or AutoTokenizer is None or AutoModelForCausalLM is None or pipeline is None:
            return False

        try:
            logging.info("[Planner] Loading local orchestrator model %s", self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                local_files_only=True,
                torch_dtype="auto",
            )
            pipe_device = 0 if self.device == "cuda" else -1
            if self.device == "cuda":
                model = model.to("cuda")
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=pipe_device,
            )
            return True
        except Exception as exc:
            self._llm_failed = True
            logging.warning(
                "[Planner] Falling back to heuristic planning because local orchestrator weights are unavailable: %s",
                exc,
            )
            return False

    def _llm_plan(
        self,
        query,
        modalities: list[str],
        default_use_windows: bool,
        rewrite_queries: bool,
    ) -> dict[str, Any]:
        if not self._ensure_pipeline():
            return {}

        prompt = self._build_prompt(
            query=query,
            modalities=modalities,
            default_use_windows=default_use_windows,
            rewrite_queries=rewrite_queries,
        )
        try:
            generated = self._pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                return_full_text=True,
            )[0]["generated_text"]
        except Exception as exc:
            logging.warning("[Planner] LLM planning failed for query %s: %s", query.qid, exc)
            return {}

        if "Plan JSON:" in generated:
            generated = generated.split("Plan JSON:", 1)[-1].strip()

        raw_plan = JSONParser.parse_json(generated)
        return self._normalize_llm_plan(raw_plan, modalities)

    def _build_prompt(
        self,
        query,
        modalities: list[str],
        default_use_windows: bool,
        rewrite_queries: bool,
    ) -> str:
        metadata = self._safe_metadata(query)
        return (
            "You are an orchestration model for an egocentric multimodal RAG system.\n"
            "Your job is to decide which retrieval tools to apply and in what order.\n"
            "Available tools: day_filter, time_range_filter, before_query_time_filter, after_query_time_filter, "
            "video_retrieval, window_retrieval, scene_retrieval, temporal_rerank.\n"
            "Return ONLY one valid JSON object with these keys:\n"
            '{'
            '"modalities": {"priority": [string, ...]}, '
            '"temporal": {'
            '"allowed_days": [string, ...], '
            '"time_of_day": string, '
            '"time_ranges_sec": [[number, number], ...], '
            '"relation_to_query_time": string'
            "}, "
            '"use_windows": boolean, '
            '"rewrite_query": boolean, '
            '"workflow": [string, ...]'
            '}\n'
            "Guidelines:\n"
            "- Use day_filter when the query implies a day restriction.\n"
            "- Use time_range_filter for phrases like morning, afternoon, evening, tonight.\n"
            "- If the query asks about a future period on the same day, use after_query_time_filter.\n"
            "- If the query is about past events on the same day without a specific future period, use before_query_time_filter.\n"
            "- Prefer window_retrieval for broad temporal spans inside a long day-video.\n"
            f"Available modalities: {modalities}\n"
            f"Default use_windows: {default_use_windows}\n"
            f"Default rewrite_query: {rewrite_queries}\n"
            f"Query text: {query.get_query()}\n"
            f"Query metadata: {metadata}\n"
            "Plan JSON:"
        )

    def _heuristic_plan(
        self,
        query,
        modalities: list[str],
        default_use_windows: bool,
        rewrite_queries: bool,
    ) -> dict[str, Any]:
        metadata = self._safe_metadata(query)
        qtext = (query.get_query() or "").strip()
        qtext_lower = qtext.lower()

        time_of_day, time_ranges = self._infer_time_of_day(qtext_lower)
        allowed_days, relative_day = self._infer_allowed_days(qtext_lower, metadata.get("query_date"))
        relation = self._infer_relation_to_query_time(
            qtext_lower=qtext_lower,
            query_time_sec=metadata.get("query_time_sec"),
            time_ranges=time_ranges,
            relative_day=relative_day,
        )
        prioritized_modalities = self._infer_modalities(
            qtext_lower=qtext_lower,
            metadata=metadata,
            modalities=modalities,
        )
        use_windows = self._infer_use_windows(
            qtext_lower=qtext_lower,
            time_ranges=time_ranges,
            default_use_windows=default_use_windows,
        )
        workflow = self._build_workflow(
            allowed_days=allowed_days,
            time_ranges=time_ranges,
            relation=relation,
            use_windows=use_windows,
            rewrite_query=rewrite_queries,
        )

        return {
            "planner": "heuristic",
            "modalities": {
                "priority": prioritized_modalities,
            },
            "temporal": {
                "allowed_days": allowed_days,
                "relative_day": relative_day,
                "time_of_day": time_of_day,
                "time_ranges_sec": time_ranges,
                "relation_to_query_time": relation,
            },
            "use_windows": use_windows,
            "rewrite_query": rewrite_queries,
            "workflow": workflow,
        }

    def _normalize_llm_plan(self, raw_plan: dict[str, Any], modalities: list[str]) -> dict[str, Any]:
        if not raw_plan:
            return {}

        normalized: dict[str, Any] = {}

        raw_modalities = raw_plan.get("modalities", {}) or {}
        priority = raw_modalities.get("priority", [])
        if isinstance(priority, str):
            priority = [priority]
        if isinstance(priority, list):
            cleaned = []
            for modality in priority:
                if isinstance(modality, str):
                    mod = modality.strip().lower()
                    if mod in modalities and mod not in cleaned:
                        cleaned.append(mod)
            if cleaned:
                normalized["modalities"] = {"priority": cleaned}

        raw_temporal = raw_plan.get("temporal", {}) or {}
        temporal: dict[str, Any] = {}
        allowed_days = raw_temporal.get("allowed_days", [])
        if isinstance(allowed_days, str):
            allowed_days = [allowed_days]
        if isinstance(allowed_days, list):
            temporal["allowed_days"] = self._normalize_days(allowed_days)
        time_of_day = raw_temporal.get("time_of_day")
        if isinstance(time_of_day, str) and time_of_day.strip():
            temporal["time_of_day"] = time_of_day.strip().lower().replace(" ", "_")
        time_ranges = raw_temporal.get("time_ranges_sec", [])
        if isinstance(time_ranges, list):
            temporal["time_ranges_sec"] = self._normalize_time_ranges(time_ranges)
        relation = raw_temporal.get("relation_to_query_time")
        if isinstance(relation, str) and relation.strip():
            temporal["relation_to_query_time"] = relation.strip().lower()
        if temporal:
            normalized["temporal"] = temporal

        use_windows = raw_plan.get("use_windows")
        if isinstance(use_windows, bool):
            normalized["use_windows"] = use_windows

        rewrite_query = raw_plan.get("rewrite_query")
        if isinstance(rewrite_query, bool):
            normalized["rewrite_query"] = rewrite_query

        workflow = raw_plan.get("workflow", [])
        if isinstance(workflow, str):
            workflow = [workflow]
        if isinstance(workflow, list):
            normalized["workflow"] = [
                step.strip()
                for step in workflow
                if isinstance(step, str) and step.strip()
            ]

        return normalized

    def _merge_plan(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_plan(merged[key], value)
            else:
                merged[key] = value

        modalities = merged.get("modalities", {}).get("priority", [])
        if not modalities:
            fallback = base.get("modalities", {}).get("priority", [])
            if fallback:
                merged.setdefault("modalities", {})["priority"] = fallback

        temporal = merged.get("temporal", {})
        if temporal.get("time_of_day") and not temporal.get("time_ranges_sec"):
            temporal["time_ranges_sec"] = TIME_OF_DAY_RANGES.get(temporal["time_of_day"], [])
        temporal["allowed_days"] = self._normalize_days(temporal.get("allowed_days", []))
        temporal["time_ranges_sec"] = self._normalize_time_ranges(temporal.get("time_ranges_sec", []))
        merged["temporal"] = temporal

        if not merged.get("workflow") and base.get("workflow"):
            merged["workflow"] = base["workflow"]

        return merged

    def _infer_allowed_days(self, qtext_lower: str, query_date: str | None) -> tuple[list[str], str]:
        explicit_days = re.findall(r"\bday\s*([1-7])\b", qtext_lower)
        explicit_days += re.findall(r"\bday([1-7])\b", qtext_lower)
        if explicit_days:
            days = [f"DAY{int(day_num)}" for day_num in explicit_days]
            return self._normalize_days(days), "explicit"

        if any(phrase in qtext_lower for phrase in ("the day before", "day before", "previous day")):
            return self._shift_days(query_date, -1), "previous"
        if any(phrase in qtext_lower for phrase in ("yesterday", "last night")):
            return self._shift_days(query_date, -1), "yesterday"
        if any(phrase in qtext_lower for phrase in ("tomorrow", "next day", "following day")):
            return self._shift_days(query_date, 1), "next"
        if any(phrase in qtext_lower for phrase in ("other day", "other days", "another day", "any day", "past days")):
            return [], "unrestricted"
        if query_date:
            return [str(query_date).upper()], "same_day"
        return [], "unrestricted"

    def _infer_time_of_day(self, qtext_lower: str) -> tuple[str | None, list[list[float]]]:
        phrase_map = [
            (r"\bearly morning\b", "early_morning"),
            (r"\bthis morning\b", "morning"),
            (r"\byesterday morning\b", "morning"),
            (r"\bmorning\b", "morning"),
            (r"\bthis afternoon\b", "afternoon"),
            (r"\byesterday afternoon\b", "afternoon"),
            (r"\bafternoon\b", "afternoon"),
            (r"\bthis evening\b", "evening"),
            (r"\byesterday evening\b", "evening"),
            (r"\bevening\b", "evening"),
            (r"\blate night\b", "late_night"),
            (r"\btonight\b", "night"),
            (r"\blast night\b", "night"),
            (r"\bnight\b", "night"),
            (r"\bnoon\b", "noon"),
            (r"\blunchtime\b", "lunch"),
            (r"\blunch\b", "lunch"),
            (r"\bbreakfast\b", "breakfast"),
            (r"\bdinner\b", "dinner"),
            (r"\bsunrise\b", "sunrise"),
            (r"\bdawn\b", "dawn"),
        ]
        for pattern, label in phrase_map:
            if re.search(pattern, qtext_lower):
                return label, deepcopy(TIME_OF_DAY_RANGES.get(label, []))
        return None, []

    def _infer_relation_to_query_time(
        self,
        qtext_lower: str,
        query_time_sec: float | None,
        time_ranges: list[list[float]],
        relative_day: str,
    ) -> str:
        if relative_day in {"previous", "yesterday", "next"}:
            return "unrestricted"
        if any(phrase in qtext_lower for phrase in ("after that", "later that day", "after lunch", "after dinner")):
            return "after_query_time"
        if any(phrase in qtext_lower for phrase in ("before that", "before lunch", "earlier that day")):
            return "before_query_time"

        if not query_time_sec:
            return "unrestricted"

        if time_ranges:
            earliest = min(start for start, _ in time_ranges)
            latest = max(end for _, end in time_ranges)
            if query_time_sec <= earliest:
                return "after_query_time"
            if query_time_sec >= latest:
                return "before_query_time"
            return "around_query_time"

        if any(phrase in qtext_lower for phrase in ("today", "this day", "earlier", "before")):
            return "before_query_time"
        return "unrestricted"

    def _infer_modalities(
        self,
        qtext_lower: str,
        metadata: dict[str, Any],
        modalities: list[str],
    ) -> list[str]:
        preferred: list[str] = []
        audio_hint = bool(metadata.get("need_audio")) or any(word in qtext_lower for word in self.AUDIO_KEYWORDS)
        visual_hint = any(word in qtext_lower for word in self.VISUAL_KEYWORDS)

        if audio_hint and "audio" in modalities:
            preferred.append("audio")
        if visual_hint and "video" in modalities:
            preferred.append("video")
        if "text" in modalities:
            preferred.append("text")

        for modality in modalities:
            if modality not in preferred:
                preferred.append(modality)
        return preferred

    def _infer_use_windows(
        self,
        qtext_lower: str,
        time_ranges: list[list[float]],
        default_use_windows: bool,
    ) -> bool:
        # Respect caller's explicit False — don't enable windows if the dataset has none.
        if not default_use_windows:
            return False
        if time_ranges:
            return True
        if any(token in qtext_lower for token in ("when", "moment", "around", "earlier", "later", "before", "after")):
            return True
        return default_use_windows

    def _build_workflow(
        self,
        allowed_days: list[str],
        time_ranges: list[list[float]],
        relation: str,
        use_windows: bool,
        rewrite_query: bool,
    ) -> list[str]:
        workflow = []
        if rewrite_query:
            workflow.append("rewrite_query")
        if allowed_days:
            workflow.append("day_filter")
        if time_ranges:
            workflow.append("time_range_filter")
        if relation == "before_query_time":
            workflow.append("before_query_time_filter")
        elif relation == "after_query_time":
            workflow.append("after_query_time_filter")
        workflow.append("video_retrieval")
        if use_windows:
            workflow.append("window_retrieval")
        workflow.append("scene_retrieval")
        if time_ranges or relation != "unrestricted":
            workflow.append("temporal_rerank")
        return workflow

    def _safe_metadata(self, query) -> dict[str, Any]:
        decomposed = getattr(query, "decomposed", {}) or {}
        metadata = decomposed.get("metadata", {}) or {}
        return dict(metadata)

    def _shift_days(self, query_date: str | None, delta: int) -> list[str]:
        if not query_date:
            return []
        match = re.search(r"DAY(\d+)", str(query_date), re.IGNORECASE)
        if not match:
            return []
        day_num = int(match.group(1)) + delta
        if day_num < 1 or day_num > 7:
            return []
        return [f"DAY{day_num}"]

    def _normalize_days(self, allowed_days: list[Any]) -> list[str]:
        normalized = []
        for day in allowed_days:
            if not isinstance(day, str):
                continue
            match = re.search(r"DAY\s*(\d+)", day.strip(), re.IGNORECASE)
            if not match:
                continue
            cleaned = f"DAY{int(match.group(1))}"
            if cleaned not in normalized:
                normalized.append(cleaned)
        return normalized

    def _normalize_time_ranges(self, time_ranges: list[Any]) -> list[list[float]]:
        normalized = []
        for item in time_ranges:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            try:
                start = float(item[0])
                end = float(item[1])
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            # Auto-detect unit: LLMs often output hours (0-24) or minutes instead of seconds.
            # Heuristic: if max value < 24 → hours; if < 1440 → minutes; else → seconds.
            max_val = max(abs(start), abs(end))
            if max_val < 24:
                start *= 3600.0
                end *= 3600.0
            elif max_val < 1440:
                start *= 60.0
                end *= 60.0
            start = max(0.0, start)
            end = min(24 * 3600.0, end)
            if end <= start:
                continue
            normalized.append([start, end])
        return normalized
