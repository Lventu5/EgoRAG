import torch
import logging
import numpy as np
from tqdm import tqdm
from transformers import (
    XCLIPProcessor,
    XCLIPModel,
    ClapProcessor,
    ClapModel,
    AutoProcessor,
    AutoModel,
    AutoTokenizer,
)
from sentence_transformers import SentenceTransformer
from torch.nn.functional import normalize, cosine_similarity

from data.video_dataset import VideoDataset, Window, Scene
from data.query import Query, QueryDataset
import data.datatypes as types
from retrieval.rewriter import QueryRewriterLLM
from retrieval.query_orchestrator import RetrievalOrchestratorLLM
from .fuser import Fuser
from .scene_merger import SceneMerger
from configuration.config import CONFIG
from retrieval.query_tagger import QueryTagger

import sys
import os
import gc
from pathlib import Path
import re
sys.path.append(os.path.join(Path(os.getcwd()), 'external/InternVideo/InternVideo2'))
sys.path.append(os.path.join(Path(os.getcwd()), 'external/InternVideo/InternVideo2/multi_modality'))
import interface


class HierarchicalRetriever:
    def __init__(
        self, 
        video_dataset: VideoDataset,
        fuser: Fuser | None = None,
        device: str = "cuda",
    ):
        self.video_dataset = video_dataset
        self.device = device
        logging.info(f"Retriever running on {self.device}")
        self.video_model_name = CONFIG.retrieval.video_model_name
        self.audio_model_name = CONFIG.retrieval.audio_model_id
        self.text_model_name = CONFIG.retrieval.text_model_id
    
        video_embed_size = 512  # XCLIP outputs 512-dim embeddings
        
        self.sizes = {
            "video": {
                "size": video_embed_size,  
                "model": self.video_model_name
            },
            "audio": {
                "size": 512,
                "model": self.audio_model_name
            },
            "text": {
                "size": 768,
                "model": self.text_model_name
            },
        }
        self.rewriter = CONFIG.retrieval.rewriter_model_id
        planner_config = getattr(CONFIG.retrieval, "orchestrator", None)
        self.use_orchestrator = bool(getattr(planner_config, "enabled", False))
        planner_model_name = getattr(planner_config, "model_name", self.rewriter)
        planner_use_llm = bool(getattr(planner_config, "use_llm", True))
        planner_max_new_tokens = int(getattr(planner_config, "max_new_tokens", 192))
        self.orchestrator = RetrievalOrchestratorLLM(
            model_name=planner_model_name,
            device=self.device,
            use_llm=planner_use_llm,
            max_new_tokens=planner_max_new_tokens,
        )
        self.current_modality = None
        self.processor = None
        self.embedder = None
        
        if fuser is None:
            logging.warning("Fuser not specified, using a RRF fuser")
            self.fuser = Fuser()
        else:
            self.fuser = fuser
        
        # Initialize scene merger if enabled in config
        merger_config = getattr(CONFIG.retrieval, 'scene_merger', None)
        if merger_config and getattr(merger_config, 'enabled', False):
            self.scene_merger = SceneMerger(
                max_gap=getattr(merger_config, 'max_gap', 1.0),
                min_scenes_to_merge=getattr(merger_config, 'min_scenes_to_merge', 2),
                max_scenes_to_merge=getattr(merger_config, 'max_scenes_to_merge', 5),
            )
            self.merge_score_aggregation = getattr(merger_config, 'score_aggregation', 'max')
            logging.info("Scene merger enabled")
        else:
            self.scene_merger = None
            self.merge_score_aggregation = 'max'
            logging.info("Scene merger disabled")

        # QueryTagger for CLIP-based query tagging (lazy load)
        self.query_tagger = QueryTagger(device=self.device)
        self._tagger_loaded = False

    def unload_models(self):
        """
        Explicitly unload all models from GPU to free memory.
        Call this when done with retrieval to free up GPU resources.
        """
        if hasattr(self, 'embedder') and self.embedder is not None:
            try:
                self.embedder.cpu()
            except:
                pass
            del self.embedder
            self.embedder = None
        
        if hasattr(self, 'processor') and self.processor is not None:
            del self.processor
            self.processor = None
        
        if hasattr(self, 'query_tagger') and self.query_tagger is not None:
            self.query_tagger.unload_model() if hasattr(self.query_tagger, 'unload_model') else None
            self._tagger_loaded = False
        
        self.current_modality = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        logging.info("All models unloaded from GPU")

    def _load_models_for_modality(self, modality: str):
        """
        Loads the model specified by a specific modality 
        Args:
            modality: the modality to load the model of
        """
        if self.current_modality == modality:
            return
        
        # Aggressive cleanup before loading new model
        if hasattr(self, 'embedder') and self.embedder is not None:
            # Move model to CPU first to free GPU memory
            try:
                self.embedder.cpu()
            except:
                pass
            del self.embedder
        if hasattr(self, 'processor') and self.processor is not None:
            del self.processor
        self.processor = None
        self.embedder = None
        
        # Force garbage collection and clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        target_modality = modality

        # Text encoder
        if target_modality == "text":
            self.embedder = SentenceTransformer(
                self.sizes["text"]["model"], device=self.device
            )
        
        # Video encoder
        elif target_modality == "video":
            if self.video_model_name == "internvideo2-6b":
                model_name = CONFIG.retrieval.internvideo2_6b_id
                logging.info(f"Loading InternVideo2 model for retrieval: {model_name}")
                # Load in float16 to reduce memory usage on V100
                self.embedder = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16  # Use half precision to save ~50% GPU memory
                ).to(self.device).eval()

            elif self.video_model_name == "internvideo2-1b":
                logging.info(f"Loading InternVideo2 1B model for retrieval: {self.video_model_name}")
                config_path = "external/InternVideo/InternVideo2/multi_modality/demo/internvideo2_stage2_config.py"
                model_path = "external/InternVideo/InternVideo2/checkpoints/InternVideo2-stage2_1b-224p-f4.pt"

                model = interface.load_model(config_path, model_path)
                self.embedder = model.to(self.device).eval()

            else:  # xclip
                model_name = self.sizes["video"]["model"]
                logging.info(f"Loading XCLIP model for retrieval: {model_name}")
                self.processor = XCLIPProcessor.from_pretrained(model_name)
                self.embedder = XCLIPModel.from_pretrained(model_name).to(self.device) # type: ignore

        # Audio
        elif target_modality == "audio":
            model_name = self.sizes["audio"]["model"]
            logging.info(f"Loading CLAP model for audio retrieval: {model_name}")
            self.processor = ClapProcessor.from_pretrained(model_name)
            self.embedder = ClapModel.from_pretrained(model_name).to(self.device) # type: ignore
    
        else:
            raise ValueError(f"Unknown modality: {modality}")

        self.current_modality = target_modality


    def _rewrite_queries(self, queries: QueryDataset):
        """
        Rewrite queries using the LLM rewriter to decompose them into sub-queries per modality
        Args:
            queries: the QueryDataset to rewrite
        """
        if self.rewrite_queries: 
            rewriter = QueryRewriterLLM(
                model_name=self.rewriter, 
                device=self.device
            )
            for query in tqdm(queries, desc = "Rewriting queries"):
                if not self._should_rewrite_query(query):
                    if query.decomposed is None or not isinstance(query.decomposed, dict):
                        query.decomposed = {}
                    for mod in ["text", "video", "audio"]:
                        if not query.decomposed.get(mod):
                            query.decomposed[mod] = query.get_query()
                    continue
                decomposition = rewriter(query.get_query(), modality="decompose")
                # Preserve existing metadata (e.g., EgoLife query fields)
                existing_meta = {}
                if isinstance(query.decomposed, dict):
                    existing_meta = query.decomposed.get("metadata", {}) or {}
                if isinstance(decomposition, dict) and existing_meta:
                    decomposition["metadata"] = existing_meta
                query.decomposed = decomposition
        else:
            for query in queries:
                if query.decomposed is None or not isinstance(query.decomposed, dict):
                    query.decomposed = {}
                # Only fill missing modalities, preserve any existing metadata
                for mod in ["text", "video", "audio"]:
                    if not query.decomposed.get(mod):
                        query.decomposed[mod] = query.get_query()

    def _get_query_metadata(self, query: Query) -> dict:
        if isinstance(getattr(query, "decomposed", None), dict):
            return query.decomposed.get("metadata", {}) or {}
        return {}

    def _get_query_plan(self, query: Query) -> dict:
        plan = getattr(query, "retrieval_plan", None)
        if isinstance(plan, dict):
            return plan
        return {}

    def _plan_queries(
        self,
        queries: QueryDataset,
        modalities: list[str],
        use_windows: bool,
        rewrite_queries: bool,
    ) -> None:
        if not self.use_orchestrator:
            for query in queries:
                if not isinstance(getattr(query, "retrieval_plan", None), dict):
                    query.retrieval_plan = {}
            return

        self.orchestrator.plan_queries(
            queries=queries,
            modalities=modalities,
            default_use_windows=use_windows,
            rewrite_queries=rewrite_queries,
        )

    def _should_rewrite_query(self, query: Query) -> bool:
        plan = self._get_query_plan(query)
        if "rewrite_query" in plan:
            return bool(plan["rewrite_query"])
        return self.rewrite_queries

    def _get_modalities_for_query(self, query: Query, default_modalities: list[str]) -> list[str]:
        plan = self._get_query_plan(query)
        preferred = ((plan.get("modalities") or {}).get("priority") or [])
        selected = [mod for mod in preferred if mod in default_modalities]
        for modality in default_modalities:
            if modality not in selected:
                selected.append(modality)
        return selected or list(default_modalities)

    def _query_uses_windows(self, query: Query, default_use_windows: bool) -> bool:
        if not default_use_windows:
            return False  # Never override an explicit False (dataset has no windows)
        plan = self._get_query_plan(query)
        if "use_windows" in plan:
            return bool(plan["use_windows"])
        return default_use_windows

    def _day_to_int(self, day_str: str | None) -> int | None:
        if not day_str:
            return None
        m = re.search(r"DAY(\d+)", str(day_str), re.IGNORECASE)
        if not m:
            return None
        try:
            return int(m.group(1))
        except ValueError:
            return None

    def _extract_day_from_video_name(self, video_name: str | None) -> str | None:
        if not video_name:
            return None
        m = re.search(r"DAY\d+", str(video_name), re.IGNORECASE)
        if not m:
            return None
        return m.group(0).upper()

    def _extract_clip_start_sec_from_video_name(self, video_name: str | None) -> float | None:
        """
        Extract the absolute start time (seconds from midnight) from an EgoLife clip name.
        Clip names end with a timestamp in HHMMSSCC format, e.g. DAY1_A1_JAKE_11094208 -> 11:09:42.08.
        """
        if not video_name:
            return None
        m = re.search(r"_(\d{8})$", str(video_name))
        if not m:
            return None
        ts = m.group(1)
        try:
            hh = int(ts[0:2])
            mm = int(ts[2:4])
            ss = int(ts[4:6])
            cc = int(ts[6:8])
            return hh * 3600 + mm * 60 + ss + cc / 100.0
        except (ValueError, IndexError):
            return None

    def _infer_allowed_days(self, query: Query, query_date: str | None) -> set[str] | None:
        """
        Infer which DAYs are allowed based on query text.
        Returns None to indicate no day filtering.
        """
        qtext = (query.get_query() or "").lower()

        # Explicit DAY mentions (e.g., DAY2, day 3)
        explicit_days: set[str] = set()
        for match in re.findall(r"\bday\s*([1-7])\b", qtext):
            explicit_days.add(f"DAY{int(match)}")
        for match in re.findall(r"\bday([1-7])\b", qtext):
            explicit_days.add(f"DAY{int(match)}")
        if explicit_days:
            return explicit_days

        # Relative references
        if "yesterday" in qtext and query_date:
            day_num = self._day_to_int(query_date)
            if day_num and day_num > 1:
                return {f"DAY{day_num - 1}"}
        if "tomorrow" in qtext and query_date:
            day_num = self._day_to_int(query_date)
            if day_num and day_num < 7:
                return {f"DAY{day_num + 1}"}

        # Other-day phrasing means allow all days
        other_day_phrases = [
            "other day",
            "other days",
            "another day",
            "previous day",
            "previous days",
            "past days",
            "earlier days",
            "day before",
            "days before",
            "the other day",
            "the other days",
        ]
        if any(phrase in qtext for phrase in other_day_phrases):
            return None

        # Default: restrict to query_date if available
        if query_date:
            return {query_date}
        return None

    def _get_egolife_constraints(self, query: Query) -> dict | None:
        """
        Extract EgoLife-specific constraints (day/time) from query metadata.
        Returns None if no EgoLife metadata is present.
        """
        meta = self._get_query_metadata(query)
        plan = self._get_query_plan(query)
        temporal = plan.get("temporal", {}) if isinstance(plan, dict) else {}

        if not meta and not temporal:
            return None

        query_date = meta.get("query_date")
        query_time_sec = meta.get("query_time_sec")
        allowed_days = temporal.get("allowed_days")
        if allowed_days is None:
            inferred_days = self._infer_allowed_days(query, query_date)
            allowed_days = sorted(inferred_days) if inferred_days else []

        time_ranges = temporal.get("time_ranges_sec") or []
        relation = temporal.get("relation_to_query_time", "unrestricted")
        return {
            "query_date": query_date,
            "query_time_sec": query_time_sec,
            "allowed_days": allowed_days,
            "time_ranges_sec": time_ranges,
            "relation_to_query_time": relation,
        }

    def _video_passes_time_filter(self, video_name: str, time_ranges: list, relation: str, query_time_sec) -> bool:
        """Check if a video's clip time overlaps with the given time ranges and relation constraint."""
        clip_start = self._extract_clip_start_sec_from_video_name(video_name)
        if clip_start is None:
            return True  # can't determine, let it through
        clip_end = clip_start + 30.0  # clips are ~30s

        if time_ranges:
            overlaps = any(clip_end >= float(s) and clip_start <= float(e) for s, e in time_ranges)
            if not overlaps:
                return False

        if query_time_sec and query_time_sec > 0:
            if relation == "before_query_time" and clip_start > float(query_time_sec):
                return False
            if relation == "after_query_time" and clip_end < float(query_time_sec):
                return False

        return True

    def _apply_egolife_video_filters(
        self,
        queries: QueryDataset,
        candidate_videos_per_query: dict[str, set[str]],
    ) -> dict[str, set[str]]:
        """
        Apply EgoLife day-level and time-range filtering to candidate videos.
        Filters are applied with fallback: if a filter produces no candidates,
        the previous (less-filtered) set is kept.
        """
        filtered = {}
        for query in queries:
            candidates = set(candidate_videos_per_query.get(query.qid, set()))
            constraints = self._get_egolife_constraints(query)
            if not constraints:
                filtered[query.qid] = candidates
                continue

            allowed_days = constraints.get("allowed_days")
            time_ranges = constraints.get("time_ranges_sec") or []
            relation = constraints.get("relation_to_query_time", "unrestricted")
            query_time_sec = constraints.get("query_time_sec")

            # Step 1: day filter
            after_day = candidates
            if allowed_days:
                day_filtered = {v for v in candidates
                                if self._extract_day_from_video_name(v) in allowed_days}
                if day_filtered:
                    after_day = day_filtered
                else:
                    logging.warning(
                        f"[EgoLife] Day filter produced no candidates for query {query.qid}; "
                        f"falling back to unfiltered candidates."
                    )

            # Step 2: time range + relation filter on day-filtered candidates
            after_time = after_day
            if time_ranges or (relation not in ("unrestricted", "around_query_time") and query_time_sec):
                time_filtered = {v for v in after_day
                                 if self._video_passes_time_filter(v, time_ranges, relation, query_time_sec)}
                if time_filtered:
                    after_time = time_filtered
                else:
                    logging.warning(
                        f"[EgoLife] Time filter produced no candidates for query {query.qid}; "
                        f"falling back to day-filtered candidates."
                    )

            filtered[query.qid] = after_time

        return filtered

    def _scene_passes_egolife_time_filter(self, query: Query, dp, scene: Scene, scene_data: dict | None = None) -> bool:
        constraints = self._get_egolife_constraints(query)
        if not constraints:
            return True

        query_date = constraints.get("query_date")
        query_time_sec = constraints.get("query_time_sec")
        allowed_days = constraints.get("allowed_days")
        time_ranges = constraints.get("time_ranges_sec") or []
        relation = constraints.get("relation_to_query_time", "unrestricted")

        scene_meta = {}
        if scene_data and isinstance(scene_data, dict):
            scene_meta = scene_data.get("meta", {}) or {}
        if not scene_meta and hasattr(scene, "meta"):
            scene_meta = getattr(scene, "meta", {}) or {}

        return True


    def _encode_text_strings(self, texts: list[str]) -> torch.Tensor:
        """
        Encode a list of strings with the currently loaded model.

        This is the shared encoding core used by both _embed_queries (standard query
        text) and the visual-context path (VLM-generated object description).  It
        returns a CPU tensor of shape (len(texts), D).
        """
        if self.current_modality == "text":
            embs = self.embedder.encode(texts, convert_to_tensor=True, device=self.device)
            return embs.cpu()

        elif self.current_modality == "video":
            if self.video_model_name == "xclip":
                inputs = self.processor(
                    text=texts, return_tensors="pt", padding=True
                ).to(self.device)
                with torch.no_grad():
                    embs = self.embedder.get_text_features(**inputs)
                return embs.cpu()

            elif self.video_model_name == "internvideo2-6b":
                embs_list = []
                for t in tqdm(texts, desc="Encoding text (internvideo2-6b)"):
                    with torch.no_grad():
                        feat = self.embedder.get_txt_feat(t).squeeze(0).cpu()
                    embs_list.append(feat)
                    if len(embs_list) % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                return torch.stack(embs_list)

            elif self.video_model_name == "internvideo2-1b":
                return interface.extract_query_features(texts, self.embedder)

            else:
                raise ValueError(f"Model type not supported: {self.video_model_name}")

        elif self.current_modality == "audio":
            inputs = self.processor(
                text=texts, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                embs = self.embedder.get_text_features(**inputs)
            return embs.cpu()

        else:
            raise ValueError(f"Unknown modality: {self.current_modality}")

    def _embed_queries(self, queries: QueryDataset) -> torch.Tensor:
        """
        Embed queries for the currently loaded modality.

        Two things happen here:
          1. Standard query text  → stored in query.embeddings[modality]
          2. Visual context text  → stored in query.embeddings["visual_context_<modality>"]
             (only for queries that have query.decomposed["visual_context"] set, and only
              when self.use_visual_context is True)

        Both use _encode_text_strings so the same encoder is applied to both signals,
        keeping them in the same embedding space.
        """
        if self.embedder is None:
            raise RuntimeError("No model loaded for embedding. Call _load_models_for_modality first.")

        # --- Standard query embeddings ---
        mod_queries = queries.group_by_modality(self.current_modality)
        logging.info(
            f"Embedding {len(mod_queries)} queries for modality '{self.current_modality}'"
        )
        embeddings_cpu = self._encode_text_strings(mod_queries)

        for query, emb in zip(queries, embeddings_cpu):
            query.embeddings[self.current_modality] = emb.clone()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Visual context embeddings (optional) ---
        # Each query that has a visual_context description gets a second embedding
        # stored under "visual_context_<modality>".  The retriever will use it for
        # a parallel retrieval pass and fuse the two ranked lists with RRF.
        if self.use_visual_context:
            vc_key = f"visual_context_{self.current_modality}"
            vc_pairs = [
                (i, q.decomposed.get("visual_context"))
                for i, q in enumerate(queries)
                if q.decomposed.get("visual_context")
            ]
            if vc_pairs:
                indices, vc_texts = zip(*vc_pairs)
                logging.info(
                    f"Encoding visual context for {len(vc_texts)} queries "
                    f"(modality '{self.current_modality}')"
                )
                vc_embeddings = self._encode_text_strings(list(vc_texts))
                for idx, emb in zip(indices, vc_embeddings):
                    queries[idx].embeddings[vc_key] = emb.clone()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return embeddings_cpu

    def _retrieve_top_videos_per_queries(
        self, 
        queries: QueryDataset,
        candidate_videos: dict[str, str],
        modalities: list[str] | str,
        top_k: int = 1
    )-> types.TopKVideosPerQuery:
        """
        Retrieves the top videos for the given queries among the candidate videos
        Args:
            queries: the QueryDataset of queries to use
            candidate_videos: A dictionary containing the candidate videos for each query
            modalities: the list of modalities to use for retrieval
            top_k: The number of videos to retrieve
        """
        if isinstance(modalities, str):
            modalities = [modalities]

        final_results = {query.qid: {} for query in queries}

        for modality in modalities:
            logging.info(f"Retrieving for modality {modality}...\n")
            self._load_models_for_modality(modality)
            self._embed_queries(queries)
            modality_results_batch = self._retrieve_by_modality(
                queries, candidate_videos, modality, top_k
            )
            for i, query in enumerate(queries):
                final_results[query.qid][modality] = modality_results_batch[i]
                
        return final_results

    def _score_against_candidates(
        self,
        q_emb: torch.Tensor,
        candidate_names: list[str],
        candidate_embs: list[torch.Tensor],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """
        Compute cosine similarity between one query embedding and all candidate
        clip embeddings, return the top-k (name, score) pairs.

        q_emb           : (1, D) tensor already on GPU.
        candidate_embs  : list of (D,) tensors (on CPU or GPU).
        """
        max_batch = 1000
        if len(candidate_embs) <= max_batch:
            db = torch.stack(candidate_embs).to(self.device)
            scores = cosine_similarity(q_emb, db, dim=-1)
            del db
        else:
            batches = []
            for start in range(0, len(candidate_embs), max_batch):
                batch = torch.stack(candidate_embs[start: start + max_batch]).to(self.device)
                batches.append(cosine_similarity(q_emb, batch, dim=-1).cpu())
                del batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            scores = torch.cat(batches).to(self.device)

        topk = min(top_k, len(candidate_names))
        top_scores, top_indices = torch.topk(scores, k=topk)
        results = [
            (candidate_names[idx.item()], score.item())
            for score, idx in zip(top_scores, top_indices)
        ]
        del scores, top_scores, top_indices
        return results

    def _retrieve_by_modality(
        self,
        queries: QueryDataset,
        candidate_videos: dict[str, str],
        modality: str,
        top_k: int = 1
    ) -> types.TopKVideosPerModality:
        """
        Retrieve the top-k videos for a given modality.

        When self.use_visual_context is True and a query has a visual-context embedding
        (stored as query.embeddings["visual_context_<modality>"]), the retrieval runs
        TWO passes over the same candidate pool:
          - Pass A: using the standard query embedding
          - Pass B: using the visual-context embedding (VLM description of the referent)
        The two ranked lists are fused per-query with the existing Fuser (RRF by default),
        giving a single ranked list that benefits from both signals.

        Queries without a visual-context embedding fall back to pass A only.
        """
        logging.info(f"Retrieving top {top_k} results for modality '{modality}'")

        # Keep query embeddings on CPU initially, move per-query to GPU
        query_embeddings_cpu = queries.embeddings_by_modality(modality)

        all_results = []

        # Build a mapping from video_name -> datapoint for fast lookup
        dp_by_name = {dp.video_name: dp for dp in self.video_dataset.video_datapoints}

        # For each query, use precomputed candidate_videos to
        # avoid repeated tag inference and to perform the screening exactly once.
        for q_idx, query in enumerate(queries):
            # Move only current query embedding to GPU
            q_emb = query_embeddings_cpu[q_idx].unsqueeze(0).to(self.device)  # shape (1, D)

            # Take the candidate videos
            candidate_names = []
            candidate_embs = []
            videos_without_modality = []

            names_to_iterate = list(candidate_videos[query.qid])

            for name in names_to_iterate:
                dp = dp_by_name.get(name)
                if dp is None:
                    continue
                emb = dp.global_embeddings.get(modality, None)
                
                # Skip videos without embedding for this modality
                if emb is None:
                    videos_without_modality.append(name)
                    continue
                
                # Convert to tensor if needed
                if not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb)
                
                candidate_names.append(dp.video_name)
                candidate_embs.append(emb)

            if not candidate_embs:
                all_results.append([])
                # Clean up query embedding
                del q_emb
                continue

            # Log info about videos without modality for this query
            if videos_without_modality and modality == "audio":
                logging.info(f"{len(videos_without_modality)} video(s) have no audio track and will be excluded from audio-based retrieval")

            # --- Pass A: standard query embedding ---
            primary_results = self._score_against_candidates(
                q_emb, candidate_names, candidate_embs, top_k
            )

            # --- Pass B: visual context embedding (optional) ---
            # If the query was enriched with a visual-context description, run a second
            # retrieval pass using that description's embedding and fuse with pass A.
            vc_emb_tensor = query.embeddings.get(f"visual_context_{modality}")
            if self.use_visual_context and vc_emb_tensor is not None:
                vc_emb = vc_emb_tensor.unsqueeze(0).to(self.device)
                vc_results = self._score_against_candidates(
                    vc_emb, candidate_names, candidate_embs, top_k
                )
                del vc_emb
                # Fuse the two ranked lists within this modality using RRF.
                # Keys "query" and "visual_context" are arbitrary labels for the fuser.
                query_results = self.fuser.fuse({
                    "query": primary_results,
                    "visual_context": vc_results,
                })
                logging.debug(
                    f"[VC] Query {query.qid}/{modality}: fused {len(primary_results)} "
                    f"+ {len(vc_results)} → {len(query_results)} candidates"
                )
            else:
                query_results = primary_results

            all_results.append(query_results)

            # Clean up tensors after each query
            del q_emb
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return all_results

    def _filter_scenes_by_query_tags(self, query: Query, target_dp, modality: str):
        """
        Helper that returns (scenes, scene_embeddings_list) for a given video
        filtered by precomputed scene tags present in `target_dp.scene_embeddings`.
        
        If use_tagging is True and query has tags:
            - Only return scenes that have at least one matching tag
        Otherwise:
            - Return all scenes with embeddings for the requested modality
        """
        query_tag_set = set([t.lower() for t in getattr(query, 'tags', []) or []])
        apply_tag_filter = self.use_tagging and len(query_tag_set) > 0

        scenes = []
        scene_embeddings_list = []

        for scene_id, scene_data in target_dp.scene_embeddings.items():
            # Apply tag-based filtering if enabled
            if apply_tag_filter:
                scene_tags = scene_data.get("tags") or []
                scene_tag_set = set([t.lower() for t in scene_tags])
                # Skip scenes with no matching tags
                if not any(t in scene_tag_set for t in query_tag_set):
                    continue

            # Get embedding for this modality
            emb = scene_data.get(modality, None)
            if emb is None:
                continue
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb)

            # Get the Scene object
            scene_obj = target_dp.get_scene_by_id(scene_id)
            if scene_obj is None:
                logging.warning(f"Scene {scene_id} has embedding but no Scene object in video {target_dp.video_name}")
                continue

            # EgoLife time/day filtering (if applicable)
            if not self._scene_passes_egolife_time_filter(query, target_dp, scene_obj, scene_data):
                continue
            
            scenes.append(scene_obj)
            scene_embeddings_list.append(emb.to(self.device))

        if apply_tag_filter and len(scenes) > 0:
            logging.debug(f"Tag filtering: {len(scenes)} scenes match query tags {list(query_tag_set)[:3]}...")

        return scenes, scene_embeddings_list
    
    def _retrieve_best_scenes(
        self, 
        query: Query, 
        video_name: str, 
        modality: str, 
        top_k: int = 1
    ) -> types.TopKScenes:
        """
        Gets the best scene in a video (we already know it is a top-k video) for the specified query
        Returns a list with the top k scenes and their similarity score
        Args:
            query: the Query to use
            video_name: the video to retrieve from
            modality: the modality to use
            top_k: the number of scenes to retrieve
        """
        target_dp = None
        for dp in self.video_dataset.video_datapoints:
            if dp.video_name == video_name:
                target_dp = dp
                break
        
        if target_dp is None:
            raise RuntimeError(f"Video '{video_name}' not found")

        query_embedding = query.get_embedding(modality).to(self.device)

        # Use helper to pre-screen scenes by query tags (relies on precomputed scene tags)
        scenes, scene_embeddings_list = self._filter_scenes_by_query_tags(query, target_dp, modality)
        
        if not scene_embeddings_list:
            # Check if this is expected (e.g., video without audio)
            if modality == "audio" and hasattr(target_dp, 'has_audio') and not target_dp.has_audio:
                logging.debug(f"Video '{video_name}' has no audio track - skipping audio scene retrieval")
            else:
                logging.error(f"No scene embeddings found for video '{video_name}' and modality '{modality}'")
            return []

        # Perform cosine similarity between query and scenes
        scene_embeddings = torch.stack(scene_embeddings_list)  # Shape: (N, D)
        
        # Ensure query_embedding is 2D: (1, D) for proper broadcasting
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)  # (D,) -> (1, D)
        
        # cosine_similarity with dim=-1 computes similarity along feature dimension
        # query_embedding: (1, D), scene_embeddings: (N, D) -> broadcasts to (N,)
        sim_vector = cosine_similarity(query_embedding, scene_embeddings, dim=-1)  # Shape: (N,)
        
        # Debug logging for similarity scores
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"[Scene Retrieval] video={video_name}, modality={modality}")
            logging.debug(f"  query_emb shape: {query_embedding.shape}")
            logging.debug(f"  scene_embs shape: {scene_embeddings.shape}")
            logging.debug(f"  sim_vector shape: {sim_vector.shape}")
            logging.debug(f"  sim_vector: min={sim_vector.min().item():.4f}, max={sim_vector.max().item():.4f}, mean={sim_vector.mean().item():.4f}")

        # topk on 1D tensor (N,) returns indices 0..N-1 corresponding to scenes
        top_scores, top_indices = torch.topk(
            sim_vector, k=min(top_k, len(scenes))
        )
        results = []
        for score, scene_idx in zip(top_scores, top_indices):
            results.append((scenes[scene_idx.item()], score.item())) # type: ignore
        
        # Clean up GPU tensors
        del query_embedding, scene_embeddings, sim_vector, top_scores, top_indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Apply scene merging if enabled
        if self.scene_merger is not None:
            results = self.scene_merger.merge_top_k_scenes(
                results,
                score_aggregation=self.merge_score_aggregation
            )
        
        return results

    def _retrieve_best_windows(
        self, 
        query: Query, 
        video_name: str, 
        modality: str, 
        top_k: int = 1
    ) -> list[tuple[Window, float]]:
        """
        Retrieves the best windows in a video for the specified query.
        Returns a list with the top k windows and their similarity scores.
        
        Args:
            query: the Query to use
            video_name: the video to retrieve from
            modality: the modality to use
            top_k: the number of windows to retrieve
        """
        target_dp = None
        for dp in self.video_dataset.video_datapoints:
            if dp.video_name == video_name:
                target_dp = dp
                break
        
        if target_dp is None:
            raise RuntimeError(f"Video '{video_name}' not found")

        # Check if video has windows
        if not hasattr(target_dp, 'windows') or not target_dp.windows:
            logging.warning(f"Video '{video_name}' has no windows - falling back to scene-level retrieval")
            return []
        
        if not hasattr(target_dp, 'window_embeddings') or not target_dp.window_embeddings:
            logging.warning(f"Video '{video_name}' has no window embeddings - falling back to scene-level retrieval")
            return []

        query_embedding = query.get_embedding(modality).to(self.device)

        # Filter windows by query tags if enabled
        query_tag_set = set([t.lower() for t in getattr(query, 'tags', []) or []])
        apply_tag_filter = self.use_tagging and len(query_tag_set) > 0

        # Collect window embeddings for the requested modality
        windows = []
        window_embeddings_list = []
        
        constraints = self._get_egolife_constraints(query)
        dp_day = self._extract_day_from_video_name(target_dp.video_name)

        for window in target_dp.windows:
            # EgoLife day/time filtering at window level (if applicable)
            if constraints:
                allowed_days = constraints.get("allowed_days")
                query_date = constraints.get("query_date")
                query_time_sec = constraints.get("query_time_sec")
                time_ranges = constraints.get("time_ranges_sec") or []
                relation = constraints.get("relation_to_query_time", "unrestricted")
                if allowed_days and dp_day and dp_day not in allowed_days:
                    continue
                if time_ranges:
                    overlaps_range = False
                    for start_sec, end_sec in time_ranges:
                        if window.end_time >= float(start_sec) and window.start_time <= float(end_sec):
                            overlaps_range = True
                            break
                    if not overlaps_range:
                        continue
                if query_date and query_time_sec and query_time_sec > 0 and dp_day and dp_day.upper() == str(query_date).upper():
                    if relation == "before_query_time" and window.start_time > float(query_time_sec):
                        continue
                    if relation == "after_query_time" and window.end_time < float(query_time_sec):
                        continue

            wid = window.window_id
            if wid not in target_dp.window_embeddings:
                logging.warning(f"Window id {wid} not found in {target_dp.window_embeddings.keys()}")
                continue
            
            window_data = target_dp.window_embeddings[wid]
            
            # Apply tag filtering if enabled
            if apply_tag_filter:
                window_tags = window_data.get("tags") or []
                window_tag_set = set([t.lower() for t in window_tags])
                # Skip windows with no matching tags
                if not any(t in window_tag_set for t in query_tag_set):
                    continue
            
            emb = window_data.get(modality)
            if emb is not None:
                if not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb)
                windows.append(window)
                window_embeddings_list.append(emb.to(self.device))
        
        if apply_tag_filter and len(windows) > 0:
            logging.debug(f"Tag filtering (windows): {len(windows)} windows match query tags {list(query_tag_set)[:3]}...")

        if not window_embeddings_list:
            logging.error(f"No window embeddings found for video '{video_name}' and modality '{modality}'")
            return []

        # Perform cosine similarity between query and windows
        window_embeddings = torch.stack(window_embeddings_list)  # Shape: (N, D)
        
        # Ensure query_embedding is 2D: (1, D) for proper broadcasting
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)  # (D,) -> (1, D)
        
        # cosine_similarity with dim=-1 computes similarity along feature dimension
        sim_vector = cosine_similarity(query_embedding, window_embeddings, dim=-1)  # Shape: (N,)

        # topk on 1D tensor (N,) returns indices 0..N-1 corresponding to windows
        top_scores, top_indices = torch.topk(
            sim_vector, k=min(top_k, len(windows))
        )
        
        results = []
        for score, window_idx in zip(top_scores, top_indices):
            results.append((windows[window_idx.item()], score.item()))
        
        # Clean up GPU tensors
        del query_embedding, window_embeddings, sim_vector, top_scores, top_indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results

    def _retrieve_best_scenes_from_windows(
        self, 
        query: Query, 
        video_name: str, 
        windows: list[Window],
        modality: str, 
        top_k: int = 1
    ) -> types.TopKScenes:
        """
        Retrieves the best scenes from the given windows for the specified query.
        Only considers scenes that are contained in the provided windows.
        
        Args:
            query: the Query to use
            video_name: the video to retrieve from
            windows: list of Window objects to search within
            modality: the modality to use
            top_k: the number of scenes to retrieve
        """
        target_dp = None
        for dp in self.video_dataset.video_datapoints:
            if dp.video_name == video_name:
                target_dp = dp
                break
        
        if target_dp is None:
            raise RuntimeError(f"Video '{video_name}' not found")

        query_embedding = query.get_embedding(modality).to(self.device)

        # Collect scene IDs from all provided windows (deduplicate)
        candidate_scene_ids = set()
        for window in windows:
            candidate_scene_ids.update(window.scene_ids)
        
        # Filter scenes based on window membership and query tags
        query_tag_set = set([t.lower() for t in getattr(query, 'tags', []) or []])
        apply_tag_filter = self.use_tagging and len(query_tag_set) > 0
        
        scenes = []
        scene_embeddings_list = []

        for scene_id in candidate_scene_ids:
            if scene_id not in target_dp.scene_embeddings:
                continue
                
            scene_data = target_dp.scene_embeddings[scene_id]
            
            # Apply tag filtering if enabled
            if apply_tag_filter:
                scene_tags = scene_data.get("tags") or []
                scene_tag_set = set([t.lower() for t in scene_tags])
                if not any(t in scene_tag_set for t in query_tag_set):
                    continue

            emb = scene_data.get(modality, None)
            if emb is None:
                continue
                
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb)

            scene_obj = target_dp.get_scene_by_id(scene_id)
            if scene_obj is None:
                logging.warning(f"Scene {scene_id} has embedding but no Scene object in video {target_dp.video_name}")
                continue

            # EgoLife time/day filtering (if applicable)
            if not self._scene_passes_egolife_time_filter(query, target_dp, scene_obj, scene_data):
                continue
            
            scenes.append(scene_obj)
            scene_embeddings_list.append(emb.to(self.device))
        
        if apply_tag_filter and len(scenes) > 0:
            logging.debug(f"Tag filtering (windows): {len(scenes)} scenes match query tags {list(query_tag_set)[:3]}...")

        if not scene_embeddings_list:
            if modality == "audio" and hasattr(target_dp, 'has_audio') and not target_dp.has_audio:
                logging.debug(f"Video '{video_name}' has no audio track - skipping audio scene retrieval")
            else:
                logging.warning(f"No scene embeddings found in windows for video '{video_name}' and modality '{modality}'")
            return []

        # Perform cosine similarity between query and scenes from windows
        scene_embeddings = torch.stack(scene_embeddings_list)  # Shape: (N, D)
        
        # Ensure query_embedding is 2D: (1, D) for proper broadcasting
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)  # (D,) -> (1, D)
        
        # cosine_similarity with dim=-1 computes similarity along feature dimension
        sim_vector = cosine_similarity(query_embedding, scene_embeddings, dim=-1)  # Shape: (N,)

        # topk on 1D tensor (N,) returns indices 0..N-1 corresponding to scenes
        top_scores, top_indices = torch.topk(
            sim_vector, k=min(top_k, len(scenes))
        )
        
        results = []
        for score, scene_idx in zip(top_scores, top_indices):
            results.append((scenes[scene_idx.item()], score.item()))
        
        # Clean up GPU tensors
        del query_embedding, scene_embeddings, sim_vector, top_scores, top_indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Apply scene merging if enabled
        if self.scene_merger is not None:
            results = self.scene_merger.merge_top_k_scenes(
                results,
                score_aggregation=self.merge_score_aggregation
            )
        
        return results

    
    def _perform_tag_based_filtering_videos(
        self, 
        queries: QueryDataset,
    ):
        """
        Performs video-level filtering based on tags.
        
        If use_tagging is True:
            - Tags queries using QueryTagger (CLIP-based, same as VisionTagger)
            - Uses rewritten video modality from query decomposition
            - Returns only videos with at least one matching tag
        Otherwise:
            - Returns all videos for all queries (no filtering)
        
        Args:
            queries: the queries to retrieve against
        
        Returns:
            Dictionary mapping query_id to set of candidate video names
        """
        filtered = {}
        
        # Load QueryTagger if needed
        if self.use_tagging:
            if not self._tagger_loaded:
                logging.info("[Retriever] Loading CLIP-based QueryTagger for video filtering...")
                self.query_tagger.load_model()  # Same CLIP model as VisionTagger
                self._tagger_loaded = True
            
            # Tag all queries using CLIP (same space as video tags)
            logging.info(f"[Retriever] Tagging {len(queries)} queries with CLIP...")
            for query in queries:
                # Use the rewritten video modality (visual description) for better CLIP alignment
                qtext = query.decomposed.get("video", query.get_query())
                qtags = self.query_tagger.tag_query(qtext)  # CLIP-based tagging
                query.tags = [t.lower() for t in qtags]
                logging.debug(f"[Retriever] Query {query.qid}: '{qtext[:50]}...' -> tags: {query.tags}")
        else:
            # No tagging: set empty tags for all queries
            for query in queries:
                query.tags = []

        # Filter videos for each query
        for query in queries:
            candidates = set()
            
            if self.use_tagging and query.tags:
                # Apply tag-based filtering
                query_tag_set = set(query.tags)
                for dp in self.video_dataset.video_datapoints:
                    dp_tags = dp.global_embeddings.get("tags") or []
                    dp_tag_set = set([t.lower() for t in dp_tags])
                    # Include video if it has at least one matching tag
                    if any(t in dp_tag_set for t in query_tag_set):
                        candidates.add(dp.video_name)
                
                logging.info(f"[Retriever] Query {query.qid} tags={query.tags[:3]}{'...' if len(query.tags) > 3 else ''} -> {len(candidates)}/{len(self.video_dataset.video_datapoints)} candidate videos")
            else:
                # No filtering: include all videos
                candidates = {dp.video_name for dp in self.video_dataset.video_datapoints}
                if self.use_tagging:
                    logging.debug(f"[Retriever] Query {query.qid} has no tags, using all {len(candidates)} videos")
            
            filtered[query.qid] = candidates
    
        return filtered

    def retrieve_hierarchically(
        self,
        queries: QueryDataset,
        modalities: list[str] | str,
        top_k_videos: int = 3,
        top_k_windows: int = 2,
        top_k_scenes: int = 1,
        use_windows: bool = True,
        skip_video_retrieval: bool = False,
        use_tagging: bool = True,
        rewrite_queries: bool = False,
        use_visual_context: bool = False,
    ) -> types.RetrievalResults:
        """
        Given a set of queries, retrieves the top scenes and videos for those queries.
        Uses a hierarchical approach: Videos -> Windows -> Scenes
        
        Args:
            queries: the queries to answer
            modalities: the modalities to use for retrieval
            top_k_videos: the number of videos to look for
            top_k_windows: the number of windows to retrieve per video (only if use_windows=True)
            top_k_scenes: the number of scenes to extract from the selected videos/windows
            use_windows: whether to use intermediate window-level retrieval
            skip_video_retrieval: if True, skip video retrieval and use the ground truth video
                                  (query.video_uid) directly. Useful for testing scene-level 
                                  retrieval in isolation. Requires each query to have video_uid set.
        """

        if isinstance(modalities, str):
            modalities = [modalities]

        self.use_tagging = use_tagging
        self.rewrite_queries = rewrite_queries
        self.use_visual_context = use_visual_context

        # Build a per-query retrieval plan before rewriting so we preserve
        # the original temporal cues from the user query.
        self._plan_queries(
            queries=queries,
            modalities=modalities,
            use_windows=use_windows,
            rewrite_queries=rewrite_queries,
        )

        # Rewrite queries decomposed into modalities
        self._rewrite_queries(queries)

        # Filter videos based on tags
        candidate_videos_per_query = self._perform_tag_based_filtering_videos(queries)
        # Apply EgoLife day-level filtering if metadata is present
        candidate_videos_per_query = self._apply_egolife_video_filters(queries, candidate_videos_per_query)
        results = types.RetrievalResults()

        if skip_video_retrieval:
            for modality in modalities:
                logging.info(f"Embedding queries for modality {modality}...\n")
                self._load_models_for_modality(modality)
                self._embed_queries(queries)
            logging.info("Skipping video retrieval - using ground truth videos directly for scene-only evaluation")
            
            # Validate that all queries have video_uid
            missing_video_uid = [q.qid for q in queries if not q.video_uid]
            if missing_video_uid:
                raise ValueError(
                    f"skip_video_retrieval=True requires all queries to have video_uid set. "
                    f"Missing for: {missing_video_uid[:5]}{'...' if len(missing_video_uid) > 5 else ''}"
                )
            
            # Set the ground truth video as the only "retrieved" video with score 1.0
            for query in queries:
                results.results[query.qid] = {}
                for modality in modalities:
                    results.results[query.qid][modality] = [(query.video_uid, 1.0)]
                results.results[query.qid]["fused"] = [(query.video_uid, 1.0)]
        else:
            # Step 1: Extract relevant videos
            logging.info(f"Step 1: Retrieving top {top_k_videos} videos globally...")
            results.add_top_level(
                top_level_results = self._retrieve_top_videos_per_queries(
                    queries=queries, 
                    candidate_videos = candidate_videos_per_query,
                    modalities=modalities, 
                    top_k=top_k_videos
                )
            )
            # Perform fusion of video rankings per query
            for query in queries:
                query_modalities = self._get_modalities_for_query(query, modalities)
                fused_video_ranking = self.fuser.fuse(
                    {modality: results[query.qid].get(modality, []) for modality in query_modalities}
                )
                results[query.qid]["fused"] = fused_video_ranking[:top_k_videos]
            
        # Retrieve the top scenes within the top videos
        detailed_results = {query.qid: [] for query in queries}

        logging.info("Step 2: Retrieving scenes using the per-query orchestration plan...")

        for query in queries:
            fused_video_list = results[query.qid]["fused"]
            query_modalities = self._get_modalities_for_query(query, modalities)
            query_use_windows = self._query_uses_windows(query, use_windows)

            if query_use_windows:
                all_windows_with_scores: list[tuple[str, Window, float]] = []

                for video_name, global_score in fused_video_list:
                    modality_window_rankings = {}
                    for modality in query_modalities:
                        modality_window_rankings[modality] = self._retrieve_best_windows(
                            query=query,
                            video_name=video_name,
                            modality=modality,
                            top_k=top_k_windows,
                        )

                    has_windows = any(len(rankings) > 0 for rankings in modality_window_rankings.values())
                    if has_windows:
                        fused_window_ranking = self.fuser.fuse(modality_window_rankings)
                        for window, score in fused_window_ranking:
                            all_windows_with_scores.append((video_name, window, score))

                all_windows_with_scores.sort(key=lambda x: x[2], reverse=True)
                top_global_windows = all_windows_with_scores[:top_k_windows]

                all_scenes_with_scores: list[tuple[str, Scene, float]] = []
                if top_global_windows:
                    windows_by_video: dict[str, list[Window]] = {}
                    for video_name, window, score in top_global_windows:
                        if video_name not in windows_by_video:
                            windows_by_video[video_name] = []
                        windows_by_video[video_name].append(window)

                    for video_name, selected_windows in windows_by_video.items():
                        modality_scene_rankings = {}
                        for modality in query_modalities:
                            modality_scene_rankings[modality] = self._retrieve_best_scenes_from_windows(
                                query=query,
                                video_name=video_name,
                                windows=selected_windows,
                                modality=modality,
                                top_k=top_k_scenes * 2,
                            )
                        fused_scene_ranking = self.fuser.fuse(modality_scene_rankings)
                        for scene, score in fused_scene_ranking:
                            all_scenes_with_scores.append((video_name, scene, score))
                else:
                    logging.warning(f"No windows found for query {query.qid}, falling back to direct scene retrieval")
                    for video_name, video_score in fused_video_list:
                        modality_scene_rankings = {}
                        for modality in query_modalities:
                            modality_scene_rankings[modality] = self._retrieve_best_scenes(
                                query=query,
                                video_name=video_name,
                                modality=modality,
                                top_k=top_k_scenes * 2,
                            )
                        fused_scene_ranking = self.fuser.fuse(modality_scene_rankings)
                        for scene, score in fused_scene_ranking:
                            all_scenes_with_scores.append((video_name, scene, score))
            else:
                all_scenes_with_scores = []
                for video_name, global_score in fused_video_list:
                    modality_scene_rankings = {}
                    for modality in query_modalities:
                        modality_scene_rankings[modality] = self._retrieve_best_scenes(
                            query=query,
                            video_name=video_name,
                            modality=modality,
                            top_k=top_k_scenes * 2,
                        )
                    fused_scene_ranking = self.fuser.fuse(modality_scene_rankings)
                    for scene, score in fused_scene_ranking:
                        all_scenes_with_scores.append((video_name, scene, score))

            all_scenes_with_scores.sort(key=lambda x: x[2], reverse=True)
            top_global_scenes = all_scenes_with_scores[:top_k_scenes]

            video_scenes: dict[str, list[tuple[Scene, float]]] = {}
            for video_name, scene, score in top_global_scenes:
                if video_name not in video_scenes:
                    video_scenes[video_name] = []
                video_scenes[video_name].append((scene, score))

            video_scores = {v: s for v, s in fused_video_list}
            for video_name, scenes in video_scenes.items():
                global_score = video_scores.get(video_name, 0.0)
                detailed_results[query.qid].append(
                    (video_name, global_score, scenes)
                )
        
        results.add_detailed_results(detailed_results)

        return results
