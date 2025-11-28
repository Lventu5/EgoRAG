import torch
import logging
import numpy as np
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
from torch.nn.functional import normalize

from data.video_dataset import VideoDataset, Window, Scene
from data.query import Query, QueryDataset
import data.datatypes as types
from retrieval.rewriter import QueryRewriterLLM
from .fuser import Fuser
from .scene_merger import SceneMerger
from configuration.config import CONFIG
from indexing.components.tagger import Tagger

import sys
import os
from pathlib import Path
sys.path.append(os.path.join(Path(os.getcwd()), 'external/InternVideo/InternVideo2'))
sys.path.append(os.path.join(Path(os.getcwd()), 'external/InternVideo/InternVideo2/multi_modality'))
import interface


class HierarchicalRetriever:
    def __init__(
        self, 
        video_dataset: VideoDataset,
        fuser: Fuser | None = None,
        device: str = "cuda",
        use_tagging: bool = False,
    ):
        self.video_dataset = video_dataset
        self.device = device
        logging.info(f"Retriever running on {self.device}")
        video_model_name = CONFIG.retrieval.video_model_id
        audio_model_name = CONFIG.retrieval.audio_model_id
        text_model_name = CONFIG.retrieval.text_model_id
        
        # Determine video model type from model name or indexing config
        if "InternVideo2" in video_model_name or CONFIG.indexing.video.model_name == "internvideo2":
            self.video_model_type = "internvideo2"
        else:
            self.video_model_type = CONFIG.indexing.video.model_name  # "xclip" or other
        
        # Set video embedding size based on model type
        if self.video_model_type == "internvideo2":
            video_embed_size = 512  # InternVideo2 outputs 512-dim embeddings
        else:  # xclip
            video_embed_size = 512  # XCLIP outputs 512-dim embeddings
        
        self.sizes = {
            "video": {
                "size": video_embed_size,  
                "model": video_model_name
            },
            "audio": {
                "size": 512,
                "model": audio_model_name
            },
            "text": {
                "size": 768,
                "model": text_model_name
            },
        }
        self.rewriter = CONFIG.retrieval.rewriter_model_id
        self.current_modality = None
        self.processor = None
        self.embedder = None
        self.use_tagging = use_tagging
        logging.info(f"[INFO] Use of tagging set to {use_tagging}")
        
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

        # Tagger for query and dataset filtering (lazy load)
        self.tagger = Tagger(device=self.device)
        self._tagger_loaded = False


    def _load_models_for_modality(self, modality: str):
        """
        Loads the model specified by a specific modality 
        Args:
            modality: the modality to load the model of
        """
        if self.current_modality == modality:
            return
        
        self.processor = None
        self.embedder = None
        target_modality = modality

        # Text encoder
        if target_modality == "text":
            self.embedder = SentenceTransformer(
                self.sizes["text"]["model"], device=self.device
            )
        
        # Video encoder
        elif target_modality == "video":
            if self.video_model_type == "internvideo2":
                # model_name = self.sizes["video"]["model"]
                # logging.info(f"Loading InternVideo2 model for retrieval: {model_name}")
                # self.embedder = AutoModel.from_pretrained(
                    # model_name,
                    # trust_remote_code=True
                # ).to(self.device).eval()

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
        rewriter = QueryRewriterLLM(
            model_name=self.rewriter, 
            device=self.device
        )
        for query in queries:
            decomposition = rewriter(query.get_query(), modality="decompose")
            query.decomposed = decomposition


    def _embed_queries(self, queries: QueryDataset) -> torch.Tensor:
        """
        Embeds the queries in the various modalities
        """
        if self.embedder is None:
            raise RuntimeError("No model loaded for embedding. Call _load_models_for_modality first.")
        
        mod_queries = queries.group_by_modality(self.current_modality)

        if self.current_modality == "text":
            embeddings = self.embedder.encode(
                mod_queries, convert_to_tensor=True, device=self.device
            )
        elif self.current_modality == "video":
            if self.video_model_type == "xclip":
                inputs = self.processor(
                    text=mod_queries, return_tensors="pt", padding=True # type: ignore
                ).to(self.device)
                with torch.no_grad():
                    embeddings = self.embedder.get_text_features(**inputs) # type: ignore
            
            elif self.video_model_type == "internvideo2":
                # InternVideo2 uses get_txt_feat for text encoding
                # embeddings_list = []
                # for query_text in mod_queries:
                    # text_feat = self.embedder.get_txt_feat(query_text)
                    # embeddings_list.append(text_feat.squeeze(0))
                # embeddings = torch.stack(embeddings_list)

                # Internvideo 1B
                embeddings = interface.extract_query_features(
                     # base_dataset, 
                     mod_queries,
                     self.embedder
                )
            else:
                raise ValueError(f"Model type not supported: {self.video_model_type}")        
            
        elif self.current_modality == "audio":
            inputs = self.processor(
                text=mod_queries, return_tensors="pt", padding=True # type: ignore
            ).to(self.device)
            with torch.no_grad():
                embeddings = self.embedder.get_text_features(**inputs) # type: ignore
        else:
            raise ValueError(f"Unknown modality: {self.current_modality}")
        
        for query, emb in zip(queries, embeddings):
            query.embeddings[self.current_modality] = emb.cpu()

        return embeddings

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

    def _retrieve_by_modality(
        self, 
        queries: QueryDataset, 
        candidate_videos: dict[str, str],
        modality: str,
        top_k: int = 1
    ) -> types.TopKVideosPerModality:
        """
        Retrieves the top-k videos for a given modality, and returns them in a format
        List (each element corresponds to a query) of lists (each element corresponds to one of
        the top-k elements) of tuples (video, score)
        Args:
            queries: The QueryDataset to use
            candidate_videos: dict with the query id and the name of the videos available for that query
            modality: the modality to use for retrieval
            top_k: the number of videos to retrieve
        """
        logging.info(f"Retrieving top {top_k} results for modality '{modality}'")

        query_embeddings = queries.embeddings_by_modality(modality).to(self.device)

        all_results = []

        # Build a mapping from video_name -> datapoint for fast lookup
        dp_by_name = {dp.video_name: dp for dp in self.video_dataset.video_datapoints}

        # For each query, use precomputed candidate_videos to
        # avoid repeated tag inference and to perform the screening exactly once.
        for q_idx, query in enumerate(queries):
            q_emb = query_embeddings[q_idx].unsqueeze(0)  # shape (1, D)

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
                continue

            # Log info about videos without modality for this query
            if videos_without_modality and modality == "audio":
                logging.info(f"{len(videos_without_modality)} video(s) have no audio track and will be excluded from audio-based retrieval")

            # Create the database for our queries
            db = torch.stack(candidate_embs).to(self.device)

            # Normalize all embeddings and perform cosine similarity
            qn = normalize(q_emb, p=2, dim=-1)
            dbn = normalize(db, p=2, dim=-1)

            scores = torch.matmul(qn, dbn.T).squeeze(0)
            
            # Debug logging for video-level retrieval
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"[Video Retrieval] query={query.qid}, modality={modality}")
                logging.debug(f"  query_emb shape: {q_emb.shape}, candidates: {len(candidate_names)}")
                logging.debug(f"  scores: min={scores.min().item():.4f}, max={scores.max().item():.4f}, mean={scores.mean().item():.4f}")

            topk = min(top_k, len(candidate_names))
            top_scores, top_indices = torch.topk(scores, k=topk)

            query_results = []
            for score, idx in zip(top_scores, top_indices):
                query_results.append((candidate_names[idx.item()], score.item()))

            all_results.append(query_results)

        return all_results

    def _filter_scenes_by_query_tags(self, query: Query, target_dp, modality: str):
        """
        Helper that returns (scenes, scene_embeddings_list) for a given video
        filtered by precomputed scene tags present in `target_dp.scene_embeddings`.
        Args:
            query.tags` is expected to be a list of lowercase tags (may be empty).
            If `query.tags` is empty, all scenes that have an embedding for the
            requested modality are returned.
            Scenes lacking precomputed `tags` will be excluded when `query.tags`
            is non-empty (no on-demand tagging performed here).
        """
        query_tag_set = set([t.lower() for t in getattr(query, 'tags', []) or []])

        scenes = []
        scene_embeddings_list = []

        for scene_id, scene_data in target_dp.scene_embeddings.items():
            if query_tag_set:
                scene_tags = scene_data.get("tags") or []
                scene_tag_set = set([t.lower() for t in scene_tags])
                if not any(t in scene_tag_set for t in query_tag_set):
                    if self.use_tagging:
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
            
            scenes.append(scene_obj)
            scene_embeddings_list.append(emb.to(self.device))

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

        # Perform cosine similarity between scenes and queries
        scene_embeddings = torch.stack(scene_embeddings_list)

        query_embedding_norm = normalize(query_embedding, p=2, dim=-1)
        scene_embeddings_norm = normalize(scene_embeddings, p=2, dim=-1)

        sim_vector = torch.matmul(query_embedding_norm, scene_embeddings_norm.T).squeeze(0)
        
        # Debug logging for similarity scores
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"[Scene Retrieval] video={video_name}, modality={modality}")
            logging.debug(f"  query_emb shape: {query_embedding.shape}, norm: {query_embedding.norm().item():.4f}")
            logging.debug(f"  scene_embs shape: {scene_embeddings.shape}")
            logging.debug(f"  sim_vector: min={sim_vector.min().item():.4f}, max={sim_vector.max().item():.4f}, mean={sim_vector.mean().item():.4f}")

        top_scores, top_indices = torch.topk(
            sim_vector, k=min(top_k, len(scenes))
        )
        results = []
        for score, scene_idx in zip(top_scores, top_indices):
            results.append((scenes[scene_idx.item()], score.item())) # type: ignore
        
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

        # Collect window embeddings for the requested modality
        windows = []
        window_embeddings_list = []
        
        for window in target_dp.windows:
            wid = window.window_id
            if wid in target_dp.window_embeddings:
                emb = target_dp.window_embeddings[wid].get(modality)
                if emb is not None:
                    if not isinstance(emb, torch.Tensor):
                        emb = torch.tensor(emb)
                    windows.append(window)
                    window_embeddings_list.append(emb.to(self.device))
            else:
                logging.warning(f"Window id {wid} not found in {target_dp.window_embeddings.keys()}")

        if not window_embeddings_list:
            logging.error(f"No window embeddings found for video '{video_name}' and modality '{modality}'")
            return []

        # Perform cosine similarity between windows and query
        window_embeddings = torch.stack(window_embeddings_list)

        query_embedding_norm = normalize(query_embedding, p=2, dim=-1)
        window_embeddings_norm = normalize(window_embeddings, p=2, dim=-1)

        sim_vector = torch.matmul(query_embedding_norm, window_embeddings_norm.T).squeeze(0)

        top_scores, top_indices = torch.topk(
            sim_vector, k=min(top_k, len(windows))
        )
        
        results = []
        for score, window_idx in zip(top_scores, top_indices):
            results.append((windows[window_idx.item()], score.item()))
        
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
        
        scenes = []
        scene_embeddings_list = []

        for scene_id in candidate_scene_ids:
            if scene_id not in target_dp.scene_embeddings:
                continue
                
            scene_data = target_dp.scene_embeddings[scene_id]
            
            # Apply tag filtering if query has tags
            if query_tag_set:
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
            
            scenes.append(scene_obj)
            scene_embeddings_list.append(emb.to(self.device))

        if not scene_embeddings_list:
            if modality == "audio" and hasattr(target_dp, 'has_audio') and not target_dp.has_audio:
                logging.debug(f"Video '{video_name}' has no audio track - skipping audio scene retrieval")
            else:
                logging.warning(f"No scene embeddings found in windows for video '{video_name}' and modality '{modality}'")
            return []

        # Perform cosine similarity between scenes and query
        scene_embeddings = torch.stack(scene_embeddings_list)

        query_embedding_norm = normalize(query_embedding, p=2, dim=-1)
        scene_embeddings_norm = normalize(scene_embeddings, p=2, dim=-1)

        sim_vector = torch.matmul(query_embedding_norm, scene_embeddings_norm.T).squeeze(0)

        top_scores, top_indices = torch.topk(
            sim_vector, k=min(top_k, len(scenes))
        )
        
        results = []
        for score, scene_idx in zip(top_scores, top_indices):
            results.append((scenes[scene_idx.item()], score.item()))
        
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
        Takes queries as an input and scans the video dataset looking for videos containing the same tags
        Args:
            queries: the queries to retrieve against
        """
        filtered = {}
        if self.use_tagging:
            if not self._tagger_loaded:
                self.tagger.load_model()
                self._tagger_loaded = True

        for query in queries:
            if self.use_tagging:
                qtext = query.get_query()
                qtags = []
                qtags = self.tagger.infer_tags_from_text(qtext)
                query.tags = [t.lower() for t in qtags]

            candidates = set()
            for dp in self.video_dataset.video_datapoints:
                if self.use_tagging:
                    dp_tags = dp.global_embeddings.get("tags") or []
                    dp_tag_set = set([t.lower() for t in (dp_tags or [])])
                    if any(t in dp_tag_set for t in query.tags):
                        candidates.add(dp.video_name)
                else:
                    candidates.add(dp.video_name)
            filtered[query.qid] = candidates
            logging.debug(f"[Retriever] Query {query.qid} tags={query.tags} -> {len(filtered[query.qid])} candidate videos")
    
        return filtered

    def retrieve_hierarchically(
        self,
        queries: QueryDataset,
        modalities: list[str] | str,
        top_k_videos: int = 3,
        top_k_windows: int = 2,
        top_k_scenes: int = 1,
        use_windows: bool = True
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
        """

        if isinstance(modalities, str):
            modalities = [modalities]

        # Rewrite queries decomposed into modalities
        self._rewrite_queries(queries)

        # Filter videos based on tags
        candidate_videos_per_query = self._perform_tag_based_filtering_videos(queries)
        results = types.RetrievalResults()

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
            fused_video_ranking = self.fuser.fuse(results[query.qid])
            results[query.qid]["fused"] = fused_video_ranking[:top_k_videos]
            
        # Retrieve the top scenes within the top videos
        detailed_results = {query.qid: [] for query in queries}

        if use_windows:
            # Hierarchical retrieval: Videos -> Windows -> Scenes
            logging.info(f"Step 2: Retrieving top {top_k_windows} windows within top videos...")
            logging.info(f"Step 3: Retrieving top {top_k_scenes} scenes GLOBALLY across all videos...")
            
            for query in queries:
                fused_video_list = results[query.qid]["fused"]
                # Collect ALL scenes from ALL top videos, then rank globally
                all_scenes_with_scores: list[tuple[str, Scene, float]] = []  # (video_name, scene, score)
                
                for video_name, global_score in fused_video_list:
                    # Step 2: Retrieve top windows within each video
                    modality_window_rankings = {}
                    for modality in modalities:
                        modality_window_rankings[modality] = self._retrieve_best_windows(
                            query=query,
                            video_name=video_name,
                            modality=modality,
                            top_k=top_k_windows
                        )
                    
                    # Check if any windows were found
                    has_windows = any(
                        len(rankings) > 0 
                        for rankings in modality_window_rankings.values()
                    )
                    
                    if has_windows:
                        # Fuse window rankings across modalities
                        fused_window_ranking = self.fuser.fuse(modality_window_rankings)
                        top_windows = [window for window, score in fused_window_ranking[:top_k_windows]]
                        
                        # Step 3: Retrieve scenes from the selected windows (get more for global ranking)
                        modality_scene_rankings = {}
                        for modality in modalities:
                            modality_scene_rankings[modality] = self._retrieve_best_scenes_from_windows(
                                query=query,
                                video_name=video_name,
                                windows=top_windows,
                                modality=modality,
                                top_k=top_k_scenes * 2  # Get more candidates for global ranking
                            )
                        fused_scene_ranking = self.fuser.fuse(modality_scene_rankings)
                    else:
                        # Fallback to direct scene retrieval if no windows available
                        logging.debug(f"No windows found for video {video_name}, falling back to direct scene retrieval")
                        modality_scene_rankings = {}
                        for modality in modalities:
                            modality_scene_rankings[modality] = self._retrieve_best_scenes(
                                query=query,
                                video_name=video_name,
                                modality=modality,
                                top_k=top_k_scenes * 2  # Get more candidates for global ranking
                            )
                        fused_scene_ranking = self.fuser.fuse(modality_scene_rankings)

                    # Collect all scenes with their scores and video name
                    for scene, score in fused_scene_ranking:
                        all_scenes_with_scores.append((video_name, scene, score))
                
                # Global ranking: sort ALL scenes by score and take top_k_scenes
                all_scenes_with_scores.sort(key=lambda x: x[2], reverse=True)
                top_global_scenes = all_scenes_with_scores[:top_k_scenes]
                
                # Group by video for the detailed_results format
                video_scenes: dict[str, list[tuple[Scene, float]]] = {}
                for video_name, scene, score in top_global_scenes:
                    if video_name not in video_scenes:
                        video_scenes[video_name] = []
                    video_scenes[video_name].append((scene, score))
                
                # Store in detailed_results with original video scores
                video_scores = {v: s for v, s in fused_video_list}
                for video_name, scenes in video_scenes.items():
                    global_score = video_scores.get(video_name, 0.0)
                    detailed_results[query.qid].append(
                        (video_name, global_score, scenes)
                    )
        else:
            # Direct scene retrieval (original behavior): Videos -> Scenes
            logging.info(f"Step 2: Retrieving top {top_k_scenes} scenes GLOBALLY across all videos...")
            for query in queries:
                fused_video_list = results[query.qid]["fused"]
                # Collect ALL scenes from ALL top videos, then rank globally
                all_scenes_with_scores: list[tuple[str, Scene, float]] = []  # (video_name, scene, score)
                
                for video_name, global_score in fused_video_list:
                    modality_scene_rankings = {}
                    for modality in modalities:
                        modality_scene_rankings[modality] = self._retrieve_best_scenes(
                            query=query,
                            video_name=video_name,
                            modality=modality,
                            top_k=top_k_scenes * 2  # Get more candidates for global ranking
                        )
                    fused_scene_ranking = self.fuser.fuse(modality_scene_rankings)

                    # Collect all scenes with their scores and video name
                    for scene, score in fused_scene_ranking:
                        all_scenes_with_scores.append((video_name, scene, score))
                
                # Global ranking: sort ALL scenes by score and take top_k_scenes
                all_scenes_with_scores.sort(key=lambda x: x[2], reverse=True)
                top_global_scenes = all_scenes_with_scores[:top_k_scenes]
                
                # Group by video for the detailed_results format
                video_scenes: dict[str, list[tuple[Scene, float]]] = {}
                for video_name, scene, score in top_global_scenes:
                    if video_name not in video_scenes:
                        video_scenes[video_name] = []
                    video_scenes[video_name].append((scene, score))
                
                # Store in detailed_results with original video scores
                video_scores = {v: s for v, s in fused_video_list}
                for video_name, scenes in video_scenes.items():
                    global_score = video_scores.get(video_name, 0.0)
                    detailed_results[query.qid].append(
                        (video_name, global_score, scenes)
                    )
        
        results.add_detailed_results(detailed_results)

        return results
