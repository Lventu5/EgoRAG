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
    LlavaNextVideoForConditionalGeneration,
    CLIPProcessor,
    CLIPTextModel,
    AutoTokenizer,
)
from sentence_transformers import SentenceTransformer
from torch.nn.functional import normalize
from utils.cache_manager import setup_smart_cache

from data.video_dataset import VideoDataset
from data.query import Query, QueryDataset
import data.datatypes as types
from retrieval.rewriter import QueryRewriterLLM
from .fuser import Fuser
from .scene_merger import SceneMerger
from configuration.config import CONFIG

sys.path.append(os.path.join(Path(os.getcwd()), 'external/InternVideo/InternVideo2'))
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
        video_model_name = CONFIG.retrieval.video_model_id
        audio_model_name = CONFIG.retrieval.audio_model_id
        text_model_name = CONFIG.retrieval.text_model_id
        caption_model_name = CONFIG.retrieval.caption_model_id
        
        # Determine video model type from model name or indexing config
        if "InternVideo2" in video_model_name or CONFIG.indexing.video.model_name == "internvideo2":
            self.video_model_type = "internvideo2"
        elif "Qwen" in video_model_name or CONFIG.indexing.video.model_name == "qwen2-vl":
            self.video_model_type = "qwen2-vl"
        else:
            self.video_model_type = CONFIG.indexing.video.model_name  # "xclip" or other
        
        # Set video embedding size based on model type
        if self.video_model_type == "internvideo2":
            video_embed_size = 512  # InternVideo2 outputs 512-dim embeddings
        elif self.video_model_type == "qwen2-vl":
            video_embed_size = 1024  # Qwen2-VL vision embeddings
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
            "caption": {
                "size": 384,
                "model": caption_model_name
            }
        }
        self.rewriter = CONFIG.retrieval.rewriter_model_id
        self.current_modality = None
        self.processor = None
        self.embedder = None
        self._fast_cache_setup_done = False
        
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

    def _load_models_for_modality(self, modality: str):
        if self.current_modality == modality:
            return
        
        self.processor = None
        self.embedder = None
        target_modality = modality

        # Ensure fast cache is configured once before loading heavy models
        if not self._fast_cache_setup_done:
            try:
                setup_smart_cache(verbose=True)
            except Exception:
                logging.warning("Fast cache setup failed or skipped")
            self._fast_cache_setup_done = True

        if target_modality == "text" or target_modality == "caption":
            self.embedder = SentenceTransformer(
                self.sizes["text"]["model"], device=self.device
            )
        
        elif target_modality == "video":
            if self.video_model_type == "qwen2-vl":
                    qwen_id = getattr(CONFIG.indexing.video, "qwen2_vl_id", None) or CONFIG.retrieval.video_model_id
                    logging.info(f"Loading Qwen2-VL processor+model: {qwen_id}")
                    self.processor = AutoProcessor.from_pretrained(qwen_id)
                    self.tokenizer = AutoTokenizer.from_pretrained(qwen_id, use_fast=True)

                    self.embedder = AutoModel.from_pretrained(qwen_id).to(self.device).eval()
            
            elif self.video_model_type == "internvideo2":
                # model_name = self.sizes["video"]["model"]
                # logging.info(f"Loading InternVideo2 model for retrieval: {model_name}")
                # self.embedder = AutoModel.from_pretrained(
                #     model_name,
                #     trust_remote_code=True
                # ).to(self.device).eval()

                config_path = "external/InternVideo/InternVideo2/multi_modality/demo/internvideo2_stage2_config.py"
                model_path = "external/InternVideo/InternVideo2/checkpoints/internvideo2_stage2_1b.pth"

                model = interface.load_model(config_path, model_path)
                self.embedder = model.to(self.device).eval()

            else:  # xclip
                model_name = self.sizes["video"]["model"]
                logging.info(f"Loading XCLIP model for retrieval: {model_name}")
                self.processor = XCLIPProcessor.from_pretrained(model_name)
                self.embedder = XCLIPModel.from_pretrained(model_name).to(self.device) # type: ignore

        elif target_modality == "audio":
            model_name = self.sizes["audio"]["model"]
            logging.info(f"Loading CLAP model for audio retrieval: {model_name}")
            self.processor = ClapProcessor.from_pretrained(model_name)
            self.embedder = ClapModel.from_pretrained(model_name).to(self.device) # type: ignore
    
        else:
            raise ValueError(f"Unknown modality: {modality}")

        self.current_modality = target_modality

    def _rewrite_queries(self, queries: QueryDataset):
        rewriter = QueryRewriterLLM(
            model_name=self.rewriter, 
            device=self.device
        )

        for query in queries:
            decomposition = rewriter(query.get_query(), modality="decompose")
            query.decomposed = decomposition


    def _embed_queries(self, queries: QueryDataset) -> torch.Tensor:

        if self.embedder is None:
            raise RuntimeError("No model loaded for embedding. Call _load_models_for_modality first.")
        
        mod_queries = queries.group_by_modality(self.current_modality)

        if self.current_modality == "text" or self.current_modality == "caption":
            embeddings = self.embedder.encode(
                mod_queries, convert_to_tensor=True, device=self.device
            ) # type: ignore
        elif self.current_modality == "video":
            if self.video_model_type == "qwen2-vl":
                if self.embedder is None:
                    raise RuntimeError("Qwen model not loaded. Call _load_models_for_modality first.")

                # Prefer processor if available, otherwise fallback to tokenizer
                proc_inputs = None
                if hasattr(self, "processor") and self.processor is not None:
                    proc_inputs = self.processor(text=mod_queries, return_tensors="pt", padding=True, truncation=True)

                if proc_inputs is None and hasattr(self, "tokenizer") and self.tokenizer is not None:
                    proc_inputs = self.tokenizer(mod_queries, return_tensors="pt", padding=True, truncation=True)

                if proc_inputs is None:
                    raise RuntimeError("No tokenizer/processor available for Qwen text encoding")

                # Filter out any visual inputs (pixel_values / pixel_values_videos etc.)
                allowed_keys = {"input_ids", "attention_mask", "token_type_ids", "position_ids", "labels", "decoder_input_ids", "decoder_attention_mask"}
                inputs = {k: v.to(self.device) for k, v in proc_inputs.items() if k in allowed_keys}

                with torch.no_grad():
                    out = self.embedder(**inputs)

                # Prefer pooler_output if provided, otherwise mean-pool last_hidden_state
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    embeddings = out.pooler_output
                else:
                    last_hidden = out.last_hidden_state
                    attention_mask = inputs.get("attention_mask", None)
                    if attention_mask is not None:
                        mask = attention_mask.unsqueeze(-1)
                        embeddings = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                    else:
                        embeddings = last_hidden[:, 0, :]
                    
            elif self.video_model_type == "xclip":
                inputs = self.processor(
                    text=mod_queries, return_tensors="pt", padding=True # type: ignore
                ).to(self.device)
                with torch.no_grad():
                    embeddings = self.embedder.get_text_features(**inputs) # type: ignore
            
            elif self.video_model_type == "internvideo2":
                # InternVideo2 uses get_txt_feat for text encoding
                # embeddings_list = []
                # for query_text in mod_queries:
                #     text_feat = self.embedder.get_txt_feat(query_text)
                #     embeddings_list.append(text_feat.squeeze(0))
                # embeddings = torch.stack(embeddings_list)

                # Internvideo 1B
                embeddings = interface.extract_query_features(
                    base_dataset, 
                    model
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

    def retrieve_queries_list(
        self, 
        queries: QueryDataset,
        modalities: list[str] | str,
        top_k: int = 1
    )-> types.TopKVideosPerQuery:
        if isinstance(modalities, str):
            modalities = [modalities]

        final_results = {query.qid: {} for query in queries}

        for modality in modalities:
            self._load_models_for_modality(modality)
            self._embed_queries(queries)
            modality_results_batch = self._retrieve_for_modality(
                queries, modality, top_k
            )
            for i, query in enumerate(queries):
                final_results[query.qid][modality] = modality_results_batch[i]
                
        return final_results

    def _retrieve_for_modality(
        self, 
        queries: QueryDataset, 
        modality: str,
        top_k: int = 1
    ) -> types.TopKVideosPerModality:
        """
        Retrieves the top-k videos for a given modality, and returns them in a format
        List (each element corresponds to a query) of lists (each element corresponds to one of
        the top-k elements) of tuples (video, score)
        """
        logging.info(f"Retrieving top {top_k} results for modality '{modality}'")

        query_embeddings = queries.embeddings_by_modality(modality).to(self.device)
        
        video_names = []
        db_embeddings_list = []
        videos_without_modality = []
        
        for dp in self.video_dataset.video_datapoints:
            emb = dp.global_embeddings.get(modality, None)
            if emb is not None:
                video_names.append(dp.video_name)
                db_embeddings_list.append(emb)
            else:
                videos_without_modality.append(dp.video_name)
        
        if not db_embeddings_list:
            logging.warning(f"No embeddings available for modality '{modality}'")
            return [[] for _ in range(query_embeddings.shape[0])]
        
        # Log information about videos without this modality
        if videos_without_modality and modality == "audio":
            logging.info(f"{len(videos_without_modality)} video(s) have no audio track and will be excluded from audio-based retrieval")
        elif videos_without_modality:
            logging.warning(f"{len(videos_without_modality)} video(s) missing '{modality}' embeddings")

        db_embeddings = torch.stack(db_embeddings_list).to(self.device)
        
        query_embeddings_norm = normalize(query_embeddings, p=2, dim=-1)
        db_embeddings_norm = normalize(db_embeddings, p=2, dim=-1)
        
        sim_matrix = torch.matmul(query_embeddings_norm, db_embeddings_norm.T)
        
        all_results = []
        for query_idx in range(sim_matrix.shape[0]):
            scores = sim_matrix[query_idx]
            
            top_scores, top_indices = torch.topk(
                scores, k=min(top_k, len(video_names))
            )
            
            query_results = []
            for score, db_idx in zip(top_scores, top_indices):
                query_results.append((video_names[db_idx.item()], score.item())) # type: ignore
            
            all_results.append(query_results)
            
        return all_results
    
    def retrieve_best_scene(
        self, 
        query: Query, 
        video_name: str, 
        modality: str, 
        top_k: int = 1
    ) -> types.TopKScenes:
        """
        Gets the best scene in a video (we already know it is a top-k video) for the specified query
        Returns a list with the top k scenes and their similarity score
        """
        target_dp = None
        for dp in self.video_dataset.video_datapoints:
            if dp.video_name == video_name:
                target_dp = dp
                break
        
        if target_dp is None:
            raise RuntimeError(f"Video '{video_name}' not found")

        query_embedding = query.get_embedding(modality).to(self.device)

        scenes = []
        scene_embeddings_list = []
        
        for scene_id, scene_data in target_dp.scene_embeddings.items():
            emb = scene_data.get(modality, None) 
            if emb is not None:
                if not isinstance(emb, torch.Tensor):
                   try:
                       emb = torch.tensor(emb)
                   except Exception as e:
                       logging.warning(f"Unable to convert embedding of scene {scene_id} to tensor: {e}. Skipping.")
                       continue
                scenes.append(target_dp.get_scene_by_id(scene_id))
                scene_embeddings_list.append(emb.to(self.device))
        
        if not scene_embeddings_list:
            # Check if this is expected (e.g., video without audio)
            if modality == "audio" and hasattr(target_dp, 'has_audio') and not target_dp.has_audio:
                logging.debug(f"Video '{video_name}' has no audio track - skipping audio scene retrieval")
            else:
                logging.error(f"No scene embeddings found for video '{video_name}' and modality '{modality}'")
            return []

        scene_embeddings = torch.stack(scene_embeddings_list)

        query_embedding_norm = normalize(query_embedding, p=2, dim=-1)
        scene_embeddings_norm = normalize(scene_embeddings, p=2, dim=-1)

        sim_vector = torch.matmul(query_embedding_norm, scene_embeddings_norm.T).squeeze(0)

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


    def retrieve_hierarchically(
        self,
        queries: QueryDataset,
        modalities: list[str] | str,
        top_k_videos: int = 3, 
        top_k_scenes: int = 1  
    ) -> types.RetrievalResults:

        if isinstance(modalities, str):
            modalities = [modalities]

        self._rewrite_queries(queries)
        results = types.RetrievalResults()

        logging.info(f"Step 1: Retrieving top {top_k_videos} videos globally...")
        results.add_top_level(
            top_level_results = self.retrieve_queries_list(
                queries=queries, 
                modalities=modalities, 
                top_k=top_k_videos
            )
        )

        for query in queries:
            fused_video_ranking = self.fuser.fuse(results[query.qid])
            results[query.qid]["fused"] = fused_video_ranking[:top_k_videos]
            
        
        detailed_results = {query.qid: [] for query in queries}

        logging.info(f"Step 2: Retrieving top {top_k_scenes} scenes within top videos...")
        for query in queries:
            fused_video_list = results[query.qid]["fused"]
            for video_name, global_score in fused_video_list:
                modality_scene_rankings = {}
                for modality in modalities:
                    modality_scene_rankings[modality] = self.retrieve_best_scene(
                        query=query,
                        video_name=video_name,
                        modality=modality,
                        top_k=top_k_scenes
                    )
                fused_scene_ranking = self.fuser.fuse(modality_scene_rankings)

                detailed_results[query.qid].append(
                    (video_name, global_score, fused_scene_ranking[:top_k_scenes])
                )
        
        results.add_detailed_results(detailed_results)

        return results
