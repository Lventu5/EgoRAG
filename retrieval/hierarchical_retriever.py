import torch
import logging
from transformers import (
    XCLIPProcessor,
    XCLIPModel,
    ClapProcessor,
    ClapModel
)
from sentence_transformers import SentenceTransformer
from torch.nn.functional import normalize

from data.video_dataset import VideoDataset
from data.query import Query, QueryDataset
import data.datatypes as types
from retrieval.rewriter import QueryRewriterLLM
from .fuser import Fuser
from configuration.config import CONFIG

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
        self.sizes = {
            "video": {
                "size": 512,
                "model": video_model_name
            },
            "audio": {
                "size": 512,
                "model": audio_model_name
            },
            "text": {
                "size": 384,
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
        if fuser is None:
            logging.warning("Fuser not specified, using a RRF fuser")
            self.fuser = Fuser()
        else:
            self.fuser = fuser

    def _load_models_for_modality(self, modality: str):
        if self.current_modality == modality:
            return
        
        self.processor = None
        self.embedder = None
        target_modality = modality

        if target_modality == "text" or target_modality == "caption":
            self.embedder = SentenceTransformer(
                self.sizes["text"]["model"], device=self.device
            )
        
        elif target_modality == "video":
            model_name = self.sizes["video"]["model"]
            self.processor = XCLIPProcessor.from_pretrained(model_name)
            self.embedder = XCLIPModel.from_pretrained(model_name).to(self.device) # type: ignore

        elif target_modality == "audio":
            model_name = self.sizes["audio"]["model"]
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
            inputs = self.processor(
                text=mod_queries, return_tensors="pt", padding=True # type: ignore
            ).to(self.device)
            with torch.no_grad():
                embeddings = self.embedder.get_text_features(**inputs) # type: ignore
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
