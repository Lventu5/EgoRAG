import logging
import torch
from typing import List, Dict, Optional, Tuple
from transformers import CLIPProcessor, CLIPModel
from indexing.components.tags_new import TAGS_CLIP
from configuration.config import CONFIG


class QueryTagger:
    """Tagger that predicts which visual tags are relevant to a text query
    using CLIP's text-to-text similarity.
    
    This works by:
    1. Encoding the query text using CLIP's text encoder
    2. Computing similarity with pre-encoded tag descriptions
    3. Selecting tags above confidence threshold
    
    Designed to work with QueryRewriterLLM:
    - Tags the "video" modality from decomposed queries
    - Provides visual grounding for text-based retrieval
    - Enables tag-based filtering to match VisionTagger outputs
    """

    def __init__(
        self, 
        device: str = "cuda", 
        confidence_threshold: float = 0.40,  # Slightly higher than VisionTagger since we have less signal
        use_adaptive_threshold: bool = True,
        adaptive_percentile: float = 0.70,  # Use 70th percentile for query (less strict than video's 75%)
        max_tags: int = 10,  # Limit number of tags per query
    ):
        self.model_name = getattr(CONFIG.indexing.tag, "vision_model_id", "openai/clip-vit-base-patch32")
        self.device = device if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold
        self.use_adaptive_threshold = use_adaptive_threshold
        self.adaptive_percentile = adaptive_percentile
        self.max_tags = max_tags

        self.processor: Optional[CLIPProcessor] = None
        self.model: Optional[CLIPModel] = None
        self.text_features: Optional[torch.Tensor] = None
        self.tag_keys: List[str] = []

    def load_model(self):
        """
        Load CLIP model and pre-compute tag embeddings.
        """
        logging.info(f"[QueryTagger] Loading model {self.model_name} on {self.device}")
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        logging.info("[QueryTagger] Pre-computing text embeddings for tags...")
        self._precompute_tags()

    def _precompute_tags(self):
        """
        Encodes all tag descriptions from TAGS_CLIP into vectors.
        Same strategy as VisionTagger: average multiple descriptions per category.
        """
        self.tag_keys = []
        category_embeddings = []
        
        for category_key, descriptions in TAGS_CLIP.items():
            # Skip metadata entries (start with underscore)
            if category_key.startswith("_"):
                continue
            self.tag_keys.append(category_key)
            
            # Encode all descriptions for this category
            inputs = self.processor(text=descriptions, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                desc_features = self.model.get_text_features(**inputs)
                # Normalize each description embedding
                desc_features = desc_features / desc_features.norm(p=2, dim=-1, keepdim=True)
                # Average all description embeddings for this category
                category_embedding = desc_features.mean(dim=0, keepdim=True)
                # Normalize the averaged embedding
                category_embedding = category_embedding / category_embedding.norm(p=2, dim=-1, keepdim=True)
                category_embeddings.append(category_embedding)
        
        # Stack all category embeddings: shape [num_categories, embedding_dim]
        self.text_features = torch.cat(category_embeddings, dim=0)
        
        logging.info(f"[QueryTagger] Precomputed embeddings for {len(self.tag_keys)} tag categories")

    def tag_query(self, query_text: str) -> List[str]:
        """
        Tag a single query string with relevant visual tags.
        
        Args:
            query_text: The query to tag (ideally the 'video' modality from decomposed query)
        
        Returns:
            List of tag keys sorted by relevance score
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not query_text or not query_text.strip():
            logging.warning("[QueryTagger] Empty query text provided")
            return []
        
        # Encode query text
        inputs = self.processor(text=[query_text], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            query_features = self.model.get_text_features(**inputs)
            query_features = query_features / query_features.norm(p=2, dim=-1, keepdim=True)
            
            # Compute similarity with all tag categories: [1, num_categories]
            similarity = query_features @ self.text_features.T
            similarity = similarity.squeeze(0)  # [num_categories]
        
        # Apply adaptive or fixed thresholding
        if self.use_adaptive_threshold:
            adaptive_thresh = torch.quantile(similarity, self.adaptive_percentile)
            adaptive_thresh = max(adaptive_thresh.item(), self.confidence_threshold)
        else:
            adaptive_thresh = self.confidence_threshold
        
        # Filter tags above threshold
        mask = similarity > adaptive_thresh
        found_indices = mask.nonzero(as_tuple=True)[0]
        
        # Collect results with scores
        results = []
        for idx in found_indices:
            score = similarity[idx].item()
            tag_key = self.tag_keys[idx.item()]
            results.append((tag_key, score))
        
        # Sort by score descending and limit
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:self.max_tags]
        
        return [r[0] for r in results]

    def tag_batch(self, query_texts: List[str]) -> List[List[str]]:
        """
        Tag multiple queries in a batch for efficiency.
        
        Args:
            query_texts: List of query strings to tag
        
        Returns:
            List of tag lists, one per query
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not query_texts:
            return []
        
        # Filter out empty queries
        valid_indices = [i for i, q in enumerate(query_texts) if q and q.strip()]
        valid_queries = [query_texts[i] for i in valid_indices]
        
        if not valid_queries:
            return [[] for _ in query_texts]
        
        # Encode all queries
        inputs = self.processor(text=valid_queries, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            query_features = self.model.get_text_features(**inputs)
            query_features = query_features / query_features.norm(p=2, dim=-1, keepdim=True)
            
            # Compute similarity: [batch_size, num_categories]
            similarity = query_features @ self.text_features.T
        
        # Process each query
        all_results = []
        for i, sim in enumerate(similarity):
            # Apply adaptive or fixed thresholding
            if self.use_adaptive_threshold:
                adaptive_thresh = torch.quantile(sim, self.adaptive_percentile)
                adaptive_thresh = max(adaptive_thresh.item(), self.confidence_threshold)
            else:
                adaptive_thresh = self.confidence_threshold
            
            # Filter tags above threshold
            mask = sim > adaptive_thresh
            found_indices = mask.nonzero(as_tuple=True)[0]
            
            # Collect results with scores
            results = []
            for idx in found_indices:
                score = sim[idx].item()
                tag_key = self.tag_keys[idx.item()]
                results.append((tag_key, score))
            
            # Sort by score descending and limit
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:self.max_tags]
            
            all_results.append([r[0] for r in results])
        
        # Map results back to original indices (including empty queries)
        final_results = [[] for _ in query_texts]
        for orig_idx, res_idx in enumerate(valid_indices):
            final_results[res_idx] = all_results[orig_idx]
        
        return final_results

    def tag_with_scores(self, query_text: str) -> List[Tuple[str, float]]:
        """
        Tag a query and return both tags and their confidence scores.
        
        Args:
            query_text: The query to tag
        
        Returns:
            List of (tag_key, score) tuples sorted by score
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not query_text or not query_text.strip():
            return []
        
        # Encode query text
        inputs = self.processor(text=[query_text], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            query_features = self.model.get_text_features(**inputs)
            query_features = query_features / query_features.norm(p=2, dim=-1, keepdim=True)
            
            # Compute similarity with all tag categories
            similarity = query_features @ self.text_features.T
            similarity = similarity.squeeze(0)
        
        # Apply thresholding
        if self.use_adaptive_threshold:
            adaptive_thresh = torch.quantile(similarity, self.adaptive_percentile)
            adaptive_thresh = max(adaptive_thresh.item(), self.confidence_threshold)
        else:
            adaptive_thresh = self.confidence_threshold
        
        mask = similarity > adaptive_thresh
        found_indices = mask.nonzero(as_tuple=True)[0]
        
        results = []
        for idx in found_indices:
            score = similarity[idx].item()
            tag_key = self.tag_keys[idx.item()]
            results.append((tag_key, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:self.max_tags]

    def pretty_print_tags(self, query_text: str, color: bool = False) -> None:
        """
        Print query tags in a readable format.
        
        Args:
            query_text: The query to tag
            color: Whether to use colored output
        """
        GREEN = "\033[92m" if color else ""
        YELLOW = "\033[93m" if color else ""
        RESET = "\033[0m" if color else ""
        
        tags_with_scores = self.tag_with_scores(query_text)
        
        print("=" * 80)
        print(f"Query: {YELLOW}{query_text}{RESET}")
        print(f"Predicted Tags ({len(tags_with_scores)}):")
        
        if not tags_with_scores:
            print("  (no tags above threshold)")
        else:
            for tag, score in tags_with_scores:
                print(f"  {GREEN}{tag}{RESET}: {score:.3f}")
        
        print()
