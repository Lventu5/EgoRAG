import json
import os
import torch
import logging
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from PIL import Image
from decord import VideoReader, cpu

from transformers import (
    AutoProcessor, 
    LlavaNextVideoForConditionalGeneration,
    AutoModel,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration
)

from data.query import QueryDataset
from data.dataset import VideoDataset


@dataclass
class RefactoredQuery:
    """Represents a refactored query with metadata"""
    query_idx: int
    original_query: str
    refactored_query: str
    video_uid: str
    clip_uid: str
    video_start_sec: float
    video_end_sec: float
    video_start_frame: int
    video_end_frame: int
    clip_start_sec: float
    clip_end_sec: float
    template: Optional[str] = None
    slot_x: Optional[str] = None
    verb_x: Optional[str] = None
    
    def to_ego4d_format(self) -> Dict:
        """Convert to Ego4D NLQ annotation format"""
        result = {
            "clip_start_sec": self.clip_start_sec,
            "clip_end_sec": self.clip_end_sec,
            "video_start_sec": self.video_start_sec,
            "video_end_sec": self.video_end_sec,
            "video_start_frame": self.video_start_frame,
            "video_end_frame": self.video_end_frame,
            "query": self.refactored_query,
        }
        if self.template:
            result["template"] = self.template
        if self.slot_x:
            result["slot_x"] = self.slot_x
        if self.verb_x:
            result["verb_x"] = self.verb_x
        result["raw_tags"] = [
            self.template or "",
            self.refactored_query,
            self.slot_x or "",
            self.verb_x or ""
        ]
        return result


class QueryRefactorer:
    """
    Refactors ambiguous queries by using a Vision-Language Model to analyze video content
    and make queries more specific and contextual.
    """
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        model_type: str = "qwen",
        video_dataset: Optional[VideoDataset] = None,
        device: str = "cuda",
        verbose: bool = True
    ):
        self.model_id = model_id
        self.model_type = model_type.lower()
        self.video_dataset = video_dataset
        self.device = device
        self.verbose = verbose
        
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.refactored_queries: List[RefactoredQuery] = []
        
        if verbose:
            logging.basicConfig(level=logging.INFO)
            
    def load_vllm(self):
        """Load the Vision-Language Model"""
        if self.model is not None:
            return
            
        if self.verbose:
            logging.info(f"Loading VLLM: {self.model_id}")
        
        if self.model_type == "llava":
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code = True)
            self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.model.eval()
        elif self.model_type == "qwen":
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.model.config.vision_feature_select_strategy = "mean"
            self.model.config.vision_feature_layer = -1

            self.model.eval()
        elif self.model_type == "internvideo":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            ).half().cuda().to(torch.bfloat16)
            self.model.eval()
            
    def unload_vllm(self):
        """Unload the Vision-Language Model to free memory"""
        if self.verbose:
            logging.info("Unloading VLLM")
        if self.model:
            del self.model
        if self.processor:
            del self.processor
        if self.tokenizer:
            del self.tokenizer
        self.model = None
        self.processor = None
        self.tokenizer = None
        torch.cuda.empty_cache()
            
    def _load_video_frames(self, video_path: str, max_frames: int = 6) -> List[Image.Image]:
        """Load video frames for processing"""
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
        
        frames = []
        for idx in frame_indices:
            frame = vr[idx].asnumpy()
            img = Image.fromarray(frame)

            w, h = img.size
            new_w = 420
            new_h = int(h * (420 / w))
            img = img.resize((new_w, new_h), Image.BICUBIC)

            frames.append(img)
        
        return frames
    
    def _build_refactoring_prompt(self, query: str) -> str:
        """Build the prompt for the VLLM to refactor the query"""
        prompt = f"""You are watching an egocentric video showing a person's actions and surroundings. You need to query a VQA system in a clear and unambiguous way.

Your task: Make this query more specific and unambiguous based on what you see in the video. Do not provide the answer to the query, just add
relevant information that make the query refer to the single instant of time provided as a ground truth. You do not need to necessarly add information,
just do it if the query might be ambiguous.
Keep the questions with the I pronoun if they contain it, do not add invented information.

Consider:
- Specific objects mentioned (which table, which door, what color, etc.)
- Actions and their context (while entering/exiting, before/after doing something)
- Spatial relationships (left/right, near/far from something)
- Temporal context (at the beginning/end, after doing X)
- Any distinguishing features that make objects or actions unique

For instance, if the query is: 'Did I close the door?' You might rephrase it as 'Did I close the door when I left the clothing shop?', in case the ground truth shows a person leaving a clothing shop.

Provide ONLY the refactored query, nothing else. Be concise but specific.

Original query: "{query}"
"""
        return prompt
        
    def _generate_with_qwen(self, frames: List[Image.Image], prompt: str) -> str:
        """Generate text using Qwen model"""
        
        content = [{"type": "image", "image": img} for img in frames]
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        # Chat template (identico all’AnswerGenerator)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=frames,
            padding=True,
            return_tensors="pt"
        ).to(self.device, torch.bfloat16)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )

        answer_ids = generated_ids[0][inputs.input_ids.shape[1]:]
        answer = self.processor.decode(answer_ids, skip_special_tokens=True)

        return answer.strip()
        
    def _generate_with_internvideo(self, video_path: str, prompt: str) -> str:
        """Generate text using InternVideo model"""
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        num_frames = min(8, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        
        frames = []
        for idx in frame_indices:
            frame = vr[idx].asnumpy()
            frames.append(Image.fromarray(frame))
        
        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(num_frames)])
        question = video_prefix + prompt
        
        generation_config = dict(
            do_sample=False,
            temperature=0.0,
            max_new_tokens=256,
            top_p=0.1,
            num_beams=1
        )
        
        with torch.no_grad():
            output, _ = self.model.chat(
                self.tokenizer, 
                frames, 
                question, 
                generation_config, 
                num_patches_list=[1]*num_frames,
                history=None,
                return_history=True
            )
        
        return output.strip()
        
    def refactor_query(
        self,
        query: str,
        video_path: str,
        ground_truth: Dict[str, float],
        video_uid: str = "",
        clip_uid: str = "",
        query_idx: int = 0
    ) -> RefactoredQuery:
        """
        Refactor a single query using the VLLM
        
        Args:
            query: Original query text
            video_path: Path to video file
            ground_truth: Dict with video timing info (video_start_sec, video_end_sec, etc.)
            video_uid: Video identifier
            clip_uid: Clip identifier
            query_idx: Query index
            
        Returns:
            RefactoredQuery object
        """
        if self.model is None:
            self.load_vllm()
            
        prompt = self._build_refactoring_prompt(query)
        
        if self.verbose:
            logging.info(f"Refactoring query {query_idx}: '{query}'")
            
        if self.model_type == "llava":
            frames = self._load_video_frames(video_path, max_frames=6)
            refactored_text = self._generate_with_qwen(frames, prompt)
        elif self.model_type == "qwen":
            frames = self._load_video_frames(video_path, max_frames=6)
            refactored_text = self._generate_with_qwen(frames, prompt)
        elif self.model_type == "internvideo":
            refactored_text = self._generate_with_internvideo(video_path, prompt)
        
        refactored_text = refactored_text.strip()
        
        if self.verbose:
            logging.info(f"  Original:   '{query}'")
            logging.info(f"  Refactored: '{refactored_text}'")
            
        refactored_query = RefactoredQuery(
            query_idx=query_idx,
            original_query=query,
            refactored_query=refactored_text,
            video_uid=video_uid,
            clip_uid=clip_uid,
            video_start_sec=ground_truth.get("video_start_sec", 0.0),
            video_end_sec=ground_truth.get("video_end_sec", 0.0),
            video_start_frame=ground_truth.get("video_start_frame", 0),
            video_end_frame=ground_truth.get("video_end_frame", 0),
            clip_start_sec=ground_truth.get("clip_start_sec", 0.0),
            clip_end_sec=ground_truth.get("clip_end_sec", 0.0),
            template=ground_truth.get("template"),
            slot_x=ground_truth.get("slot_x"),
            verb_x=ground_truth.get("verb_x")
        )
        
        self.refactored_queries.append(refactored_query)
        return refactored_query
        
    def refactor_from_ego4d_annotations(
        self,
        annotation_path: str,
        video_dir: str,
        max_queries: Optional[int] = None,
        video_uids: Optional[List[str]] = None
    ) -> List[RefactoredQuery]:
        """
        Refactor queries from Ego4D NLQ annotation file
        
        Args:
            annotation_path: Path to Ego4D NLQ JSON file
            video_dir: Directory containing video files
            max_queries: Maximum number of queries to process (for testing)
            video_uids: List of specific video UIDs to process (None = all)
            
        Returns:
            List of RefactoredQuery objects
        """
        with open(annotation_path, 'r') as f:
            data = json.load(f)
            
        videos = data.get("videos", [])
        
        if video_uids:
            videos = [v for v in videos if v.get("video_uid") in video_uids]
            
        processed_count = 0
        
        for video_entry in videos:
            video_uid = video_entry.get("video_uid", "")
            video_path = os.path.join(video_dir, f"{video_uid}.mp4")
            
            if not os.path.exists(video_path):
                if self.verbose:
                    logging.warning(f"Video not found: {video_path}")
                continue
                
            for clip in video_entry.get("clips", []):
                clip_uid = clip.get("clip_uid", "")
                
                for annotation_group in clip.get("annotations", []):
                    for idx, lang_query in enumerate(annotation_group.get("language_queries", [])):
                        if max_queries and processed_count >= max_queries:
                            return self.refactored_queries
                            
                        query_text = lang_query.get("query", "")
                        if not query_text:
                            continue
                            
                        ground_truth = {
                            "video_start_sec": lang_query.get("video_start_sec", 0.0),
                            "video_end_sec": lang_query.get("video_end_sec", 0.0),
                            "video_start_frame": lang_query.get("video_start_frame", 0),
                            "video_end_frame": lang_query.get("video_end_frame", 0),
                            "clip_start_sec": lang_query.get("clip_start_sec", 0.0),
                            "clip_end_sec": lang_query.get("clip_end_sec", 0.0),
                            "template": lang_query.get("template"),
                            "slot_x": lang_query.get("slot_x"),
                            "verb_x": lang_query.get("verb_x")
                        }
                        
                        self.refactor_query(
                            query=query_text,
                            video_path=video_path,
                            ground_truth=ground_truth,
                            video_uid=video_uid,
                            clip_uid=clip_uid,
                            query_idx=processed_count
                        )
                        
                        processed_count += 1
                        
        return self.refactored_queries
        
    def refactor_query_dataset(
        self,
        query_dataset: QueryDataset,
        video_dir: str
    ) -> List[RefactoredQuery]:
        """
        Refactor queries from a QueryDataset object
        
        Args:
            query_dataset: QueryDataset with queries to refactor
            video_dir: Directory containing video files
            
        Returns:
            List of RefactoredQuery objects
        """
        for idx, query in enumerate(query_dataset.queries):
            video_uid = query.video_uid
            video_path = os.path.join(video_dir, f"{video_uid}.mp4")
            
            if not os.path.exists(video_path):
                if self.verbose:
                    logging.warning(f"Video not found: {video_path}")
                continue
                
            ground_truth = {
                "video_start_sec": query.gt.get("start_sec", 0.0),
                "video_end_sec": query.gt.get("end_sec", 0.0),
                "video_start_frame": query.gt.get("start_frame", 0),
                "video_end_frame": query.gt.get("end_frame", 0),
                "clip_start_sec": 0.0,
                "clip_end_sec": 0.0,
            }
            
            self.refactor_query(
                query=query.query_text,
                video_path=video_path,
                ground_truth=ground_truth,
                video_uid=video_uid,
                clip_uid="",
                query_idx=idx
            )
            
        return self.refactored_queries
        
    def save_to_ego4d_format(self, output_path: str, base_annotation_path: Optional[str] = None):
        """
        Save refactored queries in Ego4D NLQ format, replacing original queries
        
        Args:
            output_path: Path to save the JSON file
            base_annotation_path: Optional base annotation file to preserve structure.
                                 If provided, loads this file and replaces matching queries.
        """
        if base_annotation_path and os.path.exists(base_annotation_path):
            with open(base_annotation_path, 'r') as f:
                data = json.load(f)
        else:
            data = {"version": "1.0", "split": "train", "videos": []}
        
        # Create a lookup map for refactored queries
        refactored_map = {}
        for refactored in self.refactored_queries:
            key = (refactored.video_uid, refactored.clip_uid, refactored.original_query)
            refactored_map[key] = refactored
        
        # Iterate through the original structure and replace queries
        for video_entry in data.get("videos", []):
            video_uid = video_entry.get("video_uid", "")
            
            for clip in video_entry.get("clips", []):
                clip_uid = clip.get("clip_uid", "")
                
                for annotation_group in clip.get("annotations", []):
                    lang_queries = annotation_group.get("language_queries", [])
                    
                    for i, lang_query in enumerate(lang_queries):
                        original_query = lang_query.get("query", "")
                        key = (video_uid, clip_uid, original_query)
                        
                        # If we have a refactored version, replace it
                        if key in refactored_map:
                            refactored = refactored_map[key]
                            lang_queries[i] = refactored.to_ego4d_format()
                    
                    annotation_group["language_queries"] = lang_queries
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        if self.verbose:
            logging.info(f"Saved {len(self.refactored_queries)} refactored queries to {output_path}")
            
    def save_simple_json(self, output_path: str):
        """
        Save refactored queries in a simple JSON format (list of queries)
        
        Args:
            output_path: Path to save the JSON file
        """
        data = [asdict(q) for q in self.refactored_queries]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        if self.verbose:
            logging.info(f"Saved {len(self.refactored_queries)} refactored queries to {output_path}")
            
    def get_statistics(self) -> Dict:
        """Get statistics about the refactoring process"""
        if not self.refactored_queries:
            return {}
            
        stats = {
            "total_queries": len(self.refactored_queries),
            "unique_videos": len(set(q.video_uid for q in self.refactored_queries)),
            "avg_original_length": sum(len(q.original_query) for q in self.refactored_queries) / len(self.refactored_queries),
            "avg_refactored_length": sum(len(q.refactored_query) for q in self.refactored_queries) / len(self.refactored_queries),
            "length_increase": sum(len(q.refactored_query) - len(q.original_query) for q in self.refactored_queries) / len(self.refactored_queries)
        }
        
        return stats
        
    def clear_queries(self):
        """Clear all stored refactored queries"""
        self.refactored_queries = []
