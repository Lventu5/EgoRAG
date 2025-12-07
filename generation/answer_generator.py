"""
Answer Generator for Ego4D NLQ using Qwen3-VL.

Takes QueryDataset and Scene objects from HierarchicalRetriever to generate answers.
"""
import logging
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from data.query import QueryDataset, Query
from data.video_dataset import Scene
import data.datatypes as types
import json
import shutil


class AnswerGenerator:
    """
    Generates answers for Ego4D NLQ queries.
    Takes QueryDataset (already created) and Scene objects from retrieval results.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        temp_dir: str = "../../mronconi/ego4d_data/temp_clips",
        max_clips_per_query: int = 3,
        max_pixels: int = 360 * 420, # Limit resolution to save tokens
        fps: float = 1.0, # Sample 1 frame per second
        max_new_tokens: int = 128,
    ):
        """
        Args:
            model_name: model path
            device: Device for inference
            temp_dir: Directory for temporary video clips
            max_clips_per_query: Max number of scenes per query
            max_pixels: Max resolution for video (saves tokens)
            fps: Frame sampling rate
            max_new_tokens: Max tokens to generate
        """
        self.model_name = model_name
        self.device = device
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.max_clips_per_query = max_clips_per_query
        self.max_pixels = max_pixels
        self.fps = fps
        self.max_new_tokens = max_new_tokens
        
        # Models will be loaded separately via load_model()
        self.model: Optional[Qwen3VLForConditionalGeneration] = None
        self.processor: Optional[AutoProcessor] = None
        
        logging.info(f"[AnswerGenerator] Initialized. Temp dir: {self.temp_dir}")
    
    def load_model(self):
        """
        Load model and processor.
        Call this method before generating answers.
        """
        if self.model is not None:
            logging.info("[AnswerGenerator] Model already loaded.")
            return
        
        logging.info(f"[AnswerGenerator] Loading {self.model_name}...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model.eval()
        
        logging.info(f"[AnswerGenerator] Model loaded successfully.")
    
    def extract_scene_clip(
        self,
        scene: Scene,
        video_base_path: str,
        output_path: str
    ) -> bool:
        """
        Extract scene clip from video using ffmpeg.
        
        Args:
            scene: Scene object with timing
            video_base_path: Base directory with videos
            output_path: Output clip path
        
        Returns:
            True if successful
        """
        # Get video name from scene
        video_name = getattr(scene, 'video_name', None)
        if not video_name:
            logging.error(f"[AnswerGenerator] Scene {scene.scene_id} missing video_name")
            return False
        
        video_path = Path(video_base_path) / f"{video_name}.mp4"
        
        if not video_path.exists():
            logging.warning(f"[AnswerGenerator] Video not found: {video_path}")
            return False
        
        duration = scene.end_time - scene.start_time
        
        # ffmpeg extraction
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(scene.start_time),
            "-i", str(video_path),
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-loglevel", "error",
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"[AnswerGenerator] ffmpeg failed: {e.stderr.decode()}")
            return False
    
    def concatenate_clips(
        self,
        clip_paths: List[str],
        output_path: str
    ) -> bool:
        """
        Concatenate clips into single video.
        
        Args:
            clip_paths: List of clip paths
            output_path: Output path
        
        Returns:
            True if successful
        """
        if not clip_paths:
            return False
        
        if len(clip_paths) == 1:
            shutil.copy(clip_paths[0], output_path)
            return True
        
        # Create concat file
        concat_file = self.temp_dir / f"concat_{Path(output_path).stem}.txt"
        with open(concat_file, 'w') as f:
            for clip_path in clip_paths:
                f.write(f"file '{os.path.abspath(clip_path)}'\n")
        
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            "-loglevel", "error",
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            concat_file.unlink()
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"[AnswerGenerator] concat failed: {e.stderr.decode()}")
            return False
    
    @torch.no_grad()
    def _generate_from_video(self, video_path: str, question: str) -> str:
        """
        Generate answer using the model.
        
        Args:
            video_path: Path to video clip
            question: Question text
        
        Returns:
            Generated answer
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Prepare messages in Qwen format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": self.max_pixels,
                        "fps": self.fps,
                    },
                    {
                        "type": "text",
                        "text": f"You are watching an egocentric video. Answer this question based on what you see: {question}"
                    },
                ],
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process vision info (extracts frames from video)
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Generate
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False
        )
        
        # Trim input and decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text.strip()
    
    def generate_answer_for_query(
        self,
        query: Query,
        retrieval_results: types.RetrievalResults,
        video_base_path: str,
        use_concatenated: bool = True
    ) -> str:
        """
        Generate answer for single query using retrieved scenes.
        
        Args:
            query: Query object
            retrieval_results: Results from HierarchicalRetriever
            video_base_path: Base directory with videos
            use_concatenated: Concatenate scenes or process individually
        
        Returns:
            Generated answer
        """
        # Get detailed results for this query
        query_results = retrieval_results.results.get(query.qid)
        if not query_results:
            logging.warning(f"[AnswerGenerator] No results for {query.qid}")
            return ""
        
        # Extract scenes: format is list of (video_name, video_score, [(Scene, score), ...])
        all_scenes: List[Tuple[Scene, float]] = []
        for video_name, video_score, scene_list in query_results:
            all_scenes.extend(scene_list)
        
        # Sort by score and take top-k
        all_scenes.sort(key=lambda x: x[1], reverse=True)
        top_scenes = all_scenes[:self.max_clips_per_query]
        
        if not top_scenes:
            logging.warning(f"[AnswerGenerator] No scenes for {query.qid}")
            return ""
        
        # Extract clips
        clip_paths = []
        for i, (scene, score) in enumerate(top_scenes):
            clip_path = self.temp_dir / f"{query.qid}_scene_{i}_{scene.scene_id}.mp4"
            success = self.extract_scene_clip(
                scene=scene,
                video_base_path=video_base_path,
                output_path=str(clip_path)
            )
            if success:
                clip_paths.append(str(clip_path))
                logging.debug(
                    f"[AnswerGenerator] Extracted {scene.scene_id}: "
                    f"{scene.start_time:.1f}s-{scene.end_time:.1f}s (score={score:.3f})"
                )
        
        if not clip_paths:
            return ""
        
        # Generate answer
        if use_concatenated:
            # Concatenate all clips
            concat_path = self.temp_dir / f"{query.qid}_concat.mp4"
            if self.concatenate_clips(clip_paths, str(concat_path)):
                answer = self._generate_from_video(str(concat_path), query.query_text)
                clip_paths.append(str(concat_path))
            else:
                answer = ""
        else:
            # Process each clip separately
            answers = []
            for clip_path in clip_paths:
                ans = self._generate_from_video(clip_path, query.query_text)
                if ans:
                    answers.append(ans)
            
            # Combine answers
            if len(answers) == 1:
                answer = answers[0]
            elif len(answers) > 1:
                answer = " ".join(answers)
            else:
                answer = ""
        
        # Cleanup
        self._cleanup_clips(clip_paths)
        
        return answer
    
    def generate_answers(
        self,
        query_dataset: QueryDataset,
        retrieval_results: types.RetrievalResults,
        video_base_path: str,
        use_concatenated: bool = True
    ) -> Dict[str, str]:
        """
        Generate answers for all queries in dataset.
        
        Args:
            query_dataset: QueryDataset (already created)
            retrieval_results: Results from HierarchicalRetriever
            video_base_path: Base directory with videos
            use_concatenated: Concatenate scenes or not
        
        Returns:
            Dict mapping query_id -> answer
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        answers = {}
        
        logging.info(f"[AnswerGenerator] Generating answers for {len(query_dataset)} queries...")
        
        for query in tqdm(query_dataset, desc="Generating answers"):
            try:
                answer = self.generate_answer_for_query(
                    query=query,
                    retrieval_results=retrieval_results,
                    video_base_path=video_base_path,
                    use_concatenated=use_concatenated
                )
                answers[query.qid] = answer
            except Exception as e:
                logging.error(f"[AnswerGenerator] Failed for {query.qid}: {e}")
                answers[query.qid] = ""
        
        return answers
    
    def _cleanup_clips(self, clip_paths: List[str]):
        """Remove temporary clips."""
        for clip_path in clip_paths:
            try:
                if os.path.exists(clip_path):
                    os.remove(clip_path)
            except Exception as e:
                logging.debug(f"[AnswerGenerator] Cleanup failed: {e}")
    
    def save_answers(self, answers: Dict[str, str], output_path: str):
        """
        Save answers to JSON.
        
        Args:
            answers: Dict of query_id -> answer
            output_path: Output path
        """
        
        output_data = {
            "model": self.model_name,
            "num_queries": len(answers),
            "answers": [
                {"query_id": qid, "answer": ans}
                for qid, ans in answers.items()
            ]
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logging.info(f"[AnswerGenerator] Saved {len(answers)} answers to {output_path}")
