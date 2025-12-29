import torch
import logging
import re
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfFolder, whoami
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_encoder import BaseEncoder
from configuration.config import CONFIG
import os
import sys
import gc

class TextEncoder(BaseEncoder):
    """
    Encodes textual descriptions into semantic embeddings using SentenceTransformers.
    Also generates screenplay-style summaries using an LLM (Qwen2-VL).
    """
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.sbert_model: SentenceTransformer = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.model_name = CONFIG.indexing.text.text_model_id
        self.llm_model_name = CONFIG.indexing.text.llm_model_id

    def load_models(self):
        print("[TextEncoder] python:", sys.executable, flush=True)
        print(f"[TextEncoder] Loading SBERT model: {self.model_name}", flush=True)

        token = (
        os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HF_TOKEN")
            or HfFolder.get_token()
        )

        st_kwargs = {}
        if token is not None:
            st_kwargs["token"] = token

        self.sbert_model = SentenceTransformer(self.model_name, **st_kwargs)
        logging.info(f"[{self.__class__.__name__}] Model loaded.")
        
        # Load LLM for screenplay generation
        logging.info(f"[{self.__class__.__name__}] Loading LLM {self.llm_model_name}...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        logging.info(f"[{self.__class__.__name__}] LLM loaded.")

    def encode(self, text: str) -> torch.Tensor:
        """
        Public method to encode a string of text.
        
        Args:
            text: The input string.
            
        Returns:
            A torch.Tensor containing the embedding.
        """
        if not text or not isinstance(text, str):
            logging.warning("No valid text provided to TextEncoder.")
            # Return a zero vector of the correct dimension
            return torch.zeros(self.sbert_model.get_sentence_embedding_dimension(), dtype=torch.float32)
            
        with torch.inference_mode():
            embedding = self.sbert_model.encode(
                text, 
                convert_to_tensor=True, 
                device=self.device
            )
        return embedding.cpu()
    
    def generate_screenplay_summary(self, scene_data: dict) -> str:
        """
        Generate a screenplay-style summary for a single scene.
        Combines visual description, dialogue, sound effects, and speaker attribution.
        
        Args:
            scene_data: Dictionary with keys 'caption_text', 'transcript', 'audio_events', 'speaker_segments'
        
        Returns:
            Screenplay-style text summary
        """
        caption = scene_data.get("caption_text", "")
        transcript = scene_data.get("transcript", "")
        audio_events = scene_data.get("audio_events", [])
        speaker_segments = scene_data.get("speaker_segments", [])
        
        # Build context for LLM
        context_parts = []
        
        if caption and len(caption) > 0:
            context_parts.append(f"Visual description: {caption}")
        
        if transcript and len(transcript) > 0:
            context_parts.append(f"Dialogue/Speech: {transcript}")
        
        if audio_events and len(audio_events) > 0:
            events_str = ", ".join([f"{e['label']} ({e['confidence']:.2f})" for e in audio_events[:3]])
            context_parts.append(f"Audio events: {events_str}")
        
        if speaker_segments and len(speaker_segments) > 0:
            speakers = list(set([seg['speaker'] for seg in speaker_segments]))
            context_parts.append(f"Speakers detected: {', '.join(speakers)}")
        
        if not context_parts:
            return "[No information available for this scene]"
        
        # Create prompt for LLM
        prompt = f"""You are an exper writer and journalist. Based on the following information about a video, write a concise report-style description (2-3 sentences) 
                    that captures what is happening visually (actions performed, objects used and who performed those - if the person wearing the camera or a different subject), 
                    what is being said, and any notable sounds. Format it like a screenplay with action lines and dialogue. 
                    Don't add personal considerations or useless descriptions, and don't make it a fancy narration, it only has to synthetize the actions mentioned in the input descriptions.
                    Scene Information:
                    {chr(10).join(context_parts)}

                    Screenplay description:"""
                            
        try:
            messages = [{"role": "user", "content": prompt}]
            text = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.device)
            
            with torch.inference_mode():
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    gen_ids = self.llm_model.generate(
                        **model_inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=self.llm_tokenizer.eos_token_id
                    )
            
            # Decode only the generated part (skip input prompt)
            gen_ids = gen_ids[:, model_inputs.input_ids.shape[1]:]
            screenplay = self.llm_tokenizer.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0].strip()
            
            # Cleanup
            screenplay = re.sub(r"\n[-]{3,}.*$", "", screenplay, flags=re.DOTALL).strip()
            
            del model_inputs, gen_ids
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return screenplay
            
        except Exception as e:
            logging.error(f"[Screenplay] Generation failed: {e}")
            import traceback
            logging.error(traceback.format_exc())
            # Fallback to simple concatenation
            return " ".join([p.split(": ", 1)[-1] for p in context_parts])
    
    def generate_global_screenplay(self, scene_screenplays: list) -> str:
        """
        Generate a global screenplay summary for the entire video.
        Summarizes all scene screenplays into a coherent narrative.
        
        Args:
            scene_screenplays: List of tuples (scene_id, screenplay_text)
        
        Returns:
            Global screenplay summary
        """
        if not scene_screenplays:
            return "[No information available for this video]"
        
        # Format scene screenplays
        formatted_scenes = [f"Scene {sid}: {text}" for sid, text in scene_screenplays if text]
        
        if not formatted_scenes:
            return "[No information available for this video]"
        
        # For short videos (< 5 scenes), just concatenate
        if len(formatted_scenes) <= 5:
            return "\n\n".join(formatted_scenes)
        
        # For longer videos, ask LLM to summarize
        prompt = f"""You are an expert reporter. Summarize the following descriptions, captions and transcripts into a coherent 3-4 sentence summary that captures what happens in the video. Focus on capturing what concretely is happening and being said, specifying actions, subjects, speakers, sounds and objects. Avoid connective or narrative language, just focus on the facts.

Scenes:
{chr(10).join(formatted_scenes)}

Overall summary:"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            text = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.device)
            
            with torch.inference_mode():
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    gen_ids = self.llm_model.generate(
                        **model_inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        pad_token_id=self.llm_tokenizer.eos_token_id
                    )
            
            # Decode only the generated part (skip input prompt)
            gen_ids = gen_ids[:, model_inputs.input_ids.shape[1]:]
            summary = self.llm_tokenizer.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0].strip()
            
            # Cleanup
            summary = re.sub(r"\n[-]{3,}.*$", "", summary, flags=re.DOTALL).strip()
            
            del model_inputs, gen_ids
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return summary
            
        except Exception as e:
            logging.error(f"[Global Screenplay] Generation failed: {e}")
            import traceback
            logging.error(traceback.format_exc())
            # Fallback to concatenation
            return "\n\n".join(formatted_scenes)


    def generate_window_screenplay(self, scene_screenplays: list, window_id: str = "") -> str:
        """
        Generate a screenplay summary for a window of scenes.
        Similar to generate_global_screenplay but optimized for shorter windows.
        
        Args:
            scene_screenplays: List of tuples (scene_id, screenplay_text)
            window_id: ID of the window (for logging)
        
        Returns:
            Window screenplay summary
        """
        if not scene_screenplays:
            return "[No information available for this window]"
        
        # Format scene screenplays
        formatted_scenes = [f"Scene {sid}: {text}" for sid, text in scene_screenplays if text]
        
        if not formatted_scenes:
            return "[No information available for this window]"
        
        # For very short windows (< 3 scenes), just concatenate
        if len(formatted_scenes) <= 2:
            return "\n\n".join(formatted_scenes)
        
        # For longer windows, ask LLM to summarize
        prompt = f"""Summarize the following scene descriptions into a concise 2-3 sentence narrative that captures what happens in this video segment. Focus on concrete actions, subjects, objects, and any dialogue or sounds.

Scenes:
{chr(10).join(formatted_scenes)}

Summary:"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            text = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.device)
            
            with torch.inference_mode():
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    gen_ids = self.llm_model.generate(
                        **model_inputs,
                        max_new_tokens=256,  # Shorter than global summary
                        do_sample=False,
                        pad_token_id=self.llm_tokenizer.eos_token_id
                    )
            
            # Decode only the generated part (skip input prompt)
            gen_ids = gen_ids[:, model_inputs.input_ids.shape[1]:]
            summary = self.llm_tokenizer.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0].strip()
            
            # Cleanup
            summary = re.sub(r"\n[-]{3,}.*$", "", summary, flags=re.DOTALL).strip()
            
            del model_inputs, gen_ids
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return summary
            
        except Exception as e:
            logging.warning(f"[Window Screenplay] Generation failed for {window_id}: {e}")
            # Fallback to concatenation
            return "\n\n".join(formatted_scenes)

    def unload_models(self):
        if hasattr(self, "sbert_model"):
            del self.sbert_model
        if hasattr(self, "llm_model"):
            del self.llm_model
        if hasattr(self, "llm_tokenizer"):
            del self.llm_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
