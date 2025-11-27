import logging
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from indexing.components.tags import TAG_LIST
from configuration.config import CONFIG

class Tagger:
    """Tagger that predicts which tags (from TAG_LIST) are present in a video
    based only on the textual description available in a VideoDataPoint.

    Usage:
        tagger = Tagger(model_name="Qwen2.5-VL", device="cuda")
        tagger.load_model()
        tags = tagger.tag_datapoint(dp)

    The implementation asks the LLM to return which tags from the provided
    `TAG_LIST` apply to the video. The result is parsed conservatively by
    searching for tag strings in the model output (case-insensitive).
    """

    def __init__(self, device: str = "cuda", batch_size: int = 4):
        self.model_name = CONFIG.indexing.tag.tagger_model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None

    def load_model(self):
        logging.info(f"[Tagger] Loading model {self.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        # prefer bfloat16 for CUDA if available
        dtype = torch.bfloat16 if (self.device == "cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype).to(self.device)
        self.model.eval()

    '''
    def _build_prompt(self, text: str) -> str:
        # Build a deterministic prompt listing tags and asking for a comma-separated answer
        tags_block = ", ".join(TAG_LIST)
        prompt = (
            "You are an expert assistant, specialized in tagging video clips. Given the following textual description of a video, "
            "return which tags from the provided list are present in the video. "
            "Only use tags from the list and return a comma-separated list of tags. Not all the tags from "
            "the list need to be used, select only the relevant ones strictly based on the screenplay of that scene. "
            "Do not add additional commentary.\n\n"
            "Tags list: " + tags_block + "\n\n"
            "Video description:\n" + text + "\n\n"
            "Example of reasoning: 'I first must look for objects, actions and settings.\n"
            "I have now identified the following relevant elements: __list of actions, objects, subjects and settings EXPLICITELY mentioned in the text description__\n"
            "I am now looking for a tag, among the tags list, that encapsulates the elements I found. "
            "I found the following relevant tags: __list of tags__\n"
            "I now check in the tag list if the tags are present and I replace the absent ones with the ones from the list that best align making sure to only select the ones mentioned in the video without inferring a lot'\n\n"
            "Explain your reasoning steps. Start your answer here: "
        )
        return prompt
    '''
    def _build_prompt(self, text: str):
        tags_block = ", ".join(TAG_LIST)

        system_message = (
            "You are an expert assistant specialized in tagging video clips. "
            "Follow the instructions and output format precisely."
        )

        user_message = (
            "TASK:\n"
            "Given the following textual description of a video, select ALL and ONLY the tags from the list that are "
            "directly supported by the description.\n\n"

            "RULES:\n"
            "1. Use only tags from the list, exactly as written.\n"
            "2. Select a tag only if it is clearly or explicitly mentioned.\n"
            "3. Do NOT guess, infer, or hallucinate information.\n"
            "4. If nothing applies, return an empty set.\n"
            "5. Prefer specific tags over generic ones.\n\n"

            f"TAGS LIST:\n{tags_block}\n\n"

            "VIDEO DESCRIPTION:\n"
            f"{text}\n\n"

            "OUTPUT FORMAT (STRICT):\n"
            "Return EXACTLY one line:\n"
            "TAGS: tag1, tag2, tag3\n"
            "If no tag applies, write exactly:\n"
            "TAGS: none\n"
            "Do not add anything else.\n\n"

            "=== START OUTPUT ==="
        )

        # 
        return system_message + "\n" + user_message

    def _parse_response(self, response: str) -> List[str]:
        found = []

        if not response:
            return found
        resp_low = response.lower()
        for tag in TAG_LIST:
            if not isinstance(tag, str):
                continue
            tag_low = tag.lower()
            if tag_low in resp_low:
                found.append(tag)
        return found

    def tag_datapoint(self, dp, tag_scenes: bool = False) -> List[str]:
        """Tag a single VideoDataPoint using its text modality.

        Args:
            dp: VideoDataPoint
            tag_scenes: if True, also tag each scene individually and store tags
                        in `dp.scene_embeddings[sid]['tags']`.

        The function will look for `dp.global_embeddings['text_raw']` or fallbacks.
        The predicted tags are written into `dp.global_embeddings['tags']` and
        returned as a list.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Extract text description
        text = None
        ge = getattr(dp, "global_embeddings", {}) or {}
        text = ge.get("text_raw") or ge.get("text") or ge.get("caption_text")

        # Global tagging
        prompt = self._build_prompt(text)
        inputs = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            gen_ids = self.model.generate(**inputs, max_new_tokens=512)
        out = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        marker = "=== START OUTPUT ==="
        if marker in out:
            out = out.split(marker, 1)[1].strip()
        tags = self._parse_response(out)
        dp.global_embeddings["tags"] = tags
        
        scenes = getattr(dp, "scene_embeddings", {}) or {}
        scene_items = []
        scene_ids = []
        for sid, sd in sorted(scenes.items()):
            if not isinstance(sd, dict):
                continue
            scene_text = sd.get("text_raw") or sd.get("caption_text") or sd.get("transcript")
            if scene_text:
                p = self._build_prompt(str(scene_text))
                scene_items.append(p)
                scene_ids.append(sid)
        # Batch-generate for scenes
        for i in range(0, len(scene_items), self.batch_size):
            batch_prompts = scene_items[i : i + self.batch_size]
            batch_ids = scene_ids[i : i + self.batch_size]
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                gen_ids = self.model.generate(**inputs, max_new_tokens=256)
            outs = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            marker = "=== START OUTPUT ==="
            for sid, out_text in zip(batch_ids, outs):
                if marker in out_text:
                    out_text = out_text.split(marker, 1)[1].strip()
                scene_tags = self._parse_response(out_text)
                scenes[sid]["tags"] = scene_tags
        return tags

    def tag_dataset(self, dataset) -> None:
        """Tag all datapoints in a VideoDataset (in-place)."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        for dp in dataset.video_datapoints:
            try:
                self.tag_datapoint(dp)
            except Exception as e:
                logging.error(f"[Tagger] Failed to tag datapoint {getattr(dp, 'video_name', '<unknown>')}: {e}")

    def infer_tags_from_text(self, text: str) -> List[str]:
        """Infer tags from a raw text string using the loaded model.

        Loads the model if not already loaded.
        """
        if not text:
            return []
        if self.tokenizer is None or self.model is None:
            try:
                self.load_model()
            except Exception as e:
                logging.warning(f"[Tagger] Could not load model to infer tags: {e}")
                return []

        prompt = self._build_prompt(text)
        inputs = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            gen_ids = self.model.generate(**inputs, max_new_tokens=128)
        out = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return self._parse_response(out)

    def _truncate(self, text: str, max_len: int = 200, suffix: str = "...") -> str:
        if not text:
            return ""
        t = str(text)
        if len(t) <= max_len:
            return t
        return t[:max_len].rstrip() + suffix

    def pretty_print_datapoint(
        self,
        dp,
        max_text_len: int = 200,
        show_scenes: bool = True,
        truncate_suffix: str = "...",
        color: bool = False,
    ) -> None:
        """Print a VideoDataPoint's global screenplay/text and tags, and per-scene text/tags.

        - `dp` is a VideoDataPoint (or any object with `global_embeddings` and `scene_embeddings`).
        - `color` enables simple ANSI coloring for tags (works in most terminals).
        """
        GREEN = "\033[92m" if color else ""
        YELLOW = "\033[93m" if color else ""
        RESET = "\033[0m" if color else ""

        name = getattr(dp, "video_name", getattr(dp, "video_path", "<unknown>"))
        print("=" * 80)
        print(f"Video: {name}")

        ge = getattr(dp, "global_embeddings", {}) or {}
        txt = ge.get("text_raw")
        txt = self._truncate(txt, max_text_len, truncate_suffix)
        tags = ge.get("tags") or getattr(dp, "tags", []) or []
        tags_str = ", ".join(tags) if tags else "(none)"

        print(f" Global text: {YELLOW}{txt}{RESET}")
        print(f" Global tags: {GREEN}{tags_str}{RESET}")

        if show_scenes:
            scenes = getattr(dp, "scene_embeddings", {}) or {}
            if not scenes:
                print(" No scene embeddings.")
            else:
                print(" Scenes:")
                for sid, sd in sorted(scenes.items()):
                    if not isinstance(sd, dict):
                        continue
                    st = sd.get("text_raw")
                    st = self._truncate(st, max_text_len, truncate_suffix)
                    stag = sd.get("tags") or []
                    stag_str = ", ".join(stag) if stag else "(none)"
                    print(f"  - {sid}: {YELLOW}{st}{RESET}")
                    print(f"     tags: {GREEN}{stag_str}{RESET}")

        print()

    def pretty_print_dataset(
        self,
        dataset,
        max_text_len: int = 200,
        show_scenes: bool = True,
        truncate_suffix: str = "...",
        color: bool = False,
    ) -> None:
        """Pretty-print all datapoints in a VideoDataset."""
        for dp in dataset.video_datapoints:
            self.pretty_print_datapoint(
                dp,
                max_text_len=max_text_len,
                show_scenes=show_scenes,
                truncate_suffix=truncate_suffix,
                color=color,
            )
