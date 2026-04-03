"""
Visual Context Extractor
========================
Before retrieval, this module "looks at" the moment in time when the query was asked
and generates a textual description of the object/person the query refers to.

WHY: Some queries have an implicit visual referent that is not captured in the text
alone. E.g. "Where was this marker before I picked it up?" — the word "this" only
makes sense if you can see the marker in the query-time clip. The description produced
here is then used as a *second retrieval signal* (Strategy 3: parallel retrieval + late
RRF fusion), so the final ranked list is the fusion of:
    - results from the original query text
    - results from the visual-context description

FLOW:
  1. For each query, read query_time.date + query_time.time from metadata.
  2. Find the corresponding video clip (same logic used for GT clip lookup).
  3. Generate a description:
       a. VLM mode  (use_vllm=True) : extract N frames → Qwen3-VL → one-sentence description
       b. Fallback  (use_vllm=False): load the pre-encoded pkl → read text_raw (the
          caption already computed during indexing). Zero extra inference cost.
  4. Store the description in query.decomposed["visual_context"].

The extractor must be called BEFORE the retriever (which loads InternVideo2/Gemma) to
avoid having two large models in GPU memory simultaneously. The correct call order is:

    extractor.load_model()        # loads Qwen3-VL
    extractor.extract(queries)    # fills query.decomposed["visual_context"]
    extractor.unload_model()      # frees GPU
    retriever.retrieve_hierarchically(queries, use_visual_context=True)
"""

import gc
import logging
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class VisualContextExtractor:
    """
    Enriches queries with a textual description of the visual referent present at
    query time.  The description is stored in query.decomposed["visual_context"] and
    later used by the retriever as a parallel retrieval signal.

    Args:
        video_dir   : Root directory of EgoLife videos (contains person sub-folders).
        pkl_dir     : Directory with pre-encoded clip pkls (*_encoded.pkl).
        model_name  : Qwen3-VL model id (only used when use_vllm=True).
        use_vllm    : If True, run Qwen3-VL on the query-time clip. If False, fall back
                      to the text_raw caption from the clip's pkl (free — no inference).
        num_frames  : Number of frames to sample from the clip for VLM input.
        temp_dir    : Temporary directory for extracted frames.
        device      : Device string (not used directly; Qwen3-VL uses device_map="auto").
    """

    # Prompt template for the VLM grounding step.
    _VLM_PROMPT = (
        "You are analysing frames from a first-person (egocentric) video.\n"
        "Given the following question:\n"
        "  \"{query}\"\n"
        "Describe in one or two sentences the specific object, person, or thing "
        "that is visible in the video and that the question is referring to. "
        "Be concise and focus only on its appearance and location."
    )

    def __init__(
        self,
        video_dir: str,
        pkl_dir: str,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        use_vllm: bool = True,
        num_frames: int = 4,
        temp_dir: str = "/tmp/vc_frames",
        device: str = "cuda",
    ):
        self.video_dir = video_dir
        self.pkl_dir = pkl_dir
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.num_frames = num_frames
        self.temp_dir = Path(temp_dir)
        self.device = device

        # Qwen3-VL model + processor (loaded lazily)
        self.model = None
        self.processor = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load Qwen3-VL. Must be called before extract() when use_vllm=True."""
        if self.model is not None:
            return
        logger.info(f"[VCE] Loading {self.model_name} for visual context extraction ...")
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model.eval()
        logger.info("[VCE] Model loaded.")

    def unload_model(self) -> None:
        """Unload Qwen3-VL and free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("[VCE] Model unloaded.")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def extract(self, queries, person_folder: str) -> int:
        """
        Enrich every query in `queries` with a visual context description.

        For each query:
          - The query-time clip is located via query_time metadata
            (query.decomposed["metadata"]["query_date"] / "query_time_sec").
          - If use_vllm=True and the model is loaded: run Qwen3-VL on sampled frames.
          - Otherwise fall back to text_raw from the clip's pkl.
          - The result is stored in query.decomposed["visual_context"].

        Args:
            queries       : QueryDataset — queries are modified in place.
            person_folder : e.g. "A1_JAKE". Used to locate the day sub-folder.

        Returns:
            Number of queries successfully enriched.
        """
        enriched = 0
        queries_list = list(queries)

        for query in queries_list:
            clip_path = self._find_query_time_clip(query, person_folder)
            if clip_path is None:
                logger.debug(f"[VCE] No query-time clip found for query {query.qid}")
                continue

            description = None

            # Try VLM path first
            if self.use_vllm and self.model is not None:
                description = self._describe_object_vllm(clip_path, query.get_query())
                if not description:
                    logger.debug(
                        f"[VCE] VLM returned empty description for query {query.qid}; "
                        f"falling back to text_raw."
                    )

            # Fall back to pre-computed caption from pkl
            if not description:
                description = self._get_text_raw_fallback(clip_path)

            if description:
                query.decomposed["visual_context"] = description
                enriched += 1
                logger.debug(
                    f"[VCE] Query {query.qid}: visual_context = '{description[:100]}...'"
                )
            else:
                logger.debug(
                    f"[VCE] Could not produce visual context for query {query.qid}."
                )

        logger.info(
            f"[VCE] Enriched {enriched}/{len(queries_list)} queries with visual context."
        )
        return enriched

    # ------------------------------------------------------------------
    # Clip finding
    # ------------------------------------------------------------------

    def _find_query_time_clip(self, query, person_folder: str) -> Optional[str]:
        """
        Return the path of the clip that was being recorded at query time.
        Uses the same "largest clip-timestamp ≤ target" logic as the benchmark scripts.
        """
        meta = (getattr(query, "decomposed", {}) or {}).get("metadata", {}) or {}
        query_date = meta.get("query_date")       # e.g. "DAY4"
        query_time_sec = meta.get("query_time_sec")  # float, seconds from midnight

        if not query_date or query_time_sec is None:
            return None

        day_path = os.path.join(self.video_dir, person_folder, str(query_date).upper())
        if not os.path.isdir(day_path):
            return None

        candidates = []
        for fname in os.listdir(day_path):
            if not fname.lower().endswith(".mp4"):
                continue
            parts = os.path.splitext(fname)[0].split("_")
            clip_sec = self._parse_timestamp(parts[-1] if parts else "")
            candidates.append((clip_sec, os.path.join(day_path, fname)))

        if not candidates:
            return None

        valid = [(s, p) for s, p in candidates if s <= query_time_sec]
        if valid:
            return max(valid, key=lambda x: x[0])[1]
        return min(candidates, key=lambda x: x[0])[1]

    @staticmethod
    def _parse_timestamp(ts: str) -> float:
        """Parse HHMMSSCC string to total seconds."""
        if not ts or len(ts) < 6:
            return 0.0
        try:
            hh = int(ts[0:2]); mm = int(ts[2:4]); ss = int(ts[4:6])
            cc = int(ts[6:8]) if len(ts) >= 8 else 0
            return hh * 3600 + mm * 60 + ss + cc / 100.0
        except (ValueError, IndexError):
            return 0.0

    # ------------------------------------------------------------------
    # Description methods
    # ------------------------------------------------------------------

    def _describe_object_vllm(self, clip_path: str, query_text: str) -> Optional[str]:
        """
        Extract N frames from the clip and ask Qwen3-VL to describe the visual referent
        of the query.
        """
        frame_dir = self.temp_dir / f"vc_{Path(clip_path).stem}"
        frame_paths = self._extract_frames(clip_path, str(frame_dir))
        if not frame_paths:
            return None

        prompt = self._VLM_PROMPT.format(query=query_text)

        try:
            content = [{"type": "image", "image": fp} for fp in frame_paths]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]

            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text_input],
                images=frame_paths,
                padding=True,
                return_tensors="pt",
            ).to(next(self.model.parameters()).device)

            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs, max_new_tokens=128, do_sample=False
                )
            ans_ids = out_ids[0][inputs.input_ids.shape[1]:]
            description = self.processor.decode(
                ans_ids, skip_special_tokens=True
            ).strip()
            return description or None

        except Exception as exc:
            logger.warning(f"[VCE] VLM inference failed for {clip_path}: {exc}")
            return None
        finally:
            shutil.rmtree(str(frame_dir), ignore_errors=True)

    def _get_text_raw_fallback(self, clip_path: str) -> Optional[str]:
        """
        Load the pre-encoded pkl for the clip and return its text_raw caption
        (computed during indexing with the LLaVA captioner). Zero inference cost.
        """
        clip_base = os.path.splitext(os.path.basename(clip_path))[0]
        pkl_path = os.path.join(self.pkl_dir, f"{clip_base}_encoded.pkl")
        if not os.path.exists(pkl_path):
            return None
        try:
            with open(pkl_path, "rb") as f:
                clip_ds = pickle.load(f)
            dp = clip_ds.video_datapoints[0]
            text_raw = dp.scene_embeddings.get("scene_0", {}).get("text_raw")
            return str(text_raw).strip() if text_raw else None
        except Exception as exc:
            logger.debug(f"[VCE] Could not load pkl {pkl_path}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def _extract_frames(self, clip_path: str, out_dir: str) -> list[str]:
        """
        Extract self.num_frames evenly-spaced frames from the clip using ffmpeg.
        Assumes clips are ~30 s, so fps = num_frames/30 gives correct spacing.
        Falls back to a single frame at t=1s on ffmpeg failure.
        """
        os.makedirs(out_dir, exist_ok=True)
        fps_val = self.num_frames / 30.0  # e.g. 4/30 ≈ 0.133 → 4 frames over 30 s
        cmd = [
            "ffmpeg", "-y", "-i", clip_path,
            "-vf", f"fps={fps_val:.4f},scale=420:-1",
            os.path.join(out_dir, "frame_%04d.jpg"),
            "-loglevel", "error",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            # Fallback: single frame
            cmd2 = [
                "ffmpeg", "-y", "-ss", "1", "-i", clip_path,
                "-frames:v", "1", "-vf", "scale=420:-1",
                os.path.join(out_dir, "frame_0001.jpg"),
                "-loglevel", "error",
            ]
            subprocess.run(cmd2, capture_output=True)

        frames = sorted(Path(out_dir).glob("*.jpg"))
        return [str(f) for f in frames[: self.num_frames]]
