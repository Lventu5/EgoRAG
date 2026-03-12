"""
EgoLife QA Benchmark - Three evaluation modes:

1. llm_only : Text-only LLM baseline. Feed question + 4 choices to Qwen3-VL, no video.
2. gt_video : Feed the ground-truth video clip + question to Qwen3-VL and measure accuracy.
3. egorag   : Use the EgoRAG retrieval pipeline to retrieve a scene, then feed it to Qwen3-VL.

Edit the CONFIG block at the top of this file to change paths/settings, then run:
    python -m test.benchmark_egolife_qa --mode gt_video
or:
    python -m test.benchmark_egolife_qa --mode llm_only
    python -m test.benchmark_egolife_qa --mode egorag
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIG — edit here
# ---------------------------------------------------------------------------
MODE           = "llm_only"   # "llm_only" | "gt_video" | "egorag"
QA_DIR         = "/cluster/project/cvg/data/EgoLife/EgoLifeQA"
VIDEO_DIR      = "/cluster/project/cvg/data/EgoLife"
PKL_DIR        = "/cluster/project/cvg/students/tnanni/ego4d_data/v2/egolife_full"
MODEL_NAME     = "Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR     = "./results/benchmark_qa2"
TEMP_DIR       = "/tmp/benchmark_frames"
PERSONS        = None   # None = all available; or a list like ["A1_JAKE", "A2_ALICE"]
MAX_QUESTIONS  = None   # None = all; or an int to limit per person (useful for quick tests)
TOPK_SCENES    = 3      # how many retrieved scenes to try in egorag mode

# ---------------------------------------------------------------------------
# QA loading helpers
# ---------------------------------------------------------------------------

def load_qa_file(path: str) -> List[dict]:
    """Load a single EgoLifeQA JSON file and return the list of entries."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def build_mc_prompt(entry: dict, include_trigger: bool = True) -> str:
    """Build the multiple-choice prompt text from a QA entry."""
    trigger = entry.get("trigger", "").strip()
    question = entry.get("question", "").strip()
    a = entry.get("choice_a", "")
    b = entry.get("choice_b", "")
    c = entry.get("choice_c", "")
    d = entry.get("choice_d", "")

    context = f"Context: {trigger}\n" if (include_trigger and trigger) else ""
    prompt = (
        f"{context}"
        f"Question: {question}\n\n"
        f"Options:\n"
        f"  A. {a}\n"
        f"  B. {b}\n"
        f"  C. {c}\n"
        f"  D. {d}\n\n"
        f"Answer with only the letter of the correct option (A, B, C, or D). "
        f"Do not add any explanation."
    )
    return prompt


def parse_answer(raw: str) -> str:
    """Extract a single letter (A/B/C/D) from the model's raw output."""
    raw = raw.strip()
    # First, try to match a standalone letter at the very start
    m = re.match(r"^\s*([ABCD])\b", raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: find the first occurrence of A/B/C/D anywhere
    m = re.search(r"\b([ABCD])\b", raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return ""


def compute_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    correct = sum(p == g for p, g in zip(predictions, ground_truths))
    return correct / len(ground_truths) if ground_truths else 0.0


# ---------------------------------------------------------------------------
# Timestamp / clip-finding helpers
# ---------------------------------------------------------------------------

def parse_timestamp(time_str: str) -> float:
    """Parse hhmmsscc -> total seconds (cc = centiseconds)."""
    if not time_str or len(time_str) < 6:
        return 0.0
    hh = int(time_str[0:2])
    mm = int(time_str[2:4])
    ss = int(time_str[4:6])
    cc = int(time_str[6:8]) if len(time_str) >= 8 else 0
    return hh * 3600 + mm * 60 + ss + cc / 100.0


def find_gt_clip(video_dir: str, person_folder: str, date: str, time_str: str) -> Optional[str]:
    """
    Find the video clip file that covers the target timestamp.

    Returns the path to the clip file whose timestamp is the largest that is
    still <= the target timestamp (i.e. the clip that was recording at that moment).
    """
    day_path = os.path.join(video_dir, person_folder, date)
    if not os.path.isdir(day_path):
        # Try uppercase variant (e.g. "Day1" -> "DAY1")
        day_path_upper = os.path.join(video_dir, person_folder, date.upper())
        if os.path.isdir(day_path_upper):
            day_path = day_path_upper
        else:
            logger.warning(f"Day folder not found: {day_path}")
            return None

    target_sec = parse_timestamp(time_str)

    # Gather all mp4 files and their timestamps
    candidates = []
    for fname in os.listdir(day_path):
        if not fname.lower().endswith(".mp4"):
            continue
        parts = os.path.splitext(fname)[0].split("_")
        clip_time_str = parts[-1] if parts else ""
        clip_sec = parse_timestamp(clip_time_str)
        candidates.append((clip_sec, os.path.join(day_path, fname)))

    if not candidates:
        logger.warning(f"No mp4 clips found in {day_path}")
        return None

    # Pick the clip whose start timestamp is <= target and closest to it
    valid = [(sec, path) for sec, path in candidates if sec <= target_sec]
    if valid:
        return max(valid, key=lambda x: x[0])[1]
    # If all clips start after the target, pick the earliest one
    return min(candidates, key=lambda x: x[0])[1]


def find_pkl_for_clip(pkl_dir: str, clip_path: str) -> Optional[str]:
    """Return the encoded pkl path for a given clip file."""
    clip_base = os.path.splitext(os.path.basename(clip_path))[0]
    pkl_path = os.path.join(pkl_dir, f"{clip_base}_encoded.pkl")
    return pkl_path if os.path.exists(pkl_path) else None


def filter_to_embedded_entries(
    qa_entries: List[dict],
    video_dir: str,
    person_folder: str,
    pkl_dir: str,
) -> List[dict]:
    """Keep only entries whose GT clip has a pre-computed pkl in pkl_dir.

    Applied to all modes so they evaluate on the same subset.
    """
    entries, skipped = [], 0
    for entry in qa_entries:
        tt = entry.get("target_time", {})
        clip_path = find_gt_clip(video_dir, person_folder, tt.get("date", ""), tt.get("time", ""))
        if clip_path and find_pkl_for_clip(pkl_dir, clip_path):
            entries.append(entry)
        else:
            skipped += 1
    if skipped:
        logger.info(
            f"Skipping {skipped} entries with no embedded clip "
            f"(evaluating on {len(entries)} entries)."
        )
    return entries


# ---------------------------------------------------------------------------
# Frame extraction (shared by gt_video and egorag modes)
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, out_dir: str, fps: int = 1) -> List[str]:
    """Extract frames from a video at `fps` frames/sec into `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"fps={fps},scale=420:-1",
        os.path.join(out_dir, "frame_%05d.jpg"),
        "-loglevel", "error",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg failed for {video_path}: {e.stderr.decode()}")
    frames = sorted(Path(out_dir).glob("*.jpg"))
    # Fallback: extract single frame if fps extraction produced nothing
    if not frames:
        cmd2 = [
            "ffmpeg", "-y", "-ss", "0", "-i", video_path,
            "-frames:v", "1", "-vf", "scale=420:-1",
            os.path.join(out_dir, "frame_00001.jpg"),
            "-loglevel", "error",
        ]
        subprocess.run(cmd2, check=True, capture_output=True)
        frames = sorted(Path(out_dir).glob("*.jpg"))
    return [str(f) for f in frames]


# ---------------------------------------------------------------------------
# Qwen3-VL model wrapper
# ---------------------------------------------------------------------------

class QwenVLModel:
    """Thin wrapper around Qwen3-VL for multi-choice QA."""

    def __init__(self, model_name: str = MODEL_NAME, max_new_tokens: int = 64):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None

    def load(self):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        logger.info(f"Loading model {self.model_name} ...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model.eval()
        logger.info("Model loaded.")

    def unload(self):
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def answer_text_only(self, prompt: str) -> str:
        """Answer a text-only prompt (no video frames)."""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        inputs = inputs.to(next(self.model.parameters()).device)
        out_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        ans_ids = out_ids[0][inputs.input_ids.shape[1]:]
        return self.processor.decode(ans_ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def answer_with_frames(self, frame_paths: List[str], prompt: str) -> str:
        """Answer a prompt given a list of image frame paths."""
        content = [{"type": "image", "image": fp} for fp in frame_paths]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[fp for fp in frame_paths],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(next(self.model.parameters()).device)
        out_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        ans_ids = out_ids[0][inputs.input_ids.shape[1]:]
        return self.processor.decode(ans_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Test 1: LLM-only baseline
# ---------------------------------------------------------------------------

def run_llm_only(
    qa_entries: List[dict],
    model: QwenVLModel,
    max_questions: Optional[int] = None,
) -> Tuple[List[dict], float]:
    """Feed question + choices to the LLM (no video). Return (records, accuracy)."""
    entries = qa_entries[:max_questions] if max_questions else qa_entries
    records = []
    preds, gts = [], []

    for entry in tqdm(entries, desc="LLM-only"):
        prompt = build_mc_prompt(entry, include_trigger=True)
        raw = model.answer_text_only(prompt)
        pred = parse_answer(raw)
        gt = entry.get("answer", "").strip().upper()

        records.append({
            "id": entry.get("ID"),
            "question": entry.get("question"),
            "gt": gt,
            "pred": pred,
            "raw_output": raw,
            "correct": pred == gt,
        })
        preds.append(pred)
        gts.append(gt)

    acc = compute_accuracy(preds, gts)
    return records, acc


# ---------------------------------------------------------------------------
# Test 2: vLLM + ground-truth video clip
# ---------------------------------------------------------------------------

def run_gt_video(
    qa_entries: List[dict],
    model: QwenVLModel,
    video_dir: str,
    person_folder: str,
    temp_dir: str,
    max_questions: Optional[int] = None,
) -> Tuple[List[dict], float]:
    """Feed the GT clip + question to the vLLM. Return (records, accuracy)."""
    entries = qa_entries[:max_questions] if max_questions else qa_entries
    records = []
    preds, gts = [], []

    for entry in tqdm(entries, desc="GT-video"):
        target_time = entry.get("target_time", {})
        date = target_time.get("date", "")
        time_str = target_time.get("time", "")
        gt = entry.get("answer", "").strip().upper()

        clip_path = find_gt_clip(video_dir, person_folder, date, time_str)
        prompt = build_mc_prompt(entry, include_trigger=True)
        raw = ""
        pred = ""

        if clip_path and os.path.exists(clip_path):
            clip_id = f"{entry.get('ID', 'unk')}_{person_folder}"
            frame_out = os.path.join(temp_dir, clip_id)
            try:
                frames = extract_frames(clip_path, frame_out, fps=1)
                if frames:
                    raw = model.answer_with_frames(frames, prompt)
                    pred = parse_answer(raw)
                else:
                    logger.warning(f"No frames extracted for entry {entry.get('ID')}")
            except Exception as e:
                logger.error(f"Error processing clip {clip_path}: {e}")
            finally:
                shutil.rmtree(frame_out, ignore_errors=True)
        else:
            logger.warning(
                f"GT clip not found for entry {entry.get('ID')} "
                f"(date={date}, time={time_str})"
            )

        records.append({
            "id": entry.get("ID"),
            "question": entry.get("question"),
            "gt": gt,
            "pred": pred,
            "raw_output": raw,
            "correct": pred == gt,
            "clip_path": clip_path or "",
        })
        preds.append(pred)
        gts.append(gt)

    acc = compute_accuracy(preds, gts)
    return records, acc


# ---------------------------------------------------------------------------
# Test 3: EgoRAG pipeline retrieval + vLLM
# ---------------------------------------------------------------------------

def _load_egolife_with_embeddings(
    video_dir: str,
    person_folder: str,
    pkl_dir: str,
    annotation_path: str,
) -> "VideoDataset":
    """
    Load each pre-encoded clip PKL for the given person as its own VideoDataPoint.
    Each clip already has global_embeddings (video + text) properly filled, so we
    use them directly for retrieval instead of grouping clips into day-level datapoints.
    """
    import pickle
    import glob as glob_module
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.video_dataset import VideoDataset

    pattern = os.path.join(pkl_dir, f"*{person_folder.upper()}*_encoded.pkl")
    pkl_files = sorted(glob_module.glob(pattern))

    if not pkl_files:
        logger.warning(f"No encoded PKL files found for {person_folder} in {pkl_dir}")
        ds = VideoDataset(video_files=[])
        ds.encoded = True
        return ds

    datapoints = []
    for pkl_path in tqdm(pkl_files, desc=f"Loading {person_folder} clips"):
        try:
            with open(pkl_path, "rb") as f:
                clip_ds: VideoDataset = pickle.load(f)
            dp = clip_ds.video_datapoints[0]
            datapoints.append(dp)
        except Exception as e:
            logger.debug(f"Could not load {pkl_path}: {e}")

    logger.info(f"Loaded {len(datapoints)} clip datapoints for {person_folder}.")

    ds = VideoDataset(video_files=[])
    ds.video_files = [dp.video_path for dp in datapoints]
    ds.video_datapoints = datapoints
    ds.encoded = True
    return ds


def run_egorag(
    qa_entries: List[dict],
    model: QwenVLModel,
    video_dir: str,
    person_folder: str,
    pkl_dir: str,
    annotation_path: str,
    temp_dir: str,
    topk_scenes: int = 5,
    max_questions: Optional[int] = None,
) -> Tuple[List[dict], float]:
    """Use EgoRAG retrieval to find the relevant scene, then answer with the vLLM."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.query import Query, QueryDataset
    from retrieval.hierarchical_retriever import HierarchicalRetriever
    from configuration.config import CONFIG

    entries = qa_entries[:max_questions] if max_questions else qa_entries

    # Load dataset with embeddings
    logger.info("Loading EgoLife video dataset with embeddings...")
    video_dataset = _load_egolife_with_embeddings(
        video_dir, person_folder, pkl_dir, annotation_path
    )
    logger.info(
        f"Loaded {len(video_dataset.video_datapoints)} day-videos for {person_folder}"
    )

    # Build queries for retrieval (one per QA entry)
    queries = []
    for entry in entries:
        qt = entry.get("query_time", {})
        question = entry.get("question", "")
        trigger = entry.get("trigger", "")
        query_text = f"{trigger}. {question}" if trigger else question
        qid = str(entry.get("ID", "unk"))
        queries.append(Query(
            qid=qid,
            query_text=query_text,
            video_uid=None,
        ))
    query_dataset = QueryDataset(queries)

    # Run retrieval (text modality only — video embeddings may vary)
    modalities = getattr(CONFIG.retrieval, "modalities", ["text", "video"])
    logger.info(f"Running retrieval with modalities={modalities}, topk_scenes={topk_scenes}")
    retriever = HierarchicalRetriever(video_dataset=video_dataset, device="cuda")
    retrieval_results = retriever.retrieve_hierarchically(
        queries=query_dataset,
        modalities=modalities,
        top_k_videos=len(video_dataset.video_datapoints),
        top_k_scenes=topk_scenes,
        skip_video_retrieval=False,
        use_windows=False,
        use_tagging=False,
    )
    retriever.unload_models()
    torch.cuda.empty_cache()

    # Generate answers using retrieved scenes
    records = []
    preds, gts = [], []

    for entry, query in tqdm(zip(entries, queries), total=len(entries), desc="EgoRAG"):
        gt = entry.get("answer", "").strip().upper()
        prompt = build_mc_prompt(entry, include_trigger=True)

        qid = query.qid
        entry_results = retrieval_results.get(qid, {})
        fused = entry_results.get("fused", []) if isinstance(entry_results, dict) else entry_results

        # Collect top scenes sorted by score
        scenes_with_scores = []
        for video_name, _vscore, scene_ranking in fused:
            for scene, sscore in (scene_ranking or []):
                scenes_with_scores.append((scene, sscore))
        scenes_with_scores.sort(key=lambda x: x[1], reverse=True)

        raw = ""
        pred = ""
        used_clip = ""

        # Try scenes in order until we get a successful inference
        for scene, _score in scenes_with_scores[:topk_scenes]:
            clip_path = getattr(scene, "source_path", None)
            if not clip_path or not os.path.exists(clip_path):
                continue
            frame_out = os.path.join(temp_dir, f"egorag_{qid}")
            try:
                frames = extract_frames(clip_path, frame_out, fps=1)
                if frames:
                    raw = model.answer_with_frames(frames, prompt)
                    pred = parse_answer(raw)
                    used_clip = clip_path
                    break
            except Exception as e:
                logger.error(f"Error on scene {scene.scene_id}: {e}")
            finally:
                shutil.rmtree(frame_out, ignore_errors=True)

        records.append({
            "id": entry.get("ID"),
            "question": entry.get("question"),
            "gt": gt,
            "pred": pred,
            "raw_output": raw,
            "correct": pred == gt,
            "retrieved_clip": used_clip,
        })
        preds.append(pred)
        gts.append(gt)

    acc = compute_accuracy(preds, gts)
    return records, acc


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    records: List[dict],
    accuracy: float,
    mode: str,
    person: str,
    output_dir: str,
    model_name: str,
):
    os.makedirs(output_dir, exist_ok=True)
    out = {
        "mode": mode,
        "person": person,
        "model": model_name,
        "num_questions": len(records),
        "accuracy": accuracy,
        "num_correct": sum(r["correct"] for r in records),
        "records": records,
    }
    fname = f"{mode}_{person}.json"
    out_path = os.path.join(output_dir, fname)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Results saved to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main():
    import argparse
    sys.path.insert(0, str(Path(__file__).parent.parent))
    parser = argparse.ArgumentParser(description="Run EgoLife QA benchmark.")
    parser.add_argument(
        "--mode",
        choices=["llm_only", "gt_video", "egorag"],
        default=MODE,
        help="Benchmark mode. Defaults to MODE in the config block.",
    )
    args = parser.parse_args()
    mode = args.mode

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["llm_only", "gt_video", "egorag"],
                        default=MODE, help="Evaluation mode (overrides MODE in the config block)")
    args, _ = parser.parse_known_args()
    mode = args.mode

    # Discover QA files
    qa_dir = Path(QA_DIR)
    qa_files = sorted(qa_dir.glob("EgoLifeQA_*.json"))
    if not qa_files:
        logger.error(f"No QA JSON files found in {QA_DIR}")
        sys.exit(1)

    # Filter by requested persons
    if PERSONS:
        qa_files = [f for f in qa_files if any(p in f.name for p in PERSONS)]
    if not qa_files:
        logger.error("No QA files matched the requested persons.")
        sys.exit(1)

    logger.info(f"Mode: {mode} | Files: {[f.name for f in qa_files]}")

    # Load model once, shared across all persons
    model = QwenVLModel(model_name=MODEL_NAME, max_new_tokens=64)
    model.load()

    all_correct, all_total = 0, 0

    for qa_file in qa_files:
        # EgoLifeQA_A1_JAKE.json -> A1_JAKE
        person = qa_file.stem.replace("EgoLifeQA_", "")
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing person: {person} | file: {qa_file.name}")

        qa_entries = load_qa_file(str(qa_file))
        logger.info(f"Loaded {len(qa_entries)} QA entries")

        # Restrict all modes to entries whose GT clip is embedded — ensures a fair comparison
        qa_entries = filter_to_embedded_entries(qa_entries, VIDEO_DIR, person, PKL_DIR)
        if not qa_entries:
            logger.warning(f"No embedded clips found for {person}, skipping.")
            continue

        if mode == "llm_only":
            records, acc = run_llm_only(qa_entries, model, max_questions=MAX_QUESTIONS)

        elif mode == "gt_video":
            records, acc = run_gt_video(
                qa_entries, model,
                video_dir=VIDEO_DIR,
                person_folder=person,
                temp_dir=TEMP_DIR,
                max_questions=MAX_QUESTIONS,
            )

        elif mode == "egorag":
            records, acc = run_egorag(
                qa_entries, model,
                video_dir=VIDEO_DIR,
                person_folder=person,
                pkl_dir=PKL_DIR,
                annotation_path=str(qa_file),
                temp_dir=TEMP_DIR,
                topk_scenes=TOPK_SCENES,
                max_questions=MAX_QUESTIONS,
            )

        else:
            logger.error(f"Unknown mode: {mode!r}. Choose llm_only, gt_video, or egorag.")
            sys.exit(1)

        logger.info(f"[{person}] Accuracy: {acc*100:.2f}% ({sum(r['correct'] for r in records)}/{len(records)})")
        save_results(records, acc, mode, person, OUTPUT_DIR, MODEL_NAME)

        all_correct += sum(r["correct"] for r in records)
        all_total += len(records)

    overall_acc = all_correct / all_total if all_total else 0.0
    logger.info(f"\n{'='*60}")
    logger.info(f"OVERALL ACCURACY ({mode}): {overall_acc*100:.2f}% ({all_correct}/{all_total})")

    summary = {
        "mode": mode,
        "model": MODEL_NAME,
        "overall_accuracy": overall_acc,
        "num_correct": all_correct,
        "num_total": all_total,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"{mode}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {OUTPUT_DIR}/{mode}_summary.json")

    model.unload()


if __name__ == "__main__":
    main()
