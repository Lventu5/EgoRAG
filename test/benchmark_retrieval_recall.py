"""
EgoLife Retrieval Recall Benchmark

Measures recall@k: for how many questions the ground-truth clip appears
among the top-k retrieved clips. Sweeps k over TOPK_LIST and runs every
combination of modalities × orchestrator defined in the CONFIG block below.

No LLM generation — pure retrieval, so 1 GPU is enough.

Edit the CONFIG block at the top of this file, then run:
    python -m test.benchmark_retrieval_recall
"""

import json
import logging
import os
import sys
import pickle
import glob as glob_module
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIG — edit here
# ---------------------------------------------------------------------------
QA_DIR        = "/cluster/project/cvg/data/EgoLife/EgoLifeQA"
VIDEO_DIR     = "/cluster/project/cvg/data/EgoLife"
PKL_DIR       = "/cluster/project/cvg/students/tnanni/ego4d_data/v2/egolife_full"
OUTPUT_DIR    = "./results/retrieval_recall"
PERSONS       = None   # None = all available; or e.g. ["A1_JAKE"]
MAX_QUESTIONS  = None   # None = all; set an int for quick tests
QUESTION_SLICE = None   # None = all; or (start, end) e.g. (0, 20) to use first 20
FILTER_NEED_NAME = True    # True = exclude questions with need_name=True

TOPK_LIST = [1, 3, 5, 10, 20, 25, 40, 50, 75, 100, 200]

# --- Visual context ---
# When USE_VISUAL_CONTEXT=True the query-time clip is looked up for every query,
# a description of the visual referent is generated (VLM or text_raw fallback),
# and a second retrieval pass is fused (RRF) with the standard query pass.
USE_VISUAL_CONTEXT  = False   # Master flag
VC_USE_VLLM         = False   # False = text_raw fallback (no extra GPU cost); True = Qwen3-VL
VC_NUM_FRAMES       = 4
VC_TEMP_DIR         = f"/tmp/vc_frames_{os.getenv('USER', 'unknown')}"

# Each entry is one retrieval configuration to evaluate.
# "modalities"      : subset of ["text", "video", "audio"]
# "orchestrator"    : True / False
# "visual_context"  : True / False (overrides USE_VISUAL_CONTEXT per-config)
CONFIGURATIONS = [
    {"modalities": ["text"],           "orchestrator": False, "visual_context": False},
    {"modalities": ["text"],           "orchestrator": True,  "visual_context": False},
    {"modalities": ["video"],          "orchestrator": False, "visual_context": False},
    {"modalities": ["video"],          "orchestrator": True,  "visual_context": False},
    {"modalities": ["text", "video"],  "orchestrator": False, "visual_context": False},
    {"modalities": ["text", "video"],  "orchestrator": True,  "visual_context": False},
]


# ---------------------------------------------------------------------------
# QA / clip helpers
# ---------------------------------------------------------------------------

def load_qa_file(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


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
    day_path = os.path.join(video_dir, person_folder, date)
    if not os.path.isdir(day_path):
        day_path_upper = os.path.join(video_dir, person_folder, date.upper())
        if os.path.isdir(day_path_upper):
            day_path = day_path_upper
        else:
            return None

    target_sec = parse_timestamp(time_str)
    candidates = []
    for fname in os.listdir(day_path):
        if not fname.lower().endswith(".mp4"):
            continue
        parts = os.path.splitext(fname)[0].split("_")
        clip_sec = parse_timestamp(parts[-1] if parts else "")
        candidates.append((clip_sec, os.path.join(day_path, fname)))

    if not candidates:
        return None
    valid = [(s, p) for s, p in candidates if s <= target_sec]
    if valid:
        return max(valid, key=lambda x: x[0])[1]
    return min(candidates, key=lambda x: x[0])[1]


def find_pkl_for_clip(pkl_dir: str, clip_path: str) -> Optional[str]:
    clip_base = os.path.splitext(os.path.basename(clip_path))[0]
    pkl_path = os.path.join(pkl_dir, f"{clip_base}_encoded.pkl")
    return pkl_path if os.path.exists(pkl_path) else None


def filter_to_embedded_entries(
    qa_entries: List[dict], video_dir: str, person_folder: str, pkl_dir: str
) -> List[dict]:
    entries, skipped = [], 0
    for entry in qa_entries:
        tt = entry.get("target_time", {})
        clip = find_gt_clip(video_dir, person_folder, tt.get("date", ""), tt.get("time", ""))
        if clip and find_pkl_for_clip(pkl_dir, clip):
            entries.append(entry)
        else:
            skipped += 1
    if skipped:
        logger.info(f"Skipped {skipped} entries with no embedded clip ({len(entries)} remaining).")
    return entries


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_egolife_with_embeddings(video_dir: str, person_folder: str, pkl_dir: str):
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
                clip_ds = pickle.load(f)
            dp = clip_ds.video_datapoints[0]
            scene_emb = dp.scene_embeddings.get("scene_0", {})
            for key in ("video", "text"):
                if dp.global_embeddings.get(key) is None and scene_emb.get(key) is not None:
                    dp.global_embeddings[key] = scene_emb[key]
            datapoints.append(dp)
        except Exception as e:
            logger.debug(f"Could not load {pkl_path}: {e}")

    logger.info(f"Loaded {len(datapoints)} clip datapoints for {person_folder}.")
    ds = VideoDataset(video_files=[])
    ds.video_files = [dp.video_path for dp in datapoints]
    ds.video_datapoints = datapoints
    ds.encoded = True
    return ds


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def run_one_config(
    qa_entries: List[dict],
    video_dataset,
    video_name_to_path: Dict[str, str],
    person_folder: str,
    modalities: List[str],
    use_orchestrator: bool,
    topk_list: List[int],
    use_visual_context: bool = False,
) -> Dict:
    """
    Run retrieval once at max(topk_list) and compute recall@k for all k.
    Returns a result dict with per-question details and recall_at_k summary.
    """
    from data.query import Query, QueryDataset
    from retrieval.hierarchical_retriever import HierarchicalRetriever
    from configuration.config import CONFIG

    max_topk = max(topk_list)
    CONFIG.retrieval.orchestrator.enabled = use_orchestrator

    queries = []
    for entry in qa_entries:
        qt = entry.get("query_time", {})
        trigger = entry.get("trigger", "")
        question = entry.get("question", "")
        query_text = f"{trigger}. {question}" if trigger else question
        query_time_sec = None
        if qt.get("time"):
            try:
                query_time_sec = parse_timestamp(qt["time"])
            except Exception:
                pass
        queries.append(Query(
            qid=str(entry.get("ID", "unk")),
            query_text=query_text,
            video_uid=None,
            decomposed={
                "text": query_text,
                "audio": query_text,
                "video": query_text,
                "metadata": {
                    "query_date": qt.get("date"),
                    "query_time_sec": query_time_sec,
                },
            },
        ))
    query_dataset = QueryDataset(queries)

    # --- Visual context enrichment (optional) ---
    # Runs BEFORE the retriever so that Qwen3-VL (if used) is unloaded before
    # InternVideo2/Gemma are loaded — they cannot both fit in GPU memory.
    if use_visual_context:
        from retrieval.visual_context_extractor import VisualContextExtractor
        from configuration.config import CONFIG as _CFG
        vc_cfg = getattr(_CFG.retrieval, "visual_context", None)
        vc_extractor = VisualContextExtractor(
            video_dir=VIDEO_DIR,
            pkl_dir=PKL_DIR,
            model_name=getattr(vc_cfg, "model_name", "Qwen/Qwen3-VL-8B-Instruct") if vc_cfg else "Qwen/Qwen3-VL-8B-Instruct",
            use_vllm=VC_USE_VLLM,
            num_frames=VC_NUM_FRAMES,
            temp_dir=VC_TEMP_DIR,
        )
        if VC_USE_VLLM:
            vc_extractor.load_model()
        vc_extractor.extract(query_dataset, person_folder)
        if VC_USE_VLLM:
            vc_extractor.unload_model()
        torch.cuda.empty_cache()

    retriever = HierarchicalRetriever(video_dataset=video_dataset, device="cuda")
    retrieval_results = retriever.retrieve_hierarchically(
        queries=query_dataset,
        modalities=modalities,
        top_k_videos=max_topk,
        top_k_scenes=max_topk,
        skip_video_retrieval=False,
        use_windows=False,
        use_tagging=False,
        use_visual_context=use_visual_context,
    )
    retriever.unload_models()
    torch.cuda.empty_cache()

    per_question = []
    for entry, query in zip(qa_entries, queries):
        tt = entry.get("target_time", {})
        gt_clip = find_gt_clip(VIDEO_DIR, person_folder, tt.get("date", ""), tt.get("time", ""))
        gt_clip_abs = os.path.abspath(gt_clip) if gt_clip else None

        entry_results = retrieval_results.get(query.qid, {})
        fused = entry_results.get("fused", []) if isinstance(entry_results, dict) else entry_results

        scenes_ranked = []
        for video_name, _vscore, scene_ranking in fused:
            for scene, sscore in (scene_ranking or []):
                clip_path = getattr(scene, "source_path", None) or video_name_to_path.get(video_name)
                scenes_ranked.append((sscore, clip_path))
        scenes_ranked.sort(key=lambda x: x[0], reverse=True)

        hits_at_k = {
            k: gt_clip_abs is not None and gt_clip_abs in {
                os.path.abspath(c) for _, c in scenes_ranked[:k] if c
            }
            for k in topk_list
        }

        top5_clips = [
            os.path.basename(c) for _, c in scenes_ranked[:5] if c
        ]

        # Orchestrator accuracy: did the plan include the GT day / GT time?
        plan = getattr(query, "retrieval_plan", None) or {}
        temporal = plan.get("temporal", {}) or {}
        orch_allowed_days = temporal.get("allowed_days") or []
        orch_time_ranges  = temporal.get("time_ranges_sec") or []

        gt_day     = str(tt.get("date", "")).upper() or None   # e.g. "DAY1"
        gt_time_sec = parse_timestamp(tt.get("time", ""))

        # Day accuracy (only meaningful when orchestrator restricts to specific days)
        day_filtered = bool(orch_allowed_days)
        if day_filtered and gt_day:
            day_hit = gt_day in orch_allowed_days
        else:
            day_hit = None  # unrestricted — no filter was applied

        # Time accuracy (only meaningful when orchestrator restricts to time ranges)
        time_filtered = bool(orch_time_ranges)
        if time_filtered and gt_time_sec is not None:
            time_hit = any(
                float(s) <= gt_time_sec <= float(e)
                for s, e in orch_time_ranges
            )
        else:
            time_hit = None  # unrestricted — no filter was applied

        per_question.append({
            "id": entry.get("ID"),
            "question": entry.get("question"),
            "gt_clip": gt_clip or "",
            "gt_day": gt_day or "",
            "gt_time_sec": gt_time_sec,
            "top5_retrieved": top5_clips,
            "num_retrieved": len(scenes_ranked),
            "hits_at_k": hits_at_k,
            "retrieval_plan": plan if use_orchestrator else None,
            # orchestrator accuracy fields
            "orch_allowed_days": orch_allowed_days,
            "orch_time_ranges_sec": orch_time_ranges,
            "day_filtered": day_filtered,
            "day_hit": day_hit,
            "time_filtered": time_filtered,
            "time_hit": time_hit,
        })

    n = len(per_question)
    recall_at_k = {
        k: {
            "hits": sum(q["hits_at_k"][k] for q in per_question),
            "total": n,
            "recall": sum(q["hits_at_k"][k] for q in per_question) / n if n else 0.0,
        }
        for k in topk_list
    }

    # Orchestrator filter accuracy summary
    orch_accuracy = _compute_orchestrator_accuracy(per_question)

    return {
        "modalities": modalities,
        "use_orchestrator": use_orchestrator,
        "use_visual_context": use_visual_context,
        "num_questions": n,
        "recall_at_k": recall_at_k,
        "orchestrator_accuracy": orch_accuracy,
        "per_question": per_question,
    }


# ---------------------------------------------------------------------------
# Orchestrator accuracy
# ---------------------------------------------------------------------------

def _compute_orchestrator_accuracy(per_question: List[Dict]) -> Dict:
    """
    Summarise how often the orchestrator's day/time filters include the GT.

    day_filtered  / time_filtered : questions where a filter was actually applied
    day_hit_rate  / time_hit_rate : fraction of *filtered* questions where GT was inside
    day_coverage  / time_coverage : fraction of *all* questions where GT was inside
                                    (unrestricted questions count as a hit)
    """
    day_filtered_qs  = [q for q in per_question if q["day_filtered"]]
    time_filtered_qs = [q for q in per_question if q["time_filtered"]]

    n = len(per_question)

    def _rate(questions, field):
        hits = sum(1 for q in questions if q[field] is True)
        total = len(questions)
        return {"hits": hits, "total": total, "rate": hits / total if total else None}

    def _coverage(questions_filtered, field, n_all):
        """Fraction of ALL questions where GT day/time was included.
        Unrestricted questions (no filter) are treated as 'covered'."""
        uncovered = sum(1 for q in questions_filtered if q[field] is False)
        covered = n_all - uncovered
        return {"covered": covered, "total": n_all, "rate": covered / n_all if n_all else None}

    return {
        "num_questions": n,
        # --- day ---
        "day": {
            "num_day_filtered": len(day_filtered_qs),
            "hit_rate_when_filtered": _rate(day_filtered_qs, "day_hit"),
            "coverage_all": _coverage(day_filtered_qs, "day_hit", n),
        },
        # --- time of day ---
        "time_of_day": {
            "num_time_filtered": len(time_filtered_qs),
            "hit_rate_when_filtered": _rate(time_filtered_qs, "time_hit"),
            "coverage_all": _coverage(time_filtered_qs, "time_hit", n),
        },
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_recall_table(results: List[Dict], topk_list: List[int]):
    col_w = 11
    config_w = 28
    header = f"{'Configuration':<{config_w}}" + "".join(
        f"{'@'+str(k):>{col_w}}" for k in topk_list
    )
    sep = "=" * len(header)
    logger.info(f"\n{sep}\n{header}\n{'-' * len(header)}")
    for res in results:
        mod_str = "+".join(res["modalities"])
        orch_str = "orch" if res["use_orchestrator"] else "no_orch"
        vc_str   = "+vc" if res.get("use_visual_context") else ""
        label = f"{mod_str}/{orch_str}{vc_str}"
        row = f"{label:<{config_w}}" + "".join(
            f"{res['recall_at_k'][k]['recall']*100:>{col_w-1}.1f}%"
            for k in topk_list
        )
        logger.info(row)
    logger.info(sep)


def print_orchestrator_accuracy_table(results: List[Dict]):
    """Print a summary table of orchestrator day/time filter accuracy."""
    orch_results = [r for r in results if r["use_orchestrator"]]
    if not orch_results:
        return

    sep = "=" * 88
    logger.info(f"\n{sep}")
    logger.info("ORCHESTRATOR FILTER ACCURACY")
    logger.info(f"{'Configuration':<28}  {'Day filtered':>12}  {'Day hit@filt':>12}  {'Day coverage':>13}  {'Time filtered':>13}  {'Time hit@filt':>13}  {'Time coverage':>13}")
    logger.info("-" * 88)
    for res in orch_results:
        mod_str = "+".join(res["modalities"])
        vc_str  = "+vc" if res.get("use_visual_context") else ""
        label = f"{mod_str}/orch{vc_str}"
        acc = res.get("orchestrator_accuracy", {})
        d = acc.get("day", {})
        t = acc.get("time_of_day", {})
        n = acc.get("num_questions", 0)

        d_filt  = d.get("num_day_filtered", 0)
        d_hitr  = d.get("hit_rate_when_filtered", {})
        d_cov   = d.get("coverage_all", {})
        t_filt  = t.get("num_time_filtered", 0)
        t_hitr  = t.get("hit_rate_when_filtered", {})
        t_cov   = t.get("coverage_all", {})

        def _pct(d, key="rate"):
            v = d.get(key)
            return f"{v*100:.1f}%" if v is not None else "  n/a"

        logger.info(
            f"{label:<28}  {d_filt:>5}/{n:<6}  {_pct(d_hitr):>12}  {_pct(d_cov):>13}  "
            f"{t_filt:>6}/{n:<6}  {_pct(t_hitr):>13}  {_pct(t_cov):>13}"
        )
    logger.info(sep)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(all_results: List[Dict], topk_list: List[int], person: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out_path = os.path.join(OUTPUT_DIR, f"retrieval_recall_{person}.json")
    with open(out_path, "w") as f:
        json.dump({"person": person, "topk_list": topk_list, "configurations": all_results}, f, indent=2)
    logger.info(f"Full results saved to {out_path}")

    summary_path = os.path.join(OUTPUT_DIR, f"retrieval_recall_{person}_summary.json")
    summary = {
        "person": person,
        "topk_list": topk_list,
        "configurations": [
            {
                "modalities": r["modalities"],
                "use_orchestrator": r["use_orchestrator"],
                "use_visual_context": r.get("use_visual_context", False),
                "num_questions": r["num_questions"],
                "recall_at_k": r["recall_at_k"],
                "orchestrator_accuracy": r.get("orchestrator_accuracy"),
            }
            for r in all_results
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))

    qa_dir = Path(QA_DIR)
    qa_files = sorted(qa_dir.glob("EgoLifeQA_*.json"))
    if not qa_files:
        logger.error(f"No QA JSON files found in {QA_DIR}")
        sys.exit(1)
    if PERSONS:
        qa_files = [f for f in qa_files if any(p in f.name for p in PERSONS)]
    if not qa_files:
        logger.error("No QA files matched the requested persons.")
        sys.exit(1)

    logger.info(f"Persons: {[f.stem for f in qa_files]}")
    logger.info(f"Configurations: {CONFIGURATIONS}")
    logger.info(f"Topk list: {TOPK_LIST}")

    for qa_file in qa_files:
        person = qa_file.stem.replace("EgoLifeQA_", "")
        logger.info(f"\n{'='*60}\nPerson: {person}")

        qa_entries = load_qa_file(str(qa_file))
        if FILTER_NEED_NAME:
            before = len(qa_entries)
            qa_entries = [e for e in qa_entries if not e.get("need_name", False)]
            logger.info(f"Filtered out {before - len(qa_entries)} need_name entries ({len(qa_entries)} remaining).")
        qa_entries = filter_to_embedded_entries(qa_entries, VIDEO_DIR, person, PKL_DIR)
        if not qa_entries:
            logger.warning(f"No embedded clips for {person}, skipping.")
            continue
        if QUESTION_SLICE:
            start, end = QUESTION_SLICE
            qa_entries = qa_entries[start:end]
        elif MAX_QUESTIONS:
            qa_entries = qa_entries[:MAX_QUESTIONS]
        logger.info(f"{len(qa_entries)} questions.")

        # Load dataset once, reuse across all configs
        video_dataset = load_egolife_with_embeddings(VIDEO_DIR, person, PKL_DIR)
        video_name_to_path = {
            dp.video_name: dp.video_path for dp in video_dataset.video_datapoints
        }

        all_results = []
        for cfg in CONFIGURATIONS:
            modalities    = cfg["modalities"]
            use_orch      = cfg["orchestrator"]
            use_vc        = cfg.get("visual_context", USE_VISUAL_CONTEXT)
            logger.info(
                f"\n--- modalities={modalities}  orchestrator={use_orch}  "
                f"visual_context={use_vc} ---"
            )

            result = run_one_config(
                qa_entries=qa_entries,
                video_dataset=video_dataset,
                video_name_to_path=video_name_to_path,
                person_folder=person,
                modalities=modalities,
                use_orchestrator=use_orch,
                topk_list=TOPK_LIST,
                use_visual_context=use_vc,
            )
            all_results.append(result)

            inline = "  |  ".join(
                f"@{k}: {result['recall_at_k'][k]['recall']*100:.1f}%"
                for k in TOPK_LIST
            )
            logger.info(f"Recall  {inline}")

        print_recall_table(all_results, TOPK_LIST)
        print_orchestrator_accuracy_table(all_results)
        save_results(all_results, TOPK_LIST, person)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
