import csv
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

from configuration.config import CONFIG
from data.datatypes import RetrievalResults
from data.query import Query
from data.video_dataset import Scene, VideoDataset
from generation.answer_generator import AnswerGenerator
from retrieval.questioner import Ego4D_NLQ_Runner
from utils.extract_video_ids import extract_video_ids


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def _load_first_video_ids(nlq_json_path: str, limit: int = 8) -> List[str]:
    video_ids = extract_video_ids(nlq_json_path)
    return video_ids[:limit]


def _build_gt_scene(video_uid: str, entry: Dict[str, Any], idx: int) -> Scene:
    start_sec = float(entry["start_sec"])
    end_sec = float(entry["end_sec"])
    start_frame = entry.get("start_frame")
    end_frame = entry.get("end_frame")
    return Scene(
        scene_id=f"gt_{video_uid}_{idx}",
        start_time=start_sec,
        end_time=end_sec,
        video_name=video_uid,
        start_frame=start_frame,
        end_frame=end_frame,
    )


def _build_retrieval_results(query_id: str, video_uid: str, scene: Scene) -> RetrievalResults:
    results = {
        query_id: [
            (video_uid, 1.0, [(scene, 1.0)])
        ]
    }
    return RetrievalResults(results=results)


def run_gt_scene_generation_test(output_csv: str) -> None:
    nlq_path = CONFIG.data.annotation_path
    video_base_path = CONFIG.data.video_path

    first_video_ids = _load_first_video_ids(nlq_path, limit=8)
    if not first_video_ids:
        raise RuntimeError("No video IDs found in NLQ validation annotations.")

    runner = Ego4D_NLQ_Runner(
        nlq_annotations_path=nlq_path,
        dataset=VideoDataset([]),
        device=CONFIG.device,
    )

    answer_generator = AnswerGenerator(
        model_name=CONFIG.retrieval.answer_generation.model_name,
        device=CONFIG.device,
        temp_dir=CONFIG.retrieval.answer_generation.temp_dir,
        max_clips_per_query=1,
        max_pixels=CONFIG.retrieval.answer_generation.max_pixels,
        fps=CONFIG.retrieval.answer_generation.fps,
        max_new_tokens=CONFIG.retrieval.answer_generation.max_new_tokens,
    )
    answer_generator.load_model()

    repo_root = Path(__file__).resolve().parents[1]
    output_path = Path(output_csv)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Writing CSV to: {output_path.resolve()}")

    with output_path.open("w", newline="", buffering=1) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_uid",
                "query_id",
                "query",
                "answer",
                "gt_start_sec",
                "gt_end_sec",
                "gt_length_sec",
                "error",
            ],
        )
        writer.writeheader()
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
        logging.info("Wrote CSV header")

        for video_uid in first_video_ids:
            entries = runner.load_video_nlq_gt(nlq_path, video_uid)
            logging.info(f"Processing video {video_uid} with {len(entries)} queries")

            for i, entry in enumerate(entries):
                start_sec = float(entry["start_sec"])
                end_sec = float(entry["end_sec"])
                if end_sec <= start_sec:
                    logging.warning(
                        f"Skipping query with invalid GT window: {video_uid} idx={i}"
                    )
                    continue

                gt_len = end_sec - start_sec
                logging.info(
                    f"GT length {gt_len:.3f}s for {video_uid} idx={i}"
                )

                query = Query(
                    qid=f"{video_uid}_{i}",
                    query_text=entry["query"],
                    video_uid=video_uid,
                    gt={
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "start_frame": entry.get("start_frame"),
                        "end_frame": entry.get("end_frame"),
                    },
                )

                gt_scene = _build_gt_scene(video_uid, entry, i)
                retrieval_results = _build_retrieval_results(query.qid, video_uid, gt_scene)

                error_msg = ""
                logging.info(
                    f"Generating answer for {query.qid} (GT {gt_len:.3f}s)"
                )
                try:
                    answer = answer_generator.generate_answer_for_query(
                        query=query,
                        retrieval_results=retrieval_results,
                        video_base_path=video_base_path,
                        use_concatenated=True,
                    )
                except Exception as e:
                    logging.error(
                        f"Answer generation failed for {query.qid}: {e}"
                    )
                    answer = ""
                    error_msg = str(e)

                logging.info(
                    f"Answer for {query.qid}: {answer if answer else '[EMPTY]'}"
                )

                writer.writerow(
                    {
                        "video_uid": video_uid,
                        "query_id": query.qid,
                        "query": query.query_text,
                        "answer": answer,
                        "gt_start_sec": start_sec,
                        "gt_end_sec": end_sec,
                        "gt_length_sec": gt_len,
                        "error": error_msg,
                    }
                )
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
                try:
                    size_bytes = output_path.stat().st_size
                    logging.info(
                        f"CSV size after write: {size_bytes} bytes"
                    )
                except Exception:
                    pass
                logging.info(f"Wrote row for {query.qid}")

    logging.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    run_gt_scene_generation_test(
        output_csv="./results/gt_scene_generation_first8_videos.csv"
    )
