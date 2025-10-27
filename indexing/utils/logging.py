import logging
import sys
from typing import Dict

class LevelAwareFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno >= logging.WARNING:
            # Per WARNING ed ERROR -> mostra [LEVEL]
            fmt = "[%(levelname)s] %(message)s"
        else:
            # Per INFO e DEBUG -> solo messaggio
            fmt = "%(message)s"
        formatter = logging.Formatter(fmt)
        return formatter.format(record)

def pretty_print_retrieval(results: Dict[str, Dict[str, list[tuple]]], max_videos: int = 3, max_scenes: int = 1):
    """
    Pretty-print retrieval results.
    Args:
        results: Dict mapping query IDs to retrieval results per modality.
        max_videos: maximum number of videos to print per query
        max_scenes: maximum number of scenes to print per video
    """
    for qid, per_mod in results.items():
        print(f"\n=== Query {qid} ===")
        for mod, items in per_mod.items():
            print(f"  [{mod}]")
            for vi, (video_name, global_score, scenes) in enumerate(items[:max_videos], 1):
                print(f"    {vi}. {video_name} (video score: {global_score:.4f})")
                for si, (scene_obj, s_score) in enumerate(scenes[:max_scenes], 1):
                    if hasattr(scene_obj, "start_time"):
                        span = f"{scene_obj.start_time:.2f}s–{scene_obj.end_time:.2f}s"
                    else:
                        span = str(scene_obj)
                    print(f"       - scene {si}: {span} (score: {s_score:.4f})")
