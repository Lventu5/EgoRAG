"""Utility to merge multiple pickled VideoDataset/VideoDataPoint files into a single pickle.

This script looks for all .pkl files in a given directory, loads them, and attempts to
extract VideoDataPoint objects (or VideoDataset.video_datapoints) merging them into
one VideoDataset which is saved to `output_path`.

Usage: edit the variables in the __main__ block or import and call `merge_pickles`.
"""
from __future__ import annotations

import os
import os.path as osp
import pickle
from typing import List

from data.video_dataset import VideoDataset, VideoDataPoint


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def merge_pickles(input_dir: str, output_path: str, recursive: bool = False, deduplicate: bool = False, drop_keyframes: bool = False) -> VideoDataset:
    """Merge all .pkl files in input_dir into a single VideoDataset and save to output_path.

    Args:
        input_dir: directory containing .pkl files (non-recursive by default)
        output_path: destination path for the merged VideoDataset pickle
        recursive: if True, walk subdirectories as well

    Returns:
        The merged VideoDataset instance.
    """
    pkl_paths: List[str] = []
    if recursive:
        for root, _, files in os.walk(input_dir):
            for fn in files:
                if fn.lower().endswith(".pkl"):
                    pkl_paths.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(input_dir):
            if fn.lower().endswith(".pkl"):
                pkl_paths.append(os.path.join(input_dir, fn))

    if not pkl_paths:
        raise FileNotFoundError(f"No .pkl files found in {input_dir}")

    merged_video_datapoints: List[VideoDataPoint] = []

    for p in sorted(pkl_paths):
        try:
            obj = load_pickle(p)
        except Exception as e:
            print(f"Warning: unable to load pickle {osp.abspath(p)}: {e}")
            continue

        # If it's a VideoDataset, extend with its datapoints
        if isinstance(obj, VideoDataset):
            merged_video_datapoints.extend(obj.video_datapoints)
            continue

        # If it's a VideoDataPoint, append
        if isinstance(obj, VideoDataPoint):
            merged_video_datapoints.append(obj)
            continue

        # If it's a dict-like or has attribute video_datapoints, try to extract
        if hasattr(obj, "video_datapoints") and isinstance(getattr(obj, "video_datapoints"), list):
            merged_video_datapoints.extend(getattr(obj, "video_datapoints"))
            continue

        # If it's a list, try to filter VideoDataPoint elements
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, VideoDataPoint):
                    merged_video_datapoints.append(item)
            continue

        print(f"Warning: pickle {p} contains unsupported object of type {type(obj)}. Skipping.")

    # Optionally deduplicate by video_uid (keep first occurrence)
    if deduplicate:
        seen_uids = set()
        deduped = []
        for dp in merged_video_datapoints:
            uid = getattr(dp, "video_uid", None) or os.path.splitext(os.path.basename(getattr(dp, "video_path", "")))[0]
            if uid in seen_uids:
                continue
            seen_uids.add(uid)
            deduped.append(dp)
        merged_video_datapoints = deduped

    # Build a new VideoDataset from the merged datapoints
    video_files = [v.video_path for v in merged_video_datapoints]

    merged_dataset = VideoDataset(video_files)
    # replace the auto-created datapoints with our merged list
    merged_dataset.video_datapoints = merged_video_datapoints
    merged_dataset.encoded = True

    # Optionally drop keyframes from scene_embeddings before saving to keep files small
    if drop_keyframes:
        for dp in merged_dataset.video_datapoints:
            se = getattr(dp, "scene_embeddings", {})
            if not isinstance(se, dict):
                continue
            for sid, scene_dict in list(se.items()):
                if isinstance(scene_dict, dict) and "keyframes" in scene_dict:
                    try:
                        scene_dict.pop("keyframes")
                    except Exception:
                        continue

    # Ensure output dir exists (handle case with no dirname)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(merged_dataset, f)

    print(f"Merged {len(merged_video_datapoints)} videos from {len(pkl_paths)} files into {output_path}")
    return merged_dataset


if __name__ == "__main__":
    # EDIT THESE PATHS as needed (no argparse here - edit and run)
    input_dir = "../ego4d_data/v2/redone_internvideo_encoded" 
    output_path = "../ego4d_data/v2/redone_internvideo_encoded/merged_redone_internvideo_videos.pkl"
    recursive = False

    merged = merge_pickles(input_dir=input_dir, output_path=output_path, recursive=recursive)
    print("Done.")

