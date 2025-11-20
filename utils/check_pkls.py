#!/usr/bin/env python3
"""Inspect a pickled VideoDataset and report approximate memory usage per component.

Designed for the case where the pickle contains a VideoDataset with a single
VideoDataPoint (common when encoding videos one-by-one). The script prints:
- dataset / datapoint info
- per-global-embedding sizes
- per-scene breakdown (per-modality and per-scene totals)
- totals and top contributors

It uses a conservative estimation for torch tensors (element_size() * nelement())
and numpy-like objects (nbytes). Other Python containers are scanned recursively
and small objects fall back to sys.getsizeof.
"""
from __future__ import annotations

import pickle
import os
import sys
from typing import Any, Dict, Set, Tuple

import torch
import numpy as np


def sizeof(obj: Any, seen: Set[int]) -> int:
    """Estimate size in bytes of obj. Avoid double-counting using seen set of ids."""
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    # Torch tensor
    try:
        if isinstance(obj, torch.Tensor):
            return int(obj.element_size() * obj.nelement())
    except Exception:
        pass

    # NumPy array
    try:
        if isinstance(obj, np.ndarray):
            return int(obj.nbytes)
    except Exception:
        pass

    # bytes-like
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return len(obj)

    # dict-like
    if isinstance(obj, dict):
        s = 0
        for k, v in obj.items():
            s += sizeof(k, seen)
            s += sizeof(v, seen)
        return s

    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        s = 0
        for v in obj:
            s += sizeof(v, seen)
        return s

    # fallback: try to use nbytes or getsizeof
    if hasattr(obj, "nbytes") and isinstance(getattr(obj, "nbytes"), int):
        try:
            return int(obj.nbytes)
        except Exception:
            pass

    try:
        return sys.getsizeof(obj)
    except Exception:
        return 0


def human(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes} B"
    nb = float(nbytes)
    for unit in ("KB", "MB", "GB", "TB"):
        nb /= 1024.0
        if nb < 1024.0:
            return f"{nb:.2f} {unit}"
    return f"{nb:.2f} PB"


def inspect_dataset(path: str) -> None:
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded object: {type(data)}")

    # Expect VideoDataset-like object with .video_datapoints
    vds = getattr(data, "video_datapoints", None)
    if vds is None:
        print("No attribute 'video_datapoints' found on loaded object. Aborting.")
        return

    print(f"# video datapoints: {len(vds)}")
    if len(vds) == 0:
        return

    # For large pickles you mentioned there is a single datapoint
    for idx, dp in enumerate(vds):
        print("\n--- VideoDataPoint[%d] ---" % idx)
        video_name = getattr(dp, "video_name", None) or getattr(dp, "video_path", "<unknown>")
        video_uid = getattr(dp, "video_uid", None) or os.path.splitext(os.path.basename(str(video_name)))[0]
        print(f"video_name: {video_name}")
        print(f"video_uid: {video_uid}")

        seen: Set[int] = set()

        # Global embeddings breakdown
        print("\nGlobal embeddings:")
        global_tot = 0
        for k, v in getattr(dp, "global_embeddings", {}).items():
            s = sizeof(v, seen)
            print(f"  {k}: {human(s)}")
            global_tot += s
        print(f"  -> total global embeddings: {human(global_tot)}")

        # Scene embeddings breakdown
        print("\nScene embeddings (per scene):")
        scene_totals: Dict[str, int] = {}
        modality_aggregate: Dict[str, int] = {}
        scenes = getattr(dp, "scene_embeddings", {})
        for scene_id, scene_dict in scenes.items():
            s_scene = 0
            print(f"  {scene_id}:")
            if not isinstance(scene_dict, dict):
                print(f"    (unexpected type: {type(scene_dict)})")
                continue
            for mod_key, val in scene_dict.items():
                s_val = sizeof(val, seen)
                print(f"    {mod_key}: {human(s_val)}")
                s_scene += s_val
                modality_aggregate[mod_key] = modality_aggregate.get(mod_key, 0) + s_val
            scene_totals[scene_id] = s_scene
            print(f"    -> scene total: {human(s_scene)}")

        total_scenes = sum(scene_totals.values())
        print(f"\nTotal scenes memory: {human(total_scenes)}")

        print("\nAggregate per-modality across scenes:")
        for mod, val in modality_aggregate.items():
            print(f"  {mod}: {human(val)}")

        grand_total = global_tot + total_scenes
        print(f"\nGRAND TOTAL (global + scenes): {human(grand_total)}")

        # Top contributors (by re-scanning without dedup to get per-object sizes)
        # Build a flat list of (name, size) for display purposes
        print("\nTop-level contributors (approx):")
        contribs: Dict[str, int] = {}
        # globals
        seen2: Set[int] = set()
        for k, v in getattr(dp, "global_embeddings", {}).items():
            contribs[f"global:{k}"] = sizeof(v, seen2)
        # scenes
        for scene_id, scene_dict in scenes.items():
            for mod_key, val in (scene_dict.items() if isinstance(scene_dict, dict) else []):
                contribs[f"{scene_id}:{mod_key}"] = sizeof(val, seen2)

        # sort and print top 10
        for name, size in sorted(contribs.items(), key=lambda kv: kv[1], reverse=True)[:20]:
            print(f"  {name}: {human(size)}")


def check_for_keyframes(path: str) -> None:
    """Check if pickle contains keyframes data that should have been deleted"""
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    vds = getattr(data, "video_datapoints", None)
    if not vds:
        print("No video_datapoints found")
        return
    
    for idx, dp in enumerate(vds):
        print(f"\n=== VideoDataPoint {idx} ===")
        
        # Check for _temp_keyframes attribute
        if hasattr(dp, "_temp_keyframes"):
            print(f"⚠️  WARNING: dp._temp_keyframes still exists!")
            print(f"   Size: {human(sizeof(dp._temp_keyframes, set()))}")
        
        # Check scene_embeddings for keyframes
        scenes = getattr(dp, "scene_embeddings", {})
        keyframe_scenes = []
        for scene_id, scene_dict in scenes.items():
            if isinstance(scene_dict, dict) and "keyframes" in scene_dict:
                kf = scene_dict["keyframes"]
                kf_size = sizeof(kf, set())
                keyframe_scenes.append((scene_id, kf_size))
        
        if keyframe_scenes:
            print(f"⚠️  WARNING: Found keyframes in scene_embeddings:")
            for sid, size in keyframe_scenes:
                print(f"   {sid}: {human(size)}")
            total_kf = sum(s for _, s in keyframe_scenes)
            print(f"   TOTAL keyframes: {human(total_kf)}")
        else:
            print("✅ No keyframes found in scene_embeddings")


def check_for_captions(path: str) -> None:
    """Check if pickle contains per-scene captions and a global caption for the video.

    Reports which scenes contain a caption (embedding or text) and whether a global
    caption is present in `dp.global_embeddings` or as a temporary attribute.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    vds = getattr(data, "video_datapoints", None)
    if not vds:
        print("No video_datapoints found")
        return

    for idx, dp in enumerate(vds):
        print(f"\n=== VideoDataPoint {idx} ===")
        scenes = getattr(dp, "scene_embeddings", {})

        scenes_with_caption = []
        for scene_id, scene_dict in scenes.items():
            if not isinstance(scene_dict, dict):
                continue
            # Check for caption embedding or caption text
            caption_obj = None
            caption_text = None
            if "caption" in scene_dict:
                caption_obj = scene_dict.get("caption")
            if "caption_text" in scene_dict:
                caption_text = scene_dict.get("caption_text")

            if caption_obj is not None or (isinstance(caption_text, str) and caption_text.strip()):
                size_obj = sizeof(caption_obj, set()) if caption_obj is not None else 0
                size_txt = sizeof(caption_text, set()) if caption_text is not None else 0
                scenes_with_caption.append((scene_id, size_obj + size_txt))

        if scenes_with_caption:
            print(f"⚠️  Found captions in {len(scenes_with_caption)} scene(s):")
            total = 0
            for sid, sz in scenes_with_caption:
                print(f"   {sid}: {human(sz)}")
                total += sz
            print(f"   TOTAL scene captions: {human(total)}")
        else:
            print("✅ No per-scene captions found")

        # Check for global caption in global_embeddings or temporary attribute
        global_caption_size = 0
        ge = getattr(dp, "global_embeddings", {})
        global_caption = None
        if isinstance(ge, dict):
            # possible keys: 'caption_text' or 'caption'
            if "caption_text" in ge and isinstance(ge.get("caption_text"), str) and ge.get("caption_text").strip():
                global_caption = ge.get("caption_text")
            elif "caption" in ge and ge.get("caption") is not None:
                global_caption = ge.get("caption")

        # Also check temporary attribute used during encoding
        if global_caption is None and hasattr(dp, "_temp_global_caption") and getattr(dp, "_temp_global_caption"):
            global_caption = getattr(dp, "_temp_global_caption")

        if global_caption is not None:
            global_caption_size = sizeof(global_caption, set())
            print(f"⚠️  Global caption present: {human(global_caption_size)}")
        else:
            print("✅ No global caption found in this datapoint")


if __name__ == "__main__":
    # Load and inspect the specific pickle the user asked for
    path = "../ego4d_data/v2/internvideo_encoded_videos/2b5569df-5deb-4ebd-8a45-dd6524330eb8_encoded.pkl"
    
    print("=" * 80)
    print("CHECKING FOR KEYFRAMES")
    print("=" * 80)
    check_for_keyframes(path)
    
    print("\n" + "=" * 80)
    print("FULL MEMORY BREAKDOWN")
    print("=" * 80)
    # Also check captions presence
    print("\n" + "=" * 80)
    print("CHECKING FOR CAPTIONS")
    print("=" * 80)
    check_for_captions(path)

    print("\n" + "=" * 80)
    print("FULL MEMORY BREAKDOWN")
    print("=" * 80)
    inspect_dataset(path)