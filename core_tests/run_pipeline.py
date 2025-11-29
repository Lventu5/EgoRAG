"""Orchestrator script: encode videos, merge pickles, run retrieval.

Edit the variables under CONFIGURATION and run this script.

Steps:
 1. Build a VideoDataset from `video_dir` (looks for .mp4 files)
 2. Run MultiModalEncoder.encode_videos() to encode scenes/windows/text
 3. Save one pickle per video into `per_video_pkl_dir`
 4. Call utils.merge_pickles.merge_pickles(...) to create a merged pickle
 5. Call test.retrieval_from_json.main(...) to run retrieval + evaluation

This script is intended as a convenience orchestrator; edit paths and flags below.
"""
import os
import glob
import pickle
import logging
from pathlib import Path

from data.video_dataset import VideoDataset
from indexing.multimodal_encoder import MultiModalEncoder
from utils.merge_pickles import merge_pickles
from test.retrieval_from_json import main as retrieval_main
from configuration.config import CONFIG

# ---------------- CONFIGURATION (edit as needed) ----------------
ENCODING_NAME = "internvideo6b_5s_window"
video_dir = CONFIG.data.video_path
per_video_pkl_dir = f"/cluster/project/cvg/students/tnanni/ego4d_data/v2/{ENCODING_NAME}"
merged_output_path = f"/cluster/project/cvg/students/tnanni/ego4d_data/v2/{ENCODING_NAME}/merged_{ENCODING_NAME}_videos.pkl"
recursive_search = False
drop_keyframes_when_merging = True
run_retrieval = True
# Retrieval settings (passed to retrieval_from_json.main)
annotations_path = CONFIG.data.annotation_path
modalities = [
    ("text_only", ["text"]),
    ("video_only", ["video"]),
    ("video_text", ["video", "text"]),
]
topk_videos = CONFIG.retrieval.top_k_videos
topk_scenes = CONFIG.retrieval.top_k_scenes
device = CONFIG.device
# -----------------------------------------------------------------

logging.basicConfig(level=logging.INFO)

os.makedirs(per_video_pkl_dir, exist_ok=True)

# Find video files
if recursive_search:
    pattern = os.path.join(video_dir, "**", "*.mp4")
else:
    pattern = os.path.join(video_dir, "*.mp4")

video_files = sorted(glob.glob(pattern, recursive=recursive_search))
if not video_files:
    logging.error(f"No video files found in {video_dir} (pattern: {pattern})")
    raise SystemExit(1)

logging.info(f"Found {len(video_files)} video(s) to encode")

# Build VideoDataset
vd = VideoDataset(video_files)

# Instantiate encoder
encoder = MultiModalEncoder(video_dataset=vd, device=device, max_workers=2)

# Run encoding (this will modify vd in-place)
logging.info("Starting encoding of videos (scene/window/global pipeline)")
encoded_dataset = encoder.encode_videos(force=False)
logging.info("Encoding complete")

# Save each VideoDataPoint as separate pickle in per_video_pkl_dir
for dp in encoded_dataset.video_datapoints:
    uid = getattr(dp, 'video_uid', None) or os.path.splitext(os.path.basename(getattr(dp, 'video_path', 'unnamed')))[0]
    out_path = os.path.join(per_video_pkl_dir, f"{uid}.pkl")
    try:
        with open(out_path, 'wb') as f:
            pickle.dump(dp, f)
        logging.info(f"Wrote per-video pickle: {out_path}")
    except Exception as e:
        logging.error(f"Failed to write pickle for {uid}: {e}")

# Merge pickles into a single dataset pickle
logging.info("Merging per-video pickles into a single merged pickle...")
merged_ds = merge_pickles(per_video_pkl_dir, merged_output_path, recursive=False, deduplicate=False, drop_keyframes=drop_keyframes_when_merging)
logging.info(f"Merged pickle saved to: {merged_output_path}")

# Optionally run retrieval+evaluation
if run_retrieval:
    logging.info("Running retrieval and evaluation on merged pickle...")
    # `retrieval_from_json.main` expects (video_pickle, annotations, modalities, topk_videos, topk_scenes, device, skip_video_retrieval, save_path)
    save_path = f"./results/{ENCODING_NAME}_retrieval.csv"
    retrieval_main(merged_output_path, annotations_path, modalities, topk_videos, topk_scenes, device, False, save_path)
    logging.info(f"Retrieval finished, results saved to {save_path}")

logging.info("Pipeline complete")
