import os
import json
import pickle
import cv2
from tqdm import tqdm
from demo.config import Config, eval_dict_leaf
from demo.utils import _frame_from_video, setup_internvideo2, frames2tensor, get_text_feat_dict
import decord
from decord import VideoReader
from models.backbones.bert.tokenization_bert import BertTokenizer
import torch


def load_model(config_path, model_path, device='cuda'):
    config = Config.from_file(config_path)
    config = eval_dict_leaf(config)
    # IMPORTANT: Set vision_encoder.pretrained = None to prevent the vision encoder builder
    # from trying to load the DeepSpeed checkpoint directly (it doesn't handle the format).
    # The checkpoint loading is handled properly by setup_internvideo2 via pretrained_path,
    # which correctly extracts weights from DeepSpeed Stage 1 checkpoint format.
    config.model.vision_encoder.pretrained = None
    config['pretrained_path'] = model_path

    model, _ = setup_internvideo2(config)

    print("DEBUG load_model right after setup, type(model.tokenizer) =", type(model.tokenizer))

    model.to(device)
    return model


def extract_video_features(video_paths, model, device='cuda', fn=4, size_t=224):
    results = {}

    for video_path in tqdm(video_paths):
        if not video_path.endswith(('.mp4', '.webm')):
            print(f'[WARNING] Video path does not end with .mp4 or .webm: {video_path}')
            continue

        try:
            vr = VideoReader(video_path, ctx=decord.cpu(), num_threads=1)
        except Exception as e:
            print(f'[ERROR] Failed to load video: {video_path}')
            print(f'[ERROR] Error: {e}')
            continue

        total_frames = len(vr)
        if total_frames == 0:
            print('[ERROR] Video length is zero', video_path)
            continue

        indices = [x + (total_frames // fn) // 2 for x in range(0, total_frames, total_frames // fn)[:fn]]
        indices[-1] = min(indices[-1], total_frames - 1)
        frames = [cv2.resize(x[..., ::-1], (size_t, size_t)) for x in vr.get_batch(indices).asnumpy()]
        frames_tensor = frames2tensor(frames, fnum=fn, target_size=(size_t, size_t), device=device)

        video_feature = model.get_vid_feat(frames_tensor).cpu().numpy()
        key = os.path.splitext(os.path.basename(video_path))[0]
        results[key] = video_feature

    return results


def extract_query_features(queries_list, model):
    embeddings_list = []
    for i, query in enumerate(queries_list):
        feature_dict = get_text_feat_dict([query], model)
        feat = next(iter(feature_dict.values()))
        embeddings_list.append(feat.squeeze(0))
    print(f"DEBUG embeddings_list length = {len(embeddings_list)}; example element type = {type(embeddings_list[0])}, shape = {embeddings_list[0].shape}")
    return torch.stack(embeddings_list, dim=0)