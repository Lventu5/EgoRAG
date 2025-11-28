import pickle
import numpy as np

try:
    import torch
except Exception:
    torch = None


def _print_embedding_summary(name: str, emb):
    if emb is None:
        print(f"{name}: None")
        return
    if torch is not None and isinstance(emb, torch.Tensor):
        print(f"{name}: torch.Tensor shape={tuple(emb.shape)} dtype={emb.dtype} on_cuda={emb.is_cuda}")
        try:
            print("  sample:", emb.detach().cpu().numpy().ravel())
        except Exception:
            pass
    elif isinstance(emb, np.ndarray):
        print(f"{name}: numpy.ndarray shape={emb.shape} dtype={emb.dtype}")
        try:
            print("  sample:", emb.ravel())
        except Exception:
            pass
    else:
        rep = repr(emb)
        if len(rep) > 300:
            rep = rep[:300] + '...'
        print(f"{name}: {type(emb).__name__} repr={rep}")


def print_first_scene_video_embedding(pkl_path: str):
    print(f"Loading pickle: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Find the first VideoDataPoint
    dp = None
    if hasattr(data, 'video_datapoints'):
        vds = getattr(data, 'video_datapoints')
        if isinstance(vds, (list, tuple)) and len(vds) > 0:
            dp = vds[0]
    elif isinstance(data, (list, tuple)) and len(data) > 0:
        dp = data[0]
    elif hasattr(data, 'scenes') and hasattr(data, 'scene_embeddings'):
        dp = data

    if dp is None:
        print("Could not find a VideoDataPoint in this pickle (no `video_datapoints` list).")
        return

    # Access scene_embeddings
    se = getattr(dp, 'scene_embeddings', None)
    we = getattr(dp, "window_embeddings", None)
    windows = getattr(dp, "windows", None)
    if not se:
        print("No `scene_embeddings` found or empty for the first video datapoint.")
        return
    if not we:
        print("No window_embeddings key")
    print([window.window_id for window in windows])
    print(we.keys())

    # Get first scene key
    try:
        first_scene_key = "scene_1"
    except StopIteration:
        print("`scene_embeddings` is empty.")
        return

    scene_dict = se.get(first_scene_key, {})

    print(f"First scene key: {first_scene_key}")
    # Print the scene's `video` embedding
    video_emb = scene_dict.get('video') if isinstance(scene_dict, dict) else None
    _print_embedding_summary(f"scene[{first_scene_key}].video", video_emb)


if __name__ == '__main__':
    # Edit this path to point to your pickle file and run the script.
    # Example: pkl_path = '/path/to/my_video_dataset.pkl'
    pkl_path = '/cluster/project/cvg/students/tnanni/ego4d_data/v2/redone_internvideo_encoded/9f28e782-417c-4c8b-a7ae-42fc96a0e94f_encoded.pkl'
    print_first_scene_video_embedding(pkl_path)
