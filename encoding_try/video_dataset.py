import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import cv2

class VideoDataPoint():
    def __init__(
        self, 
        video_path: str, 
        scenes_list: list | None = None,
        video_embeddings: dict[str, torch.Tensor] | None = None,
        audio_embeddings: dict[str, torch.Tensor] | None = None,
        image_embeddings: dict[str, torch.Tensor] | None = None,
        text_embeddings: dict[str, torch.Tensor] | None = None,
    ):
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.scenes_list = scenes_list
        self.video_embeddings = video_embeddings or dict()
        self.audio_embeddings = audio_embeddings or dict()
        self.image_embeddings = image_embeddings or dict()
        self.text_embeddings = text_embeddings or dict()
    
    def get_frames(
        self,
        every_n_frames: int = 1, 
        max_frames: int | None = None,
    )-> list[np.ndarray]:
    
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError(f"Impossibile aprire il video: {self.video_path}")

        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % every_n_frames == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            if max_frames and len(frames) >= max_frames:
                break
            frame_idx += 1
        cap.release()
        return frames

    

class VideoDataset(Dataset):
    def __init__(self, video_files):
        self.video_files = video_files

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        return video_file