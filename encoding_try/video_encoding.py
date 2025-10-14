# import scenedetect
# import torch
# import os.path as osp
# import os
# from pathlib import Path
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import torch.nn as nn
# import torch.nn.functional as F
# import glob
# from tqdm import tqdm
# from transformers import AutoModel, AutoProcessor, XCLIPModel, XCLIPProcessor
# import whisper
# from sentence_transformers import SentenceTransformer
# import librosa
# from moviepy.editor import VideoFileClip
# from scenedetect import VideoManager, SceneManager
# from scenedetect import ContentDetector, detect
# from decord import VideoReader, cpu


# class MultiModalEncoder:
#     def __init__(
#         self, 
#         video_path: str = "../../data/", 
#         video_encoder: str = "microsoft/xclip-large-patch14",
#         audio_encoder: str = "laion/clap-htsat-unfused",
#         image_encoder: str = "openai/clip-vit-large-patch14",
#         text_encoder: str = "all-MiniLM-L6-v2",
#         device='cpu'
#     ):
#         self.device = device
#         self.video_encoder = video_encoder
#         self.audio_encoder = audio_encoder
#         self.image_encoder = image_encoder
#         self.text_encoder = text_encoder
#         self.video_path = video_path

#     def _load_videos(self):
#         video_files = []
#         for file in Path(self.video_path).glob("*.mp4"):
#             video_files.append(str(file))
#         self.video_files = video_files
    
#     def _load_model(self):
#         # Video
#         self.video_processor = XCLIPProcessor.from_pretrained(self.video_encoder)
#         self.video_model = XCLIPModel.from_pretrained(self.video_encoder)
#         self.video_model.vision_model.config.return_dict = True
#         # Visual (Image/Tokens)
#         self.token_processor = AutoProcessor.from_pretrained(self.image_encoder)
#         self.token_model = AutoModel.from_pretrained(self.image_encoder)
#         # Audio
#         self.audio_model = AutoModel.from_pretrained(self.audio_encoder)
#         self.audio_processor = AutoProcessor.from_pretrained(self.audio_encoder)
#         # Text
#         self.asr_model = whisper.load_model("base")
#         self.text_embedder = SentenceTransformer(self.text_encoder)

#     def _encode_video(self):
#         all_embeddings = []
#         for video_path in self.video_files[0:1]:
#             scenes = self._get_scenes(video_path)
#             corpus_data = []
#             for i, (start, end) in tqdm(enumerate(scenes)):
#                 features = self.process_scene(video_path, start.get_seconds(), end.get_seconds())
#                 features["scene_id"] = f"scene_{i}"
#                 features["timestamp"] = (start.get_seconds(), end.get_seconds())
#                 corpus_data.append(features)
#             all_embeddings.append(corpus_data)
#         return all_embeddings

#     def process_scene(self, video_path, start_time, end_time):
#         # Use moviepy to cut the scene clip
#         with VideoFileClip(video_path) as video:
#             scene_clip = video.subclip(start_time, end_time)
        
#             # This part is crucial: we need a list of frames (NumPy arrays)
#             # We also need to ensure the number of frames is consistent for the model
#             # XCLIP often expects a fixed number of frames (e.g., 8 or 16)
#             # Here we'll just subsample to get 8 frames for this example
#             frames = [frame for frame in scene_clip.iter_frames()]

#             frame_indices = np.linspace(0, len(frames) - 1, num=8, dtype=int)
#             sampled_frames = [frames[i] for i in frame_indices]
#             sampled_frames = [
#                 np.ascontiguousarray((f * 255).astype(np.uint8)) if f.dtype != np.uint8 else np.ascontiguousarray(f)
#                 for f in sampled_frames
#             ]

#             # The processor takes the list of frames and prepares them
#             video_inputs = self.video_processor(images=sampled_frames, return_tensors="pt")
            
#             # The output of the processor is a dictionary. It contains the key 'pixel_values'.
#             # We call the model directly with these inputs.
#             with torch.no_grad():
#                 pooled_video_embedding = self.video_model.get_video_features(**video_inputs)
            
#             # The model's output is an object. The features are in 'last_hidden_state'.
#             # To get a single vector, we perform mean pooling.
#             last_hidden_state = pooled_video_embedding.last_hidden_state
#             pooled_video_embedding = last_hidden_state.mean(dim=[1, 2]) # Pool across frames and patches
#             # 2. Visual Token Embeddings (for reranking)
#             keyframes = [scene_clip.get_frame(t) for t in [0.25, 0.5, 0.75]] # 3 keyframes
#             token_inputs = self.token_processor(images=keyframes, return_tensors="pt")
#             # Don't pool! Keep the patch embeddings.
#             token_embeddings = self.token_model.get_image_features(**token_inputs, output_hidden_states=True).hidden_states[-1][:, 1:, :] # (batch, num_patches, dim)

#             # 3. Audio Embedding
#             audio_array = scene_clip.audio.to_soundarray(fps=48000)
#             audio_inputs = self.audio_processor(audios=audio_array, sampling_rate=48000, return_tensors="pt")
#             audio_embedding = self.audio_model.get_audio_features(**audio_inputs)

#             # 4. Text Representations
#             temp_audio_path = "temp_audio.wav"
#             scene_clip.audio.write_audiofile(temp_audio_path)
#             transcript = self.asr_model.transcribe(temp_audio_path)["text"]
#             dense_text_embedding = self.text_embedder.encode(transcript)

#             return {
#                 "pooled_video": pooled_video_embedding.detach().numpy(),
#                 "visual_tokens": token_embeddings.squeeze().detach().numpy(),
#                 "audio": audio_embedding.detach().numpy(),
#                 "transcript": transcript,
#                 "dense_text": dense_text_embedding
#             }
        
#     def _get_scenes(self, video_path):
#         scene_list = detect(video_path, ContentDetector())
#         return scene_list
        

# if __name__ == "__main__":
#     encoder = MultiModalEncoder()
#     encoder._load_videos()
#     encoder._load_model()
#     video_embeddings = encoder._encode_video()
#     print(video_embeddings[0][0]["pooled_video"].shape)
#     print(video_embeddings[0][0]["visual_tokens"].shape)
#     print(video_embeddings[0][0]["audio"].shape)
#     print(video_embeddings[0][0]["transcript"])
#     print(video_embeddings[0][0]["dense_text"].shape)

#     sentence = "The tennis player won the tournament"
#     text_embeddings = encoder.text_embedder.encode(sentence, convert_to_tensor=True)
#     print(text_embeddings.shape)

import scenedetect
import torch
import os.path as osp
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, XCLIPModel, XCLIPProcessor
import whisper
from sentence_transformers import SentenceTransformer
from moviepy.editor import VideoFileClip
from scenedetect import detect, ContentDetector
from decord import VideoReader, cpu

class MultiModalEncoder:
    def __init__(
        self,
        video_path: str = "../../data/",
        video_encoder: str = "microsoft/xclip-large-patch14",
        audio_encoder: str = "laion/clap-htsat-unfused",
        image_encoder: str = "openai/clip-vit-large-patch14",
        text_encoder: str = "all-MiniLM-L6-v2",
        device='cpu'
    ):
        self.device = device
        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.video_path = video_path

    def _load_videos(self):
        video_files = []
        for file in Path(self.video_path).glob("*.mp4"):
            video_files.append(str(file))
        self.video_files = video_files

    def _load_model(self):
        # Video
        self.video_processor = XCLIPProcessor.from_pretrained(self.video_encoder)
        self.video_model = XCLIPModel.from_pretrained(self.video_encoder)
        # Visual (Image/Tokens)
        self.token_processor = AutoProcessor.from_pretrained(self.image_encoder)
        self.token_model = AutoModel.from_pretrained(self.image_encoder)
        # Audio
        self.audio_model = AutoModel.from_pretrained(self.audio_encoder)
        self.audio_processor = AutoProcessor.from_pretrained(self.audio_encoder)
        # Text
        self.asr_model = whisper.load_model("base")
        self.text_embedder = SentenceTransformer(self.text_encoder)

    def _encode_video(self):
        all_embeddings = []
        for video_path in tqdm(self.video_files[0:1]):
            scenes = self._get_scenes(video_path)
            corpus_data = []
            for i, (start, end) in enumerate(scenes):
                features = self.process_scene(video_path, start.get_seconds(), end.get_seconds())
                if features:
                    features["scene_id"] = f"scene_{i}"
                    features["timestamp"] = (start.get_seconds(), end.get_seconds())
                    corpus_data.append(features)
            all_embeddings.append(corpus_data)
        return all_embeddings

    def process_scene(self, video_path, start_time, end_time):
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = vr.get_avg_fps()
            start_frame, end_frame = int(start_time * fps), int(end_time * fps)
            frame_indices = np.linspace(start_frame, end_frame - 1, num=8, dtype=int)
            
            frames_batch = vr.get_batch(frame_indices).asnumpy()
            sampled_frames = [frame for frame in frames_batch]
            video_inputs = self.video_processor(images=sampled_frames, return_tensors="pt")

            with torch.no_grad():
                # ## <<< NECESSARY WORKAROUND: Bypass the broken .get_video_features()
                # 1. Get the raw outputs from the vision model (it returns a tuple)
                vision_outputs = self.video_model.vision_model(**video_inputs)
                # 2. The pooled output is the second element of the tuple
                pooled_output = vision_outputs[1]
                # 3. Apply the final projection layer to get the true embedding
                pooled_video_embedding = self.video_model.visual_projection(pooled_output)

            with VideoFileClip(video_path) as video:
                scene_clip = video.subclip(start_time, end_time)
                keyframe_indices = np.linspace(start_frame, end_frame - 1, num=3, dtype=int)
                keyframes = vr.get_batch(keyframe_indices).asnumpy()
                token_inputs = self.token_processor(images=list(keyframes), return_tensors="pt")
                token_embeddings = self.token_model.get_image_features(**token_inputs)

                if scene_clip.audio is not None:
                    audio_array = scene_clip.audio.to_soundarray(fps=48000)
                    audio_inputs = self.audio_processor(audios=audio_array, sampling_rate=48000, return_tensors="pt")
                    audio_embedding = self.audio_model.get_audio_features(**audio_inputs)
                    temp_audio_path = "temp_audio.wav"
                    scene_clip.audio.write_audiofile(temp_audio_path, logger=None)
                    transcript = self.asr_model.transcribe(temp_audio_path)["text"]
                    dense_text_embedding = self.text_embedder.encode(transcript)
                    os.remove(temp_audio_path)
                else:
                    audio_embedding = torch.zeros((1, 512))
                    transcript = ""
                    dense_text_embedding = self.text_embedder.encode("")

            return {
                "pooled_video": pooled_video_embedding.cpu().detach().numpy(),
                "visual_tokens": token_embeddings.cpu().squeeze().detach().numpy(),
                "audio": audio_embedding.cpu().detach().numpy(),
                "transcript": transcript,
                "dense_text": dense_text_embedding
            }
        except Exception as e:
            print(f"\n--- ERROR processing scene from {start_time:.2f}s to {end_time:.2f}s ---")
            import traceback
            traceback.print_exc()
            return None

    def _get_scenes(self, video_path):
        return detect(video_path, ContentDetector())

if __name__ == "__main__":
    encoder = MultiModalEncoder()
    encoder._load_videos()
    encoder._load_model()
    video_embeddings = encoder._encode_video()
    
    if video_embeddings and video_embeddings[0]:
        print("\n" + "="*50)
        print("--- ✅ SUCCESS! ---")
        print("Pooled video embedding shape:", video_embeddings[0][0]["pooled_video"].shape)
        print("Visual tokens embedding shape:", video_embeddings[0][0]["visual_tokens"].shape)
        print("Audio embedding shape:", video_embeddings[0][0]["audio"].shape)
        print("Transcript:", f"'{video_embeddings[0][0]['transcript']}'")
        print("Dense text embedding shape:", video_embeddings[0][0]["dense_text"].shape)
        print("="*50 + "\n")
    else:
        print("\n--- ❌ ERROR: No embeddings were generated. Check the error messages above. ---")