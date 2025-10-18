import pickle
import os

class SceneClip:
    def __init__(self, video_path, start_time, end_time, embeds):
        self.video_path = video_path
        self.time_range = (start_time, end_time)
        self.embeds = embeds

    def __repr__(self):
        return f"SceneClip(video_path={self.video_path}, time_range={self.time_range})"

    def duration(self):
        return self.time_range[1] - self.time_range[0]
    
    def get_embed(self, mode):
        if mode in self.embeds.keys():
            return self.embeds[mode]
        else:
            raise ValueError(f"Embed mode '{mode}' not found in embeddings for the scene clip.")

    def to_dict(self):
        return {
            "video_path": self.video_path,
            "time_range": self.time_range,
            "embeds": self.embeds
        }

    def save_to_pickle(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_pickle(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

def main():
    # Example usage
    embeds = {
        "visual": [0.1, 0.2, 0.3],
        "audio": [0.4, 0.5, 0.6]
    }
    scene_clip = SceneClip("path/to/video.mp4", 10, 20, embeds)
    print(scene_clip)
    print("Duration:", scene_clip.duration())
    print("Visual Embeds:", scene_clip.get_embed("visual"))
    
    # Save to pickle
    scene_clip.save_to_pickle("encoding_try/scene_clips/scene_clip_test.pkl")

    # Load from pickle
    loaded_clip = SceneClip.load_from_pickle("encoding_try/scene_clips/scene_clip_test.pkl")
    print("Loaded Clip:", loaded_clip)
    print("Loaded Clip Duration:", loaded_clip.duration())
    print("Loaded Clip Visual Embeds:", loaded_clip.get_embed("visual"))

if __name__ == "__main__":
    main()