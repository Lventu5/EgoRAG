from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import os
import json
import glob

from .video_dataset import VideoDataset, Scene
from .query import QueryDataset, Query
from configuration.config import CONFIG

class BaseDataset(Dataset):
    def __init__(self, video_path: str, annotation_path: str):
        self.video_path = video_path
        self.annotation_path = annotation_path

    def __len__(self):
        raise NotImplementedError("Subclasses should implement this method.")  
    
    def __getitem__(self, vid):
        raise NotImplementedError("Subclasses should implement this method.")

    def load_videos(self, is_pickle: bool) -> VideoDataset:
        raise NotImplementedError("Subclasses should implement this method.")

    def load_annotations(self, video_ids: List[str]) -> List[dict]:
        raise NotImplementedError("Subclasses should implement this method.")
    
    def load_queries(self, video_ids: List[str]) -> QueryDataset:
        raise NotImplementedError("Subclasses should implement this method.")


class Ego4DDataset(BaseDataset):
    def __init__(self, video_path: str, annotation_path: str):
        super().__init__(video_path, annotation_path)
        self._load_data()

    def __len__(self):
        return len(self.load_videos(False))

    def __getitem__(self, vid) -> Tuple[List[dict], QueryDataset]:
        annotations = self.load_annotations(vid)
        return {vid : annotations}
    
    def _load_data(self):
        with open(self.annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.video_data = data["videos"]

    ## FIXME: implement for various video formats
    def load_videos(self, is_pickle: bool) -> VideoDataset:
        if is_pickle:
            dataset = VideoDataset.load_from_pickle(self.video_path)
        else:
            video_ids = glob.glob(os.path.join(self.video_path, "*.mp4"))
            # Extract pre-existing clips from annotations for each video
            clips_per_video = self._extract_clips_from_annotations(video_ids)
            dataset = VideoDataset(video_ids, scenes_per_video=clips_per_video)
        return dataset

    def _extract_clips_from_annotations(self, video_paths: List[str]) -> Dict[str, Dict[str, Scene]]:
        """
        Extract clip information from Ego4D annotations to use as pre-existing scenes.
        Returns a dict mapping video_path -> {scene_id: Scene object}
        """
        
        clips_per_video = {}
        
        # Create a mapping from video_uid to video_path
        video_uid_to_path = {}
        for path in video_paths:
            uid = os.path.splitext(os.path.basename(path))[0]
            video_uid_to_path[uid] = path
        
        for video_entry in self.video_data:
            video_uid = video_entry.get("video_uid")
            if video_uid not in video_uid_to_path:
                continue
            
            video_path = video_uid_to_path[video_uid]
            clips = video_entry.get("clips", [])
            
            if clips:
                scenes = {}
                for i, clip in enumerate(clips):
                    scene_id = f"scene_{i}"
                    scenes[scene_id] = Scene(
                        scene_id=scene_id,
                        start_time=float(clip.get("video_start_sec", 0.0)),
                        end_time=float(clip.get("video_end_sec", 0.0)),
                        start_frame=int(clip.get("video_start_frame", 0)),
                        end_frame=int(clip.get("video_end_frame", 0)),
                    )
                clips_per_video[video_path] = scenes
        
        return clips_per_video

    def load_annotations(self, video_ids: List[str]) -> Dict[str, List[dict]]:
        annotations = {}
        for v in self.video_data:
            if v["video_uid"] not in video_ids:
                continue
            
            annotations[v["video_uid"]] = [
                ann 
                for clip in v.get("clips", []) 
                for ann in clip.get("annotations", [])
            ]

        return annotations

    def load_queries(self, video_ids: List[str]) -> QueryDataset:
        annotations = self.load_annotations(video_ids)
        queries = [
            Query(
                qid=f"{vid}_{j}_{i}",
                query_text=lq.get("query", ""),
                video_uid=vid,
                gt={
                    "start_sec": float(lq.get("video_start_sec", -1.0)),
                    "end_sec": float(lq.get("video_end_sec", -1.0)),
                    "start_frame": int(lq.get("video_start_frame", -1)),
                    "end_frame": int(lq.get("video_end_frame", -1)),
                }
            )
            for vid, ann in annotations.items()
            for j, a in enumerate(ann)
            for i, lq in enumerate(a.get("language_queries", []))
        ]
        return QueryDataset(queries)

class EgoLifeDataset(BaseDataset):
    """
    Dataset wrapper for EgoLife dataset.
    Structure: video_path/A{n}_{NAME}/DAY{x}/clips
    - 6 people (A1-A6): JAKE, ALICE, SHURE, KATRINA, LUCIA, TASHA
    - 7 days per person (DAY1-DAY7)
    - Each person-day is treated as one video with clips as scenes
    - Clip naming: DAYx_An_name_hhmmss00.mp4
    """
    
    # Mapping of person IDs to names
    PERSON_NAMES = {
        1: "JAKE",
        2: "ALICE", 
        3: "TASHA",
        4: "LUCIA",
        5: "KATRINA",
        6: "SHURE"
    }
    
    def __init__(self, video_path: str, annotation_path: str = None):
        super().__init__(video_path, annotation_path)
        self.pattern = r"A\d_.*[A-Za-z]+"
        self.num_people = 6
        self.num_days = 7
        
    def __len__(self):
        """Total number of person-day videos (6 people * 7 days = 42)"""
        return self.num_people * self.num_days

    def __getitem__(self, vid):
        """Get annotations for a specific video ID"""
        annotations = self.load_annotations([vid])
        return {vid: annotations.get(vid, [])}

    def _get_person_day_folders(self) -> List[Tuple[str, int, int]]:
        """
        Get all person-day folder combinations.
        Returns: List of (folder_path, person_id, day_id) tuples
        """
        person_day_folders = []
        
        for person_id in range(1, self.num_people + 1):
            person_name = self.PERSON_NAMES.get(person_id, f"UNKNOWN_{person_id}")
            person_folder = f"A{person_id}_{person_name}"
            person_path = os.path.join(self.video_path, person_folder)
            
            if not os.path.exists(person_path):
                continue
                
            for day_id in range(1, self.num_days + 1):
                day_folder = f"DAY{day_id}"
                day_path = os.path.join(person_path, day_folder)
                
                if os.path.exists(day_path):
                    person_day_folders.append((day_path, person_id, day_id))
        
        return person_day_folders

    def _parse_clip_timestamp(self, clip_filename: str) -> float:
        """
        Extract timestamp in seconds from clip filename.
        Format: DAYx_An_name_hhmmsscc.mp4 where cc is centiseconds
        Returns: timestamp in seconds from midnight (with decimal for centiseconds)
        """
        try:
            # Remove extension
            name = os.path.splitext(clip_filename)[0]
            # Split by underscore and get the last part (timestamp)
            parts = name.split('_')
            timestamp_str = parts[-1]  # hhmmsscc
            
            if len(timestamp_str) < 6:
                print(f"Error parsing timestamp from {clip_filename}: timestamp too short")
                return 0.0
            
            # Extract hours, minutes, seconds
            hh = int(timestamp_str[0:2])
            mm = int(timestamp_str[2:4])
            ss = int(timestamp_str[4:6])
            
            # Centiseconds (optional, last 2 digits)
            cc = 0
            if len(timestamp_str) >= 8:
                cc = int(timestamp_str[6:8])
            
            # Convert to total seconds (including centiseconds as decimal)
            total_seconds = hh * 3600 + mm * 60 + ss + cc / 100.0
            return total_seconds
        except (ValueError, IndexError) as e:
            print(f"Error parsing timestamp from {clip_filename}: {e}")
            return 0.0

    def _extract_clips_for_day(self, day_path: str, person_id: int, day_id: int) -> Dict[str, Scene]:
        """
        Extract clip information for a specific person-day.
        Returns: Dict mapping scene_id to Scene object
        
        Args:
            day_path: Path to the day folder containing clips
            person_id: ID of the person (1-6) - used for validation
            day_id: ID of the day (1-7) - used for validation
        """
        # Get all mp4 files in the day folder
        clip_files = glob.glob(os.path.join(day_path, "*.mp4"))
        
        if not clip_files:
            return {}
        
        # Parse and sort clips by timestamp
        clip_data = []
        for clip_path in clip_files:
            clip_name = os.path.basename(clip_path)
            timestamp = self._parse_clip_timestamp(clip_name)
            clip_data.append((clip_path, clip_name, timestamp))
        
        # Sort by timestamp
        clip_data.sort(key=lambda x: x[2])
        
        # Create Scene objects
        scenes = {}
        for i, (clip_path, clip_name, start_timestamp) in enumerate(clip_data):
            scene_id = f"scene_{i}"
            
            # Start time is the clip's timestamp
            start_time = float(start_timestamp)
            
            # End time is 1 second before next clip, or +30s for last clip
            if i < len(clip_data) - 1:
                next_timestamp = clip_data[i + 1][2]
                end_time = float(next_timestamp - 1)
            else:
                # For the last clip, assume 30 seconds duration
                end_time = start_time + 30.0
            
            # Ensure end_time is always greater than start_time
            if end_time <= start_time:
                end_time = start_time + 30.0
            
            scenes[scene_id] = Scene(
                scene_id=scene_id,
                start_time=start_time,
                end_time=end_time,
                start_frame=None,  # Will be computed during encoding if needed
                end_frame=None
            )
        
        return scenes

    def load_videos(self, is_pickle: bool) -> VideoDataset:
        """
        Load EgoLife videos. Each person-day combination is treated as one video.
        """
        if is_pickle:
            return VideoDataset.load_from_pickle(self.video_path)
        
        person_day_folders = self._get_person_day_folders()
        video_ids = []
        clips_per_video = {}
        
        for day_path, person_id, day_id in person_day_folders:
            person_name = self.PERSON_NAMES.get(person_id, f"UNKNOWN_{person_id}")
            
            # Create a virtual video ID for this person-day
            video_id = f"A{person_id}_{person_name}_DAY{day_id}"
            video_path = day_path  # Use the day folder path as the "video" path
            
            # Extract clips as scenes
            scenes = self._extract_clips_for_day(day_path, person_id, day_id)
            
            if scenes:  # Only add if there are clips
                video_ids.append(video_path)
                clips_per_video[video_path] = scenes
        
        return VideoDataset(video_ids, scenes_per_video=clips_per_video)

    def _parse_query_timestamp(self, date: str, time: str) -> float:
        """
        Parse query timestamp from date (e.g., 'DAY1') and time (e.g., '11210217').
        Format: hhmmsscc where cc is centiseconds (hundredths of a second)
        Returns: timestamp in seconds from midnight (with decimal for centiseconds)
        """
        try:
            if not time or len(time) < 6:
                print(f"Error parsing timestamp from date={date}, time={time}: time string too short or empty")
                return 0.0
            
            # Extract hours, minutes, seconds, centiseconds from time string (hhmmsscc)
            hh = int(time[0:2])
            mm = int(time[2:4])
            ss = int(time[4:6])
            
            # Centiseconds (optional, last 2 digits)
            cc = 0
            if len(time) >= 8:
                cc = int(time[6:8])
            
            # Convert to total seconds (including centiseconds as decimal)
            total_seconds = hh * 3600 + mm * 60 + ss + cc / 100.0
            return total_seconds
        except (ValueError, IndexError) as e:
            print(f"Error parsing timestamp from date={date}, time={time}: {e}")
            return 0.0
    
    def _extract_person_from_video_id(self, video_id: str) -> str:
        """
        Extract person name from video_id like 'A1_JAKE_DAY1'.
        Returns: Person name (e.g., 'JAKE')
        """
        try:
            parts = video_id.split('_')
            if len(parts) >= 2:
                return parts[1]
            return ""
        except Exception:
            return ""

    def load_annotations(self, video_ids: List[str]) -> Dict[str, List[dict]]:
        """
        Load annotations for EgoLife videos from JSON file.
        JSON structure: List of annotation dictionaries with query_time, target_time, etc.
        """
        annotations = {}
        
        # Initialize empty lists for all videos
        for vid in video_ids:
            annotations[vid] = []
        
        if not self.annotation_path or not os.path.exists(self.annotation_path):
            return annotations
        
        try:
            with open(self.annotation_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Process each annotation
            for ann in data:
                query_date = ann.get("query_time", {}).get("date", "")
                target_date = ann.get("target_time", {}).get("date", "")
                
                # Extract day number from date (e.g., 'DAY1' -> 1)
                try:
                    day_num = int(query_date.replace("DAY", ""))
                except (ValueError, AttributeError):
                    continue
                
                # Match annotation to all people for this day
                # (since annotation doesn't specify which person)
                for vid in video_ids:
                    vid_name = os.path.basename(vid)
                    # Check if this video matches the day
                    if f"DAY{day_num}" in vid_name:
                        # Create cleaned annotation (remove Chinese fields)
                        clean_ann = {
                            "id": ann.get("ID"),
                            "query_time": ann.get("query_time"),
                            "target_time": ann.get("target_time"),
                            "type": ann.get("type"),
                            "need_audio": ann.get("need_audio", False),
                            "need_name": ann.get("need_name", False),
                            "last_time": ann.get("last_time", False),
                            "trigger": ann.get("trigger"),
                            "question": ann.get("question"),
                            "choice_a": ann.get("choice_a"),
                            "choice_b": ann.get("choice_b"),
                            "choice_c": ann.get("choice_c"),
                            "choice_d": ann.get("choice_d"),
                            "answer": ann.get("answer"),
                            "keywords": ann.get("keywords"),
                            "reason": ann.get("reason"),
                        }
                        annotations[vid].append(clean_ann)
        
        except Exception as e:
            print(f"Error loading annotations: {e}")
            import traceback
            traceback.print_exc()
        
        return annotations
    
    def load_queries(self, video_ids: List[str]) -> QueryDataset:
        """
        Load queries for EgoLife videos from annotations.
        Creates Query objects with proper ground truth timestamps.
        """
        queries = []
        annotations = self.load_annotations(video_ids)
        
        for vid, ann_list in annotations.items():
            for ann in ann_list:
                # Get query and target timestamps
                query_time_dict = ann.get("query_time", {})
                target_time_dict = ann.get("target_time", {})
                
                query_date = query_time_dict.get("date", "")
                query_time_str = query_time_dict.get("time", "")
                target_date = target_time_dict.get("date", "")
                target_time_str = target_time_dict.get("time", "")
                
                # Skip annotations with missing time information
                if not target_time_str or not target_date:
                    print(f"Skipping annotation {ann.get('id', ann.get('ID'))}: missing target time information")
                    continue
                
                # Parse timestamps
                query_sec = self._parse_query_timestamp(query_date, query_time_str)
                target_sec = self._parse_query_timestamp(target_date, target_time_str)
                
                # Skip if parsing failed
                if target_sec <= 0:
                    print(f"Skipping annotation {ann.get('id', ann.get('ID'))}: failed to parse target time")
                    continue
                
                # Create query text with context
                question = ann.get("question", "")
                trigger = ann.get("trigger", "")
                
                # Build query text
                if trigger:
                    query_text = f"{trigger}. {question}"
                else:
                    query_text = question
                
                # Add choices if this is a multiple choice question
                choices = []
                for choice_key in ["choice_a", "choice_b", "choice_c", "choice_d"]:
                    choice = ann.get(choice_key)
                    if choice:
                        choices.append(choice)
                
                if choices:
                    query_text += " Choices: " + ", ".join(choices)
                
                # Get video name for creating proper video_uid
                vid_name = os.path.basename(vid)
                
                # Create Query object
                query = Query(
                    qid=f"{vid_name}_{ann.get('id', ann.get('ID', 'unknown'))}",
                    query_text=query_text,
                    video_uid=vid_name,
                    gt={
                        "start_sec": float(target_sec),
                        "end_sec": float(target_sec + 30.0),  # Assume 30s window
                        "start_frame": -1,
                        "end_frame": -1,
                    }
                )
                
                # Store additional metadata in decomposed field
                query.decomposed = {
                    "text": question,
                    "audio": question if ann.get("need_audio") else "",
                    "video": question,
                    "metadata": {
                        "query_time_sec": query_sec,
                        "target_time_sec": target_sec,
                        "query_date": query_date,
                        "target_date": target_date,
                        "type": ann.get("type"),
                        "trigger": trigger,
                        "keywords": ann.get("keywords"),
                        "answer": ann.get("answer"),
                        "choices": {
                            "A": ann.get("choice_a"),
                            "B": ann.get("choice_b"),
                            "C": ann.get("choice_c"),
                            "D": ann.get("choice_d"),
                        }
                    }
                }
                
                queries.append(query)
        
        return QueryDataset(queries)
    
class DatasetFactory:
    @staticmethod
    def get_dataset(dataset_type: str, video_path: str, annotation_path: str = None) -> BaseDataset:
        if dataset_type.lower() == "ego4d":
            return Ego4DDataset(video_path, annotation_path)
        elif dataset_type.lower() == "egolife":
            return EgoLifeDataset(video_path, annotation_path)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

def main():
    dataset = Ego4DDataset(
        video_path=CONFIG.data.video_path,
        annotation_path=CONFIG.data.annotation_path
    )
    videos = dataset.load_videos(is_pickle=False)
    for video in videos.video_files: 
        print(video)
    queries = dataset.load_queries(videos.get_uids())
    print(len(queries))
    for q in queries.queries[:50]:
        print(q)
    # Check that no duplicate qids exist
    qid_set = set()
    for q in queries.queries:
        if q.qid in qid_set:
            print(f"\n\nDuplicate qid found: {q.qid}\n\n")
        qid_set.add(q.qid)

    # Test pickle loading
    dataset.video_path=CONFIG.data.video_dataset
    videos_pickle = dataset.load_videos(is_pickle=True)
    for video in videos_pickle.video_files: 
        print(video)

if __name__ == "__main__":
    main()