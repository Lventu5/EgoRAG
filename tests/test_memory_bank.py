"""
Unit tests for SceneMemoryBank.

Tests indexing, querying, and serialization functionality.
"""

import unittest
import tempfile
import os
import torch

from data.video_dataset import VideoDataset, VideoDataPoint, Scene


class TestSceneMemoryBank(unittest.TestCase):
    
    def setUp(self):
        """Create a minimal test dataset."""
        # Create test scenes
        scene1 = Scene(
            scene_id="scene_0",
            start_time=0.0,
            end_time=5.0,
            start_frame=0,
            end_frame=150
        )
        scene2 = Scene(
            scene_id="scene_1",
            start_time=5.0,
            end_time=10.0,
            start_frame=150,
            end_frame=300
        )
        
        # Create test datapoint
        dp = VideoDataPoint(
            video_name="test_video",
            video_path="/path/to/test.mp4"
        )
        dp.scenes = {"scene_0": scene1, "scene_1": scene2}
        dp.scene_embeddings = {
            "scene_0": {
                "video": torch.randn(768),
                "audio": torch.randn(512),
                "text": torch.randn(384),
                "caption": torch.randn(384),
                "transcript": "test transcript one",
                "caption_text": "test caption one",
                "meta": {"scene_type": "other", "speech_type": "monologue"}
            },
            "scene_1": {
                "video": torch.randn(768),
                "audio": torch.randn(512),
                "text": torch.randn(384),
                "caption": torch.randn(384),
                "transcript": "test transcript two",
                "caption_text": "test caption two",
                "meta": {"scene_type": "meeting", "speech_type": "dialogue"}
            }
        }
        dp.global_embeddings = {
            "video": torch.randn(768),
            "audio": torch.randn(512),
            "text": torch.randn(384),
            "caption": torch.randn(384)
        }
        
        # Create dataset
        self.video_dataset = VideoDataset()
        self.video_dataset.video_datapoints = [dp]
    
    def test_build_indices(self):
        """Test that indices are built correctly."""
        from indexing.memory.memory_bank import SceneMemoryBank
        
        memory_bank = SceneMemoryBank(self.video_dataset)
        memory_bank.build_indices()
        
        self.assertTrue(memory_bank.is_built())
        
        # Check that all modalities have indices
        for mod in ["video", "audio", "text", "caption"]:
            self.assertIn(mod, memory_bank.indices)
            self.assertIn(mod, memory_bank.metadata)
    
    def test_query_video_level(self):
        """Test video-level querying."""
        from indexing.memory.memory_bank import SceneMemoryBank
        
        memory_bank = SceneMemoryBank(self.video_dataset)
        memory_bank.build_indices()
        
        video_names, embeddings = memory_bank.query_video_level("video")
        
        self.assertEqual(len(video_names), 1)
        self.assertEqual(video_names[0], "test_video")
        self.assertEqual(embeddings.shape[0], 1)
        self.assertEqual(embeddings.shape[1], 768)
    
    def test_query_scene_level(self):
        """Test scene-level querying."""
        from indexing.memory.memory_bank import SceneMemoryBank
        
        memory_bank = SceneMemoryBank(self.video_dataset)
        memory_bank.build_indices()
        
        scene_ids, embeddings = memory_bank.query_scene_level("test_video", "video")
        
        self.assertEqual(len(scene_ids), 2)
        self.assertEqual(embeddings.shape[0], 2)
        self.assertEqual(embeddings.shape[1], 768)
    
    def test_query_scene_level_with_filters(self):
        """Test scene-level querying with filters."""
        from indexing.memory.memory_bank import SceneMemoryBank
        
        memory_bank = SceneMemoryBank(self.video_dataset)
        memory_bank.build_indices()
        
        # Filter by scene_type
        scene_ids, embeddings = memory_bank.query_scene_level(
            "test_video",
            "video",
            filters={"scene_type_in": ["meeting"]}
        )
        
        self.assertEqual(len(scene_ids), 1)
        self.assertIn("scene_1", scene_ids[0])
    
    def test_serialize_load(self):
        """Test serialization and loading."""
        from indexing.memory.memory_bank import SceneMemoryBank
        
        memory_bank = SceneMemoryBank(self.video_dataset)
        memory_bank.build_indices()
        
        # Serialize
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            temp_path = f.name
        
        try:
            memory_bank.serialize(temp_path)
            
            # Load into new instance
            memory_bank2 = SceneMemoryBank(self.video_dataset)
            memory_bank2.load(temp_path)
            
            self.assertTrue(memory_bank2.is_built())
            
            # Verify loaded data
            video_names, embeddings = memory_bank2.query_video_level("video")
            self.assertEqual(len(video_names), 1)
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
