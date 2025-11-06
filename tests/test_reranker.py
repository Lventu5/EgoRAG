"""
Unit tests for Reranker.

Tests modality agreement, temporal decay, and scene scoring.
"""

import unittest
from data.video_dataset import Scene


class TestReranker(unittest.TestCase):
    
    def test_modality_agreement_high(self):
        """Test agreement with similar scores."""
        from retrieval.reranker import modality_agreement
        
        scores = {"video": 0.8, "audio": 0.82, "text": 0.79}
        agreement = modality_agreement(scores)
        
        # High agreement should give high score
        self.assertGreater(agreement, 0.7)
    
    def test_modality_agreement_low(self):
        """Test agreement with dissimilar scores."""
        from retrieval.reranker import modality_agreement
        
        scores = {"video": 0.9, "audio": 0.3, "text": 0.5}
        agreement = modality_agreement(scores)
        
        # Low agreement (high variance) should give lower score
        self.assertLess(agreement, 0.6)
    
    def test_temporal_decay(self):
        """Test temporal decay function."""
        from retrieval.reranker import temporal_decay
        
        half_life = 3600.0  # 1 hour
        
        # At half-life, should be 0.5
        decay = temporal_decay(half_life, half_life)
        self.assertAlmostEqual(decay, 0.5, places=2)
        
        # At zero distance, should be 1.0
        decay = temporal_decay(0.0, half_life)
        self.assertEqual(decay, 1.0)
        
        # At 2x half-life, should be 0.25
        decay = temporal_decay(2 * half_life, half_life)
        self.assertAlmostEqual(decay, 0.25, places=2)
    
    def test_reranker_score_no_time_hint(self):
        """Test scoring without time hint."""
        from retrieval.reranker import Reranker
        
        reranker = Reranker(alpha=0.7, beta=0.2, gamma=0.1)
        
        scene = Scene(
            scene_id="test",
            start_time=10.0,
            end_time=15.0,
            start_frame=0,
            end_frame=100
        )
        
        fused_score = 0.8
        mod_scores = {"video": 0.8, "audio": 0.82, "text": 0.79}
        
        score = reranker.score_scene(
            query_time_hint=None,
            scene=scene,
            fused_score=fused_score,
            scores_by_modality=mod_scores
        )
        
        # Score should be weighted combination
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)
    
    def test_reranker_score_with_time_hint(self):
        """Test scoring with temporal proximity."""
        from retrieval.reranker import Reranker
        
        reranker = Reranker(alpha=0.7, beta=0.2, gamma=0.1, half_life_sec=60.0)
        
        scene = Scene(
            scene_id="test",
            start_time=10.0,
            end_time=20.0,
            start_frame=0,
            end_frame=100
        )
        
        fused_score = 0.8
        mod_scores = {"video": 0.8, "audio": 0.82, "text": 0.79}
        
        # Close time hint
        score_close = reranker.score_scene(
            query_time_hint=15.0,  # Middle of scene
            scene=scene,
            fused_score=fused_score,
            scores_by_modality=mod_scores
        )
        
        # Far time hint
        score_far = reranker.score_scene(
            query_time_hint=500.0,  # Very far
            scene=scene,
            fused_score=fused_score,
            scores_by_modality=mod_scores
        )
        
        # Closer time should give higher score
        self.assertGreater(score_close, score_far)


if __name__ == "__main__":
    unittest.main()
