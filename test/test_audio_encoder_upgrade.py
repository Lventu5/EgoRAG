"""
Quick test script for the upgraded AudioEncoder
Shows the new features and output format
"""

import sys
sys.path.append('.')

from indexing.components.audio_encoder import AudioEncoder
import torch
import logging

logging.basicConfig(level=logging.INFO)

def test_audio_encoder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Initialize encoder
    encoder = AudioEncoder(device=device)
    
    # Show configuration
    print("Configuration:")
    print(f"  - Using faster-whisper: {encoder.use_faster_whisper}")
    print(f"  - Audio events enabled: {encoder.use_audio_events}")
    print(f"  - Diarization enabled: {encoder.use_diarization}")
    print()
    
    # Load models
    print("Loading models...")
    encoder.load_models()
    print("Models loaded successfully!\n")
    
    # Example usage (you need to provide a real video path)
    video_path = "/work/courses/dslab/team21/data/AIDestroyingInternet.mp4"
    start_time = 0.0
    end_time = 360.0
    
    print(f"Example: Encoding audio from {start_time}s to {end_time}s")
    print("Expected output keys:")
    print("  - audio_embedding: torch.Tensor (CLAP features)")
    print("  - transcript: str (full text)")
    print("  - transcript_words: list (word-level timestamps)")
    print("  - audio_events: list (detected sounds with confidence)")
    print("  - speaker_segments: list (who spoke when)")
    print("  - has_audio: bool")
    print()
    
    # Uncomment to test with real video:
    # result = encoder.encode(video_path, start_time, end_time)
    # print("\nResult:")
    # print(f"Transcript: {result['transcript']}")
    # print(f"Audio events: {result['audio_events']}")
    # print(f"Word count: {len(result['transcript_words'])}")

if __name__ == "__main__":
    test_audio_encoder()
