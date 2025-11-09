"""
LLaVA Video Wrapper Module

This module wraps the LLaVA Video model for video encoding.
LLaVA Video is a vision-language model that can encode video frames.
"""

import torch
import torch.nn as nn
from typing import List, Optional
import numpy as np
from PIL import Image
from transformers import AutoProcessor, LlavaNextVideoForConditionalGeneration


class LLaVAVideoWrapper(nn.Module):
    """
    Wrapper for LLaVA Video model that provides a simple interface
    for video encoding compatible with our VideoEncoder class.
    """
    
    def __init__(
        self,
        model_id: str = "lmms-lab/LLaVA-Video-7B-Qwen2",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize LLaVA Video wrapper.
        
        Args:
            model_id: HuggingFace model ID for LLaVA Video
            device: Device to load the model on
            torch_dtype: Data type for model weights
        """
        super().__init__()
        
        self.device = device
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        
        print(f"Loading LLaVA Video model: {model_id}...")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Load model
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        
        self.model.eval()
        
        print(f"LLaVA Video model loaded successfully on {device}")
        
        # Get the vision tower's hidden size for embedding dimension
        self.embed_dim = self.model.config.vision_config.hidden_size
    
    @torch.no_grad()
    def encode_video(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Encode video frames into embeddings.
        
        Args:
            frames: List of numpy arrays representing video frames (H, W, C) in range [0, 255]
        
        Returns:
            Video embeddings tensor of shape (1, embed_dim)
        """
        # Convert numpy arrays to PIL Images
        pil_frames = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                if frame.max() > 1.0:
                    frame = frame.astype(np.uint8)
                pil_frame = Image.fromarray(frame)
                pil_frames.append(pil_frame)
            else:
                pil_frames.append(frame)
        
        # Process frames - LLaVA Video expects a list of PIL images
        # We need to provide a dummy conversation/prompt for processing
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": "Describe this video."},
                ],
            },
        ]
        
        # Apply chat template to format the conversation
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Process the video frames
        inputs = self.processor(
            text=prompt,
            videos=[pil_frames],  # LLaVA expects list of frame lists
            return_tensors="pt",
            padding=True,
        ).to(self.device, self.torch_dtype)
        
        # Get vision embeddings from the model
        # We extract the vision features before the language model
        with torch.cuda.amp.autocast(dtype=self.torch_dtype):
            # Forward through vision tower
            vision_outputs = self.model.vision_tower(
                inputs["pixel_values_videos"],
                output_hidden_states=True
            )
            
            # Get the last hidden state and pool it
            vision_features = vision_outputs.last_hidden_state
            
            # Apply global average pooling across spatial dimensions
            # vision_features shape: (batch, num_frames * num_patches, hidden_dim)
            pooled_features = vision_features.mean(dim=1)  # (batch, hidden_dim)
        
        return pooled_features
    
    def forward(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            frames: List of numpy arrays representing video frames
        
        Returns:
            Output embeddings
        """
        return self.encode_video(frames)


def load_llava_video_model(
    model_id: str = "lmms-lab/LLaVA-Video-7B-Qwen2",
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
) -> LLaVAVideoWrapper:
    """
    Convenience function to load LLaVA Video model.
    
    Args:
        model_id: HuggingFace model ID
        device: Device to load model on
        torch_dtype: Data type for model weights
    
    Returns:
        LLaVAVideoWrapper instance
    """
    return LLaVAVideoWrapper(
        model_id=model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
