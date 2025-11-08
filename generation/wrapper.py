from abc import ABC, abstractmethod
import torch
import numpy as np
import copy
from decord import VideoReader, cpu
from PIL import Image

from transformers import AutoModel, AutoTokenizer
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import generation.utils.loading_utils as utils

class Wrapper(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._load()

    @abstractmethod
    def _load(self):
        # Implement model loading logic here
        pass

    @abstractmethod
    def unload(self):
        # Implement model unloading logic here
        pass

    @abstractmethod
    def _load_video(self, video_path: str):
        # Implement video loading logic here
        pass

    @abstractmethod
    def generate(self, video_path: str, prompt: str) -> str:
        # Implement video generation logic here
        pass


class InternVideoWrapper(Wrapper):
    def __init__(self, model_name: str = "OpenGVLab/InternVideo2_5_Chat_8B"):
        super().__init__(model_name)
        self.chat_history = None

    def _load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).half().cuda().to(torch.bfloat16)
        self.model = self.model.eval()

    def unload(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()

    def _load_video(self, video_path: str, bound=None, input_size=448, max_num=1, num_segments=32, get_frame_by_duration=False):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        transform = utils.build_transform(input_size=input_size)
        if get_frame_by_duration:
            duration = max_frame / fps
            num_segments = utils.get_num_frames_by_duration(duration)
        frame_indices = utils.get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
            img = utils.dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list
    
    @torch.no_grad()
    def generate(self, video_path: str, prompt: str) -> str:
        max_num_frames = 512
        generation_config = dict(
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
            top_p=0.1,
            num_beams=1
        )
        num_segments = 128

        pixel_values, num_patches_list = self._load_video(video_path, num_segments=num_segments, max_num=1, get_frame_by_duration=False)
        pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)
        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
        # single-turn conversation

        question = video_prefix + prompt
        output, self.chat_history = self.model.chat(self.processor, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=self.chat_history, return_history=True)
        
        return output
    

class LLaVAVideoWrapper(Wrapper):
    def __init__(self, model_name: str = "llava_qwen", pretrained: str = "lmms-lab/LLaVA-Video-7B-Qwen2"):
        super().__init__(model_name)
        self.pretrained = pretrained
        self.dtype = "bfloat16" if torch.cuda.get_device_capability()[0] >= 8 else "float16"
        self.device_map = "auto"

    def _load(self):
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(self.pretrained, None, self.model_name, torch_dtype=self.dtype, device_map=self.device_map)

    def unload(self):
        del self.model
        del self.tokenizer
        del self.image_processor
        torch.cuda.empty_cache()

    def _load_video(self, video_path: str, max_frames_num,fps=1,force_sample=False):
        if max_frames_num == 0:
            return np.zeros((1, 336, 336, 3))
        vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps()/fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i/fps for i in frame_idx]
        if len(frame_idx) > max_frames_num or force_sample:
            sample_fps = max_frames_num
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i/vr.get_avg_fps() for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        # import pdb;pdb.set_trace()
        return spare_frames,frame_time,video_time
    
    @torch.no_grad()
    def generate(self, video_path: str, prompt: str, max_frames_num: int = 64) -> str:
        video,frame_time,video_time = self._load_video(video_path, max_frames_num, 1, force_sample=True)
        video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().half()
        video = [video]
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n{prompt}"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)
        cont = self.model.generate(
            input_ids,
            images=video,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        return text_outputs
    

def create_wrapper(model_name: str) -> Wrapper:
    model_map = {
        "InternVideo": InternVideoWrapper(),
        "LLaVAVideo": LLaVAVideoWrapper(),
    }
    try:
        wrapper = model_map[model_name]
    except KeyError:
        raise ValueError(f"Model {model_name} not supported. Available models: {list(model_map.keys())}")
    return wrapper