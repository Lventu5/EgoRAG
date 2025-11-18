import cv2
from transformers import AutoModel

from utils.modeling_internvideo2 import (retrieve_text, vid2tensor, _frame_from_video,)

if __name__ == '__main__':
    model = AutoModel.from_pretrained("OpenGVLab/InternVideo2-Stage2_6B", trust_remote_code=True).eval()

    video = cv2.VideoCapture('example1.mp4')
    frames = [x for x in _frame_from_video(video)]
    text_candidates = ["A playful dog and its owner wrestle in the snowy yard, chasing each other with joyous abandon.",
                    "A man in a gray coat walks through the snowy landscape, pulling a sleigh loaded with toys.",
                    "A person dressed in a blue jacket shovels the snow-covered pavement outside their house.",
                    "A cat excitedly runs through the yard, chasing a rabbit.",
                    "A person bundled up in a blanket walks through the snowy landscape, enjoying the serene winter scenery."]

    texts, probs = retrieve_text(frames, text_candidates, model=model, topk=5)
    for t, p in zip(texts, probs):
        print(f'text: {t} ~ prob: {p:.4f}')

    vidtensor = vid2tensor('models/InternVideo/Data/example1.mp4', fnum=4)
    feat = model.get_vid_feat(vidtensor)
    print('video feature shape:', feat.shape)