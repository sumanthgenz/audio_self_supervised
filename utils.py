import cv2
import av
import torch
import torchaudio
import torchvision
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

#Implementation from https://github.com/CannyLab/aai/blob/main/aai/utils/video/transform.py
def resize_video(video_frames: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    assert len(video_frames.shape) == 4, 'Video should have shape [N_Frames x H x W x C]'
    # print(video_frames)

    pad = 300 -  video_frames.shape[0]
    if pad > 0:
        video_frames =  np.pad(video_frames, pad_width=[(0, pad), (0,0), (0,0), (0,0)])

    output_array = np.zeros((
        video_frames.shape[0],
        target_size[0],
        target_size[1],
        video_frames.shape[3],
    ))
    for i in range(video_frames.shape[0]):
        output_array[i] = cv2.resize(video_frames[i], target_size)
    # return output_array, pad
    return output_array

def pad_spec(spec, pad_len=2000):
    to_pad = pad_len - spec.shape[-1] 
    if to_pad > 0:
        return torch.nn.functional.pad(spec, (0,to_pad,0,0))
    return spec[:,:pad_len]

#Implementation from https://github.com/CannyLab/aai/blob/4a93c14d834f045ee3fa61929c4f17ebc765d10c/aai/utils/torch/masking.py#L20
def get_src_conditional_mask(max_sequence_length):
    mask = (torch.triu(torch.ones(max_sequence_length, max_sequence_length)) == 1).transpose(0, 1)
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

#Implementation from https://github.com/CannyLab/aai/blob/4a93c14d834f045ee3fa61929c4f17ebc765d10c/aai/utils/torch/masking.py#L20
def sequence_mask(lengths, maxlen=None, right_aligned=False):
    # https://discuss.pytorch.org/t/pytorch-equivalent-for-tf-sequence-mask/39036
    if maxlen is None:
        maxlen = lengths.max()
    matrix = torch.unsqueeze(lengths, dim=-1)
    row_vector = torch.arange(0, maxlen, 1).type_as(matrix)
    if not right_aligned:
        mask = row_vector < matrix
    else:
        mask = row_vector > (-matrix + (maxlen - 1))

    return mask.bool()