import torch
import torchvision
import torchaudio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import tqdm
from tqdm import tqdm
import av
import cv2

import os
import warnings
import glob
from typing import Tuple, Optional

from metrics import *
from torchaudio_transforms import *

torchaudio.set_audio_backend("sox_io") 
os.environ["IMAGEIO_FFMPEG_EXE"] = "/home/sgurram/anaconda3/bin/ffmpeg"
warnings.filterwarnings("ignore")

def get_audio(path):
    input_ = av.open(path, 'r')
    input_stream = input_.streams.audio[0]
    aud = np.empty([input_stream.frames, 2, input_stream.frame_size])

    for idx, frame in enumerate(input_.decode(audio=0)):
        aud[idx] = frame.to_ndarray(format='sp32')
    input_.close()

    #channel avg, and flatten
    aud = torch.from_numpy(aud).mean(dim=1).type(dtype=torch.float32)
    return torch.flatten(aud)

    # aud = torch.from_numpy(aud).type(dtype=torch.float32)
    # return torch.reshape(aud, (aud.size(1), -1))[1]

#Implementation from https://github.com/CannyLab/aai/blob/main/aai/utils/video/file.py
def get_video(path):
    input_ = av.open(path, 'r')
    input_stream = input_.streams.video[0]
    vid = np.empty([input_stream.frames, input_stream.height, input_stream.width, 3], dtype=np.uint8)

    for idx, frame in enumerate(input_.decode(video=0)):
        vid[idx] = frame.to_ndarray(format='rgb24')
    input_.close()

    return torch.from_numpy(vid)

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


def get_audiovisual(path):
    input_a = av.open(path, 'r')
    input_v = av.open(path, 'r')

    a_stream = input_a.streams.audio[0]
    v_stream = input_v.streams.video[0]

    audio_channels = 2
    video_channels = 3

    vid = np.empty([v_stream.frames, v_stream.height, v_stream.width, video_channels], dtype=np.uint8)
    aud = np.empty([a_stream.frames, audio_channels, a_stream.frame_size])

    for idx, frame in enumerate(input_a.decode(audio=0)):
        aud_frame = frame.to_ndarray(format='sp32')
        pad =  a_stream.frame_size - aud_frame.shape[-1]
        if pad > 0:
            aud[idx] =  np.pad(aud_frame, pad_width=[(0,0),(0, pad)])
        else:
            aud[idx] = aud_frame

    for idx, frame in enumerate(input_v.decode(video=0)):
        vid[idx] = frame.to_ndarray(format='rgb24')

    input_a.close()
    input_v.close()

    aud = get_log_mel_spec(torch.flatten(torch.from_numpy(aud).mean(dim=1).type(dtype=torch.float32)))[:,:]
    vid = torch.from_numpy(resize_video(vid, target_size=(128,128))).type(dtype=torch.float32)
    vid = torch.reshape(vid, (vid.size(-1), vid.size(0), vid.size(1), vid.size(2)))

    #aud shape: [M * N], where M = 128, T ~ 2000
    #vid shape: [C * N * H * W], where N <= 300, H = W = 128, C = 3

    return aud, vid

def get_wave(path):
    wave, samp_freq = torchaudio.load(path)
    wave = wave.mean(dim=0) #avg both channels to get single audio strean
    return wave, samp_freq


def get_mfcc(wave, samp_freq=16000):
    return np.array((torchaudio.transforms.MFCC(sample_rate=samp_freq)(wave.unsqueeze(0))).mean(dim=0))


def get_mel_spec(wave, samp_freq=16000):
    wave = torch.unsqueeze(wave, 0)
    return (torchaudio.transforms.MelSpectrogram(sample_rate=samp_freq)(wave))[0,:,:]


def get_log_mel_spec(wave, samp_freq=16000):
    wave = torch.unsqueeze(wave, 0)
    spec = torchaudio.transforms.MelSpectrogram()(wave)
    spec = (spec.log2()[0,:,:])
    spec[torch.isinf(spec)] = 0
    return spec 


def augment(sample, wave_transform, spec_transform, threshold, fixed_crop=True):
    wave = wave_transform(threshold)(sample)
    wave = wave.type(torch.FloatTensor)
    spec = get_log_mel_spec(wave)

    if fixed_crop:
        spec = spec_transform(threshold)(SpecFixedCrop(threshold)(spec))
        spec[torch.isinf(spec)] = 0
        return spec
    
    spec = spec_transform(threshold)(SpecRandomCrop(threshold)(spec))
    spec[torch.isinf(spec)] = 0
    return spec

def get_augmented_views(path):
    sample, _ = get_wave(path)

    wave1 =  random.choice(list(wave_transforms.values()))
    spec1 =  random.choice(list(spec_transforms.values()))
    threshold1 = random.uniform(0.0, 0.5)

    wave2 =  random.choice(list(wave_transforms.values()))
    spec2 =  random.choice(list(spec_transforms.values()))
    threshold2 = random.uniform(0.0, 0.5)

    # wave1 = WaveIdentity
    # wave2 = WaveIdentity

    # spec1 = SpecIdentity
    # spec2 = SpecIdentity

    return augment(sample, wave1, spec1, threshold1), augment(sample, wave2, spec2, threshold2), (wave1, spec1), (wave2, spec2)

def get_temporal_shuffle_views(path):
    sample, _ = get_wave(path)
    wave = WaveIdentity
    spec1 = SpecIdentity
    spec2 = SpecPermutes
    threshold = random.uniform(0.0, 0.5)

    # Return (anchor, permutes), anchor is single sample, permutes is a list of samples
    return augment(sample, wave, spec1, threshold, fixed_crop=False), augment(sample, wave, spec2, threshold1)
    
if __name__ == '__main__':
    dir = "/big/davidchan/kinetics/kinetics_val_clipped"
    wav_paths = []
    for path in glob.glob(f'{dir}/*.mp4'):
        wav_paths.append(path)

    for filepath in tqdm(wav_paths[:1]):
        # wav_path = "/{dir}/kinetics_audio/train/25_riding a bike/0->--JMdI8PKvsc.wav".format(dir = data)
        # wav_path = "/big/kinetics_audio/train/668_jumping sofa/27->-L3lKNeY5mIs.wav"
        wav_path = "/big/kinetics_audio/validate/542_ice swimming/55->-Zp44Wj7soCE.wav"
        # wav_path = "/big/kinetics_audio/validate/590_playing paintball/61->-oTCio7AcabE.wav"

        # filepath = "/big/davidchan/kinetics/kinetics_train_clipped/-JMdI8PKvsc.mp4"
        filepath = "/big/davidchan/kinetics/kinetics_train_clipped/L3lKNeY5mIs.mp4"
        # filepath = "/big/davidchan/kinetics/kinetics_val_clipped/Zp44Wj7soCE.mp4"
        # filepath = "/big/davidchan/kinetics/kinetics_val_clipped/oTCio7AcabE.mp4"

        aud, vid = get_audiovisual(filepath)

        print(aud.shape)
        print(vid.shape)

        # vid = get_audio(filepath)
        # vid = get_video(filepath)
        # wav, _ = get_wave(wav_path)

        # print(vid)
        # print(vid.shape)

        # spec1 = get_log_mel_spec(wav)
        # spec2 = get_log_mel_spec(vid)

        # f = plt.figure()
        # f.add_subplot(2, 1, 1)
        # plt.imshow(spec1)


        # f.add_subplot(2, 1, 2)
        # plt.imshow(spec2)

        # plt.savefig("Desktop/pyav_spec.png")
        
        # view1, view2, _, _ = get_augmented_views(filepath)
        # permutes = get_temporal_shuffle_views(filepath)
        # view1, view2 = permutes[5], permutes[10]
    
    # print(permutes.shape)
    # f = plt.figure()
    # f.add_subplot(1, 2, 1)
    # plt.imshow(view1)

    # f.add_subplot(1, 2, 2)
    # plt.imshow(view2)
    # plt.savefig("Desktop/log_mel_two_views.png")

#Test git push on stout
