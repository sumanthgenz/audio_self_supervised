import torch
import torchvision
import torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import pickle
import tqdm
from tqdm import tqdm
import av


import warnings
import glob

from metrics import *
from torchaudio_transforms import *

torchaudio.set_audio_backend("sox_io") 
os.environ["IMAGEIO_FFMPEG_EXE"] = "/home/sgurram/anaconda3/bin/ffmpeg"
warnings.filterwarnings("ignore")


#Implementation from https://github.com/CannyLab/aai/blob/main/aai/utils/video/file.py
def get_video(path):
    input_ = av.open(path, 'r')
    input_stream = input_.streams.video[0]
    vid = np.empty([input_stream.frames, input_stream.height, input_stream.width, 3], dtype=np.uint8)

    for idx, frame in enumerate(input_.decode(video=0)):
        vid[idx] = frame.to_ndarray(format='rgb24')
    input_.close()

    return torch.from_numpy(vid)

def get_audio(path):
    input_ = av.open(path, 'r')
    input_stream = input_.streams.audio[0]
    aud = np.empty([input_stream.frames, 2, input_stream.frame_size])

    for idx, frame in enumerate(input_.decode(audio=0)):
        aud[idx] = frame.to_ndarray(format='sp16')
    input_.close()

    #channel avg, and flatten
    aud = torch.from_numpy(aud).mean(dim=1).type(dtype=torch.float32)
    return torch.flatten(aud)



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
    return spec.log2()[0,:,:]


def augment(sample, wave_transform, spec_transform, threshold, fixed_crop=True):
    wave = wave_transform(threshold)(sample)
    wave = wave.type(torch.FloatTensor)
    spec = get_log_mel_spec(wave)

    #suppressing "assert mask_end - mask_start < mask_param" for time/freq masks
    # try:
    #     return spec_transform(threshold)(SpecFixedCrop(threshold)(spec[15:]))
    # except:
    #     # return SpecFixedCrop(threshold)(spec)
    #     return spec_transform(threshold)(SpecFixedCrop(threshold)(spec[15:]))


    #cropping mel-bins by [15:] to remove NaNs
    if fixed_crop:
        return spec_transform(threshold)(SpecFixedCrop(threshold)(spec[15:]))

    return spec_transform(threshold)(SpecRandomCrop(threshold)(spec[15:]))

    # return SpecCrop(threshold)(spec)

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

    # spec1 = SpecShuffle
    # spec2 = SpecCheckerNoise

    print(wave1, spec1)
    print(wave2, spec2)

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
    for _ in tqdm(range(1)):
        # filepath = "/{dir}/kinetics_audio/train/25_riding a bike/0->--JMdI8PKvsc.wav".format(dir = data)
        # filepath = "/big/davidchan/kinetics/kinetics_val_clipped/---QUuC4vJs.mp4"
        # filepath = "/big/davidchan/kinetics/kinetics_val_clipped/-0ML-FXomBw.mp4"
        filepath = "/big/davidchan/kinetics/kinetics_val_clipped/-5jkBtJb8xU.mp4"

        vid = get_audio(filepath)
        # vid = get_video(filepath)
        # vid, _ = get_wave(filepath)

        print(vid)
        print(vid.shape)

        spec = get_log_mel_spec(vid)
        f = plt.figure()
        plt.imshow(spec)
        plt.savefig("Desktop/pyav_spec.png")
        
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
