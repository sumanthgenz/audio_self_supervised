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

import librosa
import librosa.display
from IPython.display import Audio
import pydub
from pydub import AudioSegment
from PIL import Image

import warnings
import glob

torchaudio.set_audio_backend("sox_io") 
os.environ["IMAGEIO_FFMPEG_EXE"] = "/home/sgurram/anaconda3/bin/ffmpeg"
warnings.filterwarnings("ignore")

def get_wave(path):
    wave, samp_freq = torchaudio.load(path)
    wave = wave.mean(dim=0) #avg both channels to get single audio strean
    return wave, samp_freq

def get_mfcc(wave, samp_freq=16000):
    wave = torch.unsqueeze(wave, 0)
    return np.transpose(np.array(torchaudio.compliance.kaldi.mfcc(wave, sample_frequency=samp_freq)))

def get_mel_spec(wave, samp_freq=16000):
    wave = torch.unsqueeze(wave, 0)
    return (torchaudio.transforms.MelSpectrogram(sample_rate=samp_freq)(wave))[0,:,:]

def get_log_mel_spec(wave, samp_freq=16000):
    wave = torch.unsqueeze(wave, 0)
    spec = torchaudio.transforms.MelSpectrogram()(wave)
    return spec.log2()[0,:,:]
    
def wave_identity(wave, threshold):
    return wave

def wave_segment(wave, threshold):
    size = int(wave.shape[0] * threshold)
    start = random.randint(0, (wave.shape[0] - size))
    return wave[start : (start + size)]

def wave_random_noise(wave, threshold):
    noise = threshold * 0.2
    return wave + (noise * np.random.normal(size=wave.shape[0]))

def wave_amplitude(wave, threshold):
    amp = threshold*10
    wave = torch.unsqueeze(wave, 0)
    wave = torchaudio.transforms.Vol(gain=amp, gain_type="amplitude")(wave)
    return torch.squeeze(wave, 0)

def wave_power(wave, threshold):
    amp = threshold*10
    wave = torch.unsqueeze(wave, 0)
    wave = torchaudio.transforms.Vol(gain=amp, gain_type="power")(wave)
    return torch.squeeze(wave, 0)

def wave_db(wave, threshold):
    amp = threshold*10
    wave = torch.unsqueeze(wave, 0)
    wave = torchaudio.transforms.Vol(gain=amp, gain_type="db")(wave)
    return torch.squeeze(wave, 0)

def wave_voice(wave, threshold):
    initial_size = wave.shape[0]
    reduction = threshold*10
    wave = torch.unsqueeze(wave, 0)
    wave = torchaudio.transforms.Vad(sample_rate=16000, noise_reduction_amount=reduction)(wave)
    padding = initial_size - wave.shape[1]
    wave = torch.nn.functional.pad(input=wave, pad=(padding, 0), mode='constant', value=0)
    return torch.squeeze(wave, 0)

def spec_identity(spec, threshold):
    return spec

def spec_crop(spec, threshold):
    size = int(spec.shape[1] * threshold)
    start = random.randint(0, (spec.shape[1] - size))
    return spec[:, start : (start + size)]

def spec_random_noise(spec, threshold):
    noise = threshold * 0.2
    return spec + (noise * np.random.normal(size=spec.shape))

def spec_time_mask(spec, threshold):
    size = int(spec.shape[1] * threshold)
    return torchaudio.transforms.TimeMasking(size)(specgram=spec)

def spec_freq_mask(spec, threshold):
    size = int(spec.shape[0] * threshold)
    return torchaudio.transforms.FrequencyMasking(size)(specgram=spec)

def spec_checkerboard_noise(spec, threshold):
    return spec_freq_mask(spec_time_mask(spec, threshold), threshold)

def spec_flip(spec, threshold):
    return torch.flip(spec, [0])    

def spec_time_reverse(spec, threshold):
    return torch.flip(spec, [1])    

def spec_time_stretch(spec, threshold):
    return torchaudio.transforms.TimeStretch()(spec)     # Need Complex Spectrogram

wave_augmentations = [wave_identity,
                    wave_segment,
                    wave_random_noise,
                    wave_amplitude,
                    wave_db,
                    wave_power]

spec_augmentations = [spec_identity,
                    spec_crop,
                    spec_random_noise,
                    spec_checkerboard_noise,
                    spec_flip,
                    spec_time_reverse,
                    spec_time_mask,
                    spec_freq_mask]

def augment(sample, wave_idx, spec_idx, threshold):
    # print(wave_idx)
    # print(spec_idx)
    # try:
    #     sample = wave_augmentations[wave_idx](sample, threshold)
    # except:
    #     sample = sample

    sample = wave_augmentations[wave_idx](sample, threshold)

    sample = sample.type(torch.FloatTensor)
    spec = get_log_mel_spec(sample)

    #suppressing "assert mask_end - mask_start < mask_param" for time/freq masks
    try:
        return spec_augmentations[spec_idx](spec, threshold)
    except:
        return spec

def get_augmented_views(path):
    sample, _ = get_wave(path)

    wave_idx1 =  random.randint(0, len(wave_augmentations)-1)
    spec_idx1 =  random.randint(0, len(spec_augmentations)-1)
    threshold1 = random.uniform(0.0, 0.5)

    wave_idx2 =  random.randint(0, len(wave_augmentations)-1)
    spec_idx2 =  random.randint(0, len(spec_augmentations)-1)
    threshold2 = random.uniform(0.0, 0.5)

    return augment(sample, wave_idx1, spec_idx1, threshold1), augment(sample, wave_idx2, spec_idx2, threshold2)

    
def angular_similarity(x,y):
    nx = np.linalg.norm(x.numpy())
    ny = np.linalg.norm(y.numpy())
    cos = np.dot(x, y)/(nx * ny)
    if cos > 1:
        cos = 1
    elif cos < -1:
        cos = -1
    return 1 - np.arccos(cos)/np.pi

def l2_norm(x,y):
    return np.linalg.norm(x-y)
    


for _ in tqdm(range(250)):
    filepath = "/data3/kinetics_pykaldi/train/25_riding a bike/0->--JMdI8PKvsc.wav"
    view1, view2 = get_augmented_views(filepath)

# wave, samp_frequency = get_wave(filepath)
# print(torch.min(wave))
# print(torch.mean(wave))
# print(torch.max(wave))

# res = get_log_mel_spec(wave, 16000)
# res = spec_checkerboard_noise(res, 0.5)
# res = spec_freq_mask(res, 0.5)
# res = wave + 0.1

# print(torch.min(res))
# print(torch.mean(res))
# print(torch.max(res))


# print(l2_norm(wave, res))


f = plt.figure()
f.add_subplot(1, 2, 1)
plt.imshow(view1)

f.add_subplot(1, 2, 2)
plt.imshow(view2)
plt.savefig("log_mel_two_views.png")
