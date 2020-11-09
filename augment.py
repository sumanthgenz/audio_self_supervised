import torch
import torchvision
import torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import pickle

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
    return (torchaudio.transforms.MelSpectrogram(sample_rate=samp_freq)(wave))[0,:,:].numpy()

def get_log_mel_spec(wave):
    wave = torch.unsqueeze(wave, 0)
    spec = torchaudio.transforms.MelSpectrogram()(wave)
    return spec.log2()[0,:,:].numpy()

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
    return wave

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

def wave_resample(wave):
    return wave

def spec_crop(spec):
    return spec

def spec_random_noise(spec):
    return spec

def spec_checkerboard_noise(spec):
    return spec

def spec_flip(spec):
    return spec    

def spec_time_reverse(spec):
    return spec

def spec_time_stretch(spec):
    return spec

def spec_time_mask(spec):
    return spec

def spec_freq_mask(spec):
    return spec

wave_augmentations = [wave_segment,
                    wave_random_noise,
                    wave_amplitude,
                    wave_db,
                    wave_power,
                    wave_voice,
                    wave_resample]

spec_augmentations = [spec_crop,
                    spec_random_noise,
                    spec_checkerboard_noise,
                    spec_flip,
                    spec_time_reverse,
                    spec_time_stretch,
                    spec_time_mask,
                    spec_freq_mask]

def augment(sample, wav_idx, spec_idx):
    try:
        sample = wave_augmentations[wav_idx](sample)
    except:
        sample = sample

    spec = get_log_mel_spec(sample)

    try:
        return spec_augmentations[spec_idx](spec)
    except:
        return spec

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
    

filepath = "/data3/kinetics_pykaldi/train/25_riding a bike/0->--JMdI8PKvsc.wav"
wave, samp_frequency = get_wave(filepath)
print(torch.min(wave))
print(torch.mean(wave))
print(torch.max(wave))

res = wave_voice(wave, 0.5)
# res = wave + 0.1

print(torch.min(res))
print(torch.mean(res))
print(torch.max(res))


print(l2_norm(wave, res))

# plt.figure()
# plt.imshow(res)
# plt.savefig("log_mel_spectogram.png")
