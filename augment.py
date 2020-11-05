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


os.environ["IMAGEIO_FFMPEG_EXE"] = "/home/sgurram/anaconda3/bin/ffmpeg"
warnings.filterwarnings("ignore")

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

def get_wave(path):
    wave, samp_freq = torchaudio.load(path)
    return wave, samp_freq

def get_mfcc(wave, samp_freq):
    return np.transpose(np.array(torchaudio.compliance.kaldi.mfcc(wave, sample_frequency=samp_freq)))

def get_mel_spec(wave):
    return (torchaudio.transforms.MelSpectrogram()(wave))[0,:,:].numpy()

def get_log_mel_spec(wave):
    spec = torchaudio.transforms.MelSpectrogram()(wave)
    return spec.log2()[0,:,:].numpy()

def wave_segment(wave):
    return wave

def wave_random_noise(wave):
    return wave

def wave_amplitude(wave):
    return wave

def wave_db(wave):
    return wave

def wave_power(wave):
    return wave

def wave_voice(wave):
    return wave

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
    

    

filepath = "/data3/kinetics_pykaldi/train/25_riding a bike/0->--JMdI8PKvsc.wav"
wave, samp_frequency = get_wave(filepath)
spec = get_log_mel_spec(wave)
print(spec.shape)

plt.figure()
plt.imshow(spec)
# plt.savefig("log_mel_spectogram.png")
