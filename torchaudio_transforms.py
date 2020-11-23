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

import warnings
import glob
import socket


from metrics import *

assert torch.__version__.startswith("1.7")
assert torchaudio.__version__.startswith("0.7")

torchaudio.set_audio_backend("sox_io") 

data = ""
host = socket.gethostname()
if host == "stout":
    data = "big"
elif socket.gethostname() == "greybeard":
    data = "ssd"

class WaveIdentity():

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, wave):
        return wave


class WaveSegment():

    def __init__(self, threshold):
        self.segment_size = 1 - threshold

    def __call__(self, wave):
        size = int(wave.shape[0] * self.segment_size)
        start = random.randint(0, (wave.shape[0] - size))
        return wave[start : (start + size)]


class WaveGaussianNoise():
    def __init__(self, threshold):
        self.noise_intensity = threshold
        self.constant = 0.2

    def __call__(self, wave):
        noise = self.noise_intensity * self.constant
        return wave + (noise * np.random.normal(size=wave.shape[0]))


class WaveAmplitude():
    def __init__(self, threshold):
        self.amplitude_scale = threshold
        self.constant = 10

    def __call__(self, wave):
        amp = self.amplitude_scale * self.constant
        wave = torch.unsqueeze(wave, 0)
        wave = torchaudio.transforms.Vol(gain=amp, gain_type="amplitude")(wave)
        return torch.squeeze(wave, 0)


class WavePower():
    def __init__(self, threshold):
        self.power_scale = threshold
        self.constant = 10

    def __call__(self, wave):
        amp = self.power_scale * self.constant
        wave = torch.unsqueeze(wave, 0)
        wave = torchaudio.transforms.Vol(gain=amp, gain_type="power")(wave)
        return torch.squeeze(wave, 0)


class WaveDB():
    def __init__(self, threshold):
        self.db_scale = threshold
        self.constant = 10

    def __call__(self, wave):
        amp = self.db_scale * self.constant
        wave = torch.unsqueeze(wave, 0)
        wave = torchaudio.transforms.Vol(gain=amp, gain_type="db")(wave)
        return torch.squeeze(wave, 0)


class SpecIdentity():
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, spec):
        return wave


class SpecCrop():
    def __init__(self, threshold):
        self.crop_size = 1 - threshold

    def __call__(self, spec):
        # size = int(spec.shape[1] * self.crop_size)
        size = 1000
        start = random.randint(0, (spec.shape[1] - size))
        return spec[:, start : (start + size)]


class SpecGaussianNoise():
    def __init__(self, threshold):
        self.noise_intensity = threshold
        self.constant = 0.2

    def __call__(self, spec):
        noise = self.noise_intensity * self.constant
        return spec + (noise * np.random.normal(size=spec.shape))


class SpecTimeMask():
    def __init__(self, threshold):
        self.mask_size = threshold

    def __call__(self, spec):
        size = int(spec.shape[1] * self.mask_size)
        return torchaudio.transforms.TimeMasking(size)(specgram=spec)


class SpecFreqMask():
    def __init__(self, threshold):
        self.mask_size = threshold

    def __call__(self, spec):
        size = int(spec.shape[0] * self.mask_size)
        return torchaudio.transforms.FrequencyMasking(size)(specgram=spec)


class SpecCheckerNoise():
    def __init__(self, threshold):
        self.mask_size = threshold

    def __call__(self, spec):
        f = SpecFreqMask(self.mask_size)
        t = SpecTimeMask(self.mask_size)
        return f(t(spec))


class SpecFlip():
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, spec):
        return torch.flip(spec, [0])    


class SpecTimeReverse():
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, spec):
        return torch.flip(spec, [1])    



# used to access transforms by name
wave_transforms = {
        'wave_identity': WaveIdentity,
        'wave_gaussian_noise': WaveGaussianNoise,
        'wave_amplitude,': WaveAmplitude,
        'wave_db': WaveDB,
        'wave_power': WavePower
}

spec_transforms = {
        'spec_identity': SpecIdentity,
        'spec_gaussian_noise': SpecGaussianNoise,
        'spec_checker_noise': SpecCheckerNoise,
        'spec_flip': SpecFlip,
        'spec_time_reverse': SpecTimeReverse
}



def get_wave(path):
    wave, samp_freq = torchaudio.load(path)
    wave = wave.mean(dim=0) #avg both channels to get single audio strean
    return wave, samp_freq


def get_mfcc(wave, samp_freq=16000):
    return np.array((torchaudio.transforms.MFCC(sample_rate=samp_freq)(wav.unsqueeze(0))).mean(dim=0))


def get_mel_spec(wave, samp_freq=16000):
    wave = torch.unsqueeze(wave, 0)
    return (torchaudio.transforms.MelSpectrogram(sample_rate=samp_freq)(wave))[0,:,:]


def get_log_mel_spec(wave, samp_freq=16000):
    wave = torch.unsqueeze(wave, 0)
    spec = torchaudio.transforms.MelSpectrogram()(wave)
    return spec.log2()[0,:,:]


def augment(sample, wave_transform, spec_transform, threshold):
    wave = wave_transform(threshold)(sample)
    wave = wave.type(torch.FloatTensor)
    spec = get_log_mel_spec(wave)

    #suppressing "assert mask_end - mask_start < mask_param" for time/freq masks
    try:
        return spec_transform(threshold)(SpecCrop(threshold)(spec))
    except:
        return SpecCrop(threshold)(spec)
    # return SpecCrop(threshold)(spec)

def get_augmented_views(path):
    sample, _ = get_wave(path)

    wave1 =  random.choice(list(wave_transforms.values()))
    spec1 =  random.choice(list(spec_transforms.values()))
    threshold1 = random.uniform(0.0, 0.5)

    wave2 =  random.choice(list(wave_transforms.values()))
    spec2 =  random.choice(list(spec_transforms.values()))
    threshold2 = random.uniform(0.0, 0.5)

    return augment(sample, wave1, spec1, threshold1), augment(sample, wave2, spec2, threshold2), (wave1, spec1), (wave2, spec2)
    
if __name__ == '__main__':
    for _ in tqdm(range(250)):
        filepath = "/{dir}/kinetics_audio/train/25_riding a bike/0->--JMdI8PKvsc.wav".format(dir = data)
        # filepath = "/big/kinetics_audio/train/25_riding a bike/0->--JMdI8PKvsc.wav"
        view1, view2, _, _ = get_augmented_views(filepath)
        
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(view1)

    f.add_subplot(1, 2, 2)
    plt.imshow(view2)
    plt.savefig("Desktop/log_mel_two_views.png")

#Test git push on stout
