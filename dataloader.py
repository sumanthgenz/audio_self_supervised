import torch
import torch.nn as nn
import torchvision
import torchaudio
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle
from tqdm import tqdm

import warnings
import glob
import gc 
import os
import socket

from augment import *
from metrics import *
from encoder import *
# from torchaudio_transforms import *

torchaudio.set_audio_backend("sox_io") 
os.environ["IMAGEIO_FFMPEG_EXE"] = "/home/sgurram/anaconda3/bin/ffmpeg"
warnings.filterwarnings("ignore")

data = ""
host = socket.gethostname()
if host == "stout":
    data = "big"
elif socket.gethostname() == "greybeard":
    data = "ssd"

class AudioData(Dataset):

    def __init__(self, dataType):
        self.dataType = dataType
        self.dir = "/{}/kinetics_audio/{}".format(data, dataType)
        self.num_classes = 700
        self.downsamp_factor = 2
        self.samp_freq = 22050*4
        self.seq_len = 500
        self.wav_paths = self.get_all_files()
        
    def get_all_files(self):
        wav_paths = []
        for path in glob.glob(f'{self.dir}/**/*.wav'):
            wav_paths.append(path)
        return wav_paths

    def get_pickle(self, classPath):
        with open('Desktop/kinetics_{}.pickle'.format(classPath), 'rb') as handle:
            result = pickle.load(handle)
        return result
    
    def __len__(self):
        return len(self.wav_paths)

    def getNumClasses(self):
        return self.num_classes

    def __getitem__(self, idx):
        try:
            filePath = self.wav_paths[idx]
            # num_label = int((filePath.split('/')[4]).split('_')[0]) - 1
            # wav, samp_freq = torchaudio.load(filePath)
            # feat = np.transpose(np.array(torchaudio.compliance.kaldi.mfcc(wav, sample_frequency=self.samp_freq)))
            # return feat, num_label, self.seq_len

            view1, view2, t1, t2 = get_augmented_views(filePath)
            # return view1.type(torch.FloatTensor), view2.type(torch.FloatTensor), t1, t2
            return view1, view2, t1, t2

        except:
            return None, None, None, None

class TemporalData(Dataset):

    def __init__(self, dataType):
        self.dataType = dataType
        self.dir = "/{}/kinetics_audio/{}".format(data, dataType)
        self.num_classes = 700
        self.downsamp_factor = 2
        self.samp_freq = 22050*4
        self.seq_len = 500
        self.wav_paths = self.get_all_files()
        
    def get_all_files(self):
        wav_paths = []
        for path in glob.glob(f'{self.dir}/**/*.wav'):
            wav_paths.append(path)
        return wav_paths

    def get_pickle(self, classPath):
        with open('Desktop/kinetics_{}.pickle'.format(classPath), 'rb') as handle:
            result = pickle.load(handle)
        return result
    
    def __len__(self):
        return len(self.wav_paths)

    def getNumClasses(self):
        return self.num_classes

    def __getitem__(self, idx):
        try:
            filePath = self.wav_paths[idx]
            # num_label = int((filePath.split('/')[4]).split('_')[0]) - 1
            # wav, samp_freq = torchaudio.load(filePath)
            # feat = np.transpose(np.array(torchaudio.compliance.kaldi.mfcc(wav, sample_frequency=self.samp_freq)))
            # return feat, num_label, self.seq_len

            anchor, permutes = get_temporal_shuffle_views(filePath)
            # return view1.type(torch.FloatTensor), view2.type(torch.FloatTensor), t1, t2
            return anchor, permutes

        except:
            return None, None, None, None


class AudioVisualData(Dataset):

    def __init__(self, dataType):
        self.dataType = dataType
        self.dir = "/big/davidchan/kinetics/kinetics_{}_clipped".format(dataType)
        self.num_classes = 700
        self.downsamp_factor = 2
        self.samp_freq = 22050*4
        self.seq_len = 500
        self.wav_paths = self.get_all_files()
        
    def get_all_files(self):
        wav_paths = []
        for path in glob.glob(f'{self.dir}/*.mp4'):
            wav_paths.append(path)
        return wav_paths

    def get_pickle(self, classPath):
        with open('Desktop/kinetics_{}.pickle'.format(classPath), 'rb') as handle:
            result = pickle.load(handle)
        return result
    
    def __len__(self):
        return len(self.wav_paths)

    def getNumClasses(self):
        return self.num_classes

    def __getitem__(self, idx):
        filePath = self.wav_paths[idx]

        return get_audiovisual(filePath)



if __name__ == '__main__':
    conv1 = torch.nn.Conv1d(
                in_channels=128, 
                out_channels=128, 
                kernel_size=50, 
    )

    conv2 = torch.nn.Conv1d(
                in_channels=128, 
                out_channels=128, 
                kernel_size=50, 
    )

    pool1 = nn.MaxPool1d(
            kernel_size=2,
    )
    
    pool2 = nn.MaxPool1d(
            kernel_size=2,
            stride=2,
    )

    conv = nn.Sequential(
            conv1,
            pool1,
            conv2,
            pool2,
    )

    model = EfficientNet.from_name(
        "efficientnet-b0", 
        include_top=False, 
        drop_connect_rate=0.1)

    encoder = AudioFeatureModel(
                            dropout=0.1,
                            model_dimension=1024)

    fc1 = torch.nn.Linear(1280, 1024)
    # bad_count = 0

    ad = AudioVisualData("train")
    for i in tqdm(range(1)):
        a, v = ad.__getitem__(i)
        a = conv(a.unsqueeze(0))
        print(a.shape)
        print(a)
