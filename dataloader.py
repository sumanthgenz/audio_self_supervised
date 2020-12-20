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

class AudioDataset(Dataset):

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

class TemporalDataset(Dataset):

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

            permutes = get_temporal_shuffle_views(filePath)
            # return view1.type(torch.FloatTensor), view2.type(torch.FloatTensor), t1, t2
            return permutes

        except:
            return None, None, None, None

if __name__ == '__main__':
    cnn1 = torch.nn.Conv2d(
                    in_channels=1, 
                    out_channels=3, 
                    kernel_size=3)
    model = EfficientNet.from_name(
        "efficientnet-b0", 
        include_top=False, 
        drop_connect_rate=0.1)

    encoder = AudioFeatureModel(
                            dropout=0.1,
                            model_dimension=1024)

    fc1 = torch.nn.Linear(1280, 1024)
    # bad_count = 0

    ad = AudioDataset("train")
    for i in tqdm(range(1)):
        view1, view2, t1, t2 = ad.__getitem__(250)

        # print(t1)
        # print(t2)
        # print(view1.shape)
        # view1, view2 = view1.type(torch.FloatTensor), view2.type(torch.FloatTensor)
        # print(view1.shape)
        # view1, view2 = cnn1(view1.unsqueeze(0).unsqueeze(0)), cnn1(view2.unsqueeze(0).unsqueeze(0))
        # print(view1.shape)
        # view1, view2 = model(view1), model(view2)
        # print(view1.shape)
        # view1, view2 =  view1.squeeze(3).squeeze(2), view2.squeeze(3).squeeze(2)
        # print(view1.shape)
        # view1, view2 = fc1(view1), fc1(view2)
        view1, view2 = encoder((view1.type(torch.FloatTensor)).unsqueeze(0)), encoder((view2.type(torch.FloatTensor)).unsqueeze(0))


    print(view1.detach())
    print(view2.detach())

    # print(kl_divergence(
    #                 (view1.squeeze(0)).detach(),
    #                 (view1.squeeze(0)).detach()
    #             )
    # )


    # f = plt.figure()
    # f.add_subplot(1, 2, 1)
    # plt.imshow(view1)

    # f.add_subplot(1, 2, 2)
    # plt.imshow(view2)
    # plt.savefig("Desktop/log_dataloader_two_views.png")