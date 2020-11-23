import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
import torchaudio
import torchvision.transforms as transforms
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import numpy
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle
import tqdm
from tqdm import tqdm
import sklearn

import wandb
import warnings
import glob
import gc 
import os

from torchaudio_transforms import get_augmented_views
from metrics import *

torchaudio.set_audio_backend("sox_io") 
os.environ["IMAGEIO_FFMPEG_EXE"] = "/home/sgurram/anaconda3/bin/ffmpeg"
warnings.filterwarnings("ignore")

class AudioDataset(Dataset):

    def __init__(self, dataType):
        self.dataType = dataType
        self.dir = "/ssd/kinetics_audio/{}".format(dataType)
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
            return view1, view2, t1, t2, True

        except:
            return None, None, None, None, False

if __name__ == '__main__':
    ad = AudioDataset("train")
    bad_count = 0
    for i in tqdm(range(1000)):
        view1, view2, t1, t2, flag = ad.__getitem__(i)
        if not flag:
            bad_count += 1

    print(t1)
    print(t2)
    print(bad_count)

    # f = plt.figure()
    # f.add_subplot(1, 2, 1)
    # plt.imshow(view1)

    # f.add_subplot(1, 2, 2)
    # plt.imshow(view2)
    # plt.savefig("Desktop/log_dataloader_two_views.png")