import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from efficientnet_pytorch import EfficientNet


import librosa
import openpyxl
import torch
import numpy
import numpy as np
import sklearn
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from datetime import datetime
import warnings
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
import pandas as pd 
from collections import Counter
import matplotlib.pyplot as plt
import wandb
import gc 

import torchaudio
from dataloader import AudioDataset

#Implementation from
# https://github.com/CVxTz/COLA_pytorch/blob/master/audio_encoder/encoder.py
class FeatureModel(torch.nn.Module):
    def __init__(self, drop_connect_rate=0.1):
        super(FeatureModel, self).__init__()

        self.cnn1 = torch.nn.Conv2d(1, 3, kernel_size=3)
        self.efficientnet = EfficientNet.from_name(
            "efficientnet-b0", include_top=False, drop_connect_rate=drop_connect_rate
        )

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.cnn1(x)
        x = self.efficientnet(x)

        y = x.squeeze(3).squeeze(2)
        return y

class Encoder(torch.nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self._model_dimension = 512

        self._fc1 = nn.Linear(1280, 512)
        self._fc2 = nn.Linear(512, 512, bias=False)

        self._norm1 = nn.LayerNorm(normalized_shape=512)
      
        self._dropout = nn.Dropout(p=0.15)
        self._tanh = torch.tanh()

        self._loss = nn.CrossEntropyLoss()

        self._encoder = FeatureModel(drop_connect_rate=0.1)


        #Implementation from
        #https://github.com/CannyLab/aai/blob/ddc76404bdfe15fb8218c31d9dc6859f3d5420db/aai/research/gptcaptions/models/encoders/predictive_byol.py
        self._representation_mlp = torch.nn.Sequential(
            torch.nn.Linear(self._model_dimension, self._model_dimension),
            torch.nn.BatchNorm1d(self._model_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self._model_dimension, self._model_dimension),
        )
        self._byol_predictor = torch.nn.Sequential(
            torch.nn.Linear(self._model_dimension, self._model_dimension),
            torch.nn.BatchNorm1d(self._model_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self._model_dimension, self._model_dimension),
        )

        self._translation_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * self._model_dimension, self._model_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self._model_dimension, self._model_dimension),
        )

        self._target_encoder = None
        self._target_networks_initialized = False
        self._ema_beta = ema_beta


    def get_paramaters(self,):
        params = []
        params += list(self._encoder.parameters())
        if self._target_encoder is not None:
            params += list(self._target_encoder.parameters())
        params += list(self._representation_mlp.parameters())
        params += list(self._byol_predictor.parameters())
        params += list(self._translation_mlp.parameters())

        return params

    def reset_moving_average(self):
        del self._target_encoder
        self._target_encoder = None

    def _ema_copy_model(self, online_model, target_model):
        for current_params, target_params in zip(online_model.parameters(), target_model.parameters()):
            old_weight, new_weight = target_params.data, current_params.data
            target_params.data = old_weight * self._ema_beta + (1 - self._ema_beta) * new_weight

    def update_moving_average(self):
        if self._target_encoder is not None:
            self._ema_copy_model(self._encoder, self._target_encoder)
    
    def forward(self, x):
        x1, x2 = x

        x1 = self._dropout(self._feature_model(x1))
        x1 = self._dropout(self.fc1(x1))
        x1 = self._dropout(self.tanh(self.norm1(x1)))

        x2 = self._dropout(self._feature_model(x2))
        x2 = self._dropout(self.fc1(x2))
        x2 = self._dropout(self.tanh(self.norm1(x2)))
        
        return x1, x2
    
