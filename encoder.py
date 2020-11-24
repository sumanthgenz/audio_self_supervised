import torch
import torchaudio
import torchvision
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet

import numpy as np
import pandas as pd 
import warnings
import glob
from tqdm import tqdm
import pickle
from collections import Counter
import copy
import os

from dataloader import *
from metrics import *


#Implementation from
# https://github.com/CVxTz/COLA_pytorch/blob/master/audio_encoder/encoder.py
class FeatureModel(torch.nn.Module):
    def __init__(self, dropout=0.1, model_dimension=1024):
        super(FeatureModel, self).__init__()

        self._cnn1 = torch.nn.Conv2d(
                                in_channels=1, 
                                out_channels=3, 
                                kernel_size=3)

        self._efficientnet = EfficientNet.from_name(
                                "efficientnet-b0", 
                                include_top=False, 
                                drop_connect_rate=dropout)

        self._fc1 = nn.Linear(1280, model_dimension)

        self._dropout = torch.nn.Dropout(p=dropout)

        self._layer_norm = torch.nn.LayerNorm(normalized_shape=model_dimension)


    def forward(self, x):
        #Input B * C * H * W
        x = x.unsqueeze(1)

        x = self._cnn1(x)
        x = self._efficientnet(x)
        x =  x.squeeze(3).squeeze(2)
        x = self._dropout(self._fc1(x))
        x = self._dropout(torch.tanh(self._layer_norm(x)))

        #Output B * D, D=1024
        return x

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

        self._encoder = FeatureModel(
                                dropout=0.1,
                                model_dimension=self._model_dimension)


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


    def byol_encode(self, x, online=True):
        if online:
            x = self._encoder(x)
        else:
            if not self._target_networks_initialized:
                self._target_encoder = copy.deepcopy(self._encoder)
                self._target_networks_initialized = True
            x = self._target_encoder(x)
        x = self._representation_mlp(x)
        return x

    
    def forward(self, x):
        x1, x2 = x

        x1 = self._feature_model(x1)
        x2 = self._feature_model(x2)

        return x1, x2
    
