import torch
import torchaudio
import torchvision
import torch.nn as nn
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

from encoder import *

class VideoBYOLightning(pl.LightningModule):

    def __init__(self,):
        super().__init__()

        self.encoder = BYOLEncoder()

    # Handle BYOL component: Implemetation from https://github.com/CannyLab/aai/blob/main/aai/research/gptcaptions/driver.py
    def on_before_zero_grad(self, _):
        self.model.update_moving_average()


    def training_step(self, batch, batch_idx):
        audio, video = batch
        x_online, y_online, x_target, y_target = self.encoder(audio, video)
        metrics = self.encoder.loss(x_online, y_online, x_target, y_target)

        return {'loss': metrics['total_loss'],
                'logs': metrics}

    def validation_step(self, batch, batch_idx):
        audio, video = batch
        x_online, y_online, x_target, y_target = self.encoder(audio, video)
        metrics = self.encoder.loss(x_online, y_online, x_target, y_target)

        return {'val_total_loss': metrics['total_loss'],
                'val_cosine_loss': metrics['cosine_loss'],
                'val_kldiv_loss': metrics['kldiv_loss'],
                'val_random_loss': metrics['random_loss'],}

    def test_step(self, batch, batch_idx):
        audio, video = batch
        x_online, y_online, x_target, y_target = self.encoder(audio, video)
        metrics = self.encoder.loss(x_online, y_online, x_target, y_target)

        return {'test_total_loss': metrics['total_loss'],
                'test_cosine_loss': metrics['cosine_loss'],
                'test_kldiv_loss': metrics['kldiv_loss'],
                'test_random_loss': metrics['random_loss'],}

    
    def validation_epoch_end(self, outputs):
        avg_total_loss = torch.stack([m['val_total_loss'] for m in outputs]).mean()
        avg_cosine_loss = torch.stack([m['val_cosine_loss'] for m in outputs]).mean()
        avg_kldiv_loss = torch.stack([m['val_kldiv_loss'] for m in outputs]).mean()
        avg_random_loss = torch.stack([m['val_random_loss'] for m in outputs]).mean()

        logs = {'val_total_loss': avg_total_loss,
                'val_cosine_loss': avg_cosine_loss,
                'val_kldiv_loss': avg_kldiv_loss,
                'val_random_loss': avg_random_loss,}

        return {'val_total_loss': avg_total_loss, 'log': logs}

    def train_dataloader(self):
        dataset = AudioVisualData(dataType='train')
        return torch.utils.data.DataLoader(
                                dataset,
                                batch_size=self.encoder._batch_size,
                                shuffle=True,
                                num_workers=8)

    def val_dataloader(self):
          dataset = AudioVisualData(dataType='val')
          return torch.utils.data.DataLoader(
                                dataset,
                                batch_size=self.encoder._batch_size,
                                shuffle=False,
                                num_workers=8)

    #test clipped folder does not exist on stout
    # def test_dataloader(self):
    #     dataset = AudioVisualData(dataType='test')
    #     return torch.utils.data.DataLoader(
    #                             dataset,
    #                             batch_size=self.encoder._batch_size,
    #                             shuffle=False,
    #                             num_workers=8)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.encoder._learning_rate)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=29, epochs=self.num_epochs)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.02)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70,80, 85, 90, 95], gamma=0.1)
        # return [optimizer], [scheduler]
        return optimizer
