import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger



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

from audio_self_supervised.encoder import Encoder
from audio_self_supervised.dataloader import AudioDataLoader

wandb_logger = WandbLogger(name='Remote_Audio_Transformer',project='kinetics-ablation')
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
# gc.collect()
# torch.cuda.empty_cache()


class ModalBYOL(pl.LightningModule):

    def __init__(self,):
        super().__init__()
        self.model = None


    def train_dataloader(self):
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, 
                                                        batch_size=self.batch_size, 
                                                        shuffle=False,
                                                        collate_fn=self.collate_fn,
                                                        num_workers=16)
        return self.train_loader


    def val_dataloader(self):
        self.test_loader = torch.utils.data.DataLoader(dataset=self.val_dataset, 
                                                        batch_size=self.batch_size, 
                                                        shuffle=False,
                                                        collate_fn=self.collate_fn,
                                                        num_workers=16)        
        return self.test_loader

    def training_step(self, batch, batch_idx):
        data, target, mask = batch
        mask = mask < 0
        output = self.forward(data, mask)

        loss = self.loss(output, target)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        data, target, mask = batch
        mask = mask < 0
        output = self.forward(data, mask)
        temp_loss = self.loss(output, target)
        pred = output.argmax(dim=1, keepdim=True)

        correct = pred.eq(target.view_as(pred)).sum().item()
        acc = torch.tensor(correct/self.batch_size)
        logs = {'val_loss': temp_loss, 'val_acc': acc}

        # if self.counter >= self.num_epochs-1:
            # wandb.sklearn.plot_confusion_matrix(target.cpu().numpy(), pred.flatten().cpu().numpy(), self.classes)

        return {'val_loss': temp_loss, 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        # print(outputs)
        self.counter += 1
        self.hist_count += 1
        avg_loss = torch.stack([m['val_loss'] for m in outputs]).mean()
        avg_acc = torch.stack([m['val_acc'] for m in outputs]).mean()
        logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'val_loss': avg_loss, 'log': logs}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=29, epochs=self.num_epochs)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.02)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70,80, 85, 90, 95], gamma=0.1)
        # return [optimizer], [scheduler]
        return optimizer


if __name__ == '__main__':
    

    model = Encoder()
    wandb_logger.watch(model, log='gradients', log_freq=10)
   
    # trainer = pl.Trainer(
    #     default_root_dir='/home/sgurram/good-checkpoint/', 
    #     auto_lr_find=True, gpus=[2,3], 
    #     overfit_batches=10, 
    #     max_epochs=100, 
    #     logger=wandb_logger, 
    #     accumulate_grad_batches=1, 
    #     distributed_backend='ddp')

   
    trainer = pl.Trainer(
        default_root_dir='/home/sgurram/good-checkpoint/', 
        gpus=[0, 1, 2,3], 
        overfit_batches=10, 
        max_epochs=50, 
        logger=wandb_logger, 
        accumulate_grad_batches=1, 
        distributed_backend='ddp')
    

    # trainer = pl.Trainer(
    #     default_root_dir='/home/sgurram/good-checkpoint/', 
    #     gpus=[0, 1, 2,3], 
    #     max_epochs=50, 
    #     logger=wandb_logger, 
    #     accumulate_grad_batches=200, 
    #     distributed_backend='ddp')

    # lr_finder = trainer.tuner.lr_find(model)
    # lr_finder.results
    # new_lr = lr_finder.suggestion()

    
    trainer.fit(model)