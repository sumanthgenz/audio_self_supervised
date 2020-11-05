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