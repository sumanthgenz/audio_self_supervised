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
from audio_byol.dataloader import AudioDataset


class Encoder(pl.LightningModule):

    def __init__(self):
        super(Net, self).__init__()
        self.num_features = 13
        self.input_size_half = 256
        self.input_size = 512
        self.seq_len = 500
        self.hidden_size_1 = 1024
        self.hidden_size_2 = 2048
        self.hidden_size_3 = 1024
        self.num_classes = 700
        self.num_epochs = 100
        self.batch_size = 16
        self.learning_rate = 0.00025
        self.attention_heads = 8
        self.n_layers = 16
        self.counter = 0

        self.trainPath = 'train'
        self.testPath = 'test'
        self.valPath = 'validate'
        self.classes = self.get_pickle("classes_dict")

        
        
        self.encode = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.attention_heads, dropout=0.15)
        self.transformer = nn.TransformerEncoder(self.encode, num_layers=self.n_layers)

        self.fc0 = nn.Linear(self.num_features, self.input_size_half)
        self.fc1 = nn.Linear(self.input_size_half, self.input_size)
        self.fc2 = nn.Linear(self.input_size, self.hidden_size_1) 
        self.fc3 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.fc4 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.fc5 = nn.Linear(self.hidden_size_3, self.num_classes)

        self.old_fc4 = nn.Linear(self.hidden_size_1, self.num_classes)


        self.norm1 = nn.LayerNorm(self.input_size_half)
        self.norm2 = nn.LayerNorm(self.input_size)
        self.norm3 = nn.LayerNorm(self.hidden_size_1)
        self.norm4 = nn.LayerNorm(self.hidden_size_2)
        self.norm5 = nn.LayerNorm(self.hidden_size_3)


        self.dropout = nn.Dropout(p=0.15)
        self.relu = nn.ReLU()

        self.loss = nn.CrossEntropyLoss()

        self.train_test_split_pct = 0.8
        self.train_dataset, self.test_dataset, self.val_dataset = self.prepare_data()
        self.conf_target = ()
        self.conf_pred = ()
        self.hist_count = 0

    def get_pickle(self, classPath):
            with open('Desktop/kinetics_{}.pickle'.format(classPath), 'rb') as handle:
                result = pickle.load(handle)
            return result

    def create_class_hists(self, data, name):
        hist_list = []
        for i in data.indices:
            hist_list.append(data.dataset[i][1])
        labels, values = zip(*Counter(hist_list).items())
        indexes = np.arange(len(labels))
        width = 1
        plt.bar(indexes, values, width)
        plt.xticks(indexes + width * 0.5, labels)
        plt.savefig(name)
        
    
    def forward(self, x, mask):
        expand = self.fc0(x)
        expand = self.norm1(self.dropout(expand))
        expand2 = self.fc1(expand)
        src = self.norm2(self.dropout(expand2))

        transform = self.transformer(src, src_key_padding_mask=mask)

        transform = self.relu(self.fc2(transform.permute(1, 0, 2).mean(dim=1)))
        transform = transform + self.dropout(transform)
        transform = self.norm3(transform)

        out = self.fc4(self.dropout(self.relu(self.fc3(transform))))
        out = transform + self.dropout(out)
        out = self.norm3(out)
        out = self.fc5(out)
        
        return out

    def prepare_data(self):
        train_dataset = AudioDataset(self.trainPath)
        test_dataset = AudioDataset(self.testPath)
        val_dataset = AudioDataset(self.valPath)
        return train_dataset, test_dataset, val_dataset
    
    def collate_fn(self, batch):
        # return torch.utils.data.dataloader.default_collate(batch)

        def trp(arr, n):
            arr = arr.tolist()
            seq = []
            for cmp in arr:
                new_cmp = cmp[:n] + [0.0]*(n-len(cmp))
                seq.append(new_cmp)
            result = np.transpose(np.array(seq)).tolist()
            return result

        def pad_masker(arr):
            zer = torch.zeros(self.num_features)
            ind = 0
            for t in arr:
                pad = torch.all(torch.eq(t, zer))
                if  pad == 1:
                    break
                ind += 1
            result = ([True]*ind) + ([False]*(self.seq_len - ind))
            return result
        
        batch = np.transpose(batch)
        data = list(filter(lambda x: x is not None, batch[0]))
        labels = list(filter(lambda x: x is not None, batch[1]))

        samples = torch.Tensor([(trp(t, self.seq_len))for t in data])
        mask = torch.Tensor([pad_masker(t) for t in samples])
        labels = torch.Tensor(labels).long()
        samples = samples.permute(1, 0, 2)

        return samples, labels, mask

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
