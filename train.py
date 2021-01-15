import torch
import torchaudio
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


import warnings
import glob
from tqdm import tqdm
import wandb

from absl import app, flags


from lightning import *

wandb_logger = WandbLogger(name='run',project='kinetics-Video_BYOL')
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

#Define modes
flags.DEFINE_enum('mode',
                  default='train',
                  enum_values=['train, test'])

flags.DEFINE_string('log',
                  default='gradients')

flags.DEFINE_enum('backend',
                  default='ddp',
                  enum_values=['ddp, ddp2'])

#Define pl_trainer params
flags.DEFINE_string('root_dir',
                  default='/home/sgurram/Desktop/video_byol')

flags.DEFINE_integer('num_gpus',
                  default=2)

flags.DEFINE_integer('max_epochs',
                    default=100)

flags.DEFINE_integer('accum_grad_batches',
                    default=1)

flags.DEFINE_integer('log_freq',
                    default=10)

FLAGS = flags.FLAGS

def train_byol():
    model = VideoBYOLightning()
    wandb_logger.watch(model, 
            log=FLAGS.log, 
            log_freq=FLAGS.log_freq)

    trainer = pl.Trainer(
            default_root_dir=FLAGS.root_dir, 
            gpus=FLAGS.num_gpus, 
            max_epochs=FLAGS.max_epochs, 
            accumulate_grad_batches=FLAGS.accum_grad_batches, 
            distributed_backend=FLAGS.backend,
            logger=wandb_logger,)   
    
    trainer.fit(model)


if __name__ == '__main__':

    if FLAGS.mode == "train":
        train_byol()