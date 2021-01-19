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
flags.DEFINE_string('mode',
                  default='train',
                  help='train or test mode',)

flags.DEFINE_string('log',
                  default='gradients',
                  help='log modes',)

flags.DEFINE_string('backend',
                  default='ddp',
                  help='distributed backend modes',)


#Define pl_trainer params
flags.DEFINE_string('root_dir',
                  default='/home/sgurram/Desktop/video_byol',
                  help='root directory to save checkpoints',)


flags.DEFINE_integer('num_gpus',
                    default=2,
                    help='number of gpus to be used with distributed backend',)

flags.DEFINE_integer('max_epochs',
                    default=100,
                    help='number of epochs for training',)

flags.DEFINE_integer('accum_grad_batches',
                    default=1,
                    help='number of batches between gradient descent steps',)

flags.DEFINE_integer('log_freq',
                    default=10,
                    help='number of batches in between logging gradients',)

def train_byol():
    model = VideoBYOLightning()
    # wandb_logger.watch(model, 
    #         log=FLAGS.log, 
    #         log_freq=FLAGS.log_freq)

    # trainer = pl.Trainer(
    #         default_root_dir=FLAGS.root_dir, 
    #         gpus=FLAGS.num_gpus, 
    #         max_epochs=FLAGS.max_epochs, 
    #         accumulate_grad_batches=FLAGS.accum_grad_batches, 
    #         distributed_backend=FLAGS.backend,
    #         logger=wandb_logger,)   

    #     wandb_logger.watch(model, 
    #         log=FLAGS.log, 
    #         log_freq=FLAGS.log_freq)

    wandb_logger.watch(model, 
            log='gradients', 
            log_freq=10)

    trainer = pl.Trainer(
            default_root_dir='/home/sgurram/Desktop/video_byol', 
            gpus=[0, 1], 
            max_epochs=100, 
            accumulate_grad_batches=1,
            overfit_batches=10, 
            distributed_backend='ddp',
            logger=wandb_logger,) 
    
    trainer.fit(model)


if __name__ == '__main__':
    
    FLAGS = flags.FLAGS 
    # if FLAGS.mode == "train":
    #     train_byol()
    train_byol()