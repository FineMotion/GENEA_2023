import logging
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from argparse import ArgumentParser

import torch.nn
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader

from .model import PhaseAutoEncoder
from .dataset import AutoEncoderDataset
from src.training.adamw import AdamW
from src.training.sgdr import CyclicRWithRestarts
from src.training.algem import rotmat_from_ortho6d
from src.training.geodesic import GeodesicLoss


class PAESystem(pl.LightningModule):

    @staticmethod
    def add_system_args(parent_parser: ArgumentParser):
        arg_parser = ArgumentParser(parents=[parent_parser], add_help=False)
        arg_parser.add_argument('--joints', type=int, default=26,
                                help="Number of joints")
        arg_parser.add_argument('--channels', type=int, default=3,
                                help="Degrees of freedom for joint")
        arg_parser.add_argument("--fps", type=int, default=30,
                                help="Framerate of animation")
        arg_parser.add_argument("--phases", type=int, default=8,
                                help="Number of phases")
        arg_parser.add_argument("--window", type=float, default=2.0,
                                help="Size of time window in seconds")
        arg_parser.add_argument("--add_root", action="store_true",
                                help="Option for additional feature with different number of channels")
        arg_parser.add_argument("--loss", choices=["mse", "geodesic"], default="mse")
        return arg_parser

    def __init__(self, joints: int, channels: int, phases: int, window: float, fps: int, learning_rate: float,
                 batch_size: int, trn_folder: str, val_folder: str, add_root: bool = False, loss_name: str = "mse",
                 *args, **kwargs):
        super().__init__()

        input_channels = joints*channels
        if add_root:
            input_channels += 3
        self.model = PhaseAutoEncoder(input_channels=input_channels, embedding_channels=phases,
                                      time_range=int(fps * window) + 1, channels_per_joint=channels, window=window,
                                      add_root=add_root)
        self.mse = torch.nn.MSELoss()
        self.loss_name = loss_name
        if loss_name == "geodesic":
            self.geodesic = GeodesicLoss()
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # None folders for inference
        self.trn_dataset = AutoEncoderDataset(
            Path(trn_folder).glob('*.npy'), window, fps) if trn_folder is not None else None
        self.val_dataset = AutoEncoderDataset(
            Path(val_folder).glob('*.npy'), window, fps) if val_folder is not None else None

        self.optimizer = None
        self.scheduler = None

    def custom_loss(self, y, x):
        """
        Custom loss with geodesic and MSE components
        """
        if self.loss_name == "mse":
            mse = self.mse(y, x)
            return mse, mse, None
        # batch_size, seq_len, channels (joints * 6 + 3)
        x_root = x[:, :, -3:]
        y_root = y[:, :, -3:]
        mse = self.mse(y_root, x_root)

        x_rot = x[:, :, :-3]  # bs, sl, joints * 6
        y_rot = y[:, :, :-3]  # bs, sl, joints * 6

        bs, sl, channels = x_rot.shape
        joints = channels // 6

        x_rot = x_rot.reshape(bs, sl, joints, 6)  # bs, sl, joints, 6
        y_rot = y_rot.reshape(bs, sl, joints, 6)  # bs, sl, joints, 6

        x_rot = x_rot.reshape(bs * sl * joints, 6)  # bs * sl * joints, 6
        y_rot = y_rot.reshape(bs * sl * joints, 6)  # bs * sl * joints, 6

        x_rotmats = rotmat_from_ortho6d(x_rot)
        y_rotmats = rotmat_from_ortho6d(y_rot)

        geodesic = self.geodesic(y_rotmats, x_rotmats)
        loss = mse + geodesic
        return loss, mse, geodesic

    def forward(self, x):
        # batch_size, seq_len, channels
        y, latent, signal, params = self.model(x)
        return y, latent, signal, params

    def training_step(self, batch, batch_idx):
        x = batch
        y, latent, signal, params = self.forward(x)
        loss, mse, geodesic = self.custom_loss(y, x)

        self.log('trn/loss', loss)
        self.log('trn/mse', mse)
        if geodesic is not None:
            self.log('trn/geodesic', geodesic)

        t_cur = self.scheduler.t_epoch + self.scheduler.batch_increments[self.scheduler.iteration]
        for i, (lr, wd) in enumerate(self.scheduler.get_lr(t_cur)):
            self.log(f'trn/lr_{i}', lr)
            self.log(f'trn/wd_{i}', wd)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y, latent, signal, params = self.forward(x)
        loss, mse, geodesic = self.custom_loss(y, x)

        self.log('val/loss', loss)
        self.log('val/mse', mse)
        if geodesic is not None:
            self.log('val/geodesic', geodesic)
        return loss

    def configure_optimizers(self):
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.scheduler = CyclicRWithRestarts(
            optimizer=self.optimizer, batch_size=self.batch_size, epoch_size=len(self.trn_dataset), restart_period=10,
            t_mult=2, policy="cosine", verbose=True
        )
        return [self.optimizer], [
            {"scheduler": self.scheduler, "interval": "step"},

        ]

    def on_train_epoch_start(self) -> None:
        # call epoch step on scheduler
        self.scheduler.epoch_step()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        logging.info(f"Batch size: {self.batch_size}")
        return DataLoader(self.trn_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.trn_dataset.collate_fn)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.val_dataset.collate_fn)


class PAEDataModule(pl.LightningDataModule):
    def __init__(self, trn_folder: str, val_folder: str, window: float = 2.0, fps: int = 30, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.window = window
        self.fps = fps
        self.trn_samples = Path(trn_folder).glob('*.npy')
        self.val_samples = Path(val_folder).glob('*.npy')
        self.trn_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.trn_dataset = AutoEncoderDataset(self.trn_samples, self.window, self.fps)
        self.val_dataset = AutoEncoderDataset(self.val_samples, self.window, self.fps)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.trn_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.trn_dataset.collate_fn)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.val_dataset.collate_fn)
