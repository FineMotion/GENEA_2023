from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from argparse import ArgumentParser

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader

from .model import PhaseAutoEncoder
from .dataset import AutoEncoderDataset


class PAESystem(pl.LightningModule):

    @staticmethod
    def add_system_args(parent_parser: ArgumentParser):
        arg_parser = ArgumentParser(parents=[parent_parser])
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

    def __init__(self, joints: int, channels: int, phases: int, window: float, fps: int, *args, **kwargs):
        super().__init__()
        self.model = PhaseAutoEncoder(input_channels=joints*channels, embedding_channels=phases,
                                      time_range=int(fps * window) + 1, channels_per_joint=channels, window=window)


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
