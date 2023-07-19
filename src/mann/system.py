import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from pathlib import Path
from argparse import ArgumentParser

from .model import GatingNetwork, MotionPredictionNetwork, AudioEncoder
from .dataset import ModeAdaptiveDataset
from src.training.adamw import AdamW
from src.training.sgdr import CyclicRWithRestarts


class ModeAdaptiveSystem(pl.LightningModule):
    @staticmethod
    def add_system_args(parent_parser: ArgumentParser):
        arg_parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # arg_parser.add_argument("--phases", type=int, default=8, help="Number of trained phases")
        arg_parser.add_argument("--samples", type=int, default=13,
                                help="Number of uniformly distributed features inside time window")
        arg_parser.add_argument("--window_size", type=float, default=2.0, help="Size of time window in seconds")
        arg_parser.add_argument("--fps", type=int, default=30, help="Motion data framerate")
        arg_parser.add_argument("--audio_fps", type=int, default=30, help="Audio data framerate")
        arg_parser.add_argument("--num_workers", type=int, default=1, help="Number of workers in train DataLoader")
        arg_parser.add_argument("--vel_included", action="store_true",
                                help="Flag to indicate phase velocity is already included")
        return arg_parser

    def __init__(self, trn_folder: str, val_folder: str, gating_hidden: int = 64, main_hidden: int = 1024,
                 experts: int = 8, dropout: float = 0.3, samples: int = 13, window_size: float = 2.0, fps: int = 30,
                 audio_fps: int = 30, learning_rate=1e-4, batch_size: int = 32, num_workers: int = 1, vel_included=False):
        super().__init__()
        # gating_input = phases * 2 * samples
        trn_files = Path(trn_folder).glob('*.npz') if Path(trn_folder).is_dir() else [trn_folder]
        logging.info(f"Phase velocities included: {vel_included}")
        # One sample to initialize shapes
        self.trn_dataset = ModeAdaptiveDataset(
            trn_files, samples, window_size, fps, audio_fps, vel_included=vel_included
        )
        self.val_dataset = ModeAdaptiveDataset(
            Path(val_folder).glob('*.npz'), samples, window_size, fps, audio_fps, vel_included=vel_included
        ) if val_folder is not None else None  # None for inference
        x, a, y, p = self.trn_dataset[0]
        main_input = x.shape[-1]
        main_output = y.shape[-1]
        gating_input = p.shape[-1]
        audio_input = a.shape[-1]
        audio_hidden = main_input

        self.gating = GatingNetwork(gating_input, gating_hidden, experts, dropout)
        self.motion = MotionPredictionNetwork(2*main_input, main_hidden, main_output, experts, dropout)
        self.audio_encoder = AudioEncoder(audio_input, audio_hidden, audio_hidden)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.optimizer = None
        self.scheduler = None

    def forward(self, x, a, p):
        a = self.audio_encoder(a)
        w = self.gating(p)
        x = torch.cat([x, a], dim=-1)
        y = self.motion(x, w)
        return y, w

    def training_step(self, batch, batch_idx):
        x,a, y, p = batch
        pred, _ = self.forward(x, a, p)
        loss = self.custom_loss(y, pred)
        self.log('train/loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, a, y, p = batch
        pred, _ = self.forward(x, a, p)
        loss = self.custom_loss(y, pred)
        self.log('val/loss', loss)
        return {'loss': loss}

    def custom_loss(self, y, pred):
        return F.mse_loss(pred, y)

    # AdamW and CyclicRWR
    # def configure_optimizers(self):
    #     self.optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
    #     self.scheduler = CyclicRWithRestarts(
    #         optimizer=self.optimizer, batch_size=self.batch_size, epoch_size=len(self.trn_dataset),
    #         restart_period=10, t_mult=2, policy="cosine", verbose=True
    #     )
    #     return [self.optimizer], [
    #         {"scheduler": self.scheduler, "interval": "step"}
    #     ]

    # def on_train_epoch_start(self) -> None:
    #     # call epoch step on scheduler
    #     self.scheduler.epoch_step()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # FOR PARALLEL WORKERS
        # return DataLoader(self.trn_dataset, batch_size=self.batch_size, shuffle=True,
        #               collate_fn=self.trn_dataset.collate_fn, num_workers=self.num_workers, persistent_workers=True)
        return DataLoader(self.trn_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.trn_dataset.collate_fn)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.val_dataset.collate_fn)
