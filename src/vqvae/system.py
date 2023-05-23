import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from .model import VQVAE
from .dataset import GestureDataset


class VQVAESystem(pl.LightningModule):
    def __init__(self, num_embeddings, embedding_dim, input_dim, hidden_dim, max_frames, learning_rate=1e-3):
        super(VQVAESystem, self).__init__()

        self.vqvae = VQVAE(num_embeddings, embedding_dim, input_dim, hidden_dim, max_frames).double()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.vqvae(x)

    def training_step(self, batch, batch_idx):
        x = batch
        reconstruction, qloss = self(x)
        mse_loss = F.mse_loss(reconstruction, x)
        loss = mse_loss + qloss

        self.log('train/loss', loss)
        self.log('train/mse_loss', mse_loss)
        self.log('train/qloss', qloss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x = batch
        reconstruction, qloss = self(x)
        mse_loss = F.mse_loss(reconstruction, x)
        loss = mse_loss + qloss

        self.log('val/loss', loss)
        self.log('val/mse_loss', mse_loss)
        self.log('val/qloss', qloss)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class VQVAEDataModule(pl.LightningDataModule):
    def __init__(self, trn_data_path='clip_data/trn/ma_audio_i_bvh/gestures',
                 val_data_path='clip_data/val/ma_audio_i_bvh/gestures',
                 batch_size: int = 128):
        super().__init__()
        self.val_dataset = None
        self.trn_dataset = None

        self.train_path = Path(trn_data_path)
        self.val_path = Path(val_data_path)

        self.batch_size = batch_size

    def setup(self, stage=None) -> None:
        self.trn_dataset = GestureDataset(self.train_path)
        self.max_frames = self.trn_dataset.max_frames_len
        self.val_dataset = GestureDataset(self.val_path)

    def collate_fn(self, batch):
        vectors_padded = [torch.cat([b, torch.zeros((b.shape[0],
                                                     self.max_frames - b.shape[1]))], dim=1) for b in batch]
        return torch.stack(vectors_padded)

    def train_dataloader(self):
        return DataLoader(self.trn_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)