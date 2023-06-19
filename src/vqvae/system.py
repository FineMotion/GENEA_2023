import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from .model import VQVAE
from .dataset import GestureDataset


class VQVAESystem(pl.LightningModule):
    """
        PyTorch Lightning Module for training and validation of VQVAE model.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, input_dim: int, hidden_dim: int,
                 max_frames: int, learning_rate=1e-3) -> None:
        super(VQVAESystem, self).__init__()
        self.num_embeddings = num_embeddings

        self.vqvae = VQVAE(
            num_embeddings, embedding_dim, input_dim, hidden_dim, max_frames
        ).double()
        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vqvae(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        x = batch
        reconstruction, qloss, info = self(x, training=True)
        mse_loss = F.mse_loss(reconstruction, x)
        loss = mse_loss + qloss

        self.log("train/loss", loss)
        self.log("train/mse_loss", mse_loss)
        self.log("train/qloss", qloss)
        return {"loss": loss}

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        x = batch
        reconstruction, qloss, info = self(x, training=False)
        mse_loss = F.mse_loss(reconstruction, x)
        loss = mse_loss + qloss

        self.log("val/loss", loss)
        self.log("val/mse_loss", mse_loss)
        self.log("val/qloss", qloss)
        return {"loss": loss}

    def on_train_epoch_start(self, *args) -> None:
        n_entries = self.vqvae.vq.n_entries
        self.log("train/codebook_used_var", torch.var(n_entries.double()))

        # Reset unused codes to random values
        unused_codes = self.vqvae.get_unused_codes()
        codebook_unused_val = (unused_codes.shape[0] if len(unused_codes.shape) > 0 else 0) / self.num_embeddings
        self.log("train/codebook_used", 1 - codebook_unused_val)
        self.vqvae.reset_codes(unused_codes)

        self.vqvae.zero_n_entries()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class VQVAEDataModule(pl.LightningDataModule):
    """
        PyTorch Lightning DataModule for managing data loading and preparation.
    """

    def __init__(self,
                 trn_data_path: str = "clip_data_min_beats/trn",
                 val_data_path: str = "clip_data_min_beats/val",
                 batch_size: int = 128,
                 max_frames: int = 18,
                 num_workers: int = 8
                 ) -> None:
        super().__init__()

        self.train_path = Path(trn_data_path)
        self.val_path = Path(val_data_path)

        self.batch_size = batch_size
        self.max_frames = max_frames
        self.num_workers = num_workers

        self.val_dataset = None
        self.trn_dataset = None

    def setup(self, stage=None) -> None:
        self.trn_dataset = GestureDataset(self.train_path)
        self.val_dataset = GestureDataset(self.val_path)

        print(f"Train size: {len(self.trn_dataset)}")
        print(f"Val size: {len(self.val_dataset)}")

    def collate_fn(self, batch):
        vectors_padded = []
        for b in batch:
            ones = torch.ones(b.shape[0], self.max_frames - b.shape[1])
            last_val = b[:, -1].unsqueeze(1)
            last_val = last_val.expand_as(ones)

            vectors_padded.append(torch.cat([b, ones * last_val], dim=1))
        return torch.stack(vectors_padded)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.trn_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )
