import pytorch_lightning as pl
import torch.nn
from pathlib import Path

from torch.utils.data import DataLoader

from .model import SimpleAutoEncoder
from .dataset import SimpleMotionDataset
from src.utils.geodesic import GeodesicLoss
from src.utils.algem import rotmat_from_ortho6d


class SAESystem(pl.LightningModule):
    def __init__(self, input_dim, trn_folder: str, val_folder: str, loss_name: str, batch_size: int = 128):
        super().__init__()
        self.model = SimpleAutoEncoder(input_dim=input_dim, output_dim=input_dim // 6, hidden_size=input_dim // 3)
        self.loss_name = loss_name
        self.batch_size = batch_size
        if loss_name == 'mse':
            self.loss = torch.nn.MSELoss()
        elif loss_name == 'geodesic':
            self.loss = GeodesicLoss()

        self.trn_dataset = SimpleMotionDataset(Path(trn_folder).glob('*.npy'))
        self.val_dataset = SimpleMotionDataset(Path(val_folder).glob('*.npy'))

    def custom_loss(self, y, x):
        if self.loss_name == 'mse':
            return self.loss(y, x)
        elif self.loss_name == 'geodesic':
            bs = x.shape[0]
            joints = x.shape[1] // 6
            x = x.reshape(bs, joints, 6)
            y = y.reshape(bs, joints, 6)
            x = x.reshape(bs * joints, 6)
            y = y.reshape(bs * joints, 6)
            x = rotmat_from_ortho6d(x)
            y = rotmat_from_ortho6d(y)
            return self.loss(y, x)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch
        y = self.forward(x)
        loss = self.custom_loss(y, x)
        self.log('trn/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y = self.forward(x)
        loss = self.custom_loss(y, x)
        self.log('val/loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def train_dataloader(self):
        return DataLoader(self.trn_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.trn_dataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.val_dataset.collate_fn)
