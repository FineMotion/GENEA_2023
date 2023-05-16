import torch
from torch.nn import CrossEntropyLoss
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import AudioGestureDataset
from .model import AudioEncoder, GestureEncoder, CLIPModel


class CLIPSystem(pl.LightningModule):
    def __init__(self, embedding_dim):
        super(CLIPSystem, self).__init__()
        self.audio_encoder = AudioEncoder(embedding_dim)
        self.gesture_encoder = GestureEncoder(embedding_dim)
        self.clip = CLIPModel(embedding_dim)
        self.loss = CrossEntropyLoss()

    def forward(self, audio, gestures):
        x = self.clip(audio, gestures)
        return x

    def training_step(self, batch, batch_idx):
        audio, gestures = batch
        audio_embeddings, gesture_embeddings = self.forward(audio, gestures)

        logits = (audio_embeddings @ gesture_embeddings.t()).softmax(dim=-1)
        targets = torch.arange(len(audio)).to(logits.device)
        loss = self.loss(logits, targets) / len(audio)

        self.log('train/loss', loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        audio, gestures = batch
        audio_embeddings, gesture_embeddings = self.forward(audio, gestures)

        logits = (audio_embeddings @ gesture_embeddings.t()).softmax(dim=-1)
        targets = torch.arange(len(audio)).to(logits.device)
        loss = self.loss(logits, targets) / len(audio)

        self.log('val/loss', loss)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class CLIPDataModule(pl.LightningDataModule):
    def __init__(self, trn_data_path='clips_data/trn/main-agent', val_data_path='clips_data/val/main-agent',
                 batch_size: int = 512):
        super().__init__()
        self.val_dataset = None
        self.trn_dataset = None

        self.train_audio_path = trn_data_path + "/audio"
        self.train_gestures_path = trn_data_path + "/gestures"

        self.val_audio_path = val_data_path + "/audio"
        self.val_gestures_path = val_data_path + "/gestures"

        self.batch_size = batch_size

    def setup(self, stage=None) -> None:
        self.trn_dataset = AudioGestureDataset(self.train_audio_path, self.train_gestures_path)
        self.val_dataset = AudioGestureDataset(self.val_audio_path, self.val_gestures_path)

    def collate_fn(self, batch):
        audios, gestures = zip(*batch)
        audios = torch.nn.utils.rnn.pad_sequence([audio for audio in audios], batch_first=True).double()
        gestures = torch.nn.utils.rnn.pad_sequence([gesture for gesture in gestures], batch_first=True).double()

        return audios, gestures

    def train_dataloader(self):
        return DataLoader(self.trn_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)